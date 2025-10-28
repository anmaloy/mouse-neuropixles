import matplotlib

# matplotlib.use('TkAgg')  # Set Matplotlib backend for interactive plots
matplotlib.use("Agg")  # Avoids opening GUI windows
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import math
import numpy as np
import os
import random
import scipy.io
import shutil
import spikeinterface.extractors as se
import spikeinterface.core as sic
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
from spikeinterface import extract_waveforms
from probeinterface.plotting import plot_probe
from scipy.signal import peak_prominences
import scipy.stats as stats

class NeuralDataProcessor:
    """Class for processing and analyzing neural data from SpikeGLX recordings."""

    def __init__(self, config):
        """Initialize the processor with a given experiment configuration."""
        self.config = config
        self.recording = None
        self.nidaq_recording = None
        self.recording_concat = None
        self.nidaq_concat = None
        self.sorting = None
        self.analyzer = None
        self.spike_times = None
        self.nidaq_data = None
        self.directions = None

        # Ensure processing directories exist
        os.makedirs(self.config["processing_folder"], exist_ok=True)
        for path in self.config["paths"].values():
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Cleanup previous processing files if rerun is enabled
        if self.config["rerun"]:
            self.cleanup_previous_processing()

    def cleanup_previous_processing(self):
        """Deletes existing processed files if rerun is enabled."""
        print("Rerun enabled. Deleting existing processed files...")
        processing_dirs = [
            self.config["paths"]["preprocessed_ap"],
            self.config["paths"]["preprocessed_NIDAQ"],
            self.config["paths"]["kilosort"],
            self.config["paths"]["mua"],
            self.config["paths"]["waveforms"],
            self.config["paths"]["analyzer"],
        ]
        for path in processing_dirs:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
                print(f"Deleted: {path}")

    def detect_light_transitions(self, signal, trim_front=0, trim_end=0):
        """
        Detects onset and offset points in the photodiode signal based on crossings of a dynamic threshold.
        Refines detected times by moving backward to the point where the signal transitions from stable to rapid change.
        Ensures that onset-offset pairs are within 50ms of 500ms apart.
        Allows for trimming of detected events at the beginning and end.

        Parameters:
        - signal: The photodiode signal.
        - trim_front: Time (in seconds) at the start of the recording where no detections are allowed.
        - trim_end: Time (in seconds) at the end of the recording where no detections are allowed.

        Returns:
        - paired_onsets: Array of refined onset times.
        - paired_offsets: Array of refined offset times.
        """
        fs = self.nidaq_recording.get_sampling_frequency()
        time_vector = np.arange(len(signal)) / fs

        # Use the middle 20 seconds of the recording to compute a more stable baseline
        total_duration = len(signal) / fs
        mid_start_time = (total_duration / 2) - 10
        mid_end_time = (total_duration / 2) + 10
        mid_start_idx = int(mid_start_time * fs)
        mid_end_idx = int(mid_end_time * fs)
        middle_mean = np.mean(signal[mid_start_idx:mid_end_idx]) + 0.66*np.std(signal[mid_start_idx:mid_end_idx])
        above_mean = signal > middle_mean

        # Compute derivatives
        first_derivative = np.diff(signal)
        second_derivative = np.diff(first_derivative)

        onsets = []
        offsets = []

        for i in range(1, len(signal)):
            if above_mean[i] and not above_mean[i - 1]:
                if time_vector[i] >= trim_front:
                    onsets.append(time_vector[i])
            elif not above_mean[i] and above_mean[i - 1]:
                if time_vector[i] <= (time_vector[-1] - trim_end):
                    offsets.append(time_vector[i])

        def refine_time(event_time, search_backward_ms=4):
            search_backward = int(search_backward_ms * fs / 1000)  # Convert ms to samples
            idx = np.searchsorted(time_vector, event_time) - 1
            search_start = max(0, idx - search_backward)
            search_region = second_derivative[search_start:idx]

            if len(search_region) > 0:
                if signal[idx] > middle_mean:
                    max_idx = np.argmin(search_region)  # Onset: steepest rise
                else:
                    max_idx = np.argmax(search_region)  # Offset: steepest fall
                return time_vector[search_start + max_idx]
            return event_time

        # Ensure onsets and offsets are correctly paired and refine times
        paired_onsets = []
        paired_offsets = []
        offset_idx = 0
        target_duration = 0.5  # 500ms
        tolerance = 0.05  # 50ms

        for onset in onsets:
            while offset_idx < len(offsets) and offsets[offset_idx] < onset:
                offset_idx += 1
            if offset_idx < len(offsets):
                duration = offsets[offset_idx] - onset
                if abs(duration - target_duration) <= tolerance:
                    refined_onset = refine_time(onset)
                    refined_offset = refine_time(offsets[offset_idx])
                    paired_onsets.append(refined_onset)
                    paired_offsets.append(refined_offset)
                offset_idx += 1

        print(f"\tDetected {len(paired_onsets)} valid onset-offset pairs.")
        return np.array(paired_onsets), np.array(paired_offsets)

    def detect_light_transitions_old(self, signal, onset_factor=4, offset_factor=2, min_separation=0.2,
                                 amplitude_filter=0.05, pair_tolerance=0.05):
        """
        Detects onset and offset points in the photodiode signal based
        on the rate of amplitude change.

        Parameters:
        - signal: The photodiode signal.
        - onset_threshold: The rate threshold for detecting an onset.
        - offset_threshold: The rate threshold for detecting an offset.
        - separation: Minimum time (in seconds) between consecutive detections.
        - amplitude_filter: Minimum relative amplitude change required to consider a transition.
        - pair_tolerance: Allowed deviation (in seconds) from the expected 0.5s onset-offset pairing.
        """
        fs = self.nidaq_recording.get_sampling_frequency()
        time_vector = np.arange(len(signal)) / fs

        # Compute first derivative (rate of change)
        first_derivative = np.diff(signal)
        abs_diff = np.abs(first_derivative)

        # Dynamically set thresholds based on signal standard deviation
        onset_threshold = onset_factor * np.std(first_derivative)
        offset_threshold = offset_factor * np.std(first_derivative)

        # Identify strong upward transitions (onsets)
        onset_candidates = np.where(first_derivative > onset_threshold)[0]
        offset_candidates = np.where(first_derivative < -offset_threshold)[0]

        # Ensure only the first strong change in each transition is selected
        filtered_onsets = []
        last_onset = -np.inf
        for idx in onset_candidates:
            if abs_diff[idx] > onset_threshold and (signal[idx + 1] - signal[idx]) > amplitude_filter * (
                    np.max(signal) - np.min(signal)):
                onset_time = time_vector[idx]
                if onset_time - last_onset >= min_separation:
                    filtered_onsets.append(onset_time)
                    last_onset = onset_time

        filtered_offsets = []
        last_offset = -np.inf
        for idx in offset_candidates:
            if idx + 5 < len(signal) and abs_diff[idx] > offset_threshold and (
                    signal[idx] - signal[idx + 5]) > amplitude_filter * (np.max(signal) - np.min(signal)):
                offset_time = time_vector[idx]
                if offset_time - last_offset >= min_separation:
                    filtered_offsets.append(offset_time)
                    last_offset = offset_time

        # Ensure onsets and offsets are correctly paired within expected 0.5s (+- pair_tolerance)
        paired_onsets = []
        paired_offsets = []

        offset_idx = 0
        for onset in filtered_onsets:
            while offset_idx < len(filtered_offsets) and filtered_offsets[offset_idx] < onset:
                offset_idx += 1
            if offset_idx < len(filtered_offsets) and abs(
                    filtered_offsets[offset_idx] - (onset + 0.5)) <= pair_tolerance:
                paired_onsets.append(onset)
                paired_offsets.append(filtered_offsets[offset_idx])
                offset_idx += 1

        print(f"\tDetected {len(paired_onsets)} valid onset-offset pairs.")
        return np.array(paired_onsets), np.array(paired_offsets)

    def detect_range(self):
        """Determine the expected range of channels based on insertion depth and target depth."""
        insertion_depth = self.config.get("insertion_depth", 0)  # Depth in um
        target_depth = self.config.get("target_depth", 0)  # Target depth in um
        row_spacing = 20  # um center-to-center distance between rows
        channels_per_row = 2  # Two channels per row
        needle_tip_length = 175  # um length of the needle tip

        # Compute row 1 depth
        row_1_depth = insertion_depth - needle_tip_length

        # Compute the target row index based on depth difference
        target_row = int(round((row_1_depth - target_depth) / row_spacing))
        print(f"Target channels: {target_row*channels_per_row} : {target_row*channels_per_row + 1}")
        target_channels = [target_row * channels_per_row, target_row * channels_per_row + 1]

        # Define the 40-channel-wide range centered around the target row
        ch_min = max(0, target_channels[0] - 20)
        ch_max = target_channels[1] + 19
        self.config["channels_sample"] = (ch_min, ch_max)

    def detect_sync_pulses(self, signal, threshold=None, pause_duration=3.0):
        """
        Detects the 10-second sync pulses in channel 8, ensuring only the first pulse
        in each burst is recorded, then pausing detection for a set duration.

        Parameters:
        - signal: The sync pulse signal.
        - threshold: The detection threshold (default: 50% of max signal).
        - pause_duration: The time (in seconds) to ignore after detecting a pulse.
        """
        if threshold is None:
            threshold = np.max(signal) * 0.5  # Use 50% of max signal as threshold

        fs = self.nidaq_recording.get_sampling_frequency()
        pulse_indices = np.where(signal > threshold)[0]
        pulse_times = pulse_indices / fs

        # Filter out pulses that occur within pause_duration of the last detected pulse
        filtered_pulses = []
        last_pulse_time = -np.inf
        for pulse_time in pulse_times:
            if pulse_time - last_pulse_time >= pause_duration:
                filtered_pulses.append(pulse_time)
                last_pulse_time = pulse_time

        print(f"\tDetected {len(filtered_pulses)} sync pulses.")
        return filtered_pulses

    def detect_peaks(self, time_window=None, distance=10):
        """
        Detects MUA spike peaks using prominence-based detection.
        Supports both negative and positive peak detection.

        Parameters:
            - time_window: (start, end) in seconds. If None, processes full recording.
            - distance: Minimum time in milliseconds between detected spikes.
        """
        fs = self.recording.get_sampling_frequency()
        num_channels = self.recording.get_num_channels()  # Process all channels
        direction = self.config.get("direction")

        # Determine the range of data to process
        if time_window is None:
            start_frame, end_frame = 0, self.recording.get_num_frames()
        else:
            start_frame = int(time_window[0] * fs)
            end_frame = int(time_window[1] * fs)

        # Extract traces for all channels
        traces = self.recording.get_traces(start_frame=start_frame, end_frame=end_frame)
        time_vector = np.arange(len(traces)) / fs  # Generate correct time mapping

        spike_data = {}

        for ch in range(num_channels):  # Process all channels
            centered_trace = traces[:, ch] - np.median(traces[:, ch])
            mad = np.median(np.abs(centered_trace)) / 0.6745

            if direction == "negative":
                threshold = np.median(traces[:, ch]) - (5.0 * mad)  # Lower threshold for downward spikes
            elif direction == "positive":
                threshold = np.median(traces[:, ch]) + (5.0 * mad)  # Upper threshold for upward spikes
            else:
                raise ValueError("Invalid direction! Choose 'negative' or 'positive'.")

            search_signal = traces[:, ch]

            # Step 1: Detect where the signal crosses the threshold
            if direction == "positive":
                threshold_crossings = np.where(search_signal > threshold)[0]
            else:  # "negative"
                threshold_crossings = np.where(search_signal < threshold)[0]

            # Step 2: Find the true peak near each threshold crossing
            peak_indices = []
            search_window = int(1.5 * fs / 1000)  # Search ±1.5 ms window

            for crossing in threshold_crossings:
                start = max(crossing - search_window, 0)
                end = min(crossing + search_window, len(search_signal))

                # Find the actual peak in this region
                if direction == "positive":
                    peak_idx = np.argmax(search_signal[start:end]) + start
                else:  # "negative"
                    peak_idx = np.argmin(search_signal[start:end]) + start

                if not peak_indices or peak_idx != peak_indices[-1]:
                    peak_indices.append(peak_idx)

            # Step 3: Convert peak indices to timestamps
            spike_times = time_vector[peak_indices]

            # Step 4: Compute prominence for these peaks
            prominence = peak_prominences(search_signal, peak_indices)[0]

            # Store results
            spike_data[ch] = {
                "peaks": np.array(peak_indices),
                "spike_times": spike_times,
                "threshold": threshold,
                "prominence": prominence
            }

        return spike_data

    def extract_mua_spikes(self):
        """Extracts Multi-Unit Activity (MUA) spikes using the selected method."""
        mua_path = self.config["paths"]["mua"]

        if os.path.exists(mua_path):
            print(f"Loading precomputed MUA spikes from {mua_path}...")
            self.spike_times = np.load(mua_path, allow_pickle=True).item()
            return

        print(f"Extracting MUA spikes using method: **{self.config['mua_method']}**")

        fs = self.recording.get_sampling_frequency()

        if self.config["mua_method"] == "threshold":
            print("Using **threshold-based** peak detection...")
            self.spike_times = self.detect_peaks(time_window=None)

        elif self.config["mua_method"] == "kilosort":
            print("Using **Kilosort-sorted spike trains** for MUA detection...")
            self.spike_times = {
                unit: self.sorting.get_unit_spike_train(unit) / fs
                for unit in self.sorting.unit_ids
            }

        else:
            raise ValueError("Invalid `mua_method`! Choose between 'threshold' or 'kilosort'.")

        np.save(mua_path, self.spike_times)
        print(f"MUA spike times saved to {mua_path}.")

    def extract_waveforms_data(self):
        """Extracts waveform data using built-in SpikeInterface functions."""
        waveform_folder = self.config["paths"]["waveforms"]

        if os.path.exists(waveform_folder):
            print("Loading existing waveforms...")
            we = sic.load(waveform_folder)
        else:
            print("Extracting waveforms...")
            we = extract_waveforms(
                self.recording_concat,
                self.sorting,
                folder=waveform_folder,
                ms_before=1.5,
                ms_after=2.0,
                max_spikes_per_unit=200,  # Reduce number of waveforms for clarity
                load_if_exists=None
            )

        return we

    def generate_synthetic_waveforms(self, num_spikes=100, num_channels=5, num_samples=50):
        """
        Generate synthetic spike waveforms for testing the plotting function.
        Each waveform is a simple negative peak followed by a return to baseline.
        Used in testing.
        """
        np.random.seed(42)  # For reproducibility

        # Generate time vector
        time_vector = np.linspace(0, 1.5, num_samples)  # Simulated time in ms

        # Generate synthetic waveforms: Gaussian-shaped negative peaks
        waveforms = np.exp(-((time_vector - 0.75) ** 2) / 0.05) * -50  # Baseline at 0, peak at -50 µV

        # Add variability across channels
        all_waveforms = np.tile(waveforms,
                                (num_spikes, num_channels, 1))  # Shape: (num_spikes, num_channels, num_samples)

        # Add noise for realism
        noise = np.random.normal(scale=5, size=all_waveforms.shape)
        all_waveforms += noise

        return all_waveforms, time_vector

    def get_autocorrelation(self, spike_times, bin_size=0.001, max_lag=0.05):
        """
        Computes the auto-correlation of spike times within ±max_lag, using a specified bin_size.

        Parameters
        ----------
        spike_times : np.ndarray
            1D array of spike times in seconds (sorted in ascending order).
        bin_size : float
            Width of the bins in seconds (default 1 ms).
        max_lag : float
            Maximum lag to compute the autocorrelation, in seconds (default 50 ms).

        Returns
        -------
        bins : np.ndarray
            Array of bin edges, spanning -max_lag to +max_lag.
        acorr : np.ndarray
            Autocorrelation histogram counts for each bin.
        """
        # Safety checks
        if len(spike_times) < 2:
            # With fewer than 2 spikes, we can’t form a meaningful autocorrelation
            bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
            acorr = np.zeros(len(bins) - 1)
            return bins, acorr

        # Optionally downsample if needed to avoid O(N^2) on huge spike trains
        # (Comment out or adjust the fraction below as desired)
        max_spikes_for_autocorr = 20000
        if len(spike_times) > max_spikes_for_autocorr:
            spike_times = np.random.choice(spike_times, size=max_spikes_for_autocorr, replace=False)
            spike_times = np.sort(spike_times)

        # Prepare bins
        bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
        half_window = max_lag

        # We’ll collect all pairwise differences that fall within ±max_lag
        diffs = []
        # A simple approach: for each spike, only compare to neighbors within ±max_lag
        # to avoid scanning every other spike in the entire array.
        idx_start = 0
        for i, t in enumerate(spike_times):
            # Move idx_start forward while spike_times[idx_start] < t - half_window
            while spike_times[idx_start] < t - half_window:
                idx_start += 1

            # Expand around t within ± half_window
            # We only look forward until we exceed t + half_window
            j = idx_start
            while j < len(spike_times) and spike_times[j] <= t + half_window:
                dt = spike_times[j] - t
                # Exclude zero-lag (the spike against itself)
                if dt != 0.0:
                    diffs.append(dt)
                j += 1

        # Convert to an array
        diffs = np.array(diffs)

        # Bin the differences
        acorr, _ = np.histogram(diffs, bins=bins)

        return bins, acorr

    def get_best_channels(self):
        """Finds the best channel for each unit using Kilosort template data.

        Returns:
            dict: {unit_id: best_channel}
        """
        if not self.analyzer:
            raise ValueError("SortingAnalyzer not loaded. Run 'process_analyzer()' first.")

        templates = self.analyzer.get_extension("templates").get_data("median")  # Shape: (units, channels, time)
        num_units, num_channels, _ = templates.shape

        best_channel_map = {}
        for unit in range(num_units):
            template_waveform = templates[unit]  # Shape: (channels, time)
            max_amplitudes = np.max(np.abs(template_waveform), axis=1)  # Peak amplitude per channel
            best_channel = np.argmax(max_amplitudes)  # Channel with highest response
            best_channel_map[unit] = best_channel

        return best_channel_map

    def get_channel_layout(self):
        """
        Returns a dictionary mapping each channel ID to (row, col)
        based on the physical x,y positions of the contacts in the probe.

        Assumes a probe with exactly 2 columns (i.e. two distinct x-coordinates).
        Channels in each column are sorted by ascending y.

        Example output:
          {
             17: (0, 0),    # channel 17 is row=0, col=0
             42: (1, 0),
             3:  (2, 0),
             19: (0, 1),
             27: (1, 1),
             ...
          }
        """
        probe = self.recording.get_probe()
        channel_ids = self.recording.channel_ids
        contact_positions = probe.contact_positions  # shape = (num_channels, 2), each row is (x, y)

        # Build array so each element is [channel_id, x, y]
        ch_xy = []
        for i, ch_id in enumerate(channel_ids):
            x, y = contact_positions[i]
            ch_xy.append([ch_id, x, y])
        ch_xy = np.array(ch_xy, dtype=object)

        # Identify the unique x-values. We assume exactly 2 columns:
        unique_xs = np.unique(ch_xy[:, 1])  # the x-coordinates
        if len(unique_xs) < 2:
            raise ValueError(
                "Probe appears to have fewer than 2 unique x-coordinates. "
                "Update the logic for your actual probe geometry."
            )

        # Sort x-values so the smaller x is col=0, the next is col=1, etc.
        unique_xs = np.sort(unique_xs)
        col_map = {xval: col_idx for col_idx, xval in enumerate(unique_xs)}

        # Collect channels for each column
        col_channels = [[] for _ in unique_xs]
        for row in ch_xy:
            ch_id, x, y = row
            # Find which column index to use based on the x-coord
            # If your probe has more than 2 distinct x's, col_map will simply assign
            # col=0, col=1, col=2, etc. as needed.
            col_idx = col_map[x]
            col_channels[col_idx].append((ch_id, float(y)))

        # Sort each column's channels by ascending y
        for i in range(len(col_channels)):
            col_channels[i].sort(key=lambda item: item[1])  # sort by y

        # Now assign row indices to each channel in ascending order of y
        layout = {}
        for col_idx, ch_list in enumerate(col_channels):
            for row_idx, (ch_id, y_val) in enumerate(ch_list):
                layout[ch_id] = (row_idx, col_idx)

        return layout

    def get_channel_spike_dict(self, data_structure):
        """
        Convert the per-channel trial list from process_segments()
        into a nested dictionary with the hierarchy:
           unit_spike_dict[unit_id][channel_id][direction] -> list of spike-time arrays (in seconds).

        Each entry is keyed first by the unit ID, then by channel, then by direction.
        """
        fs = self.recording.get_sampling_frequency()
        unit_channel_spike_dict = {}

        for ch, trial_list in data_structure.items():
            # Each item: (channel_rec, onset_frame, spike_array, direction, unit, amp_array)
            for (channel_rec, onset_frame, spike_array, direction, unit, amp_array) in trial_list:
                # Convert frames to seconds if not already
                spike_times_s = spike_array / fs

                # Make sure all dictionary levels exist
                if unit not in unit_channel_spike_dict:
                    unit_channel_spike_dict[unit] = {}
                if ch not in unit_channel_spike_dict[unit]:
                    unit_channel_spike_dict[unit][ch] = {}
                if direction not in unit_channel_spike_dict[unit][ch]:
                    unit_channel_spike_dict[unit][ch][direction] = []

                # Append this trial’s spike times
                unit_channel_spike_dict[unit][ch][direction].append(spike_times_s)

        return unit_channel_spike_dict

    def get_raster_psth(self, spike_trains, bin_size=0.01, window=(-0.25, 0.75), return_rate=True):
        """
        Computes raster coordinates and PSTH for a set of trials.

        Parameters
        ----------
        spike_trains : list of np.ndarray
            Each element is an array of spike times (in seconds), aligned so that
            time=0 corresponds to stimulus onset for that trial.
        bin_size : float
            Size of PSTH time bins in seconds. Defaults to 10 ms.
        window : tuple (float, float)
            Time window (start, end) in seconds relative to stimulus onset.
            Default is (-0.25, 0.75).
        return_rate : bool
            If True, PSTH is given in spikes/s (Hz). If False, PSTH is spike counts/bin.

        Returns
        -------
        raster_x : np.ndarray
            1D array of spike times (only those within the specified window),
            concatenated across all trials (for raster plotting on the x-axis).
        raster_y : np.ndarray
            1D array of the same length as raster_x, giving the trial index
            for each spike (for raster plotting on the y-axis).
        bin_centers : np.ndarray
            1D array of time bin centers for the PSTH.
        psth : np.ndarray
            1D array containing the PSTH values for each bin (either as
            firing rate in spikes/s or raw spike counts, depending on `return_rate`).
        """
        # Collect x (time) and y (trial index) for the raster
        raster_x = []
        raster_y = []

        n_trials = len(spike_trains)
        t_start, t_end = window

        # Build time bins for PSTH
        bins = np.arange(t_start, t_end + bin_size, bin_size)
        bin_count = len(bins) - 1
        psth_counts = np.zeros(bin_count, dtype=float)

        # Build raster and accumulate histogram for PSTH
        for trial_idx, spikes in enumerate(spike_trains):
            # Restrict spikes to the desired time window
            in_window = (spikes >= t_start) & (spikes <= t_end)
            spikes_in_win = spikes[in_window]

            # Raster data
            raster_x.extend(spikes_in_win)
            raster_y.extend([trial_idx] * len(spikes_in_win))

            # PSTH data (add to histogram)
            counts, _ = np.histogram(spikes_in_win, bins=bins)
            psth_counts += counts

        raster_x = np.array(raster_x)
        raster_y = np.array(raster_y)

        # Convert PSTH counts to rate if requested
        if return_rate:
            psth = psth_counts / (bin_size * n_trials)  # spikes/s
        else:
            psth = psth_counts  # raw spike counts per bin

        # Compute bin centers (useful for plotting)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        return raster_x, raster_y, bin_centers, psth

    def get_unit_channel_weights(self):
        """
        Returns a dictionary mapping each unit to a dictionary of channel weights
        based on the normalized amplitude of its template on each channel.

        Output:
        {
            unit_id: {channel_idx: weight, ...},
            ...
        }
        """
        if not self.analyzer:
            raise ValueError("SortingAnalyzer not loaded. Run 'process_analyzer()' first.")

        templates = self.analyzer.get_extension("templates").get_data("median")  # (units, channels, time)
        unit_ids = self.analyzer.sorting.unit_ids
        unit_weights = {}

        for unit_index, unit_id in enumerate(unit_ids):
            template = templates[unit_index]  # (channels, time)
            channel_amplitudes = np.max(np.abs(template), axis=1)
            total = np.sum(channel_amplitudes)
            if total == 0:
                continue
            weights = channel_amplitudes / total
            unit_weights[unit_id] = {ch: weight for ch, weight in enumerate(weights)}

        return unit_weights

    def load_directions(self):
        """Reads stimulus directions from a MATLAB .mat file and assigns numerical values."""
        try:
            mat_data = scipy.io.loadmat(self.config["paths"]["stimulus_directions"])
            # Extract the relevant variable (assuming it's named appropriately in the MATLAB file)
            if 'stimulusDirections' in mat_data:
                directions = mat_data['stimulusDirections'].flatten()
            else:
                raise ValueError("Expected variable 'stimulusDirections' not found in .mat file.")

            # Assign unique numerical values to each unique direction
            unique_directions = np.unique(directions)
            direction_mapping = {dir: idx for idx, dir in enumerate(unique_directions)}
            direction_values = np.array([direction_mapping[dir] for dir in directions])

            self.directions = direction_values
            print("Stimulus directions loaded successfully.")
        except Exception as e:
            print(f"Error loading stimulus directions: {e}")
            self.directions = None

    def load_nidaq(self):
        """Loads the NIDAQ binary file and extracts the photodiode and sync pulse channels."""
        preprocessed_path = self.config["paths"]["preprocessed_NIDAQ"]

        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed recording from {preprocessed_path}...")
            self.nidaq_recording = se.read_spikeglx(self.config["base_path"], stream_name='nidq')
            self.nidaq_concat = sic.load(preprocessed_path)

            photodiode_signal = self.nidaq_concat.get_traces(channel_ids=['nidq#XA0']).flatten()
            sync_pulse_signal = self.nidaq_concat.get_traces(channel_ids=['nidq#XD0']).flatten()

            # Detect light onset times
            light_onsets, light_offsets = self.detect_light_transitions(photodiode_signal)

            # Detect sync pulses every 10 seconds
            sync_pulses = self.detect_sync_pulses(sync_pulse_signal)
        else:
            print("Processing NIDAQ from raw data...")
            self.nidaq_recording = se.read_spikeglx(self.config["base_path"], stream_name='nidq')
            self.nidaq_concat = sic.concatenate_recordings([self.nidaq_recording])

            photodiode_signal = self.nidaq_concat.get_traces(channel_ids=['nidq#XA0']).flatten()
            sync_pulse_signal = self.nidaq_concat.get_traces(channel_ids=['nidq#XD0']).flatten()

            # Save if enabled
            if self.config["write_concat"]:
                self.nidaq_concat = self.nidaq_concat.save(format="binary", folder=preprocessed_path)

            # Detect light onset times
            light_onsets, light_offsets = self.detect_light_transitions(photodiode_signal)

            # Detect sync pulses every 10 seconds
            sync_pulses = self.detect_sync_pulses(sync_pulse_signal)

        # Ensure nidaq_data is assigned in both cases
        self.nidaq_data = {
            "photodiode": photodiode_signal,
            "sync_pulse": sync_pulse_signal,
            "light_onsets": light_onsets,
            "light_offsets": light_offsets,
            "sync_pulses": sync_pulses
        }
        print("NIDAQ data loaded and processed.")

    def load_recording(self):
        """Loads and preprocesses the SpikeGLX recording."""
        preprocessed_path = self.config["paths"]["preprocessed_ap"]

        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed recording from {preprocessed_path}...")
            self.recording = se.read_spikeglx(self.config["run_path"], stream_name=f'imec{self.config["probe"]}.ap')
            self.recording_concat = sic.load(preprocessed_path)
        else:
            print("Processing preprocessed recording from raw data...")
            self.recording = se.read_spikeglx(self.config["run_path"], stream_name=f'imec{self.config["probe"]}.ap')
            # self.config["channels"] = (min(channel_ids), max(channel_ids))

            # Preprocessing steps
            recording_hp = spre.highpass_filter(self.recording, freq_min=300)
            recording_clean = spre.common_reference(recording_hp, operator="median")
            self.recording_concat = sic.concatenate_recordings([recording_clean])

            # Save if enabled
            if self.config["write_concat"]:
                self.recording_concat = self.recording_concat.save(format="binary", folder=preprocessed_path)

    def load_spike_amplitudes(self):
        """Loads precomputed spike amplitudes from the SortingAnalyzer."""

        print("Loading precomputed spike amplitudes...")

        # Ensure the analyzer exists
        if self.analyzer is None:
            raise ValueError("SortingAnalyzer is not available. Run 'process_analyzer()' first.")

        # Retrieve amplitude data
        amplitudes_ext = self.analyzer.get_extension("spike_amplitudes")
        amplitudes = amplitudes_ext.get_data()

        # Print statistics
        print(f"Mean AP amplitude: {amplitudes.mean():.2f} µV")
        print(f"Max AP amplitude: {amplitudes.max():.2f} µV")
        print(f"Min AP amplitude: {amplitudes.min():.2f} µV")

        return amplitudes

    def run_kilosort(self):
        """Runs Kilosort or loads existing sorting results."""
        kilosort_path = self.config["paths"]["kilosort"]
        if os.path.exists(kilosort_path) and self.config["skip_sort"]:
            print(f"Kilosort output found at {kilosort_path}, skipping sorting step.")
            self.sorting = se.read_kilosort(f'{kilosort_path}/sorter_output')
        else:
            print("Running Kilosort4...")
            self.sorting = ss.run_sorter("kilosort4", self.recording_concat, folder=kilosort_path)

    def plot_empty_chart_layout(self, directions=None):
        """
        Plots a grid of channels as specified in config["channels"], each with:
          - Channel label on the left
          - Raster row (with header), PSTH row (with header), below them all directions
          - Waveform + autocorr stacked vertically to the right
        """
        plt.rc('font', size=5)

        if directions is None:
            directions = [0, 45, 90, 135, 180, 225, 270, 315]

        ch_start, ch_end = self.config["channels"]
        ch_ids = list(range(ch_start, ch_end+1))  # Inclusive lower bound, exclusive upper
        n_dirs = len(directions)
        print(ch_ids)

        # Layout: 4 columns fixed, rows auto-calculated
        n_cols = 2
        n_rows = math.ceil(len(ch_ids) / 2)

        # Scale figure size
        fig_width = 5 * n_cols
        fig_height = 4.5 * n_rows

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(n_rows, n_cols, wspace=0.1, hspace=0.1)

        def make_channel_subgrid(outer_cell, ch_id):
            dir_width = 1.5
            n_total_cols = n_dirs + 3  # 1 ch label + n_dirs + 2 (wave/acorr)
            width_ratios = [0.3] + [dir_width] * n_dirs + [dir_width, dir_width]

            outer = outer_cell.subgridspec(3, n_total_cols,
                                           height_ratios=[1, 1, 0.15],
                                           width_ratios=width_ratios,
                                           hspace=0.2, wspace=0.5)

            # Channel label
            ax_ch_label = fig.add_subplot(outer[0:2, 0])
            ax_ch_label.axis("off")
            ax_ch_label.text(0.5, 0.5, f"Ch {ch_id}", ha='center', va='center',
                             rotation=90, transform=ax_ch_label.transAxes, fontsize=5)

            # Compute center x for raster/psth titles
            col_start = 1
            col_end = 1 + n_dirs
            dir_col_centers = [fig.add_subplot(outer[0, i]).get_position().x0 for i in range(col_start, col_end)]
            center_x = sum(dir_col_centers) / len(dir_col_centers)

            fig.text(center_x, ax_ch_label.get_position().y1 + 0.01,
                     "Raster", ha='center', va='bottom', fontsize=5)
            fig.text(center_x, ax_ch_label.get_position().y0 - 0.01,
                     "PSTH", ha='center', va='top', fontsize=5)

            for d_idx, d_val in enumerate(directions):
                col_idx = d_idx + 1
                ax_raster = fig.add_subplot(outer[0, col_idx])
                ax_raster.set_xticks([])
                ax_raster.set_yticks([])

                ax_psth = fig.add_subplot(outer[1, col_idx])
                ax_psth.set_xticks([])
                ax_psth.set_yticks([])

                ax_label = fig.add_subplot(outer[2, col_idx])
                ax_label.axis("off")
                ax_label.text(0.5, 0.5, f"{d_val}°", ha='center', va='center', fontsize=5)

            ax_wave = fig.add_subplot(outer[0, -2])
            ax_wave.set_title("Waveform", fontsize=5)
            ax_wave.set_xticks([]);
            ax_wave.set_yticks([])

            ax_ac = fig.add_subplot(outer[1, -2])
            ax_ac.set_title("AutoCorr", fontsize=5)
            ax_ac.set_xticks([]);
            ax_ac.set_yticks([])

            fig.add_subplot(outer[0:2, -1]).axis("off")

            # Bounding box for this channel group
            sub_axes = fig.axes[-(n_dirs * 3 + 5):]
            bbox = mtransforms.Bbox.union([ax.get_position() for ax in sub_axes])
            pad_x = 0.03
            pad_y = 0.015
            rect = patches.Rectangle(
                (bbox.x0 - pad_x, bbox.y0 - 0),
                bbox.width + pad_x,
                bbox.height + pad_y,
                transform=fig.transFigure,
                linewidth=0.7,
                edgecolor='black',
                facecolor='none'
            )
            fig.patches.append(rect)

        # Loop through all channels and assign them to grid cells
        for i, ch_id in enumerate(ch_ids):
            row = n_rows - 1 - (i // 2)  # Flip vertically
            col = i % 2  # 0 = left, 1 = right
            make_channel_subgrid(gs[row, col], ch_id)

        fig.suptitle("Multi-Channel Layout (Empty Chart Template)", fontsize=6)

        # Save the chart
        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"empty_plot.png")
        self.save_chart(save_path)

        if self.config['show_plot']:
            plt.show()

    def process_analyzer(self):
        """Computes waveforms, templates, and amplitudes using SortingAnalyzer."""
        analyzer_path = self.config["paths"]["analyzer"]

        if os.path.exists(analyzer_path):
            print("Loading existing SortingAnalyzer...")
            self.analyzer = sic.load_sorting_analyzer(analyzer_path)
        else:
            print("Creating SortingAnalyzer and computing extensions...")
            self.analyzer = sic.create_sorting_analyzer(self.sorting, self.recording_concat, format="binary_folder",
                                                        folder=analyzer_path)

            # Compute extensions
            self.analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=200)
            self.analyzer.compute("waveforms", ms_before=1.5, ms_after=2.0)
            self.analyzer.compute("templates", operators=["average", "median", "std"])
            self.analyzer.compute("spike_amplitudes")

    def process_mua(self, data_structure, lowcut=300, smoothing_window=5, baseline_window=(-0.25, 0), segment_handling='truncate'):
        fs = self.recording.get_sampling_frequency()
        use_sdf = self.config.get("use_sdf", False)
        if use_sdf and self.config["mua_method"] == "kilosort":
            channel_weights = self.get_unit_channel_weights()

        ch_start, ch_end = self.config["channels_sample"]
        num_channels = ch_end - ch_start

        print(f"Processing MUA for {num_channels} selected channels on segmented data")

        aligned_mua = {}
        segment_lengths = {}

        for key, trials in data_structure.items():
            if isinstance(key, int):
                if ch_start <= key < ch_end:
                    aligned_mua[key] = []
                    segment_lengths[key] = []
            else:
                aligned_mua[key] = {ch: [] for ch in range(ch_start, ch_end)}
                segment_lengths[key] = []

        for key, trials in data_structure.items():
            if isinstance(key, int) and not (ch_start <= key < ch_end):
                continue

            print(f"Processing {len(trials)} trials for {key}")
            for segment, onset_frame, spike_data in trials:
                segment_traces = segment.get_traces()
                if segment_traces.shape[0] < 18:
                    print(f"Skipping short segment ({segment_traces.shape[0]} samples), below filter threshold.")
                    break

                if isinstance(key, int):
                    if use_sdf and self.config["mua_method"] == "kilosort":
                        sdf = self.compute_spike_density_function(
                            spike_times=spike_data,
                            onset_frame=onset_frame,
                            total_frames=segment.get_num_frames(),
                            fs=fs,
                            sigma=smoothing_window / 1000
                        )
                        aligned_mua[key].append(sdf)
                        segment_lengths[key].append(len(sdf))
                    else:
                        filtered_trace = spre.highpass_filter(segment, freq_min=lowcut).get_traces().flatten()
                        rectified_trace = np.abs(filtered_trace)
                        smoothing_samples = max(1, int(smoothing_window * fs / 1000))
                        smoothed_trace = np.convolve(rectified_trace, np.ones(smoothing_samples) / smoothing_samples, mode='same')
                        aligned_mua[key].append(smoothed_trace)
                        segment_lengths[key].append(len(smoothed_trace))
                else:
                    if use_sdf and self.config["mua_method"] == "kilosort":
                        for unit, spikes in spike_data.items():
                            sdf = self.compute_spike_density_function(
                                spike_times=spikes,
                                onset_frame=onset_frame,
                                total_frames=segment.get_num_frames(),
                                fs=fs,
                                sigma=smoothing_window / 1000
                            )
                            for ch, weight in channel_weights.get(unit, {}).items():
                                if ch_start <= ch < ch_end:
                                    aligned_mua[key][ch].append(weight * sdf)
                        segment_lengths[key].append(segment.get_num_frames())
                    else:
                        filtered_traces = spre.highpass_filter(segment, freq_min=lowcut).get_traces()[:, ch_start:ch_end]
                        rectified_traces = np.abs(filtered_traces)
                        smoothing_samples = max(1, int(smoothing_window * fs / 1000))
                        smoothed_traces = np.apply_along_axis(
                            lambda x: np.convolve(x, np.ones(smoothing_samples) / smoothing_samples, mode='same'), 0,
                            rectified_traces)
                        segment_lengths[key].append(smoothed_traces.shape[0])
                        for ch in range(ch_start, ch_end):
                            aligned_mua[key][ch].append(smoothed_traces[:, ch - ch_start])

        for key in data_structure.keys():
            if key not in aligned_mua:
                continue
            if segment_handling == 'truncate' and segment_lengths[key]:
                min_length = min(segment_lengths[key])
                if isinstance(key, int):
                    aligned_mua[key] = [trial[:min_length] for trial in aligned_mua[key]]
                else:
                    aligned_mua[key] = {ch: [trial[:min_length] for trial in trials] for ch, trials in aligned_mua[key].items()}
            elif segment_handling == 'pad' and segment_lengths[key]:
                max_length = max(segment_lengths[key])
                if isinstance(key, int):
                    aligned_mua[key] = [np.pad(trial, (0, max_length - len(trial)), mode='constant') for trial in aligned_mua[key]]
                else:
                    aligned_mua[key] = {
                        ch: [np.pad(trial, (0, max_length - len(trial)), mode='constant') for trial in trials]
                        for ch, trials in aligned_mua[key].items()
                    }

        psth = {}
        for key in data_structure.keys():
            if key not in aligned_mua or not aligned_mua[key]:
                continue
            if isinstance(key, int):
                psth[key] = np.mean(np.vstack(aligned_mua[key]), axis=0)
            else:
                psth[key] = {
                    ch: np.mean(np.vstack(trials), axis=0) if trials else np.zeros(min(segment_lengths[key]))
                    for ch, trials in aligned_mua[key].items()
                }

        baseline_idx = int(baseline_window[0] * fs), int(baseline_window[1] * fs)
        normalized_psth = {}
        for key in psth.keys():
            if isinstance(key, int):
                response = psth[key]
                if np.any(response[baseline_idx[0]:baseline_idx[1]]):
                    normalized_psth[key] = ((response - np.mean(response[baseline_idx[0]:baseline_idx[1]])) /
                                            np.mean(response[baseline_idx[0]:baseline_idx[1]]) * 100)
                else:
                    normalized_psth[key] = response
            else:
                normalized_psth[key] = {
                    ch: ((response - np.mean(response[baseline_idx[0]:baseline_idx[1]])) /
                         np.mean(response[baseline_idx[0]:baseline_idx[1]]) * 100)
                    if np.any(response[baseline_idx[0]:baseline_idx[1]]) else response
                    for ch, response in psth[key].items()
                }

        return normalized_psth

    def process_segments(self):
        """Segments the recording into trials and stores per-channel, per-direction, per-unit spike data and amplitudes."""
        if self.nidaq_data is None or self.directions is None:
            print("Error: NIDAQ data or stimulus directions not loaded.")
            return None

        light_onsets = self.nidaq_data.get("light_onsets", [])
        light_offsets = self.nidaq_data.get("light_offsets", [])

        if len(light_onsets) != len(light_offsets) or len(light_onsets) != len(self.directions):
            print("Error: Mismatch in the number of stimulus directions and onset/offset pairs.")
            return None

        fs = self.recording.get_sampling_frequency()
        adjusted_start_times = []
        adjusted_end_times = []
        onset_frame_indices = []

        onset_diff = 0.25
        offset_diff = 0.25

        for i in range(len(light_onsets)):
            if i == 0:
                start_time = light_onsets[i] - onset_diff
            else:
                start_time = -((light_onsets[i] - light_offsets[i - 1]) * (onset_diff / (onset_diff + offset_diff))) + \
                             light_onsets[i]

            if i == len(light_onsets) - 1:
                end_time = light_offsets[i] + offset_diff
            else:
                end_time = (light_onsets[i + 1] - light_offsets[i]) * (offset_diff / (onset_diff + offset_diff)) + \
                           light_offsets[i]

            start_frame = int(start_time * fs)
            end_frame = int(end_time * fs)
            onset_frame = int((light_onsets[i] - start_time) * fs)

            segment_length = end_frame - start_frame
            if segment_length < 10:
                print(f"Warning: Short segment detected! Start={start_frame}, End={end_frame}, Length={segment_length}")

            adjusted_start_times.append(start_frame)
            adjusted_end_times.append(end_frame)
            onset_frame_indices.append(onset_frame)

        data_structure = {ch: [] for ch in range(self.recording.get_num_channels())}
        segment_lengths = []
        best_channels = self.get_best_channels()

        for trial_idx, (start, end, direction, onset_frame) in enumerate(
                zip(adjusted_start_times, adjusted_end_times, self.directions, onset_frame_indices)):
            segment_length = end - start
            segment_lengths.append(segment_length / fs)

            if segment_length < 10:
                print(f"Skipping invalid segment: Start={start}, End={end}, Length={segment_length} samples")
                continue

            segment = self.recording.frame_slice(start_frame=start, end_frame=end)
            spike_data, spike_amps = self.process_segment_spikes(start, end, start + onset_frame)
            traces = segment.get_traces()

            for unit in spike_data:
                spike_array = spike_data[unit]
                amp_array = spike_amps.get(unit, np.zeros_like(spike_array))

                best_ch = best_channels.get(unit)
                if best_ch is None:
                    continue

                channel_trace = traces[:, best_ch].reshape(-1, 1)
                channel_recording = sic.NumpyRecording(channel_trace, fs)
                data_structure[best_ch].append(
                    (channel_recording, onset_frame, spike_array, direction, unit, amp_array))

        segment_lengths = np.array(segment_lengths)
        onset_frame_indices = np.array(onset_frame_indices) / fs
        print(f"Segment statistics (s):\n"
              f" \tSegment Length Mean: {segment_lengths.mean():.3f}, Onset Time Mean: {onset_frame_indices.mean():3f}\n"
              f" \tSegment Length Std: {segment_lengths.std():.3f}, Onset Time Std: {onset_frame_indices.std():3f}\n"
              f" \tSegment Length Min: {segment_lengths.min():.3f}, Onset Time Min: {onset_frame_indices.min():3f}\n"
              f" \tSegment Length Max: {segment_lengths.max():.3f}, Onset Time Max: {onset_frame_indices.max():3f}")

        print("Segmented and grouped recordings successfully with full unit, channel, and direction context.")
        return data_structure

    def process_segment_spikes(self, start_index, end_index, onset_index):
        """Extracts spike times and amplitudes for a given segment and aligns them to the stimulus onset."""
        if self.sorting is None:
            print("Error: Sorting data not loaded.")
            return None, None

        spikes_in_window = {}
        spike_amplitudes = {}

        if self.config["mua_method"] == "kilosort":
            amplitudes_ext = self.analyzer.get_extension("spike_amplitudes")
            all_amps = amplitudes_ext.get_data()
            spike_vector = self.sorting.to_spike_vector()
            all_spike_times = spike_vector["sample_index"]
            all_spike_units = spike_vector["unit_index"]

            for unit in self.sorting.unit_ids:
                spike_times = self.sorting.get_unit_spike_train(unit)
                unit_mask = (all_spike_units == unit)
                unit_amps = all_amps[unit_mask]

                mask = (spike_times >= start_index) & (spike_times <= end_index)
                aligned_spikes = spike_times[mask] - onset_index
                aligned_amps = unit_amps[mask]

                spikes_in_window[unit] = aligned_spikes
                spike_amplitudes[unit] = aligned_amps


        elif self.config["mua_method"] == "threshold":
            if not hasattr(self, "spike_times") or self.spike_times is None:
                print("Error: Threshold-based spike times not extracted.")
                return None, None

            for ch in range(self.config["channels"][0], self.config["channels"][1]):
                if ch in self.spike_times:
                    spike_times = np.array(self.spike_times[ch]["spike_times"])  # Spike times in frame indices
                    mask = (spike_times >= start_index) & (spike_times <= end_index)
                    aligned_spikes = spike_times[mask] - onset_index
                    # For threshold method, fake amplitude as constant or zeros
                    aligned_amps = np.zeros_like(aligned_spikes)

                    spikes_in_window[ch] = aligned_spikes
                    spike_amplitudes[ch] = aligned_amps

        else:
            print("Error: Invalid MUA method specified.")
            return None, None

        return spikes_in_window, spike_amplitudes

    def compute_spike_density_function(self, spike_times, onset_frame, total_frames, fs, sigma=0.005):
        """
        Computes the spike density function (SDF) for a given spike train, aligned to a stimulus onset.

        Parameters:
        - spike_times: np.ndarray of spike times (in frames, relative to segment start)
        - onset_frame: Index of stimulus onset within segment
        - total_frames: Total number of frames in the segment
        - fs: Sampling frequency (Hz)
        - sigma: Standard deviation of Gaussian kernel (in seconds)

        Returns:
        - sdf: np.ndarray of length total_frames, the continuous spike density function
        """
        sdf = np.zeros(total_frames)
        spike_indices = np.round(spike_times + onset_frame).astype(int)
        spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < total_frames)]

        sigma_samples = int(sigma * fs)
        if sigma_samples < 1:
            sigma_samples = 1

        kernel_size = sigma_samples * 6  # cover ±3σ
        t = np.arange(-kernel_size // 2, kernel_size // 2 + 1)
        kernel = np.exp(-0.5 * (t / sigma_samples) ** 2)
        kernel /= kernel.sum()  # Normalize

        for idx in spike_indices:
            start = idx - kernel_size // 2
            end = idx + kernel_size // 2 + 1

            k_start = max(0, -start)
            k_end = kernel.size - max(0, end - total_frames)

            start = max(start, 0)
            end = min(end, total_frames)

            sdf[start:end] += kernel[k_start:k_end]

        return sdf

    def plot_ap_amplitudes(self):
        """Plots action potential amplitude distribution."""
        if not self.analyzer:
            print("SortingAnalyzer not loaded. Run `process_analyzer()` first.")
            return

        # Retrieve amplitude data
        amplitudes_ext = self.analyzer.get_extension("spike_amplitudes")
        amplitudes = amplitudes_ext.get_data()

        # Compute and print statistics
        print(f"Amplitudes shape: {amplitudes.shape}")
        print(f"First 10 amplitudes: {amplitudes[:10]}")
        print(f"Mean AP amplitude: {amplitudes.mean():.2f} µV")
        print(f"Max AP amplitude: {amplitudes.max():.2f} µV")
        print(f"Min AP amplitude: {amplitudes.min():.2f} µV")

        # Plot histogram of amplitudes
        plt.figure(figsize=(10, 4))
        plt.hist(amplitudes, bins=50, alpha=0.7)
        plt.xlabel("Amplitude (µV)")
        plt.ylabel("Spike Count")
        plt.title("Distribution of Action Potential Amplitudes")

        # Save the chart
        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mua_method']}_ap_amplitude.png")
        self.save_chart(save_path)

        if self.config['show_plot']:
            plt.show()

    def plot_channel_templates(self):
        """Plots template (mean) waveforms for selected channels using SortingAnalyzer."""
        if not self.analyzer:
            print("SortingAnalyzer not found. Run `process_analyzer()` first.")
            return

        print("Extracting template waveforms for selected channels...")

        # Extract template waveforms from SortingAnalyzer
        templates = self.analyzer.get_extension("templates").get_data("median")  # Shape (units, channels, time)
        unit_ids = self.analyzer.sorting.unit_ids  # Available units

        # Extract necessary data
        fs = self.analyzer.sampling_frequency  # Sampling frequency (Hz)
        num_units, num_channels, num_timepoints = templates.shape

        # Manually set the time window (e.g., 0 to 0.5 ms)
        time_window = (0, 0.5)  # In milliseconds
        time_indices = np.linspace(0, num_timepoints / fs * 1000, num_timepoints)  # Convert to ms
        valid_indices = (time_indices >= time_window[0]) & (time_indices <= time_window[1])  # Select range

        # Plot a subset of templates
        fig, axes = plt.subplots(len(unit_ids[:3]), 1, figsize=(8, len(unit_ids[:3]) * 2), sharex=True)

        if len(unit_ids[:3]) == 1:
            axes = [axes]  # Ensure axes is iterable for a single unit case

        for ax, unit_idx in zip(axes, unit_ids[:3]):
            template = templates[unit_idx][:, valid_indices]  # Extract waveform in set range
            for ch in range(min(10, template.shape[0])):  # Limit to 10 channels for visibility
                ax.plot(time_indices[valid_indices], template[ch], alpha=0.7)

            ax.set_title(f"Unit {unit_idx}")
            ax.set_ylabel("Amplitude (µV)")

        # Shared X-axis for all plots
        axes[-1].set_xlabel("Time (ms)")
        axes[-1].set_xlim(time_window)  # Apply the manually set range

        plt.tight_layout()

        # Save the chart
        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mua_method']}_channel_templates.png")
        self.save_chart(save_path)

        if self.config['show_plot']:
            plt.show()

    def plot_directions_raster(self, direction_trials, bin_size=0.01, y_mode="channels"):
        """Plots a raster heatmap for each direction, allowing selection of trials, units, or channels as the y-axis.

        Parameters:
        - direction_trials: Dictionary containing spike data for each direction.
        - bin_size: Time bin size for raster heatmap.
        - y_mode: Determines y-axis labeling ("trials", "units", or "channels").
        """
        fig, axes = plt.subplots(2, 4, figsize=(12, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        fs = self.recording.get_sampling_frequency()

        # Get best channel mapping if using "channels" mode
        if y_mode == "channels":
            best_channel_map = self.get_best_channels()  # {unit_id: best_channel}

        for i, direction in enumerate(sorted(direction_trials.keys())):
            ax = axes[i]
            ax.set_title(f"Direction {direction * 45}°")

            # Collect all onset frames for this direction
            onset_frames = [onset_frame for _, onset_frame, _ in direction_trials[direction]]
            mean_onset_frame = np.mean(onset_frames)  # Average onset frame
            mean_onset_time = mean_onset_frame / fs  # Convert to seconds

            # Get the longest segment for this direction
            max_time = max(
                (segment.get_num_frames() - mean_onset_frame) / fs
                for segment, _, _ in direction_trials[direction]
            )
            min_time = -mean_onset_time  # Aligns start to negative onset mean

            # Define bins strictly within this range
            bins = np.arange(min_time, max_time + bin_size, bin_size)

            # Adjust tick range: keep 0.25s intervals but ensure they don't exceed min_time/max_time
            tick_positions = np.arange(
                np.ceil(min_time / 0.25) * 0.25,
                np.floor(max_time / 0.25) * 0.25 + 0.01,
                0.25
            )
            tick_labels = [f"{t:.2f}" for t in tick_positions]

            # Get y-axis data depending on selected mode
            if y_mode == "trials":
                num_y = len(direction_trials[direction])  # Number of trials
                y_label = "Trial Index"
            elif y_mode == "units":
                all_units = {unit for _, _, spike_data in direction_trials[direction] for unit in spike_data.keys()}
                num_y = len(all_units)  # Number of unique units
                y_label = "Unit Index"
            elif y_mode == "channels":
                all_channels = set(best_channel_map.values())  # Unique best channels
                num_y = len(all_channels)
                y_label = "Channel Index"
            else:
                raise ValueError(f"Invalid y_mode: {y_mode}. Choose from 'trials', 'units', or 'channels'.")

            num_bins = len(bins) - 1
            heatmap = np.zeros((num_y, num_bins))

            # Fill heatmap depending on y_mode
            if y_mode == "trials":
                for trial_idx, (segment, onset_frame, spike_data) in enumerate(direction_trials[direction]):
                    spike_times = np.concatenate(list(spike_data.values())) / fs
                    counts, _ = np.histogram(spike_times, bins=bins)
                    heatmap[trial_idx, :] = counts

            elif y_mode == "units":
                unit_to_idx = {unit: i for i, unit in enumerate(sorted(all_units))}
                for _, _, spike_data in direction_trials[direction]:
                    for unit, spikes in spike_data.items():
                        unit_idx = unit_to_idx[unit]
                        counts, _ = np.histogram(spikes / fs, bins=bins)
                        heatmap[unit_idx, :] += counts  # Accumulate across trials

            elif y_mode == "channels":
                ch_start, ch_end = self.config["channels"]  # Extract channel range from config
                num_y = ch_end - ch_start  # Number of channels analyzed
                y_label = "Channel Index"

                # Ensure only selected channels are mapped
                channel_to_idx = {ch: i for i, ch in enumerate(range(ch_start, ch_end))}

                # Create heatmap with correct number of channels
                heatmap = np.zeros((num_y, num_bins))  # Ensures correct size based on config

                for _, _, spike_data in direction_trials[direction]:
                    for unit, spikes in spike_data.items():
                        best_ch = best_channel_map.get(unit, None)

                        # Ensure best_ch is within the analyzed channel range
                        if best_ch is not None and ch_start <= best_ch < ch_end:
                            ch_idx = channel_to_idx[best_ch]  # This now maps correctly
                            counts, _ = np.histogram(spikes / fs, bins=bins)
                            heatmap[ch_idx, :] += counts  # Accumulate across trials

            # Plot heatmap
            img = ax.imshow(
                heatmap, aspect='auto', cmap='hot',
                extent=[bins[0], bins[-1], 0, num_y]
            )

            ax.axvline(0, color='white', linestyle='--', linewidth=1)  # Mark true mean onset
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)

        plt.tight_layout()

        # Save the chart
        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mua_method']}_directions_raster.png")
        self.save_chart(save_path)

        if self.config['show_plot']:
            plt.show()

    def plot_filtered_traces(self):
        """Plots high-pass filtered data with threshold-based peak detection."""
        time_window = (0, 10)  # Time range in seconds for visualization
        downsample_factor = 2  # Reduce points for faster plotting
        direction = self.config.get("direction", "negative")

        # Call `detect_peaks()` to get threshold crossings for the visualization window
        spike_data = self.detect_peaks(time_window=time_window)

        # Extract traces for visualization
        fs = self.recording.get_sampling_frequency()
        start_frame = int(time_window[0] * fs)
        end_frame = int(time_window[1] * fs)
        ch_start, ch_end = self.config["channels_sample"]
        traces = self.recording.get_traces(start_frame=start_frame, end_frame=end_frame)[:, ch_start:ch_end]
        time_vector = np.linspace(time_window[0], time_window[1], traces.shape[0])

        num_channels = ch_end - ch_start
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, num_channels * 2), sharex=True, sharey=True)

        for i, ch in enumerate(range(ch_start, ch_end)):
            ax = axes[i] if num_channels > 1 else axes

            # Plot trace
            ax.plot(time_vector[::downsample_factor], traces[::downsample_factor, i], color='black')

            # Plot detected spikes
            if ch in spike_data:
                ax.scatter(spike_data[ch]["spike_times"], traces[spike_data[ch]["peaks"], i], color='red', marker='x',
                           s=30)

            # Draw threshold line (handles both positive and negative)
            ax.axhline(spike_data[ch]["threshold"], color='blue', linestyle='dashed', linewidth=1)

            ax.set_ylabel(f"Ch {ch} (µV)")

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"High-Pass Filtered Data with {'Negative' if direction == 'negative' else 'Positive'} Peak Detection ({ch_start}-{ch_end})")

        # Save the chart
        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mua_method']}_filtered_traces.png")
        self.save_chart(save_path)

        if self.config['show_plot']:
            plt.show()

    def plot_mua_activity(self):
        """Plots the Multi-Unit Activity (MUA) event counts across recording channels."""

        num_channels = self.recording.get_num_channels()
        spike_counts = np.zeros(num_channels)  # Initialize spike count array

        if self.config["mua_method"] == "threshold":
            print("Plotting **threshold-based** MUA activity...")

            if not self.spike_times or len(self.spike_times) == 0:
                print("No MUA spikes detected. Cannot generate plot.")
                return

            # Extract spike counts per channel from `self.spike_times`
            for ch, data in self.spike_times.items():
                if "spike_times" in data:
                    spike_counts[ch] = len(data["spike_times"])

        elif self.config["mua_method"] == "kilosort":
            print("Plotting **Kilosort-based** MUA activity...")

            # Extract spike counts from sorted Kilosort units
            spike_channels = np.concatenate([
                np.full_like(self.sorting.get_unit_spike_train(unit_id), fill_value=unit_id)
                for unit_id in self.sorting.unit_ids
            ])
            spike_counts = np.bincount(spike_channels, minlength=num_channels)

        else:
            raise ValueError("Invalid `mua_method`! Choose between 'threshold' or 'kilosort'.")

        # Plot the spike count per channel
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(spike_counts)), spike_counts, log=True)  # Log scale for better visualization
        plt.xlim(self.config["channels"])  # Restrict x-axis to selected channels

        plt.xlabel("Channel Index")
        plt.ylabel("MUA Event Count (log scale)")
        plt.title("Multi-Unit Activity (MUA) Across Recording Sites")

        # Save the chart
        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mua_method']}_mua_activity.png")
        self.save_chart(save_path)

        if self.config['show_plot']:
            plt.show()

    def plot_mua_heatmap(self, psth, time_window=(-0.25, 0.75)):
        """Plots a heatmap of MUA responses across selected channels."""
        ch_start, ch_end = self.config["channels_sample"]

        # Build matrix from only the selected range
        mua_matrix = np.array([psth[ch] for ch in range(ch_start, ch_end) if ch in psth])

        plt.figure(figsize=(10, 6))
        plt.imshow(mua_matrix, aspect='auto', cmap='hot',
                   extent=[time_window[0], time_window[1], ch_start, ch_end])
        plt.colorbar(label="MUA Amplitude")
        plt.xlabel("Time (s)")
        plt.ylabel("Channel Index")
        plt.axvline(0, color='white', linestyle='--')
        plt.title("MUA Heatmap Aligned to Stimulus Onset")
        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mua_method']}_mua_heatmap.png")
        self.save_chart(save_path)
        if self.config['show_plot']:
            plt.show()

    def plot_mua_line(self, normalized_psth, time_window=(-0.1, 0.9), nrow=10, ncol=4):
        """Plots the mean MUA over time, handling both direction and channel-based styles."""
        if not normalized_psth:
            print("Error: normalized_psth is empty.")
            return

        fs = self.recording.get_sampling_frequency()
        first_key = next(iter(normalized_psth))
        first_value = normalized_psth[first_key]

        if isinstance(first_value, dict):  # direction-based structure
            first_inner_key = next(iter(first_value))
            first_array = first_value[first_inner_key]
        else:
            first_array = first_value

        time_axis = np.linspace(time_window[0], time_window[1], len(first_array))
        trim_samples = int(0.01 * fs)
        if len(time_axis) > 2 * trim_samples:
            time_axis = time_axis[trim_samples:-trim_samples]

        if self.config["style"] in ["direction", "both"]:
            directions = sorted([key for key in normalized_psth.keys() if isinstance(key, int)])
            fig, axes = plt.subplots(4, 2, figsize=(12, 12))
            axes = axes.flatten()

            for i, direction in enumerate(directions):
                ax = axes[i]
                mua_dict = normalized_psth[direction]

                cleaned = {}
                for ch, trace in mua_dict.items():
                    trace = np.array(trace)
                    if trace.ndim == 1:
                        trace = trace[np.newaxis, :]
                    if trace.shape[1] > 2 * trim_samples:
                        trace = trace[:, trim_samples:-trim_samples]

                    if np.max(trace) < 0.5:
                        trace[:] = 0
                    cleaned[ch] = np.mean(trace, axis=0)

                mua_data = np.array(list(cleaned.values()))
                mean_mua = np.mean(mua_data, axis=0)
                sem_mua = stats.sem(mua_data, axis=0)

                ax.plot(time_axis, mean_mua, label="Mean MUA", color='black')
                ax.fill_between(time_axis, mean_mua - sem_mua, mean_mua + sem_mua, color='gray', alpha=0.5)
                ax.axvline(0, color='red', linestyle='--')
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Normalized MUA Amplitude")
                ax.set_title(f"Direction {direction * 45}°")
                ax.legend()

            plt.tight_layout()
            plt.show()

        if self.config["style"] in ["channel", "both"]:
            ch_start, ch_end = self.config["channels_sample"]
            if ncol is None and nrow is None:
                fig, axes = plt.subplots(ch_end - ch_start, 1, figsize=(10, (ch_end - ch_start) * 2), sharex=True)
                axes = axes.flatten()
            else:
                fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10, (ch_end - ch_start) * 2), sharex=True)
                axes = axes.flatten()

            if (ch_end - ch_start) == 1:
                axes = [axes]

            for i, ch in enumerate(range(ch_start, ch_end)):
                if ch not in normalized_psth:
                    continue

                ax = axes[i]
                mua_data = np.array(normalized_psth[ch])

                if mua_data.ndim == 1:
                    mua_data = mua_data[np.newaxis, :]
                if mua_data.shape[1] > 2 * trim_samples:
                    mua_data = mua_data[:, trim_samples:-trim_samples]

                mean_mua = np.mean(mua_data, axis=0)

                ax.plot(time_axis, mean_mua, label=f"Ch {ch}", color='black')
                ax.axvline(0, color='red', linestyle='--')
                ax.set_ylabel(f"Ch {ch}", fontsize=10, rotation=90, labelpad=5)
                ax.yaxis.set_label_coords(-0.07, 0.5)
                ax.set_title(f"Ch {ch}")

            fig.legend([plt.Line2D([0], [0], color='black', lw=2),
                        plt.Line2D([0], [0], color='red', linestyle='--', lw=2)],
                       labels=["MUA", "Stimulus Onset"],
                       loc='upper center', fontsize=10, ncol=2, bbox_to_anchor=(0.5, 1.05))
            fig.supxlabel("Time (s)", fontsize=14)
            fig.supylabel("MUA Amplitude", fontsize=14, x=0.07)
            plt.tight_layout(rect=[0.01, 0.05, 1, 0.95])
            plt.subplots_adjust(hspace=0.1, wspace=0.1)

            screen_dpi = plt.rcParams['figure.dpi']
            screen_width = 3440 / screen_dpi
            screen_height = 1440 / screen_dpi
            fig.set_size_inches(screen_width, screen_height)

            save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                         f"{self.config['mua_method']}-sdf-{self.config['use_sdf']}_mua_line.png")
            self.save_chart(save_path)
            if self.config['show_plot']:
                plt.show()

    def plot_nidaq_signals(self):
        """
        Plots two charts: one for the photodiode signal and one for the sync pulse signal.
        Marks detected light_onsets and sync_pulses with vertical lines.
        Also overlays the middle-20s mean used for detection.
        """
        if not hasattr(self, 'nidaq_data'):
            print("NIDAQ data has not been loaded. Run load_nidaq() first.")
            return

        matplotlib.use('TkAgg')
        fs = self.nidaq_recording.get_sampling_frequency()
        signal = self.nidaq_data["photodiode"]
        time_vector = np.arange(len(signal)) / fs

        # Compute middle-20s mean
        total_duration = len(signal) / fs
        mid_start = int((total_duration / 2 - 10) * fs)
        mid_end = int((total_duration / 2 + 10) * fs)
        middle_mean = np.mean(signal[mid_start:mid_end]) + 0.66*np.std(signal[mid_start:mid_end])

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Photodiode plot
        axes[0].plot(time_vector, signal, color='black', label='Photodiode Signal')
        axes[0].axhline(middle_mean, color='orange', linestyle='--', linewidth=1.5, label='Middle Mean (20s)')
        for onset in self.nidaq_data["light_onsets"]:
            axes[0].axvline(onset, color='red', linestyle='dashed', alpha=0.6)
        for offset in self.nidaq_data["light_offsets"]:
            axes[0].axvline(offset, color='blue', linestyle='dashed', alpha=0.6)
        axes[0].set_title("Photodiode Signal with Light Onsets")
        axes[0].set_ylabel("Signal Amplitude")
        axes[0].legend(loc="upper right")

        # Sync pulse plot
        axes[1].plot(time_vector, self.nidaq_data["sync_pulse"], color='blue', label='Sync Pulse Signal')
        for pulse in self.nidaq_data["sync_pulses"]:
            axes[1].axvline(pulse, color='green', linestyle='dashed', alpha=0.6)
        axes[1].set_title("Sync Pulse Signal with Detected Pulses")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Signal Amplitude")
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def plot_probe_layout(self):
        """Plots the spatial layout of the probe."""
        if not self.recording:
            print("Recording not loaded. Run `load_recording()` first.")
            return

        probe = self.recording.get_probe()  # Extract probe geometry
        plot_probe(probe)  # Use `probeinterface` to visualize
        plt.show()

    def plot_raster_heatmap(self, bin_size=0.01):
        """
        Plots a raster heatmap of spike activity, limited to the channels specified in `self.config["channels"]`.

        - X-axis: Time (seconds)
        - Y-axis: Restricted channel range
        - Color intensity: Spike occurrences per time bin.

        Parameters:
            - bin_size: Bin width in seconds for spike count aggregation.
        """
        fs = self.recording.get_sampling_frequency()
        ch_start, ch_end = self.config["channels"]  # Restrict to specified range
        num_channels = ch_end - ch_start  # Number of channels in the range

        # Load the correct spike times based on method
        if self.config["mua_method"] == "threshold":
            spike_data = self.spike_times  # Use detected peaks
            print("Using threshold-based MUA spike data for raster plot.")
        elif self.config["mua_method"] == "kilosort":
            print("Using Kilosort spike data for raster plot.")
            spike_data = {
                unit: self.sorting.get_unit_spike_train(unit) / fs
                for unit in self.sorting.unit_ids
            }
        else:
            raise ValueError("Invalid `mua_method`. Must be 'threshold' or 'kilosort'.")

        # Ensure there are detected spikes
        if not spike_data or len(spike_data) == 0:
            print("No spikes detected. Cannot generate raster heatmap.")
            return

        # Determine total duration of the recording
        total_duration = self.recording.get_total_duration()
        num_bins = int(total_duration / bin_size)
        time_bins = np.linspace(0, total_duration, num_bins + 1)

        # Create an empty raster matrix: rows = channels in range, cols = time bins
        spike_counts = np.zeros((num_channels, num_bins))

        # Populate the raster heatmap matrix
        for ch in range(ch_start, ch_end):  # Only process channels in the defined range
            if ch in spike_data:
                spikes = spike_data[ch]
                spike_times = spikes["spike_times"] if isinstance(spikes, dict) else spikes  # Handle both data formats

                bin_indices = np.digitize(spike_times, time_bins) - 1  # Convert spike times to bin indices
                bin_indices = bin_indices[(bin_indices >= 0) & (bin_indices < num_bins)]  # Ensure valid indices

                for idx in bin_indices:
                    spike_counts[ch - ch_start, idx] += 1  # Adjust index to fit restricted range

        # Flip the spike count matrix so the lowest channel in the range starts at the bottom
        spike_counts = np.flipud(spike_counts)

        # Plot heatmap
        plt.figure(figsize=(12, 6))
        plt.imshow(spike_counts, aspect="auto", interpolation="nearest", cmap="hot",
                   extent=[0, total_duration, ch_start, ch_end])

        plt.colorbar(label="Spike Count")
        plt.xlabel("Time (s)")
        plt.ylabel("Channel Number")
        plt.title(
            f"Raster Heatmap of {'Threshold-Based' if self.config['mua_method'] == 'threshold' else 'Kilosort-Based'} Spiking Activity")

        # Flip Y-axis labels so the lowest selected channel is at the bottom
        plt.gca().invert_yaxis()

        # Save the chart
        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mua_method']}_raster_heatmap.png")
        self.save_chart(save_path)

        if self.config['show_plot']:
            plt.show()

    def plot_unit_response(self, traces, sparse=False):
        """
        Plots a direction-specific spike raster across time.

        - One subplot per stimulus direction (0° to 315°)
        - X-axis: Time (s) relative to trial start
        - Y-axis: Unit index (sparse=True) or channel number (sparse=False)
        - Each spike is a vertical tick
        """
        fs = self.recording.get_sampling_frequency()

        if sparse:
            # direction -> unit -> (channel, [spike_times], [amplitudes])
            direction_units = {d: {} for d in range(8)}
            all_amplitudes = []

            for ch, entries in traces.items():
                for segment, onset_frame, spike_array, direction, unit, amp_array in entries:
                    if not isinstance(spike_array, np.ndarray) or spike_array.size == 0:
                        continue
                    spike_times = spike_array / fs
                    amps = amp_array

                    if unit not in direction_units[direction]:
                        direction_units[direction][unit] = (ch, [], [])
                    direction_units[direction][unit][1].extend(spike_times)
                    direction_units[direction][unit][2].extend(amps)

                    all_amplitudes.extend(amps)

            if not all_amplitudes:
                print("No amplitudes found for any spikes. Aborting raster plot.")
                return

            # Setup colormap
            colormap = cm.viridis
            norm = mcolors.Normalize(vmin=np.percentile(all_amplitudes, 5),
                                     vmax=np.percentile(all_amplitudes, 95))

            unique_units = set()
            for unit_dict in direction_units.values():
                unique_units.update(unit_dict.keys())

            sorted_units = sorted(unique_units, key=lambda u: min(
                direction_units[d][u][0] for d in direction_units if u in direction_units[d]))
            unit_to_y = {unit: i for i, unit in enumerate(sorted_units)}

            fig, axes = plt.subplots(1, 8, figsize=(32, 6), sharey=True)

            for direction in range(8):
                ax = axes[direction]
                ax.set_title(f"{direction * 45}°")

                for unit, (ch, times, amps) in direction_units[direction].items():
                    y = unit_to_y[unit]
                    colors = [colormap(norm(a)) for a in amps]
                    ax.vlines(times, y - 0.4, y + 0.4, color=colors, linewidth=0.5)

                ax.set_xlim(-0.25, 0.75)
                ax.set_ylim(-0.5, len(sorted_units) - 0.5)
                ax.axvline(0, color="red", linestyle="--", linewidth=0.5)
                ax.add_patch(plt.Rectangle((0, -0.5), 0.5, len(sorted_units), color="red", alpha=0.1))
                ax.set_xlabel("Time (s)")
                ax.set_yticks([unit_to_y[u] for u in sorted_units])
                ax.set_yticklabels([
                    f"Unit {u} (ch {min(direction_units[d][u][0] for d in direction_units if u in direction_units[d])})"
                    for u in sorted_units], fontsize=2)

            axes[0].set_ylabel("Unit (by channel)")
            plt.suptitle("Spike Raster by Direction")

            # Add shared colorbar
            sm = cm.ScalarMappable(norm=norm, cmap=colormap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01)
            cbar.set_label("Amplitude (µV)", fontsize=10)

        else:
            ch_start, ch_end = self.config["channels"]
            direction_spikes = {d: [] for d in range(8)}

            for ch in range(ch_start, ch_end):
                if ch not in traces:
                    continue
                for segment, onset_frame, spike_array, direction, unit, _ in traces[ch]:
                    if not isinstance(spike_array, np.ndarray) or spike_array.size == 0:
                        continue
                    spike_times = spike_array / fs
                    direction_spikes[direction].extend((t, ch) for t in spike_times)

            fig, axes = plt.subplots(1, 8, figsize=(32, 6), sharey=True)

            for direction in range(8):
                ax = axes[direction]
                ax.set_title(f"{direction * 45}°")

                if direction_spikes[direction]:
                    spike_times, y_values = zip(*direction_spikes[direction])
                    ax.vlines(spike_times, [y - 0.4 for y in y_values], [y + 0.4 for y in y_values], color="black",
                              linewidth=0.5)

                ax.set_xlim(-0.25, 0.75)
                ax.set_ylim(ch_start, ch_end)
                ax.axvline(0, color="red", linestyle="--", linewidth=0.5)
                ax.add_patch(plt.Rectangle((0, ch_start), 0.5, ch_end - ch_start, color="red", alpha=0.1))
                ax.set_xlabel("Time (s)")

            axes[0].set_ylabel("Channel Number")
            plt.suptitle("Spike Raster by Direction (Full Channel View)")

        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mua_method']}_raster_by_direction-s{sparse}.png")
        self.save_chart(save_path)

        if self.config['show_plot']:
            plt.show()

    def plot_unit_spiking_data(self, unit_spike_dict, directions=None,
                               window=(-0.25, 0.75), bin_size=0.01):
        """
        Plots raster (top) and PSTH (bottom) for each unit, across channels and directions.

        Data structure must be:
            unit_spike_dict[unit_id][ch_id][direction] -> list of spike-time arrays
        where each list element is the spike times (in seconds) for one trial.

        We create one figure per unit. Inside that figure, we lay out each channel
        in a subgrid. Then, for each direction, we create two Axes: one for the raster
        and one for the PSTH, and fill them *immediately* after creating them.
        """
        if directions is None:
            directions = [0, 45, 90, 135, 180, 225, 270, 315]

        ch_start, ch_end = self.config["channels"]
        ch_ids = list(range(ch_start, ch_end + 1))

        def get_raster_psth(spike_trains):
            return self.get_raster_psth(
                spike_trains=spike_trains,
                bin_size=bin_size,
                window=window,
                return_rate=True
            )

        # for unit_id in sorted(unit_spike_dict.keys()):
        for unit_id in range(1):
            n_cols = 2
            n_rows = math.ceil(len(ch_ids) / n_cols)
            fig_width = 5 * n_cols
            fig_height = 4.5 * n_rows
            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = fig.add_gridspec(n_rows, n_cols, wspace=0.1, hspace=0.1)

            # Save all axes per channel so we can draw a bounding box later
            axes_per_channel = {}

            def make_channel_subgrid(outer_cell, ch_id):
                n_dirs = len(directions)
                dir_width = 1.5
                n_total_cols = n_dirs + 3

                outer = outer_cell.subgridspec(
                    3, n_total_cols,
                    height_ratios=[1, 1, 0.15],
                    width_ratios=[0.3] + [dir_width] * n_dirs + [dir_width, dir_width],
                    hspace=0.2, wspace=0.5
                )

                axes_for_dir = {}
                all_axes = []

                ax_ch_label = fig.add_subplot(outer[0:2, 0])
                ax_ch_label.axis("off")
                ax_ch_label.text(
                    0.5, 0.5,
                    f"Unit {unit_id}\nCh {ch_id}",
                    ha='center', va='center', rotation=90,
                    transform=ax_ch_label.transAxes, fontsize=5
                )
                all_axes.append(ax_ch_label)

                for d_idx, d_val in enumerate(directions):
                    col_idx = d_idx + 1
                    ax_raster = fig.add_subplot(outer[0, col_idx])
                    ax_psth = fig.add_subplot(outer[1, col_idx])

                    ax_raster.set_xticks([])
                    ax_raster.set_yticks([])
                    ax_psth.set_xticks([])
                    ax_psth.set_yticks([])

                    ax_label = fig.add_subplot(outer[2, col_idx])
                    ax_label.axis("off")
                    ax_label.text(0.5, 0.5, f"{d_val}°", ha='center', va='center', fontsize=5)

                    axes_for_dir[d_val] = (ax_raster, ax_psth)
                    all_axes.extend([ax_raster, ax_psth, ax_label])

                ax_wave = fig.add_subplot(outer[0, -2])
                ax_wave.set_title("Waveform", fontsize=5)
                ax_wave.set_xticks([])
                ax_wave.set_yticks([])

                ax_ac = fig.add_subplot(outer[1, -2])
                ax_ac.set_title("AutoCorr", fontsize=5)
                ax_ac.set_xticks([])
                ax_ac.set_yticks([])

                ax_dummy = fig.add_subplot(outer[0:2, -1])
                ax_dummy.axis("off")

                all_axes.extend([ax_wave, ax_ac, ax_dummy])

                return axes_for_dir, ax_wave, ax_ac, all_axes

            for i, ch_id in enumerate(ch_ids):
                row = n_rows - 1 - (i // 2)
                col = i % 2
                outer_cell = gs[row, col]

                axes_for_dir, ax_wave, ax_ac, chan_axes = make_channel_subgrid(outer_cell, ch_id)
                axes_per_channel[ch_id] = chan_axes

                for d_val in directions:
                    ax_raster, ax_psth = axes_for_dir[d_val]

                    spike_trains = (
                        unit_spike_dict.get(unit_id, {})
                        .get(ch_id, {})
                        .get(d_val, [])
                    )
                    if not spike_trains:
                        continue

                    raster_x, raster_y, bin_centers, psth = get_raster_psth(spike_trains)

                    ax_raster.scatter(raster_x, raster_y, s=2, color='black')
                    ax_raster.axvline(0, color='red', linestyle='--', linewidth=0.7)
                    ax_raster.set_xlim(window)
                    ax_raster.set_ylim(-0.5, len(spike_trains) - 0.5)

                    bar_width = (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 0.01
                    ax_psth.bar(bin_centers, psth, width=bar_width, color='gray')
                    ax_psth.axvline(0, color='red', linestyle='--', linewidth=0.7)
                    ax_psth.set_xlim(window)

            # Draw a bounding box for each channel block
            for ch_id, chan_axes in axes_per_channel.items():
                bbox = mtransforms.Bbox.union([ax.get_position(fig) for ax in chan_axes])
                pad_x = 0.03
                pad_y = 0.015
                rect = patches.Rectangle(
                    (bbox.x0 - pad_x, bbox.y0 - pad_y),
                    bbox.width + 2 * pad_x,
                    bbox.height + 2 * pad_y,
                    transform=fig.transFigure,
                    linewidth=0.7,
                    edgecolor='black',
                    facecolor='none'
                )
                fig.patches.append(rect)

            fig.suptitle(f"Spiking Data (Unit {unit_id}) by Channel & Direction", fontsize=6)
            save_path = (
                f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                f"unit_{unit_id}_spiking_data.png"
            )
            self.save_chart(save_path)

            if self.config['show_plot']:
                plt.show()

    def plot_unit_waveforms(self, unit_id, mode="median"):
        """
        Plots the aggregated waveforms for a given unit from the 'templates' extension.
        Valid modes are 'average', 'median', or 'std'.

        Returns nothing, just displays a Matplotlib figure.
        """
        if self.analyzer is None:
            raise ValueError("Analyzer not loaded. Run `process_analyzer()` first.")

        templates_ext = self.analyzer.get_extension("templates")
        # This will raise if 'mode' isn't one of the computed operators
        templates_data = templates_ext.get_data(mode)  # shape => (num_units, num_channels, num_samples)

        # Find which row in templates_data corresponds to this unit
        unit_ids = list(self.sorting.unit_ids)
        if unit_id not in unit_ids:
            print(f"Unit {unit_id} not found in sorting.")
            return
        unit_index = unit_ids.index(unit_id)

        waveform_2d = templates_data[unit_index]  # shape => (num_channels, num_samples)

        # Build time axis in ms
        fs = self.analyzer.recording.get_sampling_frequency()
        num_channels, num_samples = waveform_2d.shape
        t_axis = np.arange(num_samples) / fs * 1000.0

        # Plot
        plt.figure(figsize=(8, 6))
        offset_step = 40
        for ch_idx in range(num_channels):
            offset = ch_idx * offset_step
            plt.plot(t_axis, waveform_2d[ch_idx] + offset, color='black')

        plt.title(f"Unit {unit_id} {mode} Waveforms (Templates Extension)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude + offset (µV)")
        plt.show()

    def plot_waveform_traces(self, test_mode=False):
        """
        Plots overlapping spike waveforms, color-coded by channel.
        Only plots waveforms from channels in self.config["channels_sample"].
        Uses synthetic data if test_mode=True.
        """

        # Extract selected channel range
        ch_start, ch_end = self.config["channels_sample"]

        # Extract waveforms or generate synthetic ones for debugging
        if test_mode:
            waveforms, time_vector = self.generate_synthetic_waveforms()
            num_channels = waveforms.shape[1]
            print("Using synthetic waveforms for testing.")
        else:
            waveforms_ext = self.analyzer.get_extension("waveforms")
            waveforms = waveforms_ext.get_data()
            fs = self.analyzer.recording.get_sampling_frequency()
            num_samples = waveforms.shape[2]
            time_vector = np.linspace(0, num_samples / fs * 1000, num_samples)  # Convert to ms
            num_channels = waveforms.shape[1]

        num_spikes = waveforms.shape[0]

        # Identify indices corresponding to selected channel range
        selected_channel_indices = [ch for ch in range(num_channels) if ch_start <= ch < ch_end]

        # Select a reduced set of spikes per channel for clarity
        max_spikes_per_channel = 30  # Reduce number for visualization
        selected_waveforms = []
        selected_channels = []

        for ch in selected_channel_indices:
            available_spikes = waveforms[:, ch, :]
            num_available = available_spikes.shape[0]
            num_selected = min(max_spikes_per_channel, num_available)

            if num_available > 0:
                selected_indices = random.sample(range(num_available), num_selected)
                selected_waveforms.extend(available_spikes[selected_indices])
                selected_channels.extend([ch] * num_selected)  # Track corresponding channels

        # Convert lists to arrays
        selected_waveforms = np.array(selected_waveforms)
        selected_channels = np.array(selected_channels)

        # Assign colors based on channels
        colormap = plt.colormaps["rainbow"]
        unique_channels = np.unique(selected_channels)
        channel_colors = {ch: colormap((ch - ch_start) / (ch_end - ch_start)) for ch in unique_channels}

        # Plot waveforms
        plt.figure(figsize=(12, 6))
        for i, waveform in enumerate(selected_waveforms):
            ch = selected_channels[i]
            plt.plot(time_vector, waveform, alpha=0.5, color=channel_colors[ch])

        # Create legend with assigned colors
        legend_handles = [plt.Line2D([0], [0], color=channel_colors[ch], lw=2, label=f"Ch {ch}")
                          for ch in unique_channels]
        plt.legend(handles=legend_handles, title="Channels", loc="upper right")

        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.title(f"Overlapping Spike Waveforms ({ch_start}-{ch_end})")

        # Save the chart
        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mua_method']}_waveform-traces.png")
        self.save_chart(save_path)

        if self.config['show_plot']:
            plt.show()

    def save_chart(self, save_path):
        """Creates the directory if it doesn't exist and saves the chart."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"Chart saved to {save_path}")

    def verify_stimulus_alignment(self):
        """Verifies that the number of stimulus directions matches the number of onset/offset pairs."""
        if self.directions is None or self.nidaq_data is None:
            print("Error: Directions or NIDAQ data are not loaded.")
            return False

        light_onsets = self.nidaq_data.get("light_onsets", [])
        light_offsets = self.nidaq_data.get("light_offsets", [])

        if len(self.directions) != len(light_onsets) or len(self.directions) != len(light_offsets):
            print("Error: Mismatch in the number of stimulus directions and onset/offset pairs.")
            self.plot_nidaq_signals()
            print(f"\t{len(self.directions)} segments. {len(light_onsets)} onsets and {len(light_offsets)} offsets.")
            return False

        print("Stimulus alignment verified successfully.")
        print(f"\t{len(self.directions)} segments and {len(light_onsets)} pairs.")
        return True

# -------------------------- MAIN SCRIPT --------------------------

def configure_experiment():
    """Defines experiment metadata and paths."""
    config = {
        "rerun": False,
        "sglx_folder": "SGL_DATA",
        "mouse_id": "mouse02",
        "gate": "1",
        "probe": "0",
        "channels": (0, 3),
        "channels_sample": (0, 19),
        "skip_sort": True,
        "write_concat": True,
        "mua_method": "kilosort",
        "processing_folder": "processing",
        "direction": "negative",  # Can be "negative" or "positive" for detecting positive or negative peaks
        "show_plot": False,
        "style": "channel",     # Select between the (direction) or (channel) based analysis, or (both)
        "insertion_depth": 3000,
        "target_depth": 1100,
        "use_sdf": True
    }

    # Define paths **after** initializing config
    base_path = f'{config["sglx_folder"]}/{config["mouse_id"]}/{config["mouse_id"]}_g{config["gate"]}'
    run_path = f'{base_path}/{config["mouse_id"]}_g{config["gate"]}_imec{config["probe"]}'

    config["base_path"] = base_path
    config["run_path"] = run_path

    config["paths"] = {
        "preprocessed_ap": f"{config['processing_folder']}/concat/{config['mouse_id']}_g{config['gate']}_imec{config['probe']}/ap",
        "preprocessed_NIDAQ": f"{config['processing_folder']}/concat/{config['mouse_id']}_g{config['gate']}_imec{config['probe']}/NIDAQ",
        "kilosort": f"{config['processing_folder']}/kilosort/{config['mouse_id']}_g{config['gate']}",
        "mua": f"{config['processing_folder']}/mua_spikes/{config['mouse_id']}_g{config['gate']}.npy",
        "waveforms": f"{config['processing_folder']}/waveforms/{config['mouse_id']}_g{config['gate']}",
        "analyzer": f"{config['processing_folder']}/sorting_analyzer/{config['mouse_id']}_g{config['gate']}",
        "stimulus_directions": "processing/stimulus/StimulusDirections.mat"
    }

    return config


def test_new_helpers(processor):
    """
    Quick test to ensure our new helper methods work as expected.
    1) Pick the first unit from the sorting output.
    2) Compute and plot autocorrelation for that unit.
    3) Fetch and plot waveforms for that unit.
    4) Generate a tiny toy spike_trains list and compute a raster + PSTH to verify.
    5) Check channel layout dictionary from recording.
    """
    sorting = processor.sorting
    fs = processor.recording.get_sampling_frequency()

    if sorting is None:
        print("No sorting found. Cannot run test_new_helpers.")
        return

    # 1) Pick the first unit (if any).
    unit_ids = sorting.unit_ids
    if len(unit_ids) == 0:
        print("No units found in sorting. Cannot run test_new_helpers.")
        return

    test_unit = unit_ids[0]
    print(f"Testing helpers with unit {test_unit}...")

    # 2) Autocorrelation
    spikes = sorting.get_unit_spike_train(test_unit)  # in frames
    times_s = spikes / fs  # convert to seconds
    if len(times_s) > 1:
        bins, acorr = processor.get_autocorrelation(times_s, bin_size=0.001, max_lag=0.05)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        plt.figure(figsize=(6, 3))
        plt.bar(bin_centers, acorr, width=0.001, align='center', color='k')
        plt.axvline(0, color='r', linestyle='--', linewidth=1)
        plt.xlabel("Lag (s)")
        plt.ylabel("Count")
        plt.title(f"Autocorr - Unit {test_unit}")
        plt.tight_layout()
        plt.show(block=False)  # or block=True if you want to pause here
    else:
        print("Not enough spikes for meaningful autocorrelation. Skipping AC plot.")

    # 3) Waveforms (aggregated)
    # Make sure we have waveforms computed via process_analyzer() -> waveforms extension
    # Just directly call our plotting function
    processor.plot_unit_waveforms(unit_id=test_unit)


    # 4) Tiny test for raster/PSTH (synthetic or from real data).
    # We'll build a toy spike_trains list for 5 trials:
    fake_spike_trains = []
    rng = np.random.default_rng(42)
    for _ in range(5):
        # ~5 to 20 random spikes in [-0.2s, 0.5s]
        n_spikes = rng.integers(5, 20)
        spk_times = rng.uniform(-0.2, 0.5, size=n_spikes)
        fake_spike_trains.append(np.sort(spk_times))

    raster_x, raster_y, bin_centers, psth = processor.get_raster_psth(
        spike_trains=fake_spike_trains,
        bin_size=0.01,
        window=(-0.25, 0.75),
        return_rate=True
    )

    # Plot them side by side
    plt.figure(figsize=(6, 5))

    # Subplot 1: Raster
    plt.subplot(2, 1, 1)
    plt.scatter(raster_x, raster_y, s=8, color='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.ylabel("Trial #")
    plt.title("Raster (toy data)")

    # Subplot 2: PSTH
    plt.subplot(2, 1, 2)
    plt.bar(bin_centers, psth, width=0.01, align='center', color='gray')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (spikes/s)")
    plt.title("PSTH (toy data)")
    plt.tight_layout()
    plt.show(block=False)

    # 5) Check channel layout
    try:
        layout = processor.get_channel_layout()
        n_rows = max(rc[0] for rc in layout.values()) + 1
        n_cols = max(rc[1] for rc in layout.values()) + 1
        print(f"Channel layout: {n_rows} rows × {n_cols} columns")
        # Print snippet for first few channels
        some_ch = list(layout.keys())[:5]
        print("Example layout mapping for a few channels:")
        for ch in some_ch:
            print(f"  Channel {ch} -> (row={layout[ch][0]}, col={layout[ch][1]})")
    except Exception as e:
        print(f"Channel layout error: {e}")

    print("\nTest of new helpers completed.\n")

if __name__ == "__main__":
    config = configure_experiment()
    processor = NeuralDataProcessor(config)
    processor.load_directions()
    processor.load_nidaq()
    processor.detect_range()
    if not processor.verify_stimulus_alignment():
        exit()
    processor.load_recording()
    processor.run_kilosort()
    processor.process_analyzer()
    data_structure = processor.process_segments()
    channel_spike_dict = processor.get_channel_spike_dict(data_structure)
    print(channel_spike_dict.keys())
    processor.plot_unit_spiking_data(channel_spike_dict)
    exit()
    # traces = processor.process_segments()
    # processor.plot_unit_response(traces, sparse=True)
    # processor.plot_unit_response(traces, sparse=False)
    # processor.plot_directions_raster(traces)

    # Process MUA
    # normalized_psth = processor.process_mua(traces)

    # Extract and save MUA spike times
    # processor.extract_mua_spikes()

    # Visualization
    print("\nGenerating plots...\n")
    # processor.plot_mua_line(normalized_psth)
    # processor.plot_mua_heatmap(normalized_psth)
    # processor.plot_filtered_traces()  # Plot threshold-based spike detection
    # processor.plot_mua_activity()  # Plot MUA event count per channel
    # processor.plot_raster_heatmap()
    # processor.plot_ap_amplitudes()  # Plot action potential amplitude distributions
    # processor.plot_waveform_traces()  # Plot individual spike waveforms
    # processor.plot_channel_templates()  # Plot template waveforms for selected channels
