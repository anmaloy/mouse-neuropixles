import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.io
import shutil
import subprocess
import spikeinterface.core as sic
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import warnings

from io import BytesIO
from PIL import Image
from probeinterface.plotting import plot_probe


class ProcessUnit:
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
        self.spike_times = None
        self.square_wave_data = None
        self.square_wave_alignment = None
        self.waveforms = None
        self.stim_directions = None
        self.sampling_rate = None
        self.trial_drift_alpha = 0.0035

        if self.config["rerun"]:
            self.cleanup_previous_processing()

    def calculate_direction_evocation(self):
        """
        Calculates a per-direction evocation index for each unit, normalized by the unit's
        mean across directions and squared to emphasize contrast.

        Returns:
            Dict[int, Dict[int, float]]: unit_id → {direction → normalized_evocation_score}
        """
        if self.sorting is None or self.nidaq_data is None or self.sampling_rate is None or self.directions is None:
            raise ValueError("Missing sorting, NIDAQ data, sampling rate, or stimulus directions.")

        light_onsets = self.nidaq_data.get("light_onsets", [])
        good_units = self.load_good_units()
        sr = self.sampling_rate
        pre_window = 0.25
        post_window = 0.5

        direction_index_dict = {}

        for unit_id in good_units:
            spike_train = self.sorting.get_unit_spike_train(unit_id) / sr
            unit_raw_evocations = {}

            # First pass: compute raw evocation per direction
            for direction in np.unique(self.directions):
                count_before = 0
                count_during = 0

                for trial_idx, onset in enumerate(light_onsets):
                    if self.directions[trial_idx] != direction:
                        continue

                    pre_mask = (spike_train >= (onset - pre_window)) & (spike_train < onset)
                    during_mask = (spike_train >= onset) & (spike_train < onset + post_window)
                    count_before += np.sum(pre_mask)
                    count_during += np.sum(during_mask)

                total = (count_before + (count_during / 2))
                if total == 0:
                    evocation = 0.0
                else:
                    evocation = ((count_during / 2) - count_before) / total

                unit_raw_evocations[direction] = evocation

            # Second pass: normalize by mean across directions and square
            evocation_values = list(unit_raw_evocations.values())
            mean_evocation = np.mean(evocation_values) if evocation_values else 1.0  # avoid div by 0

            unit_direction_scores = {}
            for direction, raw in unit_raw_evocations.items():
                score = (raw / mean_evocation) if mean_evocation > 0 else 0.0
                unit_direction_scores[direction] = score

            direction_index_dict[unit_id] = unit_direction_scores
        return direction_index_dict

    def calculate_light_evocation(self):
        """
        Calculates the light evocation index for each good unit.
        The index is defined as (spikes_during - spikes_before) / (spikes_during + spikes_before)
        using the time windows: -0.25s to 0s (before) and 0s to 0.5s (during) relative to light onset.

        Returns:
            Dict[int, float]: Mapping of unit_id to evocation index.
        """
        if self.sorting is None or self.nidaq_data is None or self.sampling_rate is None:
            raise ValueError("Missing sorting, NIDAQ data, or sampling rate.")

        light_onsets = self.nidaq_data.get("light_onsets", [])
        good_units = self.load_good_units()
        evocation_index_dict = {}

        pre_window = 0.25  # seconds before stimulus onset
        post_window = 0.5  # seconds during stimulus
        sr = self.sampling_rate

        for unit_id in good_units:
            spike_train = self.sorting.get_unit_spike_train(unit_id) / sr  # convert to seconds

            count_before = 0
            count_during = 0

            for onset in light_onsets:
                pre_mask = (spike_train >= (onset - pre_window)) & (spike_train < onset)
                during_mask = (spike_train >= onset) & (spike_train < onset + post_window)
                count_before += np.sum(pre_mask)
                count_during += np.sum(during_mask)

            total = (count_before + (count_during/2))
            if total == 0:
                evocation_index = 0.0
            else:
                evocation_index = ((count_during/2) - count_before) / total

            evocation_index_dict[unit_id] = evocation_index
        return evocation_index_dict

    def calculate_raster_data(self, unit_id, channel, direction):
        if self.sorting is None or self.nidaq_data is None or self.directions is None:
            raise ValueError("Missing required spike sorting, nidaq, or direction data.")

        spike_times = self.sorting.get_unit_spike_train(unit_id)
        light_onsets = np.array(self.nidaq_data.get("light_onsets", []))
        light_offsets = np.array(self.nidaq_data.get("light_offsets", []))
        light_onsets *= int(self.sampling_rate)
        light_offsets *= int(self.sampling_rate)

        if len(light_onsets) != len(self.directions):
            raise ValueError("Mismatch between number of light_onsets and directions.")

        raster_data = {}
        padding = int(0.25 * self.sampling_rate)

        for trial_idx, (onset, offset) in enumerate(zip(light_onsets, light_offsets)):
            if self.directions[trial_idx] != direction // 45:
                continue

            extended_onset = onset - padding
            extended_offset = offset + padding

            mask = (spike_times >= extended_onset) & (spike_times <= extended_offset)
            aligned_spikes = spike_times[mask] - onset  # relative to true stimulus onset
            ### Alpha Correction v1###
            # trial_number_for_direction = sum(1 for i in range(trial_idx) if self.directions[i] == direction // 45)
            # time_shift = int(
            #     self.trial_drift_alpha * self.sampling_rate * trial_number_for_direction - 0.25 * self.sampling_rate)
            # aligned_spikes = spike_times[mask] - onset + time_shift

            if channel not in raster_data:
                raster_data[channel] = []
            raster_data[channel].append((aligned_spikes, offset - onset))

        return raster_data

    def calculate_psth(self, unit_id, direction, bin_width):
        if self.sorting is None or self.nidaq_data is None or self.directions is None or self.sampling_rate is None:
            raise ValueError("Missing required spike sorting, nidaq, direction data, or sampling rate.")

        spike_vector = self.sorting.to_spike_vector()
        all_spike_units = spike_vector["unit_index"]

        light_onsets = np.array(self.nidaq_data.get("light_onsets", []))
        light_offsets = np.array(self.nidaq_data.get("light_offsets", []))
        light_onsets *= int(self.sampling_rate)
        light_offsets *= int(self.sampling_rate)

        if len(light_onsets) != len(self.directions):
            raise ValueError("Mismatch between number of light_onsets and directions.")

        padding = int(0.25 * self.sampling_rate)
        aligned_spikes_all = []
        num_trials = 0

        unit_mask = (all_spike_units == unit_id)
        unit_spike_times = spike_vector["sample_index"][unit_mask]

        for trial_idx, (onset, offset) in enumerate(zip(light_onsets, light_offsets)):
            if self.directions[trial_idx] != direction // 45:
                continue

            extended_onset = onset - padding
            extended_offset = offset + padding

            mask = (unit_spike_times >= extended_onset) & (unit_spike_times <= extended_offset)
            aligned_spikes = (unit_spike_times[mask] - onset) / self.sampling_rate  # convert to seconds
            ### Alpha Correction v1###
            # trial_number_for_direction = sum(1 for i in range(trial_idx) if self.directions[i] == direction // 45)
            # time_shift = self.trial_drift_alpha * trial_number_for_direction - 0.25
            # aligned_spikes = (unit_spike_times[mask] - onset) / self.sampling_rate + time_shift

            aligned_spikes_all.extend(aligned_spikes)
            num_trials += 1

        if num_trials == 0:
            return {"bin_centers": np.array([]), "firing_rates": np.array([])}

        bins = np.arange(-0.25, 0.75 + bin_width, bin_width)
        counts, _ = np.histogram(aligned_spikes_all, bins=bins)
        firing_rates = counts / (num_trials * bin_width)
        bin_centers = bins[:-1] + bin_width / 2

        return {"bin_centers": bin_centers, "firing_rates": firing_rates}

    def calculate_square_alignment(self, skip_edge_pulses=False):
        """
        Calculates pulse onset times from square wave signals in AP (SY0) and NIDAQ (XD0),
        then computes cumulative time correction needed to align NIDAQ to AP.

        Stores:
            self.square_wave_alignment["ap_times"]
            self.square_wave_alignment["nidq_times"]
            self.square_wave_alignment["offsets"]
            self.square_wave_alignment["cumulative"]
        """
        def detect_rising_edges(trace, time, min_interval=0.25):
            trace = np.asarray(trace)
            threshold = 0.5 * np.max(trace)
            binary = (trace > threshold).astype(np.uint8)
            rising = np.where(np.diff(binary) == 1)[0] + 1
            rising_times = time[rising]

            # Group by time difference > min_interval
            if len(rising_times) == 0:
                return rising_times

            grouped = [rising_times[0]]
            for t in rising_times[1:]:
                if t - grouped[-1] > min_interval:
                    grouped.append(t)

            return np.array(grouped)

        ap = self.square_wave_data["ap"]
        nidq = self.square_wave_data["nidq"]

        ap_times = detect_rising_edges(ap["trace"], ap["time"])
        nidq_times = detect_rising_edges(nidq["sync"], nidq["time"])

        if skip_edge_pulses:
            ap_times = ap_times[1:-1]
            nidq_times = nidq_times[1:-1]

        if len(ap_times) != len(nidq_times):
            raise ValueError(f"Mismatch in number of pulses: AP={len(ap_times)} vs NIDAQ={len(nidq_times)}")

        offsets = nidq_times - ap_times

        self.square_wave_alignment = {
            "ap_times": ap_times,
            "nidq_times": nidq_times,
            "offsets": offsets
        }
        # print(self.square_wave_alignment)

        print(f"Extracted {len(ap_times)} AP pulses and {len(nidq_times)} NIDAQ pulses.")

    def cleanup_previous_processing(self):
        """Deletes existing processed files if rerun is enabled."""
        print("Rerun enabled. Deleting existing processed files...")
        processing_dirs = [
            self.config["paths"]["preprocessed_ap"],
            self.config["paths"]["preprocessed_NIDAQ"],
            self.config["paths"]["kilosort"],
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
        middle_mean = np.mean(signal[mid_start_idx:mid_end_idx]) + 0.73*np.std(signal[mid_start_idx:mid_end_idx])
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

    def load_good_units(self):
        if self.config.get("run_phy"):
            # Phy curation output
            info_path = os.path.join(
                "processing", "phy", f"{self.config['mouse_id']}_g{self.config['gate']}", "cluster_info.tsv"
            )
            if not os.path.exists(info_path):
                raise FileNotFoundError(f"Missing cluster_info.tsv: {info_path}")

            df = pd.read_csv(info_path, sep="\t")
            if "group" not in df.columns or "KSLabel" not in df.columns:
                raise ValueError("cluster_info.tsv must contain both 'group' and 'KSLabel' columns.")

            # Fill missing group values with KSLabel, then select good units
            df["group"] = df["group"].fillna(df["KSLabel"])
            good_units = df[df["group"] == "good"]["cluster_id"].tolist()

        else:
            # Kilosort output (prior to Phy curation)
            group_path = os.path.join(self.config["paths"]["kilosort"], "sorter_output", "cluster_group.tsv")
            if not os.path.exists(group_path):
                raise FileNotFoundError(f"Missing cluster_group.tsv: {group_path}")

            df = pd.read_csv(group_path, sep="\t")
            if "cluster_id" not in df.columns or "KSLabel" not in df.columns:
                raise ValueError("cluster_group.tsv must contain 'unit' and 'KSLabel' columns.")

            good_units = df[df["KSLabel"] == "good"]["cluster_id"].tolist()

        return good_units

    def load_nidaq(self):
        """Loads the NIDAQ binary file and extracts the photodiode and sync pulse channels."""
        preprocessed_path = self.config["paths"]["preprocessed_NIDAQ"]

        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed recording from {preprocessed_path}...")
            self.nidaq_recording = se.read_spikeglx(self.config["base_path"], stream_name='nidq')
            # self.sampling_rate = self.nidaq_recording.get_sampling_frequency()
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
            # self.sampling_rate = self.nidaq_recording.get_sampling_frequency()

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
            # self.recording = se.read_spikeglx(self.config["run_path"], stream_name=f'imec{self.config["probe"]}.ap')
            self.recording = sic.load(preprocessed_path)
            self.sampling_rate = self.recording.get_sampling_frequency()
        else:
            print("Processing preprocessed recording from raw data...")
            self.recording = se.read_spikeglx(self.config["run_path"], stream_name=f'imec{self.config["probe"]}.ap')
            self.sampling_rate = self.recording.get_sampling_frequency()

            # Preprocessing steps
            self.recording = spre.bandpass_filter(self.recording, freq_min=300, freq_max=6000)
            self.recording = spre.common_reference(self.recording, operator="median")
            self.recording_concat = sic.concatenate_recordings([self.recording])

            # Save if enabled
            if self.config["write_concat"]:
                self.recording_concat = self.recording_concat.save(format="binary", folder=preprocessed_path)

    def load_square_waves(self):
        """
        Loads the SY0 (AP) and XD0 (NIDAQ) square wave synchronization signals from their respective raw binary files.

        Stores the output in:
            self.square_wave_data["ap"] = {"trace": ..., "time": ..., "fs": ...}
            self.square_wave_data["nidq"] = {"trace": ..., "time": ..., "fs": ...}
        """

        print("Loading square wave signals from raw binaries...")

        # ---------- Load AP Band SY0 ----------
        ap_bin = os.path.join(self.config["run_path"],
                              f"{self.config['mouse_id']}_g{self.config['gate']}_t0.imec{self.config['probe']}.ap.bin")
        ap_meta = ap_bin.replace(".ap.bin", ".ap.meta")
        with open(ap_meta, "r") as f:
            ap_lines = f.readlines()
        ap_meta_dict = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in ap_lines if '=' in line}
        ap_fs = float(ap_meta_dict["imSampRate"])
        ap_n_chans = int(ap_meta_dict["nSavedChans"])

        ap_raw = np.memmap(ap_bin, dtype=np.int16, mode='r')
        ap_raw = ap_raw.reshape(-1, ap_n_chans)
        sy0_trace = ap_raw[:, 384].astype(np.int16)
        sy0_time = np.arange(len(sy0_trace)) / ap_fs

        # ---------- Load NIDAQ Band XD0 ----------
        nidq_bin = os.path.join(self.config["base_path"],
                                f"{self.config['mouse_id']}_g{self.config['gate']}_t0.nidq.bin")
        nidq_meta = nidq_bin.replace(".bin", ".meta")
        with open(nidq_meta, "r") as f:
            nidq_lines = f.readlines()
        nidq_meta_dict = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in nidq_lines if '=' in line}
        nidq_fs = float(nidq_meta_dict["niSampRate"])
        nidq_n_chans = int(nidq_meta_dict["nSavedChans"])

        nidq_raw = np.memmap(nidq_bin, dtype=np.int16, mode='r')
        nidq_raw = nidq_raw.reshape(-1, nidq_n_chans)
        xd0_trace = nidq_raw[:, 8].astype(np.int16)
        xd0_time = np.arange(len(xd0_trace)) / nidq_fs


        xd0_raw = nidq_raw[:, 8].astype(np.uint16)
        # Show which bits are used in XD0
        unique_vals = np.unique(xd0_raw)
        used_bits = set()
        for val in unique_vals:
            for bit in range(16):
                if val & (1 << bit):
                    used_bits.add(bit)

        print(f"Bits used in XD0: {sorted(used_bits)}")

        sync_trace = (xd0_raw & (1 << 0)) > 0  # Sync signal
        stim_trace = (xd0_raw & (1 << 2)) > 0  # Stimulus signal

        # Swaps photodiode onset for TTL onset
        # ---------------------------------------
        # Detect rising edges of the stimulus TTL (bit 2 of XD0) to get stimulus start times in NIDAQ clock
        stim_binary = stim_trace.astype(np.uint8)
        stim_rise_idx = np.where(np.diff(stim_binary) == 1)[0] + 1
        stim_onsets_nidq = xd0_time[stim_rise_idx]

        # Overwrite the photodiode-based onsets with TTL-based onsets
        # Keep offsets as-is (photodiode) so segment ends still come from measured light offset
        if hasattr(self, "nidaq_data"):
            self.nidaq_data["light_onsets"] = stim_onsets_nidq
        else:
            # safety fallback, but in your call order load_nidaq() already ran so nidaq_data exists
            self.nidaq_data = {
                "light_onsets": stim_onsets_nidq,
            }
        # ------------------------------------

        xd0_time = np.arange(len(sync_trace)) / nidq_fs

        # ---------- Store Results ----------
        self.square_wave_data = {
            "ap": {"trace": sy0_trace, "time": sy0_time, "fs": ap_fs},
            "nidq": {
                "sync": sync_trace.astype(np.uint8),
                "stim": stim_trace.astype(np.uint8),
                "time": xd0_time,
                "fs": nidq_fs
            }
        }

        print(f"Loaded SY0 from AP: {len(sy0_trace)} samples at {ap_fs:.2f} Hz")
        print(f"Loaded XD0 from NIDAQ: {len(xd0_trace)} samples at {nidq_fs:.2f} Hz")

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

    def plot_unit_spiking_data(self):
        unit_best_channels = sic.get_template_extremum_channel(self.analyzer, peak_sign="both", mode="extremum",
                                                               outputs="index")
        units = self.load_good_units()
        print(f"Plotting: {len(units)} Units")

        stacked_figure = []

        for unit_id in units:
        # for unit_id in range(5):
            best_channel = unit_best_channels[unit_id]
            l_ch_labels = self.analyzer.sparsity.unit_id_to_channel_ids[unit_id]
            l_channels = [int(label.split("AP")[-1]) for label in l_ch_labels]

            # Create GridSpec-based layout
            fig = plt.figure(figsize=(20, 9))
            gs = fig.add_gridspec(3, 10)
            axs = np.empty((3, 10), dtype=object)

            for row in range(3):
                for col in range(10):
                    if col == 9:
                        continue  # skip column 9 — we'll manually add ax_probe below
                    axs[row, col] = fig.add_subplot(gs[row, col])

            fig.subplots_adjust(hspace=0.4, top=0.85)
            fig.suptitle(f"Unit {unit_id} Spiking Summary (Channel {best_channel})", fontsize=16)

            direction_angles = [0, 45, 90, 135, 180, 225, 270, 315]

            ax_rasters = []
            ax_psths = []
            psth_bin_width = 0.02 # in s

            psth_data_all = {}
            global_ymax = 0

            for angle in direction_angles:
                psth_data = self.calculate_psth(unit_id, angle, psth_bin_width)
                psth_data_all[angle] = psth_data
                if psth_data["firing_rates"].size > 0:
                    global_ymax = max(global_ymax, psth_data["firing_rates"].max())

            for i, angle in enumerate(direction_angles):
                ax_raster = axs[0, i]
                ax_psth = axs[1, i]

                raster_data = self.calculate_raster_data(unit_id, best_channel, angle)
                psth_data = psth_data_all[angle]

                self.plot_raster(ax_raster, raster_data, show_y_axis=(i == 0))
                self.plot_psth(ax_psth, psth_data, psth_bin_width, ymax=global_ymax, show_y_axis=(i == 0))

                ax_rasters.append(ax_raster)
                ax_psths.append(ax_psth)

            ax_waveform = axs[0, 8]
            ax_autocorr = axs[1, 8]

            # Assign ax_probe separately to span all rows of column 9
            ax_probe = fig.add_subplot(gs[0:3, 9])

            self.plot_waveform(ax_waveform, unit_id, best_channel)
            ax_waveform.set_title(f"Waveform (ch {best_channel})", pad=4)

            self.plot_autocorrelation(ax_autocorr, unit_id)
            ax_autocorr.set_title("Autocorrelogram", pad=4)
            ax_autocorr.set_xticks([])
            ax_autocorr.set_yticks([])

            self.plot_probe_channel_map(ax_probe, l_ch_labels)
            ax_probe.set_title(f"Probe Map", pad=4)
            ax_probe.set_xticks([])

            # Bottom waveform row: find up to 9 channels centered around best
            if best_channel in l_channels:
                best_idx = l_channels.index(best_channel)
                start = max(0, best_idx - 4)
                end = min(len(l_channels), start + 9)
                start = max(0, end - 9)  # Adjust back if we're at the end
                selected_channels = l_channels[start:end]

                for i, ch in enumerate(selected_channels):
                    ax = axs[2, i]
                    self.plot_waveform(ax, unit_id, ch)
                    ax.text(0.5, -0.2, f"Ch {ch}", ha='center', va='top', transform=ax.transAxes, fontsize=12)

            # Move direction labels to row 2 (bottom row)
            for i, angle in enumerate(direction_angles):
                pos = axs[1, i].get_position()
                fig.text(pos.x0 + pos.width / 2, pos.y0 - 0.04, f"{angle}°", ha='center', va='top', fontsize=12)

            mid_raster = ax_rasters[4]
            mid_psth = ax_psths[4]
            fig.text(
                mid_raster.get_position().x0 + mid_raster.get_position().width / 2,
                mid_raster.get_position().y1 + 0.01,
                "Raster",
                ha='center', va='bottom', fontsize=12
            )
            fig.text(
                mid_psth.get_position().x0 + mid_psth.get_position().width / 2,
                mid_psth.get_position().y1 + 0.01,
                "PSTH",
                ha='center', va='bottom', fontsize=12
            )

            self.stack_plots(stacked_figure, fig)

            save_path = (
                f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                f"spiking_data/unit_{unit_id}_ch_{best_channel}_spiking_data.png"
            )
            self.save_chart(save_path)

            if self.config['show_plot']:
                plt.show()
        self.save_stacked_plot(stacked_figure, f"output/{self.config['mouse_id']}/"
                f"{self.config['mouse_id']}_gate_{self.config['gate']}_spiking_data.tiff"
            )

    def plot_autocorrelation(self, ax, unit_id):
        sw.plot_autocorrelograms(
            self.analyzer,
            unit_ids=[unit_id],
            backend="matplotlib"
        )
        fig_temp = plt.gcf()

        for fig_ax in plt.gcf().axes:
            if fig_ax.patches:
                for patch in fig_ax.patches:
                    xy = patch.get_xy()
                    width = patch.get_width()
                    height = patch.get_height()
                    ax.bar(xy[0], height, width=width, align='edge', color=patch.get_facecolor())
                ax.set_xlim(fig_ax.get_xlim())
                ax.set_ylim(fig_ax.get_ylim())
                break
        plt.close(fig_temp)

        ax.set_xticks([])
        ax.set_yticks([])

    def plot_direction_evocation(self):
        """
        Plots a row of probe maps (1 per direction) showing evocation scores
        computed per unit and direction. Channel color = max score across units at that site.
        """
        direction_data = self.calculate_direction_evocation()
        unit_best_channels = sic.get_template_extremum_channel(
            self.analyzer, peak_sign="both", mode="extremum", outputs="index"
        )
        all_ch_ids = self.recording.get_channel_ids()
        channel_id_map = [int(ch.split("AP")[-1]) for ch in all_ch_ids]
        directions = sorted(np.unique(self.directions))

        direction_channel_maps = []
        all_values = []

        # Aggregate per-direction evocation maps
        for direction in directions:
            channel_values = {}
            for unit_id, dir_scores in direction_data.items():
                ch = unit_best_channels.get(unit_id)
                if ch is None:
                    continue
                evocation = dir_scores.get(direction, 0.0)
                if ch not in channel_values or evocation > channel_values[ch]:
                    channel_values[ch] = evocation

            values = np.zeros(len(all_ch_ids))
            for i, ch_int in enumerate(channel_id_map):
                values[i] = channel_values.get(ch_int, 0.0)

            direction_channel_maps.append(values)
            all_values.append(values)

        # Use global abs max for scaling
        absmax = max(np.abs(np.concatenate(all_values)).max(), 1e-6)
        cmap = matplotlib.colormaps["bwr"]
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)

        fig, axes = plt.subplots(1, len(directions), figsize=(20, 10), sharey=True)
        fig.subplots_adjust(wspace=0.05)

        for i, direction in enumerate(directions):
            ax = axes[i]
            values = direction_channel_maps[i]
            scaled_values = values / absmax

            contacts_colors = [
                "black" if val == 0 else cmap((val + 1) / 2)
                for val in scaled_values
            ]

            probe = self.recording.get_probe()
            plot_probe(
                probe, ax=ax,
                contacts_colors=contacts_colors,
                probe_shape_kwargs=dict(edgecolor="white", facecolor="black", linewidth=0.2)
            )

            ax.set_title(f"{direction}°")
            ax.set_xticks([])
            ax.set_xlabel("")
            ax.set_ylim(-200, 3900)
            ax.set_xlim(-14, 62)

            # Channel tick labels
            tick_spacing = 15   # 15 for 2.0, 20 for 1.0
            tick_indices = np.arange(0, max(channel_id_map) + 1, tick_spacing)
            tick_positions = tick_spacing * ((tick_indices - 1) // 2)
            tick_labels = [str(ch) for ch in tick_indices]

            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel("Channel")

        # Shared colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Evocation Index (Activation ↔ Inhibition)", rotation=270, labelpad=12)
        cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

        fig.suptitle("Per-Direction Evocation Maps", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.93, 1])

        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mouse_id']}-gate_{self.config['gate']}-direction_evocation.png")
        self.save_chart(save_path, dpi=500)

        if self.config['show_plot']:
            matplotlib.use("TkAgg")
            plt.show()

    def plot_light_evocation(self):
        """
        Plots a probe map where each channel is colored by the highest evocation index
        of any unit assigned to that channel (using best channel logic).
        """
        evocation_dict = self.calculate_light_evocation()
        best_channels = sic.get_template_extremum_channel(self.analyzer, peak_sign="both",
                                                          mode="extremum", outputs="index")

        # Build channel -> evocation value map (maximum per channel)
        channel_values = {}
        for unit_id, index_value in evocation_dict.items():
            ch = best_channels.get(unit_id)
            if ch is None:
                continue
            if ch not in channel_values or index_value > channel_values[ch]:
                channel_values[ch] = index_value

        # Map string channel labels to float values
        all_ch_ids = self.recording.get_channel_ids()
        values = np.zeros(len(all_ch_ids))
        for i, ch_str in enumerate(all_ch_ids):
            try:
                ch_int = int(ch_str.split("AP")[-1])
            except (ValueError, IndexError):
                ch_int = -1  # fallback, should not match
            values[i] = channel_values.get(ch_int, 0.0)

        # New normalization
        cmap = matplotlib.colormaps["bwr"]  # Or "coolwarm" or "bwr"
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)

        # Symmetric linear normalization
        absmax = max(np.abs(values).max(), 1e-6)
        scaled_values = values / absmax  # Now in [-1.0, 1.0]

        # Directly use scaled values
        contacts_colors = [
            "black" if val == 0 else cmap((val + 1) / 2)
            for val in scaled_values
        ]

        # Plot
        fig, ax = plt.subplots(figsize=(3, 10))
        probe = self.recording.get_probe()
        plot_probe(
            probe,
            ax=ax,
            contacts_colors=contacts_colors,
            probe_shape_kwargs=dict(edgecolor="white", facecolor="black", linewidth=0.2)
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("Evocation Index (Activation ↔ Inhibition)", rotation=270, labelpad=12)
        cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_title("Stimulus-Onset Evocation Map")
        ax.set_xticks([])

        # Set y-axis to show channel numbers instead of microns
        channel_positions = np.array([
            int(ch.split("AP")[-1]) for ch in self.recording.get_channel_ids()
        ])

        # Tick every N channels, align to correct rows
        tick_spacing = 15  # 15 for 2.0, 20 for 1.0
        tick_indices = np.arange(0, channel_positions.max() + 1, tick_spacing)
        tick_positions = tick_spacing * ((tick_indices - 1) // 2)
        tick_labels = [str(ch) for ch in tick_indices]

        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.set_ylabel("Channel")
        ax.set_xlabel("")

        ax.set_ylim(-200, 3900)
        ax.set_xlim(-14, 62)
        ax.axhline(self.config["insertion_depth"], color="red", linestyle="--", linewidth=1, label="~Cortex Edge")
        plt.tight_layout()
        ax.legend(fontsize=8)

        save_path = (f"output/{self.config['mouse_id']}/gate_{self.config['gate']}/"
                     f"{self.config['mouse_id']}-gate_{self.config['gate']}-light_evocation.png")
        self.save_chart(save_path, dpi=500)

        if self.config['show_plot']:
            matplotlib.use("TkAgg")
            plt.show()

    def plot_raster(self, ax, raster_data, show_y_axis=False):
        for channel, trials in raster_data.items():
            for trial_idx, (spike_times, stim_duration) in enumerate(trials):
                ax.vlines(spike_times / self.sampling_rate, trial_idx + 0.5, trial_idx + 1.5,
                          color='black', linewidth=0.5)
                if trial_idx == 0:
                    ax.axvline(0, color='red', linestyle='dashed', linewidth=0.25)
                    ax.axvspan(0, stim_duration / self.sampling_rate, color='red', alpha=0.1)

        ax.set_xlim(-0.25, 0.75)
        ax.set_xticks([-0.25, 0.75])
        ax.set_xticklabels(["-0.25", "0.75"], fontsize=10)
        xticklabels = ax.get_xticklabels()
        if len(xticklabels) == 2:
            xticklabels[0].set_horizontalalignment('left')
            xticklabels[1].set_horizontalalignment('right')
        ax.tick_params(axis='x', labelsize=12)

        if raster_data:
            num_trials = len(next(iter(raster_data.values())))
            ax.set_ylim(0.5, num_trials + 0.5)
            ax.set_yticks([1, num_trials])
            if show_y_axis:
                ax.set_yticklabels([str(1), str(num_trials)], fontsize=12)
                ax.set_ylabel("Trial", fontsize=16)
            else:
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.set_ylabel("")

        else:
            ax.set_yticks([])
            ax.set_ylabel("")

        if not show_y_axis:
            ax.tick_params(axis='y', left=False)

    def plot_psth(self, ax, psth_data, bin_width, ymax=None, show_y_axis=False):
        if psth_data["bin_centers"].size == 0 and psth_data["firing_rates"].size == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            return

        ax.bar(psth_data["bin_centers"], psth_data["firing_rates"],
               width=bin_width, color='black', align='center', linewidth=0)
        ax.axvline(0, color='red', linestyle='dashed', linewidth=0.5)
        ax.axvspan(0, 0.5, color='red', alpha=0.05)
        ax.set_xlim(-0.25, 0.75)
        ax.set_xticks([-0.25, 0.75])
        ax.set_xticklabels(["-0.25", "0.75"], fontsize=12)
        xticklabels = ax.get_xticklabels()
        if len(xticklabels) == 2:
            xticklabels[0].set_horizontalalignment('left')
            xticklabels[1].set_horizontalalignment('right')
        ax.tick_params(axis='x', labelsize=12)

        if ymax is None:
            ymax = psth_data["firing_rates"].max()

        ymax_display = ymax * 1.1
        ax.set_ylim(0, ymax_display)
        ax.set_yticks([0, int(np.ceil(ymax))])
        if show_y_axis:
            ax.set_yticklabels([str(0), str(int(np.ceil(ymax)))], fontsize=10)
            ax.set_ylabel("Spike Count", fontsize=12)  # in psth
        else:
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_ylabel("")

        if not show_y_axis:
            ax.tick_params(axis='y', left=False)

    def plot_probe_channel_map(self, ax, probe_channel_map):
        all_ch_ids = self.recording.get_channel_ids()
        color_channels = ['red' if ch in probe_channel_map else 'lightgray' for ch in all_ch_ids]

        # Get the probe (assumes your recording has one attached)
        probe = self.recording.get_probe()

        plot_probe(probe, ax=ax, contacts_colors=color_channels,
                   probe_shape_kwargs=dict(edgecolor=color_channels, facecolor='white', linewidth=0.1))
        ax.set_ylim(-200, 3900)

    def plot_waveform(self, ax, unit_id, channel):
        if self.analyzer is None:
            return

        ch_id_label = f"imec0.ap#AP{channel}"

        # Let the widget draw its figure, then capture it
        # Annoying warning about sparsity
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sw.plot_unit_waveforms(
                self.analyzer,
                unit_ids=[unit_id],
                channel_ids=[ch_id_label],
                backend="matplotlib",
            )
        fig_temp = plt.gcf()
        src_ax = fig_temp.axes[0]

        # Extract all Y lines
        y_lines = [line.get_ydata() for line in src_ax.lines]

        # Compute a baseline offset — either:
        offset = np.mean([np.mean(y) for y in y_lines])  # mean-centered
        # offset = np.mean([np.min(y) for y in y_lines])  # min-aligned (more visual clarity)

        # Plot, corrected
        for line in src_ax.lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata() - offset
            ax.plot(xdata, ydata, color=line.get_color(), linewidth=line.get_linewidth())

        plt.close(fig_temp)
        # Defines tick labels and range (mV), include to define range and hide labels
        # ax.set_ylim(60, -90)
        ax.set_xticks([])
        # ax.set_yticklabels([])
        ax.set_ylabel("mV", fontsize=4, rotation=0, labelpad=2)
        ax.yaxis.set_label_coords(0, -0.05)
        ax.tick_params(axis='y', labelsize=4)

    def process_analyzer(self):
        """Computes waveforms, templates, and amplitudes using SortingAnalyzer."""
        analyzer_path = self.config["paths"]["analyzer"]

        if os.path.exists(analyzer_path):
            print("Loading existing SortingAnalyzer...")
            self.analyzer = sic.load_sorting_analyzer(analyzer_path)

        else:
            print("Creating SortingAnalyzer and computing extensions...")
            self.analyzer = sic.create_sorting_analyzer(self.sorting, self.recording, format="binary_folder",
                                                        folder=analyzer_path)

            # Compute extensions
            self.analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=200)
            self.analyzer.compute("waveforms", ms_before=1.5, ms_after=2.0)
            self.analyzer.compute("templates", operators=["average", "median", "std"])
            self.analyzer.compute("spike_amplitudes")
            self.analyzer.compute("correlograms", window_ms=100.0, bin_ms=1.0)

    def process_best_channels(self):
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
        best_channels = self.process_best_channels()

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

        amplitudes_ext = self.analyzer.get_extension("spike_amplitudes")
        all_amps = amplitudes_ext.get_data()
        spike_vector = self.sorting.to_spike_vector()
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
        return spikes_in_window, spike_amplitudes

    def process_spike_dict(self, data_structure):
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

    def run_kilosort(self):
        """Runs Kilosort or loads existing sorting results."""
        kilosort_path = self.config["paths"]["kilosort"]
        if os.path.exists(kilosort_path) and self.config["skip_sort"]:
            print(f"Kilosort output found at {kilosort_path}, skipping sorting step.")
            self.sorting = se.read_kilosort(f'{kilosort_path}/sorter_output')
        else:
            print("Running Kilosort4...")
            self.sorting = ss.run_sorter("kilosort4", self.recording_concat, folder=kilosort_path)

        if self.config["run_phy"]:
            self.run_phy()

    def run_phy(self):
        """Handles Phy curation: launches GUI if rerun is True, otherwise loads existing curated results if available."""
        sorter_output = os.path.join(self.config["paths"]["kilosort"], "sorter_output")
        phy_working = os.path.join("processing", "phy/working")
        phy_export = os.path.join("processing", "phy", f"{self.config['mouse_id']}_g{self.config['gate']}")
        params_path = os.path.join(phy_working, "params.py")

        if not self.config.get("rerun", True):
            cluster_info_path = os.path.join(phy_export, "cluster_info.tsv")
            if os.path.exists(cluster_info_path):
                print(f"Loading curated Phy results from: {phy_export}")
                return
            else:
                print(f"Warning: No curated Phy data found at {phy_export}. Proceeding to GUI...")

        # If rerun is True, launch GUI
        if not os.path.exists(params_path):
            print(f"params.py not found at {params_path}. Copying from sorter_output...")
            if not os.path.exists(sorter_output):
                raise FileNotFoundError(f"Cannot find sorter_output directory at {sorter_output}")
            shutil.copytree(sorter_output, phy_working, dirs_exist_ok=True)

        # Copy original Kilosort cluster_group.tsv to seed labels
        source_tsv = os.path.join(sorter_output, "cluster_group.tsv")
        dest_tsv = os.path.join(phy_working, "cluster_group.tsv")
        if not os.path.exists(dest_tsv) and os.path.exists(source_tsv):
            shutil.copy2(source_tsv, dest_tsv)

        print("\nLaunching Phy (phy_working)...\nPlease assign labels and press Ctrl+S to save before closing.")
        subprocess.run([self.config["paths"].get("phy_path", "phy"), "template-gui", params_path])

        # Save result to final export
        os.makedirs(phy_export, exist_ok=True)
        for fname in ["cluster_group.tsv", "params.py", "cluster_info.tsv"]:
            src = os.path.join(phy_working, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(phy_export, fname))
        print(f"Exported curated results to: {phy_export}")

        # Clean phy_working
        for entry in os.listdir(phy_working):
            entry_path = os.path.join(phy_working, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
        print("Cleaned Phy working directory.")

    def save_chart(self, save_path, dpi=300):
        """Creates the directory if it doesn't exist and saves the chart."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Chart saved to {save_path}")

    def save_stacked_plot(self, stack_list, output_path):
        """Stacks image arrays vertically and saves the final image"""
        stacked = np.vstack(stack_list[::-1])
        final_img = Image.fromarray(stacked)
        final_img.save(output_path, compression='tiff_deflate')
        print(f"Stacked image saved to {output_path}")

    def stack_plots(self, stack_list, fig):
        """Stacks image arrays vertically and saves the final image"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img_array = np.array(img)
        stack_list.append(img_array)

    def synchronize_data(self):
        self.load_square_waves()
        self.calculate_square_alignment()

        print("Applying alignment corrections to NIDAQ light onset/offset times...")

        onsets = self.nidaq_data["light_onsets"]
        offsets = self.nidaq_data["light_offsets"]

        nidq_pulse_times = self.square_wave_alignment["nidq_times"]
        offsets_ap_nidq = self.square_wave_alignment["offsets"]

        adjusted_onsets = []
        adjusted_offsets = []

        for i, onset in enumerate(onsets):
            idx = np.searchsorted(nidq_pulse_times, onset, side='right') - 1

            if idx < 0:
                idx = 0  # Apply first correction if before first pulse
            elif idx >= len(offsets_ap_nidq):
                raise ValueError(f"Onset at {onset:.3f}s after final sync pulse.")

            correction = offsets_ap_nidq[idx]
            adjusted_onsets.append(onset - correction)
            adjusted_offsets.append(offsets[i] - correction)

        self.nidaq_data["light_onsets"] = np.array(adjusted_onsets)
        self.nidaq_data["light_offsets"] = np.array(adjusted_offsets)

        print(f"Adjusted {len(adjusted_onsets)} photodiode events into AP time base.")

    def verify_stimulus_alignment(self):
        """Verifies that the number of stimulus directions matches the number of onset/offset pairs."""
        if self.directions is None or self.nidaq_data is None:
            print("Error: Directions or NIDAQ data are not loaded.")
            return False

        light_onsets = self.nidaq_data.get("light_onsets", [])
        light_offsets = self.nidaq_data.get("light_offsets", [])
        # self.plot_nidaq_signals()
        # exit()

        if len(self.directions) != len(light_onsets) or len(self.directions) != len(light_offsets):
            print("Error: Mismatch in the number of stimulus directions and onset/offset pairs.")
            self.plot_nidaq_signals()
            print(f"\t{len(self.directions)} segments. {len(light_onsets)} onsets and {len(light_offsets)} offsets.")
            return False

        print("Stimulus alignment verified successfully.")
        print(f"\t{len(self.directions)} segments and {len(light_onsets)} pairs.")
        return True

    def plot_raw_spike_alignment(self, unit_id, window=1.0):
        """
        Plot raw spike times relative to photodiode onset, without slicing or alignment logic.

        Parameters:
            unit_id: ID of the unit to plot (from self.sorting)
            window: seconds before/after onset to include (default: 1.0s)
        """

        matplotlib.use('TkAgg')
        if "light_onsets" not in self.nidaq_data:
            raise ValueError("Photodiode onset times not loaded.")

        onsets = self.nidaq_data["light_onsets"]
        spike_vector = self.sorting.get_unit_spike_train(unit_id=unit_id)
        fs = self.recording.get_sampling_frequency()
        spike_vector = spike_vector / fs

        plt.figure(figsize=(12, 6))
        for i, onset in enumerate(onsets):
            rel_spikes = spike_vector[(spike_vector > onset - window) & (spike_vector < onset + window)] - onset
            plt.vlines(rel_spikes, i, i + 0.9, color='black', linewidth=0.5)

        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel("Time from onset (s)")
        plt.ylabel("Trial")
        plt.title(f"Raw spike times relative to photodiode onset (Unit {unit_id})")
        plt.tight_layout()
        plt.show()


def configure_experiment():
    """Defines experiment metadata and paths."""
    config = {
        "rerun": False,
        "sglx_folder": "SGL_DATA",
        "mouse_id": "Mouse03",
        "gate": "5",
        "probe": "0",
        "skip_sort": True,
        "write_concat": False,
        "processing_folder": "processing",
        "show_plot": False,
        "insertion_depth": 3000,
        "run_phy": False
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
        "waveforms": f"{config['processing_folder']}/waveforms/{config['mouse_id']}_g{config['gate']}",
        "analyzer": f"{config['processing_folder']}/sorting_analyzer/{config['mouse_id']}_g{config['gate']}",
        "stimulus_directions": "processing/stimulus/StimulusDirections.mat",
        "phy_path": "/home/andrew/anaconda3/envs/phy2/bin/phy"
    }

    return config


if __name__ == "__main__":
    config = configure_experiment()
    processor = ProcessUnit(config)
    processor.load_directions()
    processor.load_nidaq()
    if not processor.verify_stimulus_alignment():
        exit()
    processor.synchronize_data()
    processor.load_recording()
    probe = processor.recording.get_probe()
    processor.run_kilosort()
    processor.process_analyzer()
    # processor.plot_raw_spike_alignment(80)
    # exit()
    processor.plot_direction_evocation()
    processor.plot_light_evocation()
    processor.plot_unit_spiking_data()
