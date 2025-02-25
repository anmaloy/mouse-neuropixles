import matplotlib

matplotlib.use('TkAgg')  # Set Matplotlib backend for interactive plots
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import spikeinterface.extractors as se
import spikeinterface.core as sic
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
from spikeinterface import extract_waveforms
from probeinterface.plotting import plot_probe
from scipy.signal import find_peaks, peak_prominences


class NeuralDataProcessor:
    """Class for processing and analyzing neural data from SpikeGLX recordings."""

    def __init__(self, config):
        """Initialize the processor with a given experiment configuration."""
        self.config = config
        self.recording = None
        self.recording_concat = None
        self.sorting = None
        self.analyzer = None
        self.spike_times = None

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
            self.config["paths"]["preprocessed"],
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

    def load_recording(self):
        """Loads and preprocesses the SpikeGLX recording."""
        preprocessed_path = self.config["paths"]["preprocessed"]

        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed recording from {preprocessed_path}...")
            self.recording = se.read_spikeglx(self.config["run_path"], stream_name=f'imec{self.config["probe"]}.ap')
            self.recording_concat = sic.load(preprocessed_path)
        else:
            print("Processing from raw data...")
            self.recording = se.read_spikeglx(self.config["run_path"], stream_name=f'imec{self.config["probe"]}.ap')

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

        # Save plot
        output_path = f'output/mua_charts/{self.config["mouse_id"]}_g{self.config["gate"]}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
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

        # Save and show the plot
        output_path = f'output/raster_heatmap/{self.config["mouse_id"]}_g{self.config["gate"]}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
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
        plt.show()


# -------------------------- MAIN SCRIPT --------------------------

def configure_experiment():
    """Defines experiment metadata and paths."""
    config = {
        "rerun": True,
        "sglx_folder": "SGL_DATA",
        "mouse_id": "testMouse",
        "gate": "0",
        "probe": "0",
        "channels": (0, 100),
        "channels_sample": (10, 15),
        "skip_sort": True,
        "write_concat": False,
        "mua_method": "kilosort",
        "processing_folder": "processing",
        "direction": "negative"  # Can be "negative" or "positive"
    }

    # Define paths **after** initializing config
    base_path = f'{config["sglx_folder"]}/{config["mouse_id"]}_g{config["gate"]}'
    run_path = f'{base_path}/{config["mouse_id"]}_g{config["gate"]}_imec{config["probe"]}'

    config["base_path"] = base_path
    config["run_path"] = run_path

    config["paths"] = {
        "preprocessed": f"{config['processing_folder']}/concat/{config['mouse_id']}_g{config['gate']}_imec{config['probe']}",
        "kilosort": f"{config['processing_folder']}/kilosort/{config['mouse_id']}_g{config['gate']}",
        "mua": f"{config['processing_folder']}/mua_spikes/{config['mouse_id']}_g{config['gate']}.npy",
        "waveforms": f"{config['processing_folder']}/waveforms/{config['mouse_id']}_g{config['gate']}",
        "analyzer": f"{config['processing_folder']}/sorting_analyzer/{config['mouse_id']}_g{config['gate']}",
    }

    return config


if __name__ == "__main__":
    # Load experiment configuration
    config = configure_experiment()

    # Initialize the neural data processor
    processor = NeuralDataProcessor(config)

    # Step 1: Load and preprocess the recording
    processor.load_recording()

    # Step 2: Run spike sorting with Kilosort
    processor.run_kilosort()

    # Step 3: Compute SortingAnalyzer metrics (waveforms, templates, amplitudes)
    processor.process_analyzer()

    # Step 4: Extract and save MUA spike times
    processor.extract_mua_spikes()

    # Step 5: Visualization
    print("\nGenerating plots...\n")
    processor.plot_filtered_traces()  # Plot threshold-based spike detection
    processor.plot_mua_activity()  # Plot MUA event count per channel
    processor.plot_raster_heatmap()
    processor.plot_ap_amplitudes()  # Plot action potential amplitude distributions
    # processor.plot_waveform_traces()  # Plot individual spike waveforms
    # processor.plot_channel_templates()  # Plot template waveforms for selected channels
    # processor.plot_probe_layout()  # Plot probe layout

