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
import scipy.spatial.distance as _distance
from scipy.ndimage import median_filter, uniform_filter1d, minimum_filter1d, maximum_filter1d


# ===========================================================================
# Photodiode transition detection (Allen ecephys stimulus_sync method).
# See scripts/photodiode_standalone.py for the standalone version + tests.
# The signal is thresholded loosely into raw transitions, then the transition
# SEQUENCE is reconciled against the known 500/500 ms trial periodicity.
# ===========================================================================
SMOOTH_MS = 60.0        # median-filter window: wider than the ~11 Hz monitor
                        # ripple, far narrower than the 500 ms trial.
HALF_PERIOD_S = 0.5     # ON == OFF == 0.5 s during the trial run.
NDEVS = 10              # flag_unexpected_edges deviation threshold (Allen).
MAX_HALF_OFFSET = 4     # fix_unexpected_edges snap tolerance (Allen).
DEBOUNCE_FRAC = 0.5     # drop raw transitions closer than this * half-period.
RUN_GAP_S = 2.0         # a gap longer than this ends the contiguous run.
LOCK_TOL = 0.08         # cadence-lock tolerance (photodiode-only fallback).
SNAP_WIN_S = 0.15       # TTL-edge -> nearest-photodiode-edge snap window.


def trimmed_stats(data, pctiles=(10, 90)):
    """Robust mean/std ignoring the tails. Allen verbatim."""
    low = np.percentile(data, pctiles[0])
    high = np.percentile(data, pctiles[1])
    trimmed = data[np.logical_and(data <= high, data >= low)]
    return np.mean(trimmed), np.std(trimmed)


def flag_unexpected_edges(pd_times, ndevs=NDEVS):
    """Mask (1=expected, 0=unexpected) of inter-transition intervals deviating
    from the trimmed mean by more than ndevs std. Allen verbatim."""
    pd_diff = np.diff(pd_times)
    diff_mean, diff_std = trimmed_stats(pd_diff)
    mask = np.ones(pd_diff.size)
    mask[np.logical_or(pd_diff < diff_mean - ndevs * diff_std,
                       pd_diff > diff_mean + ndevs * diff_std)] = 0
    mask[1:] = np.logical_and(mask[:-1], mask[1:])
    mask = np.concatenate([mask, [mask[-1]]])
    return mask


def fix_unexpected_edges(pd_times, ndevs=NDEVS, cycle=1, max_frame_offset=MAX_HALF_OFFSET):
    """Repair the transition sequence against the expected interval: insert
    missed edges, drop spurious ones, on the expected grid. Allen verbatim
    (cycle=1 for the trial-transition cadence rather than 60 monitor frames)."""
    pd_times = np.array(pd_times, dtype=float)
    expected_duration_mask = flag_unexpected_edges(pd_times, ndevs=ndevs)
    diff_mean, diff_std = trimmed_stats(np.diff(pd_times))
    frame_interval = diff_mean / cycle

    bad_edges = np.where(expected_duration_mask == 0)[0]
    if bad_edges.size == 0:
        return pd_times

    bad_blocks = np.sort(np.unique(np.concatenate([
        [0], np.where(np.diff(bad_edges) > 1)[0] + 1, [len(bad_edges)]])))

    output_edges = np.array([], dtype=float)
    for low, high in zip(bad_blocks[:-1], bad_blocks[1:]):
        current_bad_edge_indices = bad_edges[low: high - 1]
        if current_bad_edge_indices.size == 0:
            continue
        current_bad_edges = pd_times[current_bad_edge_indices]
        low_bound = pd_times[current_bad_edge_indices[0]]
        high_bound = pd_times[current_bad_edge_indices[-1] + 1]
        edges_missing = int(np.around((high_bound - low_bound) / diff_mean))
        expected = np.linspace(low_bound, high_bound, edges_missing + 1)
        distances = _distance.cdist(current_bad_edges[:, None], expected[:, None])
        distances = np.around(distances / frame_interval).astype(int)
        min_offsets = np.amin(distances, axis=0)
        min_offset_indices = np.argmin(distances, axis=0)
        output_edges = np.concatenate([
            output_edges,
            expected[min_offsets > max_frame_offset],
            current_bad_edges[min_offset_indices[min_offsets <= max_frame_offset]]])

    return np.sort(np.concatenate([output_edges, pd_times[expected_duration_mask > 0]]))


def correct_on_off_effects(pd_times):
    """Remove the systematic ON->OFF vs OFF->ON timing asymmetry. Allen verbatim."""
    pd_times = np.array(pd_times, dtype=float)
    pd_diff = np.diff(pd_times)
    odd_diff_mean, _ = trimmed_stats(pd_diff[1::2])
    even_diff_mean, _ = trimmed_stats(pd_diff[0::2])
    half_diff = np.diff(pd_times[0::2])
    full_period_mean, _ = trimmed_stats(half_diff)
    half_period_mean = full_period_mean / 2
    pd_times[::2] -= (odd_diff_mean - half_period_mean) / 2
    pd_times[1::2] -= (even_diff_mean - half_period_mean) / 2
    return pd_times


def extract_raw_transitions(signal, fs):
    """Loose detection of every gray<->ON transition, via drift-tracking
    hysteresis on the median-smoothed level. Only has to be roughly right;
    fix_unexpected_edges repairs the sequence afterwards."""
    med_win = max(1, int(SMOOTH_MS * fs / 1000))
    if med_win % 2 == 0:
        med_win += 1
    level = median_filter(np.asarray(signal, dtype=np.float64), size=med_win)

    win = max(1, int(1.0 * fs))
    loc_lo = minimum_filter1d(level, win, mode="nearest")
    loc_hi = maximum_filter1d(level, win, mode="nearest")
    mid = 0.5 * (loc_lo + loc_hi)
    span = loc_hi - loc_lo
    hi_thr = mid + 0.1 * span
    lo_thr = mid - 0.1 * span

    above = level > hi_thr
    below = level < lo_thr
    state = np.zeros(level.size, dtype=bool)
    cur = False
    prev = 0
    for idx in np.where(above | below)[0]:
        if above[idx] and not cur:
            state[prev:idx] = cur; cur = True; prev = idx
        elif below[idx] and cur:
            state[prev:idx] = cur; cur = False; prev = idx
    state[prev:] = cur

    times = (np.where(np.diff(state.astype(np.int8)) != 0)[0] + 1) / fs
    if times.size:
        keep = np.concatenate([[True], np.diff(times) > (DEBOUNCE_FRAC * HALF_PERIOD_S)])
        times = times[keep]
    return times
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
        self.trial_drift_alpha = 0.0035  # Testing calibration drift.
        self.unique_shanks = None

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
            print('len(light_onsets)', len(light_onsets))
            print('len(self.directions)', len(self.directions))
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

    def calculate_square_alignment(self, skip_edge_pulses=False, first_pulse_tolerance=0.05, max_shift=5):
        """
        Calculates pulse onset times from square wave signals in AP (SY0) and NIDAQ (XD0),
        then computes time correction needed to align NIDAQ to AP.

        It tolerates small mismatches in pulse counts by trimming extra pulses from the
        longer sequence, preferring to drop pulses at the edges while enforcing that
        the first matched pulses are aligned within 'first_pulse_tolerance' seconds.

        Stores:
            self.square_wave_alignment["ap_times"]
            self.square_wave_alignment["nidq_times"]
            self.square_wave_alignment["offsets"]
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

        def align_pulse_trains(ap_times, nidq_times, tol, max_shift):
            """
            Trim extra pulses from the longer sequence while ensuring the first
            aligned pulses are within tol seconds. Allows up to max_shift pulses
            to be discarded from the leading edge to fix off-by-one.
            """
            n_ap = len(ap_times)
            n_nidq = len(nidq_times)

            if n_ap == 0 or n_nidq == 0:
                raise ValueError("No pulses detected in one or both signals.")

            if n_ap == n_nidq:
                # Straightforward case
                if abs(ap_times[0] - nidq_times[0]) > tol:
                    raise ValueError(
                        f"First AP/NIDAQ pulses misaligned by {abs(ap_times[0] - nidq_times[0]):.3f}s "
                        f"(tolerance {tol}s)."
                    )
                return ap_times, nidq_times

            # Determine which is longer
            if n_ap > n_nidq:
                longer = ap_times
                shorter = nidq_times
                longer_is_ap = True
            else:
                longer = nidq_times
                shorter = ap_times
                longer_is_ap = False

            diff = abs(n_ap - n_nidq)
            # Limit how many pulses we ever consider discarding at the front
            max_shift = min(diff, max_shift)

            best_shift = None
            best_score = None

            # Try discarding 0..max_shift pulses at the *front* of the longer train;
            # the remaining extra pulses are implicitly discarded at the tail.
            for shift in range(max_shift + 1):
                if shift + len(shorter) > len(longer):
                    break
                cand_longer = longer[shift:shift + len(shorter)]

                # Enforce first pulse alignment
                if abs(cand_longer[0] - shorter[0]) > tol:
                    continue

                # Evaluate candidate by median absolute offset
                offsets = shorter - cand_longer
                score = np.median(np.abs(offsets))

                if best_score is None or score < best_score:
                    best_score = score
                    best_shift = shift

            if best_shift is None:
                raise ValueError(
                    f"Unable to align pulse trains with tolerance {tol}s "
                    f"and up to {max_shift} leading-pulse shift."
                )

            # Build final aligned sequences
            if longer_is_ap:
                aligned_ap = longer[best_shift:best_shift + len(shorter)]
                aligned_nidq = shorter
            else:
                aligned_ap = shorter
                aligned_nidq = longer[best_shift:best_shift + len(shorter)]

            return aligned_ap, aligned_nidq

        ap = self.square_wave_data["ap"]
        nidq = self.square_wave_data["nidq"]

        ap_times = detect_rising_edges(ap["trace"], ap["time"])
        nidq_times = detect_rising_edges(nidq["sync"], nidq["time"])

        if skip_edge_pulses:
            ap_times = ap_times[1:-1]
            nidq_times = nidq_times[1:-1]

        # New robust alignment that tolerates mismatched pulse counts
        ap_times_aligned, nidq_times_aligned = align_pulse_trains(
            ap_times, nidq_times, tol=first_pulse_tolerance, max_shift=max_shift
        )

        offsets = nidq_times_aligned - ap_times_aligned

        self.square_wave_alignment = {
            "ap_times": ap_times_aligned,
            "nidq_times": nidq_times_aligned,
            "offsets": offsets,
        }

        print(
            f"Extracted {len(ap_times)} AP pulses and {len(nidq_times)} NIDAQ pulses "
            f"(aligned to {len(ap_times_aligned)} pulses)."
        )

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

    def detect_light_transitions(self, signal, trim_front=0, trim_end=0, ttl=None):
        """
        Detect trial onsets/offsets in the photodiode signal (Allen ecephys
        stimulus_sync method; see scripts/photodiode_standalone.py for the
        standalone version and tests).

        Replaces the old global-threshold + 500 ms +/- 50 ms duration gate, which
        failed on session-long intensity drift and on the inconsistent crossing
        point along each edge.

        - If a clean stimulus TTL is supplied (XD0 bit 2, same NIDAQ clock), its
          rising edges define the trial grid; each is snapped to the nearest
          photodiode transition for the precise (canonical) optical time.
          Warmup/ramp transitions with no TTL partner are dropped structurally.
        - Otherwise: isolate the longest metronomic (0.5 s) cadence-locked run
          and reconcile it with fix_unexpected_edges / correct_on_off_effects.

        Parameters:
        - signal: photodiode signal.
        - trim_front / trim_end: seconds at start/end to exclude.
        - ttl: optional binary stimulus-TTL trace (same length/clock as signal).

        Returns (paired_onsets, paired_offsets), seconds on the NIDAQ clock.
        """
        fs = self.nidaq_recording.get_sampling_frequency()

        raw_times = extract_raw_transitions(signal, fs)
        if raw_times.size < 4:
            print("\tDetected 0 valid onset-offset pairs.")
            return np.array([]), np.array([])

        # ---- TTL-grid path (clean TTL present) ----
        if ttl is not None and np.any(ttl):
            ttl_arr = np.asarray(ttl).astype(np.int8)
            ttl_rise = np.where(np.diff(ttl_arr) == 1)[0] / fs
            if ttl_rise.size >= 4:
                d = np.diff(ttl_rise)
                if np.median(np.abs(d - np.median(d))) < 0.05:  # cadence-locked
                    pd_trans = raw_times
                    onsets, offsets = [], []
                    win = SNAP_WIN_S
                    for r in ttl_rise:
                        j = np.searchsorted(pd_trans, r)
                        cands = []
                        if j < len(pd_trans):
                            cands.append(pd_trans[j])
                        if j > 0:
                            cands.append(pd_trans[j - 1])
                        if not cands:
                            continue
                        pd_on = min(cands, key=lambda x: abs(x - r))
                        if abs(pd_on - r) > win:
                            pd_on = r  # photodiode dropped this edge -> use TTL
                        k = np.searchsorted(pd_trans, pd_on + HALF_PERIOD_S)
                        off_cands = []
                        if k < len(pd_trans):
                            off_cands.append(pd_trans[k])
                        if k > 0:
                            off_cands.append(pd_trans[k - 1])
                        target = pd_on + HALF_PERIOD_S
                        pd_off = (min(off_cands, key=lambda x: abs(x - target))
                                  if off_cands else target)
                        if abs(pd_off - target) > win:
                            pd_off = target
                        onsets.append(pd_on)
                        offsets.append(pd_off)

                    onsets = np.array(onsets)
                    offsets = np.array(offsets)
                    keep = (onsets >= trim_front) & (offsets <= (len(signal) / fs - trim_end))
                    onsets, offsets = onsets[keep], offsets[keep]
                    print(f"\tDetected {len(onsets)} valid onset-offset pairs "
                          f"(TTL grid: {len(ttl_rise)} edges).")
                    return onsets, offsets

        # ---- photodiode-only fallback: cadence-lock + Allen reconciliation ----
        gaps = np.diff(raw_times)
        breaks = np.where(gaps > RUN_GAP_S)[0]
        seg_bounds = np.concatenate([[0], breaks + 1, [raw_times.size]])
        biggest = int(np.argmax(np.diff(seg_bounds)))
        run = raw_times[seg_bounds[biggest]:seg_bounds[biggest + 1]]
        if run.size < 4:
            print("\tDetected 0 valid onset-offset pairs.")
            return np.array([]), np.array([])

        d = np.diff(run)
        band = (d > HALF_PERIOD_S * 0.7) & (d < HALF_PERIOD_S * 1.3)
        run_gap = np.median(d[band]) if np.any(band) else HALF_PERIOD_S
        on_cadence = np.abs(d - run_gap) < (LOCK_TOL * run_gap)
        best_len = best_start = cur_len = cur_start = 0
        for i, ok in enumerate(on_cadence):
            if ok:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        if best_len < 3:
            print("\tDetected 0 valid onset-offset pairs.")
            return np.array([]), np.array([])
        run = run[best_start: best_start + best_len + 1]

        fixed = fix_unexpected_edges(run, ndevs=NDEVS, cycle=1,
                                     max_frame_offset=MAX_HALF_OFFSET)
        fixed = correct_on_off_effects(fixed)

        on_a, off_a = fixed[0::2], fixed[1::2]
        on_b, off_b = fixed[1::2], fixed[2::2]

        def score(on, off):
            m = min(len(on), len(off))
            return np.inf if m == 0 else np.abs(np.median(off[:m] - on[:m]) - HALF_PERIOD_S)

        if score(on_b, off_b) < score(on_a, off_a):
            onsets, offsets = on_b, off_b
        else:
            onsets, offsets = on_a, off_a

        m = min(len(onsets), len(offsets))
        onsets, offsets = onsets[:m], offsets[:m]
        valid = (offsets - onsets) > (HALF_PERIOD_S * 0.5)
        onsets, offsets = onsets[valid], offsets[valid]
        keep = (onsets >= trim_front) & (offsets <= (len(signal) / fs - trim_end))
        onsets, offsets = onsets[keep], offsets[keep]

        print(f"\tDetected {len(onsets)} valid onset-offset pairs.")
        return np.array(onsets), np.array(offsets)

    def detect_sync_pulses(self, signal, threshold=None, pause_duration=0.7):
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
        # # todo updated this to run with the new TTL from the stimulus script, will probably break older data and
        #    needs updating to be conditional
        """Loads the NIDAQ binary file and extracts the photodiode and sync pulse channels."""
        preprocessed_path = self.config["paths"]["preprocessed_NIDAQ"]

        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed recording from {preprocessed_path}...")
            self.nidaq_recording = se.read_spikeglx(self.config["base_path"], stream_name='nidq')
            # self.sampling_rate = self.nidaq_recording.get_sampling_frequency()
            self.nidaq_concat = sic.load(preprocessed_path)

            photodiode_signal = self.nidaq_concat.get_traces(channel_ids=['nidq#XA0']).flatten()

            # XD0 is a 16-bit digital word: bit 0 = 1s sync wave, bit 2 = stim TTL
            xd0_raw = self.nidaq_concat.get_traces(channel_ids=['nidq#XD0']).flatten().astype(np.uint16)
            sync_pulse_signal = ((xd0_raw & (1 << 0)) > 0).astype(np.uint8)
            stim_ttl = ((xd0_raw & (1 << 2)) > 0).astype(np.uint8)  # None-safe: any()==False if unused

            # Detect light onset times (TTL grid when present, else photodiode cadence)
            light_onsets, light_offsets = self.detect_light_transitions(
                photodiode_signal, ttl=stim_ttl)

            # Detect sync pulses every 10 seconds, using clean sync bit
            sync_pulses = self.detect_sync_pulses(sync_pulse_signal)

        else:
            print("Processing NIDAQ from raw data...")
            self.nidaq_recording = se.read_spikeglx(self.config["base_path"], stream_name='nidq')
            # self.sampling_rate = self.nidaq_recording.get_sampling_frequency()

            self.nidaq_concat = sic.concatenate_recordings([self.nidaq_recording])

            photodiode_signal = self.nidaq_concat.get_traces(channel_ids=['nidq#XA0']).flatten()

            # XD0 is a 16-bit digital word: bit 0 = 1s sync wave, bit 2 = stim TTL
            xd0_raw = self.nidaq_concat.get_traces(channel_ids=['nidq#XD0']).flatten().astype(np.uint16)
            sync_pulse_signal = ((xd0_raw & (1 << 0)) > 0).astype(np.uint8)
            stim_ttl = ((xd0_raw & (1 << 2)) > 0).astype(np.uint8)

            # Save if enabled
            if self.config["write_concat"]:
                self.nidaq_concat = self.nidaq_concat.save(format="binary", folder=preprocessed_path)

            # Detect light onset times (TTL grid when present, else photodiode cadence)
            light_onsets, light_offsets = self.detect_light_transitions(
                photodiode_signal, ttl=stim_ttl)

            # Detect sync wave
            sync_pulses = self.detect_sync_pulses(sync_pulse_signal)

        # Ensure nidaq_data is assigned in both cases
        self.nidaq_data = {
            "photodiode": photodiode_signal,
            "sync_pulse": sync_pulse_signal,
            "light_onsets": light_onsets,
            "light_offsets": light_offsets,
            "sync_pulses": sync_pulses,
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
        # stim_binary = stim_trace.astype(np.uint8)
        # stim_rise_idx = np.where(np.diff(stim_binary) == 1)[0] + 1
        # stim_onsets_nidq = xd0_time[stim_rise_idx]
        #
        # # Overwrite the photodiode-based onsets with TTL-based onsets
        # # Keep offsets as-is (photodiode) so segment ends still come from measured light offset
        # # todo this block breaks code from before the ttl was used, make it conditional
        # if hasattr(self, "nidaq_data"):
        #     self.nidaq_data["light_onsets"] = stim_onsets_nidq
        # else:
        #     # safety fallback, but in your call order load_nidaq() already ran so nidaq_data exists
        #     self.nidaq_data = {
        #         "light_onsets": stim_onsets_nidq,
        #     }
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
        ax.set_ylim(20, -30)
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
        ap_pulse_times = self.square_wave_alignment["ap_times"]

        # Map NIDAQ event times onto the AP clock by LINEAR INTERPOLATION between
        # the bracketing shared sync pulses, rather than applying one nearest
        # per-pulse offset. The two clocks drift measurably across a recording
        # (observed ~1200 ppm -> ~0.8 s accumulated over 10 min), and that drift
        # is linear within each 1 s interval, so interpolation tracks it while a
        # nearest-offset step does not. np.interp clamps outside the pulse span.
        adjusted_onsets = np.interp(onsets, nidq_pulse_times, ap_pulse_times)
        adjusted_offsets = np.interp(offsets, nidq_pulse_times, ap_pulse_times)

        self.nidaq_data["light_onsets"] = adjusted_onsets
        self.nidaq_data["light_offsets"] = adjusted_offsets

        print(f"Adjusted {len(adjusted_onsets)} photodiode events into AP time base "
              f"(interpolated between {len(nidq_pulse_times)} sync pulses).")

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

    def plot_probe_shanks_debug(self, shank_ids_override=None):
        if self.unique_shanks is None:
            raise RuntimeError("Call load_recording() first.")
        matplotlib.use("TkAgg")
        probe = self.recording.get_probe()

        if shank_ids_override is None:
            try:
                shank_ids = np.asarray(probe.shank_ids)
            except Exception:
                shank_ids = np.zeros(self.recording.get_num_channels(), dtype=int)
        else:
            shank_ids = np.asarray(shank_ids_override)

        palette = plt.get_cmap("tab10")
        colors = [palette(int(s) % 10) for s in shank_ids]

        fig, ax = plt.subplots(figsize=(5 + 1.0 * max(1, len(self.unique_shanks)), 10))
        plot_probe(probe, ax=ax, contacts_colors=colors,
                   probe_shape_kwargs=dict(edgecolor="black", facecolor="white", linewidth=0.2))
        ax.set_title(f"Probe geometry by shank {self.unique_shanks}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    def _infer_shanks_from_x(self, tol_um=15.0):
        probe = self.recording.get_probe()
        pos = np.asarray(probe.contact_positions)  # shape (n, 2)
        x = pos[:, 0]
        xq = np.round(x / tol_um) * tol_um  # quantize to collapse each column
        uniq = np.sort(np.unique(xq))
        rank = {u: i for i, u in enumerate(uniq)}
        return np.array([rank[v] for v in xq], dtype=int)


def configure_experiment():
    """Defines experiment metadata and paths."""
    config = {
        "rerun": True,
        "sglx_folder": "SGL_DATA",
        "mouse_id": "mouse05",
        "gate": "1",
        "probe": "0",
        "skip_sort": True,
        "write_concat": False,
        "processing_folder": "processing",
        "show_plot": False,
        "insertion_depth": 3000,
        "run_phy": True
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
