"""Signal-processing helpers shared by live.Plotter and record.Plotter."""

from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.interpolate

class General:
    """Generic moving-window statistics, resampling and normalization helpers.

    All functions operate on 2D arrays shaped (samples, channels) and process
    each channel independently.
    """

    @staticmethod
    def sliding_window_view_complete(signal: np.ndarray, window: int) -> np.ndarray:
        """Sliding windows of a 1D signal, NaN-padded at both ends to preserve its length.

        Args:
            signal: 1D array to window.
            window: Window size; must be odd.

        Returns:
            Array of shape (len(signal), window).
        """

        if np.mod(window, 2) == 0:
            raise ValueError('window must be odd')

        x = np.lib.stride_tricks.sliding_window_view(signal, window)

        starts   = np.arange(int(-(window-1)/2), 0)
        ends     = starts + window
        x_before = []

        for start, end in zip(starts, ends):

            part = np.empty(window)
            part[:-start] = np.nan
            part[-start:] = signal[:end]

            x_before.append(part)

        x_before = np.stack(x_before)

        starts  = np.arange(int(len(signal)-window+1), int(len(signal)-(window-1)/2))
        ends    = starts + window
        x_after = []

        for start, end in zip(starts, ends):

            part = np.empty(window)
            part[len(signal)-start:] = np.nan
            part[:len(signal)-start] = signal[start:]

            x_after.append(part)

        x_after = np.stack(x_after)

        return np.concatenate([x_before, x, x_after], axis=0)

    @staticmethod
    def moving_min(signal: np.ndarray, window: int) -> np.ndarray:
        """Per-channel moving minimum over an odd-sized window.

        Args:
            signal: Array shaped (samples, channels).
            window: Window size; must be odd.

        Returns:
            Array of the same shape as signal.
        """

        def helper(signal_one_channel: np.ndarray, window: int) -> np.ndarray:
            return np.nanmin(General.sliding_window_view_complete(signal_one_channel, window), axis=1)

        return np.stack([helper(signal[:, i], window) for i in range(signal.shape[1])], axis=1)

    @staticmethod
    def moving_max(signal: np.ndarray, window: int) -> np.ndarray:
        """Per-channel moving maximum over an odd-sized window.

        Args:
            signal: Array shaped (samples, channels).
            window: Window size; must be odd.

        Returns:
            Array of the same shape as signal.
        """

        def helper(signal_one_channel: np.ndarray, window: int) -> np.ndarray:
            return np.nanmax(General.sliding_window_view_complete(signal_one_channel, window), axis=1)

        return np.stack([helper(signal[:, i], window) for i in range(signal.shape[1])], axis=1)

    @staticmethod
    def moving_mean(signal: np.ndarray, window: int) -> np.ndarray:
        """Per-channel moving mean over an odd-sized window.

        Args:
            signal: Array shaped (samples, channels).
            window: Window size; must be odd.

        Returns:
            Array of the same shape as signal.
        """

        def helper(signal_one_channel: np.ndarray, window: int) -> np.ndarray:
            return np.nanmean(General.sliding_window_view_complete(signal_one_channel, window), axis=1)

        return np.stack([helper(signal[:, i], window) for i in range(signal.shape[1])], axis=1)

    @staticmethod
    def moving_median(signal: np.ndarray, window: int) -> np.ndarray:
        """Per-channel moving median over an odd-sized window.

        Args:
            signal: Array shaped (samples, channels).
            window: Window size; must be odd.

        Returns:
            Array of the same shape as signal.
        """

        def helper(signal_one_channel: np.ndarray, window: int) -> np.ndarray:
            return np.nanmedian(General.sliding_window_view_complete(signal_one_channel, window), axis=1)

        return np.stack([helper(signal[:, i], window) for i in range(signal.shape[1])], axis=1)

    @staticmethod
    def moving_std(signal: np.ndarray, window: int) -> np.ndarray:
        """Per-channel moving standard deviation over an odd-sized window.

        Args:
            signal: Array shaped (samples, channels).
            window: Window size; must be odd.

        Returns:
            Array of the same shape as signal.
        """

        def helper(signal_one_channel: np.ndarray, window: int) -> np.ndarray:
            return np.nanstd(General.sliding_window_view_complete(signal_one_channel, window), axis=1)

        return np.stack([helper(signal[:, i], window) for i in range(signal.shape[1])], axis=1)

    @staticmethod
    def estimate_fs(timestamps: np.ndarray) -> int:
        """Estimate the sampling rate, in Hz, from a sequence of timestamps.

        Args:
            timestamps: Sample timestamps, in seconds.

        Returns:
            Estimated sampling rate, rounded to the nearest integer Hz.
        """

        return int(np.round(np.mean(1/np.diff(timestamps))))

    @staticmethod
    def resample(timestamps: np.ndarray, samples: np.ndarray, fs: float, time_start: float | None = None, time_end: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Cubic-interpolate irregularly-sampled data onto a fixed-rate time base.

        Args:
            timestamps: Original sample timestamps.
            samples: Original samples, shape (len(timestamps), channels).
            fs: Target sampling rate, in Hz.
            time_start: Start of the new time base; defaults to timestamps[0].
            time_end: End of the new time base; defaults to timestamps[-1].

        Returns:
            Tuple of (new_timestamps, new_samples).
        """

        def helper(timestamps: np.ndarray, new_timestamps: np.ndarray, samples_one_channel: np.ndarray) -> np.ndarray:
            return scipy.interpolate.griddata(timestamps, samples_one_channel, new_timestamps, method='cubic')

        if time_start is None:
            time_start = timestamps[0]
        if time_end is None:
            time_end   = timestamps[-1]

        new_timestamps = np.arange(time_start, time_end, 1/fs, dtype=np.float64)
        new_samples    = np.stack([helper(timestamps, new_timestamps, samples[:, i]) for i in range(samples.shape[-1])], axis=1)

        return new_timestamps, new_samples

    @staticmethod
    def min_max_filter(signal: np.ndarray, window: int) -> np.ndarray:
        """Per-channel min-max normalization within a moving window, to roughly [0, 1].

        Args:
            signal: Array shaped (samples, channels).
            window: Window size; must be odd.

        Returns:
            Normalized array of the same shape as signal.
        """

        min_ = General.moving_min(signal, window)
        max_ = General.moving_max(signal, window)

        def helper(signal_one_channel: np.ndarray, min_one_channel: np.ndarray, max_one_channel: np.ndarray) -> np.ndarray:

            nan_mask      = np.logical_or(np.isnan(min_one_channel), np.isnan(max_one_channel))
            zero_amp_mask = max_one_channel - min_one_channel == 0
            signal_mask   = np.logical_not(np.logical_or(nan_mask, zero_amp_mask))

            signal_filt = np.empty_like(signal_one_channel)
            signal_filt[signal_mask]   = (signal_one_channel[signal_mask] - min_one_channel[signal_mask]) / (max_one_channel[signal_mask] - min_one_channel[signal_mask])
            signal_filt[nan_mask]      = np.nan
            signal_filt[zero_amp_mask] = 0

            return signal_filt

        return np.stack([helper(signal[:, i], min_[:, i], max_[:, i]) for i in range(signal.shape[1])], axis=1)

    @staticmethod
    def std_filter(signal: np.ndarray, window: int) -> np.ndarray:
        """Per-channel normalization by a moving standard deviation.

        Args:
            signal: Array shaped (samples, channels).
            window: Window size; must be odd.

        Returns:
            Normalized array of the same shape as signal.
        """

        std_ = General.moving_std(signal, window)

        def helper(signal_one_channel: np.ndarray, std_one_channel: np.ndarray) -> np.ndarray:
            return signal_one_channel / std_one_channel

        return np.stack([helper(signal[:, i], std_[:, i]) for i in range(signal.shape[1])], axis=1)

class PPG:
    """Processing specific to the optical (PPG) signal."""

    @staticmethod
    def fetch_signals(samples: np.ndarray) -> np.ndarray:
        """Extract and sign-correct the 3 raw PPG channels from a sample array.

        Args:
            samples: Raw samples as read from a session file, shape (samples, >=3).

        Returns:
            The first 3 channels, sign-corrected, shape (samples, 3).
        """

        return -samples[:, :3]

    @staticmethod
    def bandpass_filter(signals: np.ndarray, fs: float) -> np.ndarray:
        """Zero-phase 0.5-15 Hz band-pass filter, covering the plausible heart rate range.

        Args:
            signals: PPG samples, shape (samples, channels).
            fs: Sampling rate of signals, in Hz.

        Returns:
            Filtered samples, same shape as signals.
        """

        sos = scipy.signal.butter(4, [0.5, 15], btype='bandpass', output='sos', fs=fs)
        return scipy.signal.sosfiltfilt(sos, signals, axis=0)

class ACC:
    """Processing specific to the accelerometer signal."""

    @staticmethod
    def fetch_signals(samples: np.ndarray) -> np.ndarray:
        """Convert raw accelerometer samples from milli-g to g.

        Args:
            samples: Raw samples as read from a session file, shape (samples, channels).

        Returns:
            Samples in g, same shape as samples.
        """

        return samples / 1e3

class GYR:
    """Processing specific to the gyroscope signal."""

    @staticmethod
    def fetch_signals(samples: np.ndarray) -> np.ndarray:
        """Return raw gyroscope samples (degrees per second) unchanged.

        Args:
            samples: Raw samples as read from a session file, shape (samples, channels).

        Returns:
            samples, unchanged.
        """

        return samples
