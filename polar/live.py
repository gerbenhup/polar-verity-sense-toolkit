"""Consumers of a live driver.Device stream: real-time plotting and saving to disk."""

from __future__ import annotations

import numpy as np
import time
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.signal

from .driver import Device

class Plotter:
    """Live-plots each measurement type in a sliding time window as data streams in."""

    def __init__(self, window_length: int, update_interval: float) -> None:
        """
        Args:
            window_length: Seconds of history to keep and display per measurement type.
            update_interval: Minimum seconds between plot redraws.
        """

        self.window_length   = window_length
        self.update_interval = update_interval

        self.timestamps    = {}
        self.frame_samples = {}

        self.initialize_plotter()

    def stream_callback(self, device_obj: Device, measurement_type: Device.MeasurementType, timestamps: np.ndarray, frame_samples: np.ndarray) -> None:
        """Append a decoded frame to the plotted window and redraw if due.

        Args:
            device_obj: The device the frame was received from; unused here.
            measurement_type: Which measurement type this frame belongs to.
            timestamps: Absolute per-sample timestamps, in seconds.
            frame_samples: Decoded samples, shape (samples, channels).
        """

        if measurement_type is Device.MeasurementType.PPG:
            frame_samples = -frame_samples[:, 0:1]
        elif measurement_type is Device.MeasurementType.ACC:
            frame_samples = np.sqrt(np.sum(np.square(frame_samples), axis=1, keepdims=True))

        if measurement_type not in self.timestamps:
            self.timestamps[measurement_type]    = timestamps
            self.frame_samples[measurement_type] = frame_samples
        else:
            self.timestamps[measurement_type]    = np.concatenate([self.timestamps[measurement_type], timestamps], axis=0)
            self.frame_samples[measurement_type] = np.concatenate([self.frame_samples[measurement_type], frame_samples], axis=0)

        for measurement_type in self.timestamps:
            inds = np.argwhere(self.timestamps[measurement_type] >= self.timestamps[measurement_type][-1] - self.window_length).ravel()
            self.timestamps[measurement_type]    = self.timestamps[measurement_type][inds]
            self.frame_samples[measurement_type] = self.frame_samples[measurement_type][inds, :]

        self.update_plotter()

    def preprocess(self, timestamps: np.ndarray, frame_samples: np.ndarray) -> np.ndarray:
        """Band-pass filter samples using a sampling rate estimated from timestamps.

        Args:
            timestamps: Per-sample timestamps, in seconds, used to estimate the sampling rate.
            frame_samples: Samples to filter, shape (samples, channels).

        Returns:
            Filtered samples, same shape as frame_samples.
        """

        fs  = np.mean(1/np.diff(timestamps))
        sos = scipy.signal.butter(4, [0.5, 15], 'bandpass', output='sos', fs=fs)
        frame_samples = scipy.signal.sosfiltfilt(sos, frame_samples, axis=0)

        return frame_samples

    def initialize_plotter(self) -> None:
        """(Re)create the figure with one subplot per measurement type seen so far."""

        if not hasattr(self, 'figure'):

            self.time_keeper = time.time()

            self.figure = plt.figure()

            for i, measurement_type in enumerate(self.timestamps):
                if i == 0:
                    axis = self.figure.add_subplot(len(self.timestamps), 1, i+1)
                else:
                    axis = self.figure.add_subplot(len(self.timestamps), 1, i+1, sharex=axis)

                axis.plot(self.timestamps[measurement_type], self.preprocess(self.timestamps[measurement_type], self.frame_samples[measurement_type]))

            plt.pause(0.001)

        else:

            for axis in self.figure.axes:
                axis.remove()

            for i, measurement_type in enumerate(self.timestamps):
                if i == 0:
                    axis = self.figure.add_subplot(len(self.timestamps), 1, i+1)
                else:
                    axis = self.figure.add_subplot(len(self.timestamps), 1, i+1, sharex=axis)

                axis.plot(self.timestamps[measurement_type], self.preprocess(self.timestamps[measurement_type], self.frame_samples[measurement_type]))

            self.figure.canvas.draw()
            plt.pause(0.001)

    def update_plotter(self) -> None:
        """Redraw the plot with the current window, throttled to update_interval."""

        if self.time_keeper >= time.time() - self.update_interval:
            return

        if len(self.timestamps) != len(self.figure.axes):
            self.initialize_plotter()
        else:
            for i, (measurement_type, axis) in enumerate(zip(self.timestamps, self.figure.axes)):

                frame_samples = self.frame_samples[measurement_type]
                frame_samples = self.preprocess(self.timestamps[measurement_type], frame_samples)

                for j, line in enumerate(axis.lines):
                    line.set_data(self.timestamps[measurement_type], self.preprocess(self.timestamps[measurement_type], frame_samples[:, j]))

                min_ = np.nanmin(frame_samples)
                max_ = np.nanmax(frame_samples)

                if max_ - min_ > 0:
                    diff = max_ - min_
                else:
                    diff = 1

                axis.set_ylim(min_ - 0.1*diff, max_ + 0.1*diff)

                if i == 0:
                    axis.set_xlim(self.timestamps[measurement_type][0], self.timestamps[measurement_type][-1])

            self.figure.canvas.draw()
            plt.pause(0.001)

        self.time_keeper = time.time()

class Saver:
    """Saves a live stream to per-measurement-type text files, one session directory per device."""

    def __init__(self, save_dir: Path) -> None:
        """
        Args:
            save_dir: Directory under which each session's subdirectory is created.
        """

        self.save_dir = save_dir

    def stream_callback(self, device_obj: Device, measurement_type: Device.MeasurementType, timestamps: np.ndarray, frame_samples: np.ndarray) -> None:
        """Append a decoded frame to the session's file for this measurement type, creating it if needed.

        Args:
            device_obj: The device the frame was received from; used to name the session directory.
            measurement_type: Which measurement type this frame belongs to.
            timestamps: Absolute per-sample timestamps, in seconds.
            frame_samples: Decoded samples, shape (samples, channels).
        """

        save_subdir = self.save_dir / '{}-{}'.format(datetime.datetime.fromtimestamp(device_obj.timestamp_start).strftime('%Y%m%d%H%M%S'), device_obj.device_name)
        save_file   = save_subdir / '{}.txt'.format(measurement_type.name)

        if not save_subdir.exists():
            save_subdir.mkdir()

        if not save_file.exists():

            with open(save_file, 'w') as f:

                # Write header
                f.write('Device name: {}\n'.format(device_obj.device_name))
                f.write('Start recording: {}\n\n'.format(device_obj.timestamp_start))

                # Write stream settings
                f.write('Settings: {}\n'.format(measurement_type.name))

                for setting, value in device_obj.stream_settings[measurement_type].items():
                    f.write('{}: {}\n'.format(setting.name, value))

                f.write('\n')

                # Write contents
                f.write('Measurement: {}\n'.format(measurement_type.name))
                f.write('relative_timestamp\t{}\n'.format('\t'.join(['{}{}'.format(measurement_type.name, x) for x in range(device_obj.stream_settings[measurement_type][device_obj.SettingType.CHANNELS])])))

        with open(save_file, 'a') as f:

            for timestamp, row in zip(timestamps, frame_samples):

                f.write('{:f}\t{}\n'.format(timestamp, '\t'.join('{:d}'.format(x) for x in row)))
