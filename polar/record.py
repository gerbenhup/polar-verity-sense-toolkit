"""Reading and plotting a session previously saved by live.Saver."""

from __future__ import annotations

import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from . import tools

class Reader:
    """Reads a recorded session directory's PPG/ACC/GYR/log files into memory."""

    def __init__(self, save_dir: Path) -> None:
        """
        Args:
            save_dir: Session directory to read, as created by live.Saver.
        """

        self.save_dir = save_dir

        # Fetch measurement types
        files                  = list(self.save_dir.glob('*.txt'))
        self.measurement_types = []
        for file in files:
            if file.stem != 'log':
                self.measurement_types.append(file.stem)

        # Read measurements
        for measurement_type in self.measurement_types:
            self.read_measurement(measurement_type)

        # Read log
        self.read_log()

    def read_measurement(self, measurement_type: str) -> None:
        """Parse one measurement type's `.txt` file into timestamps, samples and settings.

        Args:
            measurement_type: Name of the measurement type (e.g. "PPG"), matching a file stem.
        """

        if not (self.save_dir / '{}.txt'.format(measurement_type)).exists():
            raise RuntimeError('Measurement type {} does not exist'.format(measurement_type))

        if not hasattr(self, 'timestamps'):
            self.timestamps      = {}
            self.samples         = {}
            self.stream_settings = {}

        with open(self.save_dir / '{}.txt'.format(measurement_type), 'r') as f:

            mode = 'header'

            for line in f.readlines():

                if mode == 'header':

                    if 'Device name' in line:
                        device_name = str(line.split(':')[1].strip())

                        if hasattr(self, 'device_name') and self.device_name != device_name:
                            raise RuntimeError('Mismatch in device names between files')
                        else:
                            self.device_name = device_name

                    elif 'Start recording' in line:
                        timestamp_start = float(line.split(':')[1].strip())

                        if hasattr(self, 'timestamp_start') and self.timestamp_start != timestamp_start:
                            raise RuntimeError('Mismatch in timestamp start between files')
                        else:
                            self.timestamp_start = timestamp_start

                    elif 'Settings' in line:
                        mode = 'settings'

                if mode == 'settings':

                    if 'Settings' in line:
                        measurement_type_file = line.split(':')[1].strip()

                        if measurement_type != measurement_type_file:
                            raise RuntimeError('File name does not match settings for measurement type {}'.format(measurement_type))

                        self.stream_settings[measurement_type] = {}
                        self.timestamps[measurement_type]      = []
                        self.samples[measurement_type]         = []

                    elif 'SAMPLE_RATE' in line or 'RESOLUTION' in line or 'RANGE' in line or 'CHANNELS' in line:

                        name, value = [x.strip() for x in line.split(':')]
                        self.stream_settings[measurement_type][name] = int(value)

                    elif line == '\n':
                        continue

                    elif 'Measurement' in line:
                        mode = 'measurement'

                if mode == 'measurement':

                    if 'Measurement' in line:
                        measurement_type_file = line.split(':')[1].strip()

                        if measurement_type != measurement_type_file:
                            raise RuntimeError('Data does not match settings for measurement type {}'.format(measurement_type))

                    elif 'relative_timestamp' in line:
                        continue

                    elif line == '\n':
                        continue

                    else:
                        row = [x.strip() for x in line.split('\t')]

                        self.timestamps[measurement_type].append(float(row[0]))
                        self.samples[measurement_type].append([int(x) for x in row[1:]])

        self.timestamps[measurement_type] = np.array(self.timestamps[measurement_type], dtype=np.float64) + self.timestamp_start
        self.samples[measurement_type]    = np.array(self.samples[measurement_type], dtype=np.float64)

    def read_log(self) -> None:
        """Parse the session's log.txt into (timestamp, level, message) entries."""

        if not (self.save_dir / 'log.txt').exists():
            raise RuntimeError('Could not find log file')

        if not hasattr(self, 'log'):
            self.log = []

        with open(self.save_dir / 'log.txt', 'r') as f:

            for line in f.readlines():

                if line == '\n':
                    continue

                x = line.find(' - ')
                timestamp = datetime.datetime.strptime(line[:x], '%Y-%m-%d %H:%M:%S.%f').timestamp()
                line = line[x+3:]

                x = line.find(' - ')
                device_name = line[:x]
                line = line[x+3:]

                if hasattr(self, 'device_name') and self.device_name != device_name:
                    raise RuntimeError('Device name mismatch between logs and data files')

                x = line.find(': ')
                loglevel = line[:x]
                line = line[x+2:]

                message = line

                self.log.append([timestamp, loglevel, message])

class Plotter:
    """Resamples a recorded session's PPG/ACC/GYR to a common time base and plots it."""

    # Order in which measurement types are processed and plotted, when present
    TYPE_ORDER: list[str] = ['PPG', 'ACC', 'GYR']

    def __init__(self, record: Reader) -> None:
        """
        Args:
            record: An already-loaded session to process and plot.
        """

        self.record = record
        self.types  = [t for t in self.TYPE_ORDER if t in record.measurement_types]

        self.process()
        self.plot()

    def process(self) -> None:
        """Resample each present measurement type onto a shared time base, filtering PPG."""

        # Determine common start and end timestamps
        time_start = np.max([np.min(self.record.timestamps[t]) for t in self.types])
        time_end   = np.min([np.max(self.record.timestamps[t]) for t in self.types])

        self.processed_time    = {}
        self.processed_samples = {}

        for measurement_type in self.types:

            measurement_tools = getattr(tools, measurement_type)
            time    = self.record.timestamps[measurement_type]
            samples = measurement_tools.fetch_signals(self.record.samples[measurement_type])
            fs      = tools.General.estimate_fs(time)

            time, samples = tools.General.resample(time, samples, fs, time_start, time_end)

            if measurement_type == 'PPG':
                samples = tools.PPG.bandpass_filter(samples, fs)
                samples = tools.General.std_filter(samples, int(np.round(fs/2)*2+1))

            self.processed_time[measurement_type]    = time
            self.processed_samples[measurement_type] = samples

    def plot(self) -> None:
        """Plot each processed measurement type in its own subplot with a shared x-axis."""

        plt.figure()

        axis = None
        for i, measurement_type in enumerate(self.types):

            if i == 0:
                axis = plt.subplot(len(self.types), 1, i+1)
            else:
                axis = plt.subplot(len(self.types), 1, i+1, sharex=axis)

            plt.plot(self.processed_time[measurement_type], self.processed_samples[measurement_type])
            plt.title(measurement_type)

        plt.show()
