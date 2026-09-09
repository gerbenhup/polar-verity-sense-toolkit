"""Scan for Polar Sense devices and simultaneously record and live-plot each one found."""

from __future__ import annotations

import asyncio
from pathlib import Path
import numpy as np
from polar import Driver, Live

def main() -> None:
    """Run the device manager until interrupted, saving and plotting every captured stream."""

    save_dir = Path('.')

    saver   = Live.Saver(save_dir)
    plotter = Live.Plotter(window_length=10, update_interval=1)

    def stream_callback(device_obj: Driver.Device, measurement_type: Driver.Device.MeasurementType, timestamps: np.ndarray, frame_samples: np.ndarray) -> None:
        saver.stream_callback(device_obj, measurement_type, timestamps, frame_samples)
        plotter.stream_callback(device_obj, measurement_type, timestamps, frame_samples)

    manager = Driver.Manager(
        save_dir        = save_dir,
        stream_callback = stream_callback,
        loglevel        = Driver.Device.LogLevel.INFO
    )

    asyncio.run(manager.run())

if __name__ == '__main__':
    main()
