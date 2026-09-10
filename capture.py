"""Scan for Polar Sense devices and simultaneously record and live-plot each one found."""

from __future__ import annotations

import asyncio
from pathlib import Path
import numpy as np
import polar

def main() -> None:
    """Run the device manager until interrupted, saving and plotting every captured stream."""

    save_dir = Path('.')

    saver   = polar.live.Saver(save_dir)
    plotter = polar.live.Plotter(window_length=10, update_interval=1)

    def stream_callback(device_obj: polar.driver.Device, measurement_type: polar.driver.Device.MeasurementType, timestamps: np.ndarray, frame_samples: np.ndarray) -> None:
        saver.stream_callback(device_obj, measurement_type, timestamps, frame_samples)
        plotter.stream_callback(device_obj, measurement_type, timestamps, frame_samples)

    manager = polar.driver.Manager(
        save_dir        = save_dir,
        stream_callback = stream_callback,
        loglevel        = polar.driver.Device.LogLevel.INFO
    )

    asyncio.run(manager.run())

if __name__ == '__main__':
    main()
