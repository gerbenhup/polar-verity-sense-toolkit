"""Library for capturing, saving, live-plotting and reviewing Polar Verity
Sense BLE streams.

Organized into four namespace classes:

- ``Driver`` — connects to a device over BLE and streams PPG/ACC/GYR data
  in SDK mode.
- ``Live`` — consumes a live stream, either plotting it in real time or
  saving it to disk.
- ``Record`` — reads a saved recording back in and plots it.
- ``Tools`` — signal-processing helpers shared by ``Live`` and ``Record``.
"""

# Must run before matplotlib.pyplot is imported anywhere (by .live or .record)
import matplotlib
matplotlib.use('TkAgg')

from .driver import Driver
from .live import Live
from .record import Record
from .tools import Tools

__all__ = ['Driver', 'Live', 'Record', 'Tools']
