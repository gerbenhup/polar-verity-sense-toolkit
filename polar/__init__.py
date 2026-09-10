"""Library for capturing, saving, live-plotting and reviewing Polar Verity
Sense BLE streams.

Organized into four submodules, used as namespaces:

- ``driver`` — connects to a device over BLE and streams PPG/ACC/GYR data
  in SDK mode.
- ``live`` — consumes a live stream, either plotting it in real time or
  saving it to disk.
- ``record`` — reads a saved recording back in and plots it.
- ``tools`` — signal-processing helpers shared by ``live`` and ``record``.

Usage: ``import polar`` then ``polar.driver.Device``, ``polar.live.Saver``,
``polar.record.Reader``, ``polar.tools.PPG``, etc.
"""

# Must run before matplotlib.pyplot is imported anywhere (by .live or .record)
import matplotlib
matplotlib.use('TkAgg')

from . import driver
from . import live
from . import record
from . import tools

__all__ = ['driver', 'live', 'record', 'tools']
