# Polar Verity Sense Toolkit

Python tools for capturing and analyzing raw PPG, accelerometer, and gyroscope
data from a [Polar Verity Sense](https://www.polar.com/en/products/accessories/polar-verity-sense) optical
heart rate sensor over Bluetooth Low Energy (BLE), using the device's SDK
streaming mode.

## Features

- `polar.py` — the library. Organized into four namespace classes:
  `Driver` (BLE connection and SDK-mode streaming), `Live` (live plotting and
  saving of a stream), `Record` (reading and plotting a recorded session),
  and `Tools` (signal processing helpers).
- `capture.py` — scans for nearby Polar Sense devices, connects
  automatically, enables SDK streaming mode, and simultaneously records
  PPG/ACC/GYR data (and connection logs) to disk and live-plots it.
- `view.py` — reads a recorded session back in, resamples the three signal
  streams to a common time base, applies basic filtering, and plots the
  result.

## Requirements

- Python 3.9+
- A Polar Verity Sense device
- A computer with Bluetooth Low Energy support

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Capturing data

```bash
python capture.py
```

This continuously scans for Polar Sense devices in range and automatically
connects to and records from any it finds, live-plotting each stream while
simultaneously saving it to disk. Each session is saved to a timestamped
directory (`<YYYYMMDDHHMMSS>-<device_name>/`) containing one `.txt` file per
measurement type (`PPG.txt`, `ACC.txt`, `GYR.txt`) plus a `log.txt`
connection/debug log.

### Reading a recorded session

```bash
python view.py path/to/<YYYYMMDDHHMMSS>-<device_name>/
```

This loads the recorded session, resamples PPG/ACC/GYR to a common time base,
and plots all three signals.

## Notes

- The Bluetooth GATT protocol used here (SDK mode, delta-frame decoding) is
  documented in Polar's official
  [BLE SDK repository](https://github.com/polarofficial/polar-ble-sdk),
  including its `technical_documentation` folder (Measurement Data
  Specification, SDK mode explanation), which are useful references if you
  want to extend this toolkit.
- This is an unofficial, independent project and is not affiliated with or
  endorsed by Polar.

## License

MIT — see [LICENSE](LICENSE).
