"""View a session recorded by capture.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from polar import Record

def main() -> None:
    """Parse CLI args, load the recorded session, and plot it."""

    parser = argparse.ArgumentParser(description='View a session recorded by capture.py')
    parser.add_argument('record_path', type=Path, help='Path to a recorded session directory (e.g. 20230228130024-C0887322/)')
    args = parser.parse_args()

    record = Record.Reader(args.record_path)
    Record.Plotter(record)

if __name__ == '__main__':
    main()
