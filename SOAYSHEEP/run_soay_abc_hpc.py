#!/usr/bin/env python3
"""Run the Soay ABC Analysis.

Usage: python run_soay_abc_hpc.py DATA.csv SAMPLES.npz OUTPUT_DIRECTORY
"""

import sys

from soay_abc_model import run


if len(sys.argv) != 4:
    raise SystemExit(__doc__)

run(
    data_file=sys.argv[1],
    samples_file=sys.argv[2],
    output_directory=sys.argv[3],
)
