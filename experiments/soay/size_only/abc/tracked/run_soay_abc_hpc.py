#!/usr/bin/env python3
"""Run normal Soay ABC with proposal tracking.

Usage: python run_soay_abc_hpc.py DATA.csv SAMPLES.npz OUTPUT_DIRECTORY
"""

from experiments.paths import GENERATED, PROJECT_ROOT

import os
import sys
from pathlib import Path

PROJECT_DIRECTORY = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_DIRECTORY))
os.environ["PYTHONPATH"] = str(PROJECT_DIRECTORY) + os.pathsep + os.environ.get(
    "PYTHONPATH", ""
)

from experiments.soay.size_only.abc.tracked.soay_abc_model import run  # noqa: E402, I001


if len(sys.argv) != 4:
    raise SystemExit(__doc__)

run(sys.argv[1], sys.argv[2], sys.argv[3])
