#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Run tracked age-size Soay ABC.

Usage: python run_soay_abc_hpc_agesize.py DATA.csv SAMPLES.npz OUTPUT_DIRECTORY
"""

from experiments.paths import GENERATED, PROJECT_ROOT

import os
import sys
from pathlib import Path

PROJECT_DIRECTORY = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_DIRECTORY))
os.environ["PYTHONPATH"] = str(PROJECT_DIRECTORY) + os.pathsep + os.environ.get("PYTHONPATH", "")

from experiments.soay.age_size.abc.gp.soay_abc_model_agesize import run


if len(sys.argv) != 4:
    raise SystemExit(__doc__)

run(sys.argv[1], sys.argv[2], sys.argv[3])
