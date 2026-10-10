#!/usr/bin/env python3
# ruff: noqa: E402
"""Run tracked age-size Soay ABC with Transformer DeepRV.

Usage: python run_soay_abc_hpc_deeprv_agesize.py DATA.csv SAMPLES.npz CHECKPOINT OUTPUT_DIRECTORY SELECTION_JSON
"""

from experiments.paths import GENERATED, PROJECT_ROOT

import os
import sys
from pathlib import Path

PROJECT_DIRECTORY = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_DIRECTORY))
os.environ["PYTHONPATH"] = str(PROJECT_DIRECTORY) + os.pathsep + os.environ.get("PYTHONPATH", "")

from experiments.soay.age_size.abc.deeprv.soay_abc_model_deeprv_agesize import run

if len(sys.argv) != 6:
    raise SystemExit(__doc__)

run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
