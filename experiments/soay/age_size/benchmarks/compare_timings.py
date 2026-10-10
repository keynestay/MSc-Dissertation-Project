#!/usr/bin/env python3
"""Compare completed age-size DeepRV and parallel GPflow speed runs."""

import argparse
import json
from pathlib import Path

from experiments.soay.age_size.benchmarks.benchmark_common import write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("deeprv_run_info")
    parser.add_argument("gpflow_run_info")
    parser.add_argument("--output-file")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    deeprv = load_json(args.deeprv_run_info)
    gpflow = load_json(args.gpflow_run_info)
    if deeprv["design_sha256"] != gpflow["design_sha256"]:
        raise ValueError("The runs use different benchmark designs")
    if deeprv["number_datasets"] != gpflow["number_datasets"]:
        raise ValueError("The runs contain different numbers of datasets")

    comparison = {
        "number_datasets": deeprv["number_datasets"],
        "number_years": deeprv["number_years"],
        "deeprv_simulation_seconds": deeprv["simulation_seconds"],
        "gpflow_simulation_seconds": gpflow["simulation_seconds"],
        "deeprv_datasets_per_second": deeprv["datasets_per_second"],
        "gpflow_datasets_per_second": gpflow["datasets_per_second"],
        "deeprv_speedup_over_parallel_gpflow": gpflow["simulation_seconds"]
        / deeprv["simulation_seconds"],
        "deeprv_batch_size": deeprv["batch_size"],
        "gpflow_cpu_cores": gpflow["num_cpus"],
        "deeprv_extinctions": deeprv["extinctions"],
        "gpflow_extinctions": gpflow["extinctions"],
        "deeprv_total_requested_locations": deeprv["total_requested_locations"],
        "gpflow_total_requested_locations": gpflow["total_requested_locations"],
        "deeprv_total_unique_requested_locations": deeprv[
            "total_unique_requested_locations"
        ],
        "gpflow_total_unique_requested_locations": gpflow[
            "total_unique_requested_locations"
        ],
        "deeprv_overflows": deeprv["overflows"],
        "gpflow_overflows": gpflow["overflows"],
    }
    if args.output_file:
        write_json(args.output_file, comparison)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
