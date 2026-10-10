#!/usr/bin/env python3
"""Summarise DeepRV and GPflow speed across initial populations."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.soay.size_only.benchmarks.benchmark_common import write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directories", nargs="+")
    parser.add_argument("--output-file")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarise_method(results_directory, method):
    method_directory = Path(results_directory) / method
    run_info = load_json(method_directory / "run_info.json")
    with np.load(method_directory / "simulation_outputs.npz") as outputs:
        population_sizes = np.asarray(outputs["population_sizes"])
        overflows = np.asarray(outputs["overflow"], dtype=bool)

    capacity = int(
        run_info.get("max_locations", run_info.get("max_locations_diagnostic"))
    )
    exceeds_capacity = population_sizes > capacity
    first_overflow_year = np.where(
        exceeds_capacity.any(axis=1), exceeds_capacity.argmax(axis=1) + 1, 0
    )
    within_capacity = int(len(overflows) - overflows.sum())
    completed = within_capacity if method == "deeprv" else len(overflows)
    reported_population_sizes = population_sizes.astype(float)
    if method == "deeprv":
        years = np.arange(population_sizes.shape[1])[None]
        after_overflow = (first_overflow_year[:, None] > 0) & (
            years >= first_overflow_year[:, None]
        )
        reported_population_sizes[after_overflow] = np.nan
    seconds = float(run_info["simulation_seconds"])
    return {
        "simulation_seconds": seconds,
        "datasets_per_second": float(run_info["datasets_per_second"]),
        "completed_datasets": completed,
        "completed_datasets_per_second": completed / seconds,
        "within_capacity_datasets": within_capacity,
        "within_capacity_datasets_per_second": within_capacity / seconds,
        "extinctions": int(run_info["extinctions"]),
        "overflows": int(overflows.sum()),
        "overflow_percentage": float(100 * overflows.mean()),
        "first_overflow_counts_by_year": [
            int(np.sum(first_overflow_year == year))
            for year in range(1, population_sizes.shape[1] + 1)
        ],
        "median_population_by_year": np.nanmedian(
            reported_population_sizes, axis=0
        ).tolist(),
        "population_95th_percentile_by_year": np.nanpercentile(
            reported_population_sizes, 95, axis=0
        ).tolist(),
        "population_99th_percentile_by_year": np.nanpercentile(
            reported_population_sizes, 99, axis=0
        ).tolist(),
        "maximum_population_by_year": np.nanmax(
            reported_population_sizes, axis=0
        ).tolist(),
        "total_unique_requested_locations": int(
            run_info["total_unique_requested_locations"]
        ),
    }


def summarise_scenario(results_directory):
    results_directory = Path(results_directory).resolve()
    deeprv_info = load_json(results_directory / "deeprv" / "run_info.json")
    gpflow_info = load_json(results_directory / "gpflow" / "run_info.json")
    if deeprv_info["design_sha256"] != gpflow_info["design_sha256"]:
        raise ValueError(f"Methods use different designs in {results_directory}")
    if deeprv_info["initial_population_size"] != gpflow_info["initial_population_size"]:
        raise ValueError(
            f"Methods use different initial populations in {results_directory}"
        )

    deeprv = summarise_method(results_directory, "deeprv")
    gpflow = summarise_method(results_directory, "gpflow")
    return {
        "results_directory": str(results_directory),
        "initial_years": deeprv_info["initial_years"],
        "initial_population_size": int(deeprv_info["initial_population_size"]),
        "initial_unique_locations": int(deeprv_info["initial_unique_locations"]),
        "number_datasets": int(deeprv_info["number_datasets"]),
        "number_years": int(deeprv_info["number_years"]),
        "deeprv_speedup_over_parallel_gpflow": (
            gpflow["simulation_seconds"] / deeprv["simulation_seconds"]
        ),
        "deeprv": deeprv,
        "gpflow": gpflow,
    }


def main():
    args = parse_args()
    scenarios = [summarise_scenario(path) for path in args.results_directories]
    scenarios.sort(key=lambda result: result["initial_population_size"])
    summary = {"scenarios": scenarios}
    if args.output_file:
        write_json(args.output_file, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
