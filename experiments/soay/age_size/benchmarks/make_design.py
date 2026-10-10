#!/usr/bin/env python3
"""Create a common parameter design for the age-size IBM speed benchmark."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.soay.age_size.benchmarks.benchmark_common import (
    MODEL_NAMES,
    PARAMETER_NAMES,
    file_sha256,
    load_samples,
    parameter_values,
    sample_counts,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("samples_file")
    parser.add_argument("output_file")
    parser.add_argument("--number-datasets", type=int, default=5_000)
    parser.add_argument("--number-years", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.number_datasets <= 0:
        raise ValueError("number_datasets must be positive")
    if args.number_years <= 0:
        raise ValueError("number_years must be positive")

    data_path = Path(args.data_file).resolve()
    samples_path = Path(args.samples_file).resolve()
    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    observed_years = np.asarray(sorted(data["yr"].dropna().unique()))
    if len(observed_years) == 0:
        raise ValueError("The dataset contains no observed years")
    initial_year = observed_years[0]
    initial_population_size = len(
        data.loc[data["yr"] == initial_year, ["z", "age"]].dropna()
    )

    samples = load_samples(samples_path)
    counts = sample_counts(samples)
    rng = np.random.default_rng(args.seed)
    indices = np.column_stack(
        [
            rng.integers(0, count, args.number_datasets, dtype=np.int64)
            for count in counts
        ]
    )
    values = parameter_values(samples, indices)
    task_seeds = rng.integers(
        0, np.iinfo(np.uint32).max, args.number_datasets, dtype=np.uint32
    )
    data_hash = file_sha256(data_path)
    samples_hash = file_sha256(samples_path)

    np.savez_compressed(
        output_path,
        parameter_indices=indices,
        parameter_values=values,
        parameter_names=np.asarray(PARAMETER_NAMES),
        model_names=np.asarray(MODEL_NAMES),
        task_seeds=task_seeds,
        number_datasets=np.asarray(args.number_datasets),
        number_years=np.asarray(args.number_years),
        experiment_seed=np.asarray(args.seed),
        initial_year=np.asarray(initial_year),
        data_sha256=np.asarray(data_hash),
        samples_sha256=np.asarray(samples_hash),
    )
    write_json(
        output_path.with_suffix(".json"),
        {
            "design_file": str(output_path),
            "data_file": str(data_path),
            "samples_file": str(samples_path),
            "data_sha256": data_hash,
            "samples_sha256": samples_hash,
            "number_datasets": args.number_datasets,
            "number_years": args.number_years,
            "experiment_seed": args.seed,
            "initial_year": int(initial_year),
            "initial_population_size": initial_population_size,
            "coordinate_order": ["size", "age"],
            "model_names": list(MODEL_NAMES),
            "parameter_names": list(PARAMETER_NAMES),
        },
    )
    print(f"Saved design: {output_path}", flush=True)
    print(f"Saved metadata: {output_path.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
