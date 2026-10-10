#!/usr/bin/env python3
"""Create a common parameter design for the IBM speed benchmark."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special

from experiments.soay.size_only.benchmarks.benchmark_common import (
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
    parser.add_argument("--initial-through-year", type=int)
    parser.add_argument("--initial-population-size", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.number_datasets <= 0:
        raise ValueError("number_datasets must be positive")
    if args.number_years <= 0:
        raise ValueError("number_years must be positive")
    if args.initial_population_size is not None and args.initial_population_size <= 1:
        raise ValueError("initial_population_size must be greater than one")

    data_path = Path(args.data_file).resolve()
    samples_path = Path(args.samples_file).resolve()
    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    observed_years = np.asarray(sorted(data["yr"].dropna().unique()))
    if len(observed_years) == 0:
        raise ValueError("The dataset contains no observed years")
    if args.initial_through_year is None:
        initial_years = observed_years[:1]
    else:
        matches = np.flatnonzero(observed_years == args.initial_through_year)
        if len(matches) == 0:
            raise ValueError(
                f"initial-through-year must be one of {observed_years.tolist()}"
            )
        initial_years = observed_years[: matches[0] + 1]
    source_sizes = (
        data.loc[data["yr"].isin(initial_years), "z"].dropna().to_numpy(float)
    )
    if len(source_sizes) <= 1:
        raise ValueError("The source population must contain at least two individuals")
    if args.initial_population_size is None:
        initial_sizes = source_sizes
        initial_population_method = "observed"
    else:
        probabilities = (
            np.arange(args.initial_population_size) + 0.5
        ) / args.initial_population_size
        initial_sizes = np.quantile(source_sizes, probabilities)
        initial_population_method = "empirical_quantiles"
    initial_population_size = int(len(initial_sizes))
    initial_unique_locations = int(np.unique(initial_sizes).size)

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
    recruit_probability = float(scipy.special.expit(samples["recr_int"]).mean())
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
        initial_year=np.asarray(initial_years[0]),
        initial_years=initial_years,
        initial_through_year=np.asarray(initial_years[-1]),
        initial_sizes=initial_sizes,
        initial_population_method=np.asarray(initial_population_method),
        initial_population_size=np.asarray(initial_population_size),
        initial_unique_locations=np.asarray(initial_unique_locations),
        source_population_size=np.asarray(len(source_sizes)),
        source_unique_locations=np.asarray(np.unique(source_sizes).size),
        recruit_probability=np.asarray(recruit_probability),
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
            "initial_year": int(initial_years[0]),
            "initial_years": initial_years.tolist(),
            "initial_through_year": int(initial_years[-1]),
            "initial_population_method": initial_population_method,
            "initial_population_size": initial_population_size,
            "initial_unique_locations": initial_unique_locations,
            "source_population_size": int(len(source_sizes)),
            "source_unique_locations": int(np.unique(source_sizes).size),
            "recruit_probability": recruit_probability,
            "model_names": list(MODEL_NAMES),
            "parameter_names": list(PARAMETER_NAMES),
        },
    )
    print(f"Saved design: {output_path}", flush=True)
    print(f"Saved metadata: {output_path.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
