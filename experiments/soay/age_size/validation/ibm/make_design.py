#!/usr/bin/env python3
# ruff: noqa: E402
"""Create the shared parameter design for age-size exact-GP and DeepRV runs."""

from experiments.paths import GENERATED, PROJECT_ROOT

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIRECTORY = PROJECT_ROOT
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from experiments.soay.age_size.validation.ibm.comparison_common import (
    MODEL_NAMES,
    PARAMETER_NAMES,
    SUMMARY_NAMES,
    file_sha256,
    load_compact_samples,
    parameter_values,
    sample_counts,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("samples_file")
    parser.add_argument("output_file")
    parser.add_argument("--number-parameters", type=int, default=5_000)
    parser.add_argument("--number-replicates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.number_parameters <= 0 or args.number_replicates <= 0:
        raise ValueError("number-parameters and number-replicates must be positive")
    data_path, samples_path, output_path = map(
        lambda path: Path(path).resolve(),
        (args.data_file, args.samples_file, args.output_file),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    samples = load_compact_samples(samples_path)
    years = np.asarray(sorted(data["yr"].dropna().unique()))
    if len(years) == 0:
        raise ValueError("The dataset contains no observed years")
    pooled_sizes = pd.concat([data["z"], data["z1"], data["Rcsz"]]).dropna()
    boundaries = np.quantile(pooled_sizes, [0.2, 0.4, 0.6, 0.8])
    size_edges = np.r_[-np.inf, boundaries, np.inf]

    rng = np.random.default_rng(args.seed)
    indices = np.column_stack(
        [
            rng.integers(0, count, args.number_parameters, dtype=np.int64)
            for count in sample_counts(samples)
        ]
    )
    values = parameter_values(samples, indices)
    data_hash, samples_hash = file_sha256(data_path), file_sha256(samples_path)
    np.savez_compressed(
        output_path,
        parameter_indices=indices,
        parameter_values=values,
        parameter_names=np.asarray(PARAMETER_NAMES),
        model_names=np.asarray(MODEL_NAMES),
        number_parameters=np.asarray(args.number_parameters),
        number_replicates=np.asarray(args.number_replicates),
        experiment_seed=np.asarray(args.seed),
        years=years,
        size_edges=size_edges,
        summary_names=np.asarray(SUMMARY_NAMES),
        data_sha256=np.asarray(data_hash),
        samples_sha256=np.asarray(samples_hash),
    )

    complete_initial = data.loc[data["yr"] == years[0], ["z", "age"]].dropna()
    metadata = {
        "design_file": str(output_path),
        "data_file": str(data_path),
        "samples_file": str(samples_path),
        "data_sha256": data_hash,
        "samples_sha256": samples_hash,
        "number_parameters": args.number_parameters,
        "number_replicates": args.number_replicates,
        "number_datasets_per_method": args.number_parameters * args.number_replicates,
        "experiment_seed": args.seed,
        "number_years": len(years),
        "years": years.tolist(),
        "initial_population_size": len(complete_initial),
        "size_quantile_boundaries": boundaries.tolist(),
        "coordinate_order": ["size", "age"],
        "model_names": list(MODEL_NAMES),
        "parameter_names": list(PARAMETER_NAMES),
        "summary_names": list(SUMMARY_NAMES),
    }
    write_json(output_path.with_suffix(".json"), metadata)
    print(f"Saved design: {output_path}", flush=True)
    print(f"Saved metadata: {output_path.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
