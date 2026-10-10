#!/usr/bin/env python3
"""Create the common parameter design for exact-GP and DeepRV validation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special

from experiments.soay.size_only.validation.ibm.comparison_common import (
    MODEL_NAMES,
    SUMMARY_NAMES,
    file_sha256,
    load_compact_samples,
    parameter_names,
    parameter_values,
    sample_counts,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("samples_file")
    parser.add_argument("output_file")
    parser.add_argument("--number-parameters", type=int, default=50_000)
    parser.add_argument("--number-replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.number_parameters <= 0:
        raise ValueError("number_parameters must be positive")
    if args.number_replicates <= 0:
        raise ValueError("number_replicates must be positive")

    data_path = Path(args.data_file).resolve()
    samples_path = Path(args.samples_file).resolve()
    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    samples = load_compact_samples(samples_path)
    counts = sample_counts(samples)
    years = np.asarray(sorted(data["yr"].dropna().unique()))
    if len(years) == 0:
        raise ValueError("The dataset contains no observed years")

    pooled_sizes = pd.concat([data["z"], data["z1"], data["Rcsz"]]).dropna()
    boundaries = np.quantile(pooled_sizes, [0.2, 0.4, 0.6, 0.8])
    size_edges = np.r_[-np.inf, boundaries, np.inf]
    recruit_probability = float(scipy.special.expit(samples["recr_int"]).mean())

    rng = np.random.default_rng(args.seed)
    indices = np.column_stack(
        [
            rng.integers(0, count, args.number_parameters, dtype=np.int64)
            for count in counts
        ]
    )
    values = parameter_values(samples, indices)
    names = parameter_names()
    data_hash = file_sha256(data_path)
    samples_hash = file_sha256(samples_path)

    np.savez_compressed(
        output_path,
        parameter_indices=indices,
        parameter_values=values,
        parameter_names=names,
        model_names=np.asarray(MODEL_NAMES),
        number_parameters=np.asarray(args.number_parameters),
        number_replicates=np.asarray(args.number_replicates),
        experiment_seed=np.asarray(args.seed),
        years=years,
        size_edges=size_edges,
        recruit_probability=np.asarray(recruit_probability),
        summary_names=np.asarray(SUMMARY_NAMES),
        data_sha256=np.asarray(data_hash),
        samples_sha256=np.asarray(samples_hash),
    )

    metadata_path = output_path.with_suffix(".json")
    write_json(
        metadata_path,
        {
            "design_file": str(output_path),
            "data_file": str(data_path),
            "samples_file": str(samples_path),
            "data_sha256": data_hash,
            "samples_sha256": samples_hash,
            "number_parameters": args.number_parameters,
            "number_replicates": args.number_replicates,
            "experiment_seed": args.seed,
            "number_years": len(years),
            "years": years.tolist(),
            "initial_population_size": int(np.sum(data["yr"] == years[0])),
            "size_quantile_boundaries": boundaries.tolist(),
            "recruit_probability": recruit_probability,
            "model_names": list(MODEL_NAMES),
            "parameter_names": names.tolist(),
            "summary_names": SUMMARY_NAMES,
        },
    )
    print(f"Saved design: {output_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
