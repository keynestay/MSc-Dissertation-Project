#!/usr/bin/env python3
"""Compare exact-GP and DeepRV discrepancies for every parameter combination."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from experiments.soay.size_only.validation.ibm.comparison_common import (
    STATUS_ERROR,
    STATUS_EXTINCTION,
    STATUS_MAX_LOCATIONS,
    STATUS_SUCCESS,
    file_sha256,
    status_counts,
    write_json,
)


QUANTILES = np.asarray([0.05, 0.5, 0.95])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("design_file")
    parser.add_argument("results_directory")
    return parser.parse_args()


def load_method_results(method_directory, shape):
    number_parameters, number_replicates, number_years, number_summaries = shape
    distances = np.full(shape, np.nan)
    population_sizes = np.full(shape[:3], np.nan)
    status = np.full(shape[:2], -1, dtype=int)
    extinction_year = np.full(shape[:2], -1, dtype=int)
    seconds = np.full(shape[:2], np.nan)
    seen = np.zeros(shape[:2], dtype=bool)

    chunk_paths = sorted(Path(method_directory).glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No result chunks found in {method_directory}")
    for chunk_path in chunk_paths:
        with np.load(chunk_path) as chunk:
            parameter_id = chunk["parameter_id"].astype(int)
            replicate_id = chunk["replicate_id"].astype(int)
            if np.any(parameter_id < 0) or np.any(parameter_id >= number_parameters):
                raise ValueError(f"Invalid parameter ID in {chunk_path}")
            if np.any(replicate_id < 0) or np.any(replicate_id >= number_replicates):
                raise ValueError(f"Invalid replicate ID in {chunk_path}")
            if np.any(seen[parameter_id, replicate_id]):
                raise ValueError(f"Duplicate result in {chunk_path}")
            distances[parameter_id, replicate_id] = chunk["distances"]
            population_sizes[parameter_id, replicate_id] = chunk["population_sizes"]
            status[parameter_id, replicate_id] = chunk["status"]
            extinction_year[parameter_id, replicate_id] = chunk["extinction_year"]
            seconds[parameter_id, replicate_id] = chunk["seconds"]
            seen[parameter_id, replicate_id] = True

    if not seen.all():
        missing = np.argwhere(~seen)
        raise ValueError(
            f"{method_directory} is missing {len(missing)} parameter-replicate results"
        )
    return {
        "distances": distances,
        "population_sizes": population_sizes,
        "status": status,
        "extinction_year": extinction_year,
        "seconds": seconds,
    }


def summarise_replicates(values):
    finite_values = np.where(np.isfinite(values), values, np.nan)
    finite_count = np.sum(np.isfinite(finite_values), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(finite_values, axis=1)
        standard_deviation = np.nanstd(finite_values, axis=1, ddof=1)
        quantiles = np.nanquantile(finite_values, QUANTILES, axis=1)
    return mean, standard_deviation, quantiles, finite_count


def validate_method_info(method_directory, design_hash):
    info_path = Path(method_directory) / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("design_sha256") != design_hash:
        raise ValueError(f"{method_directory} used a different benchmark design")
    return info


def main():
    args = parse_args()
    design_path = Path(args.design_file).resolve()
    results_directory = Path(args.results_directory).resolve()
    exact_directory = results_directory / "exact_gp"
    deeprv_directory = results_directory / "deeprv"
    design_hash = file_sha256(design_path)

    with np.load(design_path) as design:
        number_parameters = int(design["number_parameters"].item())
        number_replicates = int(design["number_replicates"].item())
        years = design["years"]
        summary_names = design["summary_names"]
        parameter_indices = design["parameter_indices"]
        parameter_values = design["parameter_values"]
        parameter_names = design["parameter_names"]
    shape = (
        number_parameters,
        number_replicates,
        len(years),
        len(summary_names),
    )

    exact_info = validate_method_info(exact_directory, design_hash)
    deeprv_info = validate_method_info(deeprv_directory, design_hash)
    exact = load_method_results(exact_directory, shape)
    deeprv = load_method_results(deeprv_directory, shape)

    exact_mean, exact_sd, exact_quantiles, exact_count = summarise_replicates(
        exact["distances"]
    )
    deeprv_mean, deeprv_sd, deeprv_quantiles, deeprv_count = summarise_replicates(
        deeprv["distances"]
    )
    mean_difference = deeprv_mean - exact_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        difference_standard_error = np.sqrt(
            exact_sd**2 / exact_count + deeprv_sd**2 / deeprv_count
        )

    exact_population_mean, exact_population_sd, exact_population_quantiles, _ = (
        summarise_replicates(exact["population_sizes"])
    )
    deeprv_population_mean, deeprv_population_sd, deeprv_population_quantiles, _ = (
        summarise_replicates(deeprv["population_sizes"])
    )

    exact_status_rate = np.stack(
        [
            np.mean(exact["status"] == code, axis=1)
            for code in (
                STATUS_SUCCESS,
                STATUS_EXTINCTION,
                STATUS_MAX_LOCATIONS,
                STATUS_ERROR,
            )
        ],
        axis=1,
    )
    deeprv_status_rate = np.stack(
        [
            np.mean(deeprv["status"] == code, axis=1)
            for code in (
                STATUS_SUCCESS,
                STATUS_EXTINCTION,
                STATUS_MAX_LOCATIONS,
                STATUS_ERROR,
            )
        ],
        axis=1,
    )

    output_path = results_directory / "comparison_summary.npz"
    np.savez_compressed(
        output_path,
        parameter_indices=parameter_indices,
        parameter_values=parameter_values,
        parameter_names=parameter_names,
        years=years,
        summary_names=summary_names,
        quantile_probabilities=QUANTILES,
        exact_gp_mean=exact_mean,
        deeprv_mean=deeprv_mean,
        mean_difference=mean_difference,
        difference_standard_error=difference_standard_error,
        exact_gp_standard_deviation=exact_sd,
        deeprv_standard_deviation=deeprv_sd,
        exact_gp_quantiles=exact_quantiles,
        deeprv_quantiles=deeprv_quantiles,
        exact_gp_finite_count=exact_count,
        deeprv_finite_count=deeprv_count,
        exact_gp_population_mean=exact_population_mean,
        deeprv_population_mean=deeprv_population_mean,
        exact_gp_population_standard_deviation=exact_population_sd,
        deeprv_population_standard_deviation=deeprv_population_sd,
        exact_gp_population_quantiles=exact_population_quantiles,
        deeprv_population_quantiles=deeprv_population_quantiles,
        status_names=np.asarray(
            ["success", "extinction", "max_locations", "error"]
        ),
        exact_gp_status_rate=exact_status_rate,
        deeprv_status_rate=deeprv_status_rate,
    )

    metadata_path = results_directory / "comparison_summary.json"
    write_json(
        metadata_path,
        {
            "design_file": str(design_path),
            "design_sha256": design_hash,
            "number_parameters": number_parameters,
            "number_replicates": number_replicates,
            "years": years.tolist(),
            "number_summaries": len(summary_names),
            "summary_names": summary_names.tolist(),
            "quantile_probabilities": QUANTILES.tolist(),
            "distance_summary_rule": "finite replicates only",
            "exact_gp_status_counts": status_counts(exact["status"]),
            "deeprv_status_counts": status_counts(deeprv["status"]),
            "exact_gp_elapsed_seconds": exact_info.get("elapsed_seconds"),
            "deeprv_elapsed_seconds": deeprv_info.get("elapsed_seconds"),
            "comparison_file": str(output_path),
        },
    )
    print(f"Saved comparison: {output_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
