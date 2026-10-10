#!/usr/bin/env python3
"""Run DeepRV IBM validation with one restored GPU decoder."""

import argparse
import time
from pathlib import Path

import jax
import numpy as np
from jax import random

from experiments.soay.size_only.training.deeprv_decoder import get_decoder
from experiments.soay.size_only.validation.ibm.comparison_common import (
    JITTER,
    METHOD_DEEPRV,
    STATUS_MAX_LOCATIONS,
    empty_task_result,
    load_experiment,
    make_task_chunks,
    make_task_rng,
    make_task_seed,
    read_json,
    save_chunk,
    selected_models,
    simulate_dataset,
    status_counts,
    validate_existing_chunks,
    write_json,
)


RESULTS_PER_CHUNK = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("samples_file")
    parser.add_argument("design_file")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_directory")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def collect_status_counts(method_directory):
    statuses = []
    for chunk_path in sorted(method_directory.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            statuses.append(chunk["status"])
    return status_counts(np.concatenate(statuses)) if statuses else status_counts([])


def main():
    args = parse_args()
    devices = jax.devices()
    if not args.allow_cpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("DeepRV validation requires a JAX GPU")

    experiment = load_experiment(
        args.data_file, args.samples_file, args.design_file
    )
    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    output_directory = Path(args.output_directory).resolve()
    method_directory = output_directory / "deeprv"
    method_directory.mkdir(parents=True, exist_ok=True)
    run_info_path = method_directory / "run_info.json"
    previous_info = read_json(run_info_path) if run_info_path.exists() else {}
    previous_elapsed = float(previous_info.get("elapsed_seconds", 0.0))

    restore_started = time.perf_counter()
    runtime = get_decoder(checkpoint_path)
    restore_seconds = time.perf_counter() - restore_started
    chunks = make_task_chunks(
        experiment["number_parameters"],
        experiment["number_replicates"],
        target_chunk_size=RESULTS_PER_CHUNK,
    )
    validate_existing_chunks(
        method_directory, experiment["design_sha256"], len(chunks)
    )
    run_info = {
        "method": "deeprv",
        "design_file": experiment["design_path"],
        "design_sha256": experiment["design_sha256"],
        "checkpoint": checkpoint_path,
        "max_locations": runtime.max_locations,
        "number_parameters": experiment["number_parameters"],
        "number_replicates": experiment["number_replicates"],
        "number_years": len(experiment["years"]),
        "number_summaries": 16,
        "number_chunks": len(chunks),
        "results_per_chunk": RESULTS_PER_CHUNK,
        "jitter": JITTER,
        "devices": [str(device) for device in devices],
        "restore_seconds": restore_seconds,
        "elapsed_seconds": previous_elapsed,
    }
    write_json(run_info_path, run_info)

    started = time.perf_counter()
    completed = 0
    for chunk_id, tasks in enumerate(chunks):
        chunk_path = method_directory / f"chunk_{chunk_id:05d}.npz"
        if chunk_path.exists():
            completed += 1
            continue

        results = []
        for parameter_id, replicate_id in tasks:
            result = empty_task_result(
                parameter_id, replicate_id, len(experiment["observed_by_year"])
            )
            simulation_started = time.perf_counter()
            try:
                rng = make_task_rng(
                    experiment["experiment_seed"],
                    METHOD_DEEPRV,
                    parameter_id,
                    replicate_id,
                )
                runtime.key = random.key(
                    make_task_seed(
                        experiment["experiment_seed"],
                        METHOD_DEEPRV,
                        parameter_id,
                        replicate_id,
                    )
                )
                models = selected_models(
                    experiment["samples"],
                    experiment["parameter_indices"][parameter_id],
                )

                def draw(_model_name, locations, model):
                    return runtime.sample(
                        locations,
                        model["intercept"],
                        model["variance"],
                        model["lengthscale"],
                    )

                distances, population_sizes, status, extinction_year = simulate_dataset(
                    models,
                    draw,
                    experiment["initial_population"],
                    experiment["observed_by_year"],
                    experiment["size_edges"],
                    experiment["recruit_probability"],
                    rng,
                )
                result.update(
                    distances=distances,
                    population_sizes=population_sizes,
                    status=status,
                    extinction_year=extinction_year,
                )
            except Exception as error:
                if isinstance(error, ValueError) and "requires at most" in str(error):
                    result["status"] = STATUS_MAX_LOCATIONS
                result["error"] = f"{type(error).__name__}: {error}"
            result["seconds"] = time.perf_counter() - simulation_started
            results.append(result)

        save_chunk(chunk_path, results)
        completed += 1
        print(f"Completed chunks: {completed}/{len(chunks)}", flush=True)

    run_info.update(
        elapsed_seconds=previous_elapsed + time.perf_counter() - started,
        completed_chunks=len(list(method_directory.glob("chunk_*.npz"))),
        status_counts=collect_status_counts(method_directory),
    )
    write_json(run_info_path, run_info)
    print(f"DeepRV results: {method_directory}", flush=True)


if __name__ == "__main__":
    main()
