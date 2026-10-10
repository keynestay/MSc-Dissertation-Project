#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the age-size exact prior-GP reference with Ray CPU workers."""

from experiments.paths import GENERATED, PROJECT_ROOT

import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import ray

PROJECT_DIRECTORY = PROJECT_ROOT
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from experiments.soay.age_size.validation.ibm.comparison_common import (
    JITTER,
    METHOD_EXACT_GP,
    SUMMARY_NAMES,
    empty_task_result,
    load_experiment,
    make_task_chunks,
    make_task_rng,
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
    parser.add_argument("output_directory")
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
    )
    return parser.parse_args()


@ray.remote(num_cpus=1)
def simulate_exact_chunk(tasks, experiment):
    results = []
    number_years = len(experiment["years"])
    for parameter_id, replicate_id in tasks:
        result = empty_task_result(parameter_id, replicate_id, number_years)
        started = time.perf_counter()
        try:
            rng = make_task_rng(
                experiment["experiment_seed"],
                METHOD_EXACT_GP,
                parameter_id,
                replicate_id,
            )
            models = selected_models(
                experiment["samples"], experiment["parameter_indices"][parameter_id]
            )
            distances, population_sizes, status, extinction_year = simulate_dataset(
                models,
                experiment["initial_population"],
                experiment["observed_components"],
                experiment["size_edges"],
                rng,
            )
            result.update(
                distances=distances,
                population_sizes=population_sizes,
                status=status,
                extinction_year=extinction_year,
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        result["seconds"] = time.perf_counter() - started
        results.append(result)
    return results


def collect_status_counts(method_directory):
    statuses = []
    for chunk_path in sorted(method_directory.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            statuses.append(chunk["status"])
    return status_counts(np.concatenate(statuses)) if statuses else status_counts([])


def main():
    args = parse_args()
    if args.num_cpus <= 0:
        raise ValueError("num-cpus must be positive")
    experiment = load_experiment(args.data_file, args.samples_file, args.design_file)
    method_directory = Path(args.output_directory).resolve() / "exact_gp"
    method_directory.mkdir(parents=True, exist_ok=True)
    run_info_path = method_directory / "run_info.json"
    previous_info = read_json(run_info_path) if run_info_path.exists() else {}
    previous_elapsed = float(previous_info.get("elapsed_seconds", 0.0))

    chunks = make_task_chunks(
        experiment["number_parameters"],
        experiment["number_replicates"],
        RESULTS_PER_CHUNK,
        args.num_cpus * 4,
    )
    validate_existing_chunks(method_directory, experiment["design_sha256"], len(chunks))
    pending = [
        (chunk_id, chunk)
        for chunk_id, chunk in enumerate(chunks)
        if not (method_directory / f"chunk_{chunk_id:05d}.npz").exists()
    ]
    run_info = {
        "method": "exact_gp_cpu",
        "design_file": experiment["design_path"],
        "design_sha256": experiment["design_sha256"],
        "number_parameters": experiment["number_parameters"],
        "number_replicates": experiment["number_replicates"],
        "number_years": len(experiment["years"]),
        "number_summaries": len(SUMMARY_NAMES),
        "summary_names": list(SUMMARY_NAMES),
        "num_cpus": args.num_cpus,
        "number_chunks": len(chunks),
        "results_per_chunk": RESULTS_PER_CHUNK,
        "jitter": JITTER,
        "two_dimensional_sampler": "numpy_ard_cholesky",
        "one_dimensional_sampler": "numpy_rbf_cholesky",
        "elapsed_seconds": previous_elapsed,
    }
    write_json(run_info_path, run_info)

    started = time.perf_counter()
    if pending:
        ray.init(
            num_cpus=args.num_cpus,
            include_dashboard=False,
            log_to_driver=False,
            _temp_dir=f"/tmp/ray-{os.environ.get('SLURM_JOB_ID', os.getpid())}",
        )
        try:
            experiment_reference = ray.put(experiment)
            futures = {
                simulate_exact_chunk.remote(chunk, experiment_reference): chunk_id
                for chunk_id, chunk in pending
            }
            completed = len(chunks) - len(pending)
            while futures:
                ready, _ = ray.wait(list(futures), num_returns=1)
                future = ready[0]
                chunk_id = futures.pop(future)
                save_chunk(
                    method_directory / f"chunk_{chunk_id:05d}.npz", ray.get(future)
                )
                completed += 1
                print(f"Completed Chunks: {completed}/{len(chunks)}", flush=True)
        finally:
            ray.shutdown()

    run_info.update(
        elapsed_seconds=previous_elapsed + time.perf_counter() - started,
        completed_chunks=len(list(method_directory.glob("chunk_*.npz"))),
        status_counts=collect_status_counts(method_directory),
    )
    write_json(run_info_path, run_info)
    print(f"Exact-GP Results: {method_directory}", flush=True)


if __name__ == "__main__":
    main()
