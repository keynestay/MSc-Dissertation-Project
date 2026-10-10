#!/usr/bin/env python3
# ruff: noqa: E402
"""Run restartable, fully batched JAX DeepRV age-size validation on one GPU."""

from experiments.paths import GENERATED, PROJECT_ROOT

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_DIRECTORY = PROJECT_ROOT
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from experiments.soay.age_size.abc.deeprv.deeprv_decoder_agesize import get_runtime
from experiments.soay.age_size.validation.ibm.comparison_common import (
    JITTER,
    METHOD_DEEPRV,
    SUMMARY_NAMES,
    empty_task_result,
    load_experiment,
    make_task_chunks,
    make_task_seed,
    path_sha256,
    read_json,
    save_chunk,
    status_counts,
    validate_existing_chunks,
    write_json,
)
from experiments.soay.age_size.validation.ibm.deeprv_ibm_jax import make_deeprv_simulator

RESULTS_PER_CHUNK = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("samples_file")
    parser.add_argument("design_file")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def make_batch(experiment, tasks, start, batch_size):
    selected_tasks = tasks[start : start + batch_size]
    count = len(selected_tasks)
    parameter_ids = selected_tasks[:, 0].astype(np.int64)
    replicate_ids = selected_tasks[:, 1].astype(np.int64)
    parameters = np.empty(
        (batch_size, experiment["parameter_values"].shape[1]), dtype=np.float32
    )
    seeds = np.zeros(batch_size, dtype=np.uint32)
    valid = np.zeros(batch_size, dtype=bool)
    parameters[:count] = experiment["parameter_values"][parameter_ids]
    for row, (parameter_id, replicate_id) in enumerate(selected_tasks):
        seeds[row] = make_task_seed(
            experiment["experiment_seed"], METHOD_DEEPRV, parameter_id, replicate_id
        )
    valid[:count] = True
    if count < batch_size:
        parameters[count:] = parameters[0]
        seeds[count:] = seeds[0]
    return parameters, seeds, valid, parameter_ids, replicate_ids, count


def block(outputs):
    return jax.tree.map(lambda value: value.block_until_ready(), outputs)


def collect_status_counts(method_directory):
    statuses = []
    for chunk_path in sorted(method_directory.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            statuses.append(chunk["status"])
    return status_counts(np.concatenate(statuses)) if statuses else status_counts([])


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    devices = jax.devices()
    if not args.allow_cpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("DeepRV validation requires a JAX GPU")
    print(f"JAX Devices: {devices}", flush=True)

    experiment = load_experiment(args.data_file, args.samples_file, args.design_file)
    checkpoint_path = Path(args.checkpoint_path).resolve()
    checkpoint_hash = path_sha256(checkpoint_path)
    method_directory = Path(args.output_directory).resolve() / "deeprv"
    method_directory.mkdir(parents=True, exist_ok=True)
    run_info_path = method_directory / "run_info.json"
    previous_info = read_json(run_info_path) if run_info_path.exists() else {}
    previous_elapsed = float(previous_info.get("elapsed_seconds", 0.0))

    restore_started = time.perf_counter()
    runtime = get_runtime(checkpoint_path)
    restore_seconds = time.perf_counter() - restore_started
    simulator = make_deeprv_simulator(
        runtime,
        len(experiment["years"]),
        experiment["initial_population"],
        experiment["observed_components"],
        experiment["size_edges"],
    )
    chunks = make_task_chunks(
        experiment["number_parameters"],
        experiment["number_replicates"],
        RESULTS_PER_CHUNK,
    )
    validate_existing_chunks(
        method_directory, experiment["design_sha256"], len(chunks), checkpoint_hash
    )
    if (
        list(method_directory.glob("chunk_*.npz"))
        and previous_info.get("batch_size") != args.batch_size
    ):
        raise ValueError("Existing chunks were generated with another batch size")

    first_tasks = next(
        (
            chunk
            for chunk_id, chunk in enumerate(chunks)
            if not (method_directory / f"chunk_{chunk_id:05d}.npz").exists()
        ),
        None,
    )
    compile_seconds = float(previous_info.get("compile_seconds", 0.0))
    if first_tasks is not None:
        warmup = make_batch(experiment, first_tasks, 0, args.batch_size)
        print(f"Compiling JAX IBM With Batch Size {args.batch_size}", flush=True)
        compile_started = time.perf_counter()
        block(
            simulator(
                jnp.asarray(warmup[0]), jnp.asarray(warmup[1]), jnp.asarray(warmup[2])
            )
        )
        compile_seconds = time.perf_counter() - compile_started
        print(f"Compilation And Warm-Up: {compile_seconds:.3f} Seconds", flush=True)

    run_info = {
        "method": "deeprv_gpu_jax_ibm",
        "design_file": experiment["design_path"],
        "design_sha256": experiment["design_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "coordinate_order": ["size", "age"],
        "max_locations": runtime.max_locations,
        "number_parameters": experiment["number_parameters"],
        "number_replicates": experiment["number_replicates"],
        "number_years": len(experiment["years"]),
        "number_summaries": len(SUMMARY_NAMES),
        "summary_names": list(SUMMARY_NAMES),
        "number_chunks": len(chunks),
        "results_per_chunk": RESULTS_PER_CHUNK,
        "batch_size": args.batch_size,
        "jitter": JITTER,
        "two_dimensional_sampler": "transformer_deeprv_batched",
        "recruitment_sampler": "jax_exact_rbf_age_grid_cholesky",
        "recruit_size_sampler": "jax_exact_rbf_bucketed_cholesky",
        "random_streams": "independent_method_specific",
        "devices": [str(device) for device in devices],
        "checkpoint_restore_seconds": restore_seconds,
        "compile_seconds": compile_seconds,
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
        for start in range(0, len(tasks), args.batch_size):
            parameters, seeds, valid, parameter_ids, replicate_ids, count = make_batch(
                experiment, tasks, start, args.batch_size
            )
            batch_started = time.perf_counter()
            outputs = block(
                simulator(
                    jnp.asarray(parameters), jnp.asarray(seeds), jnp.asarray(valid)
                )
            )
            batch_seconds = time.perf_counter() - batch_started
            outputs = {
                name: np.asarray(value)[:count] for name, value in outputs.items()
            }
            for row in range(count):
                result = empty_task_result(
                    parameter_ids[row], replicate_ids[row], len(experiment["years"])
                )
                result.update(
                    distances=outputs["distances"][row],
                    population_sizes=outputs["population_sizes"][row],
                    status=int(outputs["status"][row]),
                    extinction_year=int(outputs["extinction_year"][row]),
                    seconds=batch_seconds / count,
                )
                results.append(result)
        save_chunk(chunk_path, results)
        completed += 1
        print(f"Completed Chunks: {completed}/{len(chunks)}", flush=True)

    elapsed = time.perf_counter() - started
    run_info.update(
        elapsed_seconds=previous_elapsed + elapsed,
        completed_chunks=len(list(method_directory.glob("chunk_*.npz"))),
        status_counts=collect_status_counts(method_directory),
    )
    write_json(run_info_path, run_info)
    print(f"DeepRV Results: {method_directory}", flush=True)


if __name__ == "__main__":
    main()
