#!/usr/bin/env python3
"""Benchmark the JAX-only age-size DeepRV IBM on one GPU."""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.soay.age_size.abc.deeprv.deeprv_decoder_agesize import get_runtime
from experiments.soay.age_size.benchmarks.benchmark_common import (
    combine_outputs,
    load_benchmark,
    output_summary,
    write_json,
)
from experiments.soay.age_size.benchmarks.deeprv_ibm_jax import (
    make_deeprv_simulator,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("design_file")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def make_batch(benchmark, start, batch_size):
    stop = min(start + batch_size, benchmark["number_datasets"])
    count = stop - start
    source = slice(start, stop)
    parameters = np.empty(
        (batch_size, benchmark["parameter_values"].shape[1]), dtype=np.float32
    )
    seeds = np.empty(batch_size, dtype=np.uint32)
    task_ids = np.full(batch_size, -1, dtype=np.int32)
    valid = np.zeros(batch_size, dtype=bool)
    parameters[:count] = benchmark["parameter_values"][source]
    seeds[:count] = benchmark["task_seeds"][source]
    task_ids[:count] = np.arange(start, stop, dtype=np.int32)
    valid[:count] = True
    if count < batch_size:
        parameters[count:] = parameters[0]
        seeds[count:] = seeds[0]
    return parameters, seeds, task_ids, valid, count


def block(outputs):
    return jax.tree.map(lambda value: value.block_until_ready(), outputs)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    devices = jax.devices()
    if not args.allow_cpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("The DeepRV speed benchmark requires a JAX GPU")
    print(f"JAX devices: {devices}", flush=True)

    total_started = time.perf_counter()
    benchmark = load_benchmark(args.data_file, args.design_file)
    output_directory = Path(args.output_directory).resolve() / "deeprv"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / "simulation_outputs.npz"
    run_info_file = output_directory / "run_info.json"
    if output_file.exists() or run_info_file.exists():
        raise FileExistsError(f"DeepRV output already exists in {output_directory}")

    restore_started = time.perf_counter()
    runtime = get_runtime(args.checkpoint_path)
    restore_seconds = time.perf_counter() - restore_started
    if len(benchmark["initial_sizes"]) > runtime.max_locations:
        raise ValueError("The initial population exceeds the checkpoint capacity")

    initial_sizes = np.full(runtime.max_locations, np.nan, dtype=np.float32)
    initial_ages = np.zeros(runtime.max_locations, dtype=np.int32)
    initial_count = len(benchmark["initial_sizes"])
    initial_sizes[:initial_count] = benchmark["initial_sizes"]
    initial_ages[:initial_count] = benchmark["initial_ages"]
    initial_sizes_device = jnp.asarray(initial_sizes)
    initial_ages_device = jnp.asarray(initial_ages)
    simulate = make_deeprv_simulator(runtime, benchmark["number_years"])

    warmup_batch = make_batch(benchmark, 0, args.batch_size)
    print(
        f"Compiling age-size DeepRV IBM with batch size {args.batch_size}", flush=True
    )
    compile_started = time.perf_counter()
    warmup_outputs = simulate(
        jnp.asarray(warmup_batch[0]),
        jnp.asarray(warmup_batch[1]),
        jnp.asarray(warmup_batch[2]),
        jnp.asarray(warmup_batch[3]),
        initial_sizes_device,
        initial_ages_device,
    )
    block(warmup_outputs)
    compile_seconds = time.perf_counter() - compile_started
    print(f"Compilation and warm-up: {compile_seconds:.3f} seconds", flush=True)

    parts = []
    print(
        f"Simulating {benchmark['number_datasets']} {benchmark['number_years']}-year datasets",
        flush=True,
    )
    simulation_started = time.perf_counter()
    for start in range(0, benchmark["number_datasets"], args.batch_size):
        parameters, seeds, task_ids, valid, count = make_batch(
            benchmark, start, args.batch_size
        )
        outputs = simulate(
            jnp.asarray(parameters),
            jnp.asarray(seeds),
            jnp.asarray(task_ids),
            jnp.asarray(valid),
            initial_sizes_device,
            initial_ages_device,
        )
        block(outputs)
        parts.append(
            {name: np.asarray(value)[:count] for name, value in outputs.items()}
        )
    simulation_seconds = time.perf_counter() - simulation_started
    outputs = combine_outputs(parts)

    save_started = time.perf_counter()
    np.savez_compressed(output_file, **outputs)
    save_seconds = time.perf_counter() - save_started
    total_seconds = time.perf_counter() - total_started
    run_info = {
        "method": "deeprv_gpu_jax_age_size_ibm",
        "design_file": benchmark["design_path"],
        "design_sha256": benchmark["design_sha256"],
        "checkpoint": str(Path(args.checkpoint_path).resolve()),
        "coordinate_order": ["size", "age"],
        "number_datasets": benchmark["number_datasets"],
        "number_years": benchmark["number_years"],
        "initial_population_size": initial_count,
        "max_locations": runtime.max_locations,
        "batch_size": args.batch_size,
        "devices": [str(device) for device in devices],
        "two_dimensional_sampler": "transformer_deeprv_batched",
        "recruitment_sampler": "jax_exact_rbf_age_grid_cholesky",
        "recruit_size_sampler": "jax_exact_rbf_bucketed_cholesky",
        "checkpoint_restore_seconds": restore_seconds,
        "compile_seconds": compile_seconds,
        "simulation_seconds": simulation_seconds,
        "datasets_per_second": benchmark["number_datasets"] / simulation_seconds,
        "seconds_per_dataset": simulation_seconds / benchmark["number_datasets"],
        "output_save_seconds": save_seconds,
        "total_seconds": total_seconds,
        "output_file": str(output_file),
        **output_summary(outputs),
    }
    write_json(run_info_file, run_info)
    print(f"DeepRV outputs: {output_file}", flush=True)
    print(f"DeepRV run information: {run_info_file}", flush=True)
    print(
        f"Simulation: {simulation_seconds:.3f} seconds, {run_info['datasets_per_second']:.3f} datasets/second",
        flush=True,
    )


if __name__ == "__main__":
    main()
