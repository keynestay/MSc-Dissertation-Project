#!/usr/bin/env python3
# ruff: noqa: E402
"""Benchmark the original age-size GPflow IBM with parallel Ray workers."""

import os

for variable in (
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray

from experiments.soay.age_size.abc.gp import soay_abc_model_agesize as base
from experiments.soay.age_size.benchmarks.benchmark_common import (
    PARAMETER_NAMES,
    combine_outputs,
    load_benchmark,
    output_summary,
    write_json,
)

COLUMN = {name: position for position, name in enumerate(PARAMETER_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("design_file")
    parser.add_argument("output_directory")
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
    )
    parser.add_argument("--tasks-per-batch", type=int, default=25)
    parser.add_argument("--max-locations", type=int, default=512)
    return parser.parse_args()


@ray.remote(num_cpus=1)
class GPflowWorker:
    def __init__(self, data, initial_sizes, initial_ages, number_years, capacity):
        self.models = {
            "m_grw": base.build_new_m_grw(data),
            "m_surv": base.build_new_m_surv(data),
            "m_repr": base.build_new_m_repr(data),
            "m_recr": base.build_new_m_recr(data),
            "m_rcsz": base.build_new_m_rcsz(data),
        }
        self.initial_sizes = np.asarray(initial_sizes, dtype=float)
        self.initial_ages = np.asarray(initial_ages, dtype=np.int32)
        self.number_years = int(number_years)
        self.capacity = int(capacity)

    def ready(self):
        return True

    def assign_parameters(self, values):
        settings = {
            "m_grw": (
                "growth_int",
                "growth_var",
                ("growth_ls_size", "growth_ls_age"),
                "growth_noise",
            ),
            "m_surv": ("surv_int", "surv_var", ("surv_ls_size", "surv_ls_age"), None),
            "m_repr": ("repr_int", "repr_var", ("repr_ls_size", "repr_ls_age"), None),
            "m_recr": ("recr_int", "recr_var", ("recr_ls_age",), None),
            "m_rcsz": ("rcsz_int", "rcsz_var", ("rcsz_ls_size",), "rcsz_noise"),
        }
        for model_name, names in settings.items():
            model = self.models[model_name]
            model.mean_function.c.assign(values[COLUMN[names[0]]])
            model.kernel.variance.assign(values[COLUMN[names[1]]])
            model.kernel.lengthscales.assign(
                np.asarray([values[COLUMN[name]] for name in names[2]])
            )
            if names[3] is not None:
                noise = values[COLUMN[names[3]]]
                model.likelihood.variance.assign(noise**2)

    @staticmethod
    def unique_rows(data, selected, columns):
        return len(data.loc[selected, list(columns)].dropna().drop_duplicates())

    def simulate_one(self, task_id, parameters, seed):
        np.random.seed(int(seed))
        self.assign_parameters(parameters)
        population = pd.DataFrame(
            {"z": self.initial_sizes.copy(), "age": self.initial_ages.copy()}
        )
        population_sizes = np.full(self.number_years, len(population), dtype=np.int32)
        survivor_counts = np.zeros(self.number_years, dtype=np.int32)
        breeder_counts = np.zeros(self.number_years, dtype=np.int32)
        recruit_counts = np.zeros(self.number_years, dtype=np.int32)
        requested_locations = np.zeros(self.number_years, dtype=np.int32)
        unique_requested_locations = np.zeros(self.number_years, dtype=np.int32)
        extinction_year = -1
        overflow = False

        for year in range(self.number_years):
            if len(population) <= 1:
                population_sizes[year:] = len(population)
                break

            simulated = base.simulate_one_year(population, self.models)
            survived = simulated["Surv"] == 1
            eligible = survived & (simulated["age"] >= base.MIN_REPRODUCTIVE_AGE)
            reproduced = simulated["Repr"] == 1
            female = simulated["Sex"] == 1
            recruited = simulated["Recr"] == 1

            survivor_count = int(survived.sum())
            breeder_count = int(reproduced.sum())
            recruit_count = int(recruited.sum())
            requested_locations[year] = (
                len(population)
                + survivor_count
                + int(eligible.sum())
                + int(female.sum())
                + recruit_count
            )
            unique_requested_locations[year] = (
                self.unique_rows(
                    simulated, np.ones(len(simulated), dtype=bool), ("z", "age")
                )
                + self.unique_rows(simulated, survived, ("z", "age"))
                + self.unique_rows(simulated, eligible, ("z", "age"))
                + self.unique_rows(simulated, female, ("age",))
                + self.unique_rows(simulated, recruited, ("z",))
            )
            survivor_counts[year] = survivor_count
            breeder_counts[year] = breeder_count
            recruit_counts[year] = recruit_count

            population = base.population_structure(simulated, year + 1)
            population_sizes[year] = len(population)
            overflow = overflow or len(population) > self.capacity
            if len(population) <= 1:
                extinction_year = year + 1
                if year + 1 < self.number_years:
                    population_sizes[year + 1 :] = len(population)
                break

        final_sizes = population["z"].to_numpy(float)
        return {
            "task_id": np.asarray([task_id], dtype=np.int32),
            "population_sizes": population_sizes[None],
            "survivor_counts": survivor_counts[None],
            "breeder_counts": breeder_counts[None],
            "recruit_counts": recruit_counts[None],
            "requested_locations": requested_locations[None],
            "unique_requested_locations": unique_requested_locations[None],
            "final_population_size": np.asarray([len(population)], dtype=np.int32),
            "final_size_total": np.asarray([np.sum(final_sizes)], dtype=float),
            "extinction_year": np.asarray([extinction_year], dtype=np.int32),
            "overflow": np.asarray([overflow], dtype=bool),
        }

    def simulate_batch(self, task_ids, parameters, seeds):
        results = [
            self.simulate_one(task_id, values, seed)
            for task_id, values, seed in zip(task_ids, parameters, seeds)
        ]
        return combine_outputs(results)


def main():
    args = parse_args()
    if args.num_cpus <= 0:
        raise ValueError("num_cpus must be positive")
    if args.tasks_per_batch <= 0:
        raise ValueError("tasks_per_batch must be positive")
    if args.max_locations <= 0:
        raise ValueError("max_locations must be positive")

    total_started = time.perf_counter()
    benchmark = load_benchmark(args.data_file, args.design_file)
    output_directory = Path(args.output_directory).resolve() / "gpflow"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / "simulation_outputs.npz"
    run_info_file = output_directory / "run_info.json"
    if output_file.exists() or run_info_file.exists():
        raise FileExistsError(f"GPflow output already exists in {output_directory}")

    print(f"Starting Ray with {args.num_cpus} CPU workers", flush=True)
    ray_started = time.perf_counter()
    ray.init(
        num_cpus=args.num_cpus,
        include_dashboard=False,
        log_to_driver=False,
        _temp_dir=f"/tmp/ray-{os.environ.get('SLURM_JOB_ID', os.getpid())}",
    )
    ray_start_seconds = time.perf_counter() - ray_started
    try:
        worker_started = time.perf_counter()
        workers = [
            GPflowWorker.remote(
                benchmark["data"],
                benchmark["initial_sizes"],
                benchmark["initial_ages"],
                benchmark["number_years"],
                args.max_locations,
            )
            for _ in range(args.num_cpus)
        ]
        ray.get([worker.ready.remote() for worker in workers])
        worker_initialization_seconds = time.perf_counter() - worker_started
        print(
            f"GPflow workers ready: {worker_initialization_seconds:.3f} seconds",
            flush=True,
        )

        warmup_started = time.perf_counter()
        ray.get(
            [
                worker.simulate_batch.remote(
                    np.asarray([0], dtype=np.int32),
                    benchmark["parameter_values"][:1],
                    benchmark["task_seeds"][:1],
                )
                for worker in workers
            ]
        )
        warmup_seconds = time.perf_counter() - warmup_started
        print(f"Worker warm-up: {warmup_seconds:.3f} seconds", flush=True)

        chunks = [
            np.arange(
                start,
                min(start + args.tasks_per_batch, benchmark["number_datasets"]),
                dtype=np.int32,
            )
            for start in range(0, benchmark["number_datasets"], args.tasks_per_batch)
        ]
        print(
            f"Simulating {benchmark['number_datasets']} datasets in parallel",
            flush=True,
        )
        simulation_started = time.perf_counter()
        futures = []
        for chunk_number, task_ids in enumerate(chunks):
            futures.append(
                workers[chunk_number % len(workers)].simulate_batch.remote(
                    task_ids,
                    benchmark["parameter_values"][task_ids],
                    benchmark["task_seeds"][task_ids],
                )
            )
        outputs = combine_outputs(ray.get(futures))
        simulation_seconds = time.perf_counter() - simulation_started
    finally:
        ray.shutdown()

    save_started = time.perf_counter()
    np.savez_compressed(output_file, **outputs)
    save_seconds = time.perf_counter() - save_started
    total_seconds = time.perf_counter() - total_started
    run_info = {
        "method": "gpflow_cpu_prior_age_size_ibm",
        "design_file": benchmark["design_path"],
        "design_sha256": benchmark["design_sha256"],
        "number_datasets": benchmark["number_datasets"],
        "number_years": benchmark["number_years"],
        "initial_population_size": len(benchmark["initial_sizes"]),
        "num_cpus": args.num_cpus,
        "tasks_per_batch": args.tasks_per_batch,
        "max_locations_diagnostic": args.max_locations,
        "gp_sampling": "gpflow_tensorflow_ard_cholesky_prior",
        "ray_start_seconds": ray_start_seconds,
        "worker_initialization_seconds": worker_initialization_seconds,
        "warmup_seconds": warmup_seconds,
        "simulation_seconds": simulation_seconds,
        "datasets_per_second": benchmark["number_datasets"] / simulation_seconds,
        "seconds_per_dataset": simulation_seconds / benchmark["number_datasets"],
        "output_save_seconds": save_seconds,
        "total_seconds": total_seconds,
        "output_file": str(output_file),
        **output_summary(outputs),
    }
    write_json(run_info_file, run_info)
    print(f"GPflow outputs: {output_file}", flush=True)
    print(f"GPflow run information: {run_info_file}", flush=True)
    print(
        f"Simulation: {simulation_seconds:.3f} seconds, {run_info['datasets_per_second']:.3f} datasets/second",
        flush=True,
    )


if __name__ == "__main__":
    main()
