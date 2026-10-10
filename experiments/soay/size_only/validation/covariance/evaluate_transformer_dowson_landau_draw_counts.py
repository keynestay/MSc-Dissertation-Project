"""Evaluate DeepRV Dowson--Landau distances at several nested draw counts."""

from experiments.paths import GENERATED, PROJECT_ROOT

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np
import pandas as pd

REPO_ROOT = PROJECT_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.soay.size_only.training.deeprv_decoder import JITTER, DeepRVDecoder  # noqa: E402
from experiments.soay.size_only.validation.covariance.evaluate_transformer_dowson_landau import (  # noqa: E402
    METRIC_COLUMNS,
    VITAL_RATES,
    generate_deeprv_draws,
    load_inputs,
    open_saved_array,
    psd_square_root,
    symmetric_psd,
    write_json,
)


def prepare_target(target_covariance):
    target, target_square_root, _, minimum_target_eigenvalue = psd_square_root(
        target_covariance, "Target covariance"
    )
    return {
        "square_root": target_square_root,
        "trace": float(np.trace(target)),
        "minimum_eigenvalue": minimum_target_eigenvalue,
    }


def dowson_landau_metrics(draws, target):
    draws = np.asarray(draws, dtype=np.float64)
    empirical_mean = draws.mean(axis=0)
    centered = draws - empirical_mean
    empirical_covariance = centered.T @ centered / (len(draws) - 1)
    empirical_covariance, empirical_eigenvalues, _, minimum_empirical_eigenvalue = symmetric_psd(
        empirical_covariance, "Empirical covariance"
    )
    middle = target["square_root"] @ empirical_covariance @ target["square_root"]
    _, middle_square_root, _, minimum_cross_eigenvalue = psd_square_root(
        middle, "Covariance square-root argument"
    )

    empirical_trace = float(np.trace(empirical_covariance))
    cross_trace = float(np.trace(middle_square_root))
    mean_squared = float(empirical_mean @ empirical_mean)
    covariance_squared_raw = empirical_trace + target["trace"] - 2 * cross_trace
    tolerance = 1e-8 * max(1.0, empirical_trace + target["trace"])
    if covariance_squared_raw < -tolerance:
        raise FloatingPointError(f"Dowson--Landau covariance term is negative: {covariance_squared_raw:.3e}")
    covariance_squared = max(0.0, covariance_squared_raw)
    total_squared = mean_squared + covariance_squared
    rank_tolerance = np.finfo(np.float64).eps * len(empirical_eigenvalues) * max(1.0, empirical_eigenvalues[-1])

    return np.array([
        mean_squared,
        covariance_squared,
        covariance_squared_raw,
        total_squared,
        np.sqrt(total_squared),
        empirical_trace,
        target["trace"],
        cross_trace,
        np.count_nonzero(empirical_eigenvalues > rank_tolerance),
        minimum_empirical_eigenvalue,
        target["minimum_eigenvalue"],
        minimum_cross_eigenvalue,
        np.max(np.abs(empirical_mean)),
    ], dtype=np.float64)


def prepare_storage(name, num_parameters, num_counts, output_directory, resume):
    slug = VITAL_RATES[name]["filename"]
    intermediate_directory = output_directory / "intermediates"
    progress_directory = output_directory / "progress"
    intermediate_directory.mkdir(parents=True, exist_ok=True)
    progress_directory.mkdir(parents=True, exist_ok=True)
    return {
        "metrics": open_saved_array(
            intermediate_directory / f"{slug}_scalar_metrics_by_draw_count.npy",
            (num_parameters, num_counts, len(METRIC_COLUMNS)),
            np.float64,
            resume,
            np.nan,
        ),
        "completed": open_saved_array(
            progress_directory / f"{slug}_completed.npy",
            (num_parameters,),
            np.bool_,
            resume,
            False,
        ),
    }


def flush_arrays(arrays):
    for array in arrays.values():
        array.flush()


def parameter_frame(name, parameters, num_locations, realization_counts, arrays):
    indices = np.flatnonzero(np.asarray(arrays["completed"]))
    frames = []
    for count_index, realizations in enumerate(realization_counts):
        frame = pd.DataFrame({
            "Vital Rate": name,
            "Parameter Draw": indices,
            "Variance": parameters["variance"][indices],
            "Lengthscale": parameters["lengthscale"][indices],
            "Number Of Locations": num_locations,
            "Realizations": realizations,
        })
        for metric_index, metric_name in enumerate(METRIC_COLUMNS):
            frame[metric_name] = np.asarray(arrays["metrics"])[indices, count_index, metric_index]
        target_trace = frame["Target Covariance Trace"]
        frame["Relative D Mean Squared"] = frame["D Mean Squared"] / target_trace
        frame["Relative D Covariance Squared"] = frame["D Covariance Squared"] / target_trace
        frame["Relative D DL Squared"] = frame["D DL Squared"] / target_trace
        frame["Relative D DL"] = np.sqrt(frame["Relative D DL Squared"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_rate_results(name, parameters, num_locations, realization_counts, arrays, output_directory):
    frame = parameter_frame(name, parameters, num_locations, realization_counts, arrays)
    filename = VITAL_RATES[name]["filename"]
    frame.to_csv(output_directory / "results" / f"dowson_landau_draw_counts_{filename}.csv", index=False)
    return frame


def create_summary(results):
    rows = []
    grouped = results.groupby(["Vital Rate", "Realizations"], sort=False)
    for (name, realizations), current in grouped:
        row = {
            "Vital Rate": name,
            "Realizations": int(realizations),
            "Parameter Draws": len(current),
            "Number Of Locations": int(current["Number Of Locations"].iloc[0]),
        }
        for metric in ("Relative D Mean Squared", "Relative D Covariance Squared", "Relative D DL Squared"):
            row[f"Mean {metric}"] = current[metric].mean()
            row[f"Median {metric}"] = current[metric].median()
            row[f"{metric} 5th Percentile"] = current[metric].quantile(0.05)
            row[f"{metric} 95th Percentile"] = current[metric].quantile(0.95)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_vital_rate(name, locations, parameters, decoder, args, output_directory, rate_index):
    num_parameters = len(parameters["variance"])
    num_locations = len(locations)
    if num_locations > decoder.max_locations:
        raise ValueError(f"{name} has {num_locations} locations but the decoder supports {decoder.max_locations}")

    arrays = prepare_storage(name, num_parameters, len(args.realization_counts), output_directory, args.resume)
    slug = VITAL_RATES[name]["filename"]
    np.save(output_directory / "intermediates" / f"{slug}_locations.npy", locations)
    np.savez(
        output_directory / "intermediates" / f"{slug}_parameters.npz",
        variance=parameters["variance"],
        lengthscale=parameters["lengthscale"],
    )

    squared_distances = (locations[:, None] - locations[None, :]) ** 2
    identity = np.eye(num_locations, dtype=np.float64)
    maximum_realizations = max(args.realization_counts)
    started = time.perf_counter()

    for parameter_index, (variance, lengthscale) in enumerate(zip(parameters["variance"], parameters["lengthscale"])):
        if arrays["completed"][parameter_index]:
            continue

        rng = np.random.default_rng(np.random.SeedSequence([args.seed, rate_index, parameter_index]))
        _, deeprv_draws = generate_deeprv_draws(
            decoder,
            locations,
            variance,
            lengthscale,
            maximum_realizations,
            args.decoder_batch_size,
            rng,
        )
        target_covariance = variance * np.exp(-0.5 * squared_distances / lengthscale**2) + JITTER * identity
        target = prepare_target(target_covariance)

        for count_index, realizations in enumerate(args.realization_counts):
            arrays["metrics"][parameter_index, count_index] = dowson_landau_metrics(
                deeprv_draws[:realizations], target
            )

        arrays["completed"][parameter_index] = True
        completed = parameter_index + 1
        if completed % args.save_every == 0 or completed == num_parameters:
            flush_arrays(arrays)
            save_rate_results(name, parameters, num_locations, args.realization_counts, arrays, output_directory)
            elapsed_hours = (time.perf_counter() - started) / 3600
            print(f"{name}: {completed:,}/{num_parameters:,} Parameter Draws ({elapsed_hours:.2f} Hours)", flush=True)

    flush_arrays(arrays)
    return save_rate_results(name, parameters, num_locations, args.realization_counts, arrays, output_directory)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=1)
    parser.add_argument("--realization-counts", type=int, nargs="+", default=[50, 100, 250, 500])
    parser.add_argument("--decoder-batch-size", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.realization_counts = sorted(set(args.realization_counts))
    if not args.realization_counts or min(args.realization_counts) < 2:
        raise ValueError("realization-counts must contain values of at least 2")
    if args.decoder_batch_size < 1 or args.save_every < 1:
        raise ValueError("decoder-batch-size and save-every must be positive")

    output_directory = args.output_dir.resolve()
    (output_directory / "results").mkdir(parents=True, exist_ok=True)
    metadata_path = output_directory / "run_metadata.json"
    if metadata_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite an existing run in {output_directory}")
    if metadata_path.exists() and args.resume:
        previous = json.loads(metadata_path.read_text())
        if previous.get("realization_counts") != args.realization_counts or previous.get("year") != args.year:
            raise ValueError("The realization counts or year do not match the existing run")

    locations, parameter_draws = load_inputs(args.data.resolve(), args.samples.resolve(), args.year)
    decoder = DeepRVDecoder(args.checkpoint.resolve())
    started = time.perf_counter()
    metadata = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "samples": str(args.samples.resolve()),
        "output_directory": str(output_directory),
        "year": args.year,
        "parameter_draws_per_vital_rate": 5000,
        "realization_counts": args.realization_counts,
        "maximum_realizations_generated_per_parameter_draw": max(args.realization_counts),
        "nested_realizations": True,
        "decoder_batch_size": args.decoder_batch_size,
        "save_every": args.save_every,
        "seed": args.seed,
        "jitter": JITTER,
        "maximum_decoder_locations": decoder.max_locations,
        "location_counts": {name: len(values) for name, values in locations.items()},
        "metric_columns": METRIC_COLUMNS,
        "target": "Zero-mean RBF GP with posterior variance and lengthscale",
        "exact_gp_samples_generated": False,
        "raw_deeprv_draws_saved": False,
        "latent_vectors_saved": False,
        "jax_devices": [str(device) for device in jax.devices()],
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(metadata_path, metadata)
    print("JAX Devices:", jax.devices(), flush=True)
    print("Output Directory:", output_directory, flush=True)
    print("Realization Counts:", args.realization_counts, flush=True)
    print("Location Counts:", metadata["location_counts"], flush=True)

    rate_frames = []
    for rate_index, name in enumerate(VITAL_RATES):
        print(f"\nEvaluating {name} At {len(locations[name])} First-Year Locations", flush=True)
        frame = evaluate_vital_rate(
            name,
            locations[name],
            parameter_draws[name],
            decoder,
            args,
            output_directory,
            rate_index,
        )
        rate_frames.append(frame)
        combined = pd.concat(rate_frames, ignore_index=True)
        combined.to_csv(output_directory / "results" / "dowson_landau_by_draw_count.csv", index=False)
        create_summary(combined).to_csv(
            output_directory / "results" / "dowson_landau_draw_count_summary.csv", index=False
        )

    results = pd.concat(rate_frames, ignore_index=True)
    summary = create_summary(results)
    results.to_csv(output_directory / "results" / "dowson_landau_by_draw_count.csv", index=False)
    summary.to_csv(output_directory / "results" / "dowson_landau_draw_count_summary.csv", index=False)
    metadata.update({
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "result_rows": len(results),
    })
    write_json(metadata_path, metadata)
    print("\nEvaluation Complete", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
