"""Evaluate DeepRV GP draws with the squared Dowson--Landau distance."""

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
import jax.numpy as jnp
import numpy as np
import pandas as pd

REPO_ROOT = PROJECT_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.soay.size_only.training.deeprv_decoder import DeepRVDecoder, JITTER  # noqa: E402, I001


VITAL_RATES = {
    "Survival": {"prefix": "surv", "observation": "Surv", "filename": "survival"},
    "Growth": {"prefix": "growth", "observation": "z1", "filename": "growth"},
    "Reproduction": {"prefix": "repr", "observation": "Repr", "filename": "reproduction"},
    "Recruit Size": {"prefix": "rcsz", "observation": "Rcsz", "filename": "recruit_size"},
}

METRIC_COLUMNS = [
    "D Mean Squared",
    "D Covariance Squared",
    "D Covariance Squared Raw",
    "D DL Squared",
    "D DL",
    "Empirical Covariance Trace",
    "Target Covariance Trace",
    "Cross Square Root Trace",
    "Empirical Covariance Rank",
    "Minimum Raw Empirical Eigenvalue",
    "Minimum Raw Target Eigenvalue",
    "Minimum Raw Cross Eigenvalue",
    "Maximum Absolute Empirical Mean",
]


def load_inputs(data_path, samples_path, year):
    data = pd.read_csv(data_path)
    year_mask = data["yr"].eq(year)
    locations = {
        name: data.loc[year_mask & data[settings["observation"]].notna(), "z"].dropna().to_numpy(dtype=np.float64)
        for name, settings in VITAL_RATES.items()
    }

    parameter_draws = {}
    with np.load(samples_path) as saved:
        for name, settings in VITAL_RATES.items():
            prefix = settings["prefix"]
            parameter_draws[name] = {
                "variance": np.asarray(saved[f"{prefix}_var"], dtype=np.float64).reshape(-1),
                "lengthscale": np.asarray(saved[f"{prefix}_ls"], dtype=np.float64).reshape(-1),
            }

    for name, draws in parameter_draws.items():
        lengths = {len(values) for values in draws.values()}
        if lengths != {5000}:
            raise ValueError(f"{name} must have exactly 5,000 parameter draws, found {sorted(lengths)}")
        if not all(np.all(np.isfinite(values)) for values in draws.values()):
            raise ValueError(f"{name} contains non-finite GP parameters")
        if np.any(draws["variance"] <= 0) or np.any(draws["lengthscale"] <= 0):
            raise ValueError(f"{name} contains a non-positive variance or lengthscale")
        if not len(locations[name]):
            raise ValueError(f"{name} has no locations in year {year}")

    return locations, parameter_draws


def symmetric_psd(matrix, name):
    symmetric = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T) / 2
    raw_eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(raw_eigenvalues))))
    if raw_eigenvalues[0] < -1e-8 * scale:
        raise FloatingPointError(f"{name} has a materially negative eigenvalue: {raw_eigenvalues[0]:.3e}")
    eigenvalues = np.clip(raw_eigenvalues, 0.0, None)
    projected = (eigenvectors * eigenvalues) @ eigenvectors.T
    projected = (projected + projected.T) / 2
    return projected, eigenvalues, eigenvectors, float(raw_eigenvalues[0])


def psd_square_root(matrix, name):
    projected, eigenvalues, eigenvectors, minimum_raw_eigenvalue = symmetric_psd(matrix, name)
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    square_root = (square_root + square_root.T) / 2
    return projected, square_root, eigenvalues, minimum_raw_eigenvalue


def dowson_landau_components(draws, target_covariance):
    draws = np.asarray(draws, dtype=np.float64)
    empirical_mean = draws.mean(axis=0)
    centered = draws - empirical_mean
    empirical_covariance = centered.T @ centered / (len(draws) - 1)
    empirical_covariance, empirical_eigenvalues, _, minimum_empirical_eigenvalue = symmetric_psd(
        empirical_covariance, "Empirical covariance"
    )
    target_covariance, target_square_root, target_eigenvalues, minimum_target_eigenvalue = psd_square_root(
        target_covariance, "Target covariance"
    )
    middle = target_square_root @ empirical_covariance @ target_square_root
    _, middle_square_root, cross_eigenvalues, minimum_cross_eigenvalue = psd_square_root(
        middle, "Covariance square-root argument"
    )

    empirical_trace = float(np.trace(empirical_covariance))
    target_trace = float(np.trace(target_covariance))
    cross_trace = float(np.trace(middle_square_root))
    mean_squared = float(empirical_mean @ empirical_mean)
    covariance_squared_raw = empirical_trace + target_trace - 2 * cross_trace
    tolerance = 1e-8 * max(1.0, empirical_trace + target_trace)
    if covariance_squared_raw < -tolerance:
        raise FloatingPointError(f"Dowson--Landau covariance term is negative: {covariance_squared_raw:.3e}")
    covariance_squared = max(0.0, covariance_squared_raw)
    total_squared = mean_squared + covariance_squared
    rank_tolerance = np.finfo(np.float64).eps * len(empirical_eigenvalues) * max(1.0, empirical_eigenvalues[-1])

    metrics = np.array([
        mean_squared,
        covariance_squared,
        covariance_squared_raw,
        total_squared,
        np.sqrt(total_squared),
        empirical_trace,
        target_trace,
        cross_trace,
        np.count_nonzero(empirical_eigenvalues > rank_tolerance),
        minimum_empirical_eigenvalue,
        minimum_target_eigenvalue,
        minimum_cross_eigenvalue,
        np.max(np.abs(empirical_mean)),
    ], dtype=np.float64)
    return {
        "metrics": metrics,
        "empirical_mean": empirical_mean,
        "empirical_covariance_diagonal": np.diag(empirical_covariance),
        "empirical_eigenvalues": empirical_eigenvalues,
        "target_eigenvalues": target_eigenvalues,
        "cross_eigenvalues": cross_eigenvalues,
    }


def generate_deeprv_draws(decoder, locations, variance, lengthscale, realizations, batch_size, rng):
    num_locations = len(locations)
    padded_locations = np.zeros(decoder.max_locations, dtype=np.float32)
    padded_locations[:num_locations] = locations.astype(np.float32)
    mask = np.arange(decoder.max_locations) < num_locations
    squared_distances = (padded_locations[:, None] - padded_locations[None, :]) ** 2
    covariance = np.exp(-0.5 * squared_distances / lengthscale**2).astype(np.float32)
    covariance += JITTER * np.eye(decoder.max_locations, dtype=np.float32)

    latent = rng.standard_normal((realizations, num_locations)).astype(np.float32)
    padded_latent = np.zeros((realizations, decoder.max_locations), dtype=np.float32)
    padded_latent[:, :num_locations] = latent
    unit_draws = []

    for start in range(0, realizations, batch_size):
        current_latent = padded_latent[start:start + batch_size]
        current_mask = np.repeat(mask[None], len(current_latent), axis=0)
        predicted = decoder.decode(
            jnp.asarray(current_latent),
            jnp.asarray(lengthscale, dtype=jnp.float32),
            jnp.asarray(padded_locations)[:, None],
            jnp.asarray(covariance),
            jnp.asarray(current_mask),
        )[:, :num_locations, 0]
        predicted.block_until_ready()
        unit_draws.append(np.asarray(jax.device_get(predicted), dtype=np.float32))

    draws = np.sqrt(np.float32(variance)) * np.concatenate(unit_draws, axis=0)
    return latent, draws


def generate_exact_gp_draws(target_covariance, realizations, rng):
    latent = rng.standard_normal((realizations, len(target_covariance))).astype(np.float32)
    cholesky = np.linalg.cholesky(target_covariance)
    draws = latent.astype(np.float64) @ cholesky.T
    return latent, draws.astype(np.float32)


def open_saved_array(path, shape, dtype, resume, fill=None):
    if path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {path}")
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"Stored array {path} has shape {array.shape} and dtype {array.dtype}, expected {shape} and {np.dtype(dtype)}")
        return array
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        array[...] = fill
        array.flush()
    return array


def prepare_storage(name, num_parameters, realizations, num_locations, output_directory, resume, exact_gp_benchmark):
    slug = VITAL_RATES[name]["filename"]
    raw_directory = output_directory / "raw_draws"
    intermediate_directory = output_directory / "intermediates"
    progress_directory = output_directory / "progress"
    for directory in (raw_directory, intermediate_directory, progress_directory):
        directory.mkdir(parents=True, exist_ok=True)

    arrays = {
        "latent": open_saved_array(raw_directory / f"{slug}_latent_z.npy", (num_parameters, realizations, num_locations), np.float32, resume),
        "draws": open_saved_array(raw_directory / f"{slug}_deeprv_draws.npy", (num_parameters, realizations, num_locations), np.float32, resume),
        "mean": open_saved_array(intermediate_directory / f"{slug}_empirical_mean.npy", (num_parameters, num_locations), np.float64, resume),
        "covariance_diagonal": open_saved_array(intermediate_directory / f"{slug}_empirical_covariance_diagonal.npy", (num_parameters, num_locations), np.float64, resume),
        "empirical_eigenvalues": open_saved_array(intermediate_directory / f"{slug}_empirical_eigenvalues.npy", (num_parameters, num_locations), np.float64, resume),
        "target_eigenvalues": open_saved_array(intermediate_directory / f"{slug}_target_eigenvalues.npy", (num_parameters, num_locations), np.float64, resume),
        "cross_eigenvalues": open_saved_array(intermediate_directory / f"{slug}_cross_eigenvalues.npy", (num_parameters, num_locations), np.float64, resume),
        "metrics": open_saved_array(intermediate_directory / f"{slug}_scalar_metrics.npy", (num_parameters, len(METRIC_COLUMNS)), np.float64, resume, np.nan),
        "completed": open_saved_array(progress_directory / f"{slug}_completed.npy", (num_parameters,), np.bool_, resume, False),
    }
    if exact_gp_benchmark:
        arrays.update({
            "exact_latent": open_saved_array(raw_directory / f"{slug}_exact_gp_latent_z.npy", (num_parameters, realizations, num_locations), np.float32, resume),
            "exact_draws": open_saved_array(raw_directory / f"{slug}_exact_gp_draws.npy", (num_parameters, realizations, num_locations), np.float32, resume),
            "exact_mean": open_saved_array(intermediate_directory / f"{slug}_exact_gp_empirical_mean.npy", (num_parameters, num_locations), np.float64, resume),
            "exact_covariance_diagonal": open_saved_array(intermediate_directory / f"{slug}_exact_gp_empirical_covariance_diagonal.npy", (num_parameters, num_locations), np.float64, resume),
            "exact_empirical_eigenvalues": open_saved_array(intermediate_directory / f"{slug}_exact_gp_empirical_eigenvalues.npy", (num_parameters, num_locations), np.float64, resume),
            "exact_cross_eigenvalues": open_saved_array(intermediate_directory / f"{slug}_exact_gp_cross_eigenvalues.npy", (num_parameters, num_locations), np.float64, resume),
            "exact_metrics": open_saved_array(intermediate_directory / f"{slug}_exact_gp_scalar_metrics.npy", (num_parameters, len(METRIC_COLUMNS)), np.float64, resume, np.nan),
        })
    return arrays


def flush_arrays(arrays):
    for array in arrays.values():
        array.flush()


def parameter_frame(name, draws, num_locations, realizations, arrays):
    metrics = arrays["metrics"]
    completed = arrays["completed"]
    indices = np.flatnonzero(np.asarray(completed))
    frame = pd.DataFrame({
        "Vital Rate": name,
        "Parameter Draw": indices,
        "Variance": draws["variance"][indices],
        "Lengthscale": draws["lengthscale"][indices],
        "Number Of Locations": num_locations,
        "Realizations": realizations,
    })
    for column_index, column in enumerate(METRIC_COLUMNS):
        frame[column] = np.asarray(metrics)[indices, column_index]
    if "exact_metrics" in arrays:
        for column_index, column in enumerate(METRIC_COLUMNS):
            frame[f"Exact GP {column}"] = np.asarray(arrays["exact_metrics"])[indices, column_index]
        target_trace = frame["Target Covariance Trace"]
        for component in ("Mean", "Covariance", "DL"):
            deeprv_column = f"D {component} Squared"
            exact_column = f"Exact GP D {component} Squared"
            frame[f"Excess D {component} Squared"] = frame[deeprv_column] - frame[exact_column]
            frame[f"Relative DeepRV D {component} Squared"] = frame[deeprv_column] / target_trace
            frame[f"Relative Exact GP D {component} Squared"] = frame[exact_column] / target_trace
            frame[f"Relative Excess D {component} Squared"] = frame[f"Excess D {component} Squared"] / target_trace
    return frame


def save_rate_results(name, draws, num_locations, realizations, arrays, output_directory):
    frame = parameter_frame(name, draws, num_locations, realizations, arrays)
    path = output_directory / "results" / f"dowson_landau_{VITAL_RATES[name]['filename']}.csv"
    frame.to_csv(path, index=False)
    return frame


def evaluate_vital_rate(name, locations, draws, decoder, args, output_directory, rate_index):
    num_parameters = len(draws["variance"])
    num_locations = len(locations)
    if num_locations > decoder.max_locations:
        raise ValueError(f"{name} has {num_locations} locations but the decoder supports {decoder.max_locations}")

    arrays = prepare_storage(
        name,
        num_parameters,
        args.realizations,
        num_locations,
        output_directory,
        args.resume,
        args.exact_gp_benchmark,
    )
    np.save(output_directory / "intermediates" / f"{VITAL_RATES[name]['filename']}_locations.npy", locations)
    np.savez(
        output_directory / "intermediates" / f"{VITAL_RATES[name]['filename']}_parameters.npz",
        variance=draws["variance"],
        lengthscale=draws["lengthscale"],
    )

    squared_distances = (locations[:, None] - locations[None, :]) ** 2
    identity = np.eye(num_locations, dtype=np.float64)
    started = time.perf_counter()

    for parameter_index, (variance, lengthscale) in enumerate(zip(draws["variance"], draws["lengthscale"])):
        if arrays["completed"][parameter_index]:
            continue

        rng = np.random.default_rng(np.random.SeedSequence([args.seed, rate_index, parameter_index]))
        latent, deeprv_draws = generate_deeprv_draws(
            decoder, locations, variance, lengthscale, args.realizations, args.decoder_batch_size, rng
        )
        target_covariance = variance * np.exp(-0.5 * squared_distances / lengthscale**2) + JITTER * identity
        diagnostics = dowson_landau_components(deeprv_draws, target_covariance)

        arrays["latent"][parameter_index] = latent
        arrays["draws"][parameter_index] = deeprv_draws
        arrays["mean"][parameter_index] = diagnostics["empirical_mean"]
        arrays["covariance_diagonal"][parameter_index] = diagnostics["empirical_covariance_diagonal"]
        arrays["empirical_eigenvalues"][parameter_index] = diagnostics["empirical_eigenvalues"]
        arrays["target_eigenvalues"][parameter_index] = diagnostics["target_eigenvalues"]
        arrays["cross_eigenvalues"][parameter_index] = diagnostics["cross_eigenvalues"]
        arrays["metrics"][parameter_index] = diagnostics["metrics"]

        if args.exact_gp_benchmark:
            exact_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, rate_index, parameter_index, 1])
            )
            exact_latent, exact_draws = generate_exact_gp_draws(
                target_covariance, args.realizations, exact_rng
            )
            exact_diagnostics = dowson_landau_components(exact_draws, target_covariance)
            arrays["exact_latent"][parameter_index] = exact_latent
            arrays["exact_draws"][parameter_index] = exact_draws
            arrays["exact_mean"][parameter_index] = exact_diagnostics["empirical_mean"]
            arrays["exact_covariance_diagonal"][parameter_index] = exact_diagnostics["empirical_covariance_diagonal"]
            arrays["exact_empirical_eigenvalues"][parameter_index] = exact_diagnostics["empirical_eigenvalues"]
            arrays["exact_cross_eigenvalues"][parameter_index] = exact_diagnostics["cross_eigenvalues"]
            arrays["exact_metrics"][parameter_index] = exact_diagnostics["metrics"]

        arrays["completed"][parameter_index] = True

        completed = parameter_index + 1
        if completed % args.save_every == 0 or completed == num_parameters:
            flush_arrays(arrays)
            save_rate_results(name, draws, num_locations, args.realizations, arrays, output_directory)
            elapsed_hours = (time.perf_counter() - started) / 3600
            print(f"{name}: {completed:,}/{num_parameters:,} Parameter Draws ({elapsed_hours:.2f} Hours)", flush=True)

    flush_arrays(arrays)
    return save_rate_results(name, draws, num_locations, args.realizations, arrays, output_directory)


def create_summary(results):
    rows = []
    for name, current in results.groupby("Vital Rate", sort=False):
        row = {
            "Vital Rate": name,
            "Parameter Draws": len(current),
            "Realizations Per Parameter Draw": int(current["Realizations"].iloc[0]),
            "Number Of Locations": int(current["Number Of Locations"].iloc[0]),
        }
        metrics = ["D Mean Squared", "D Covariance Squared", "D DL Squared", "D DL"]
        metrics.extend(
            metric
            for metric in (
                "Exact GP D Mean Squared",
                "Exact GP D Covariance Squared",
                "Exact GP D DL Squared",
                "Exact GP D DL",
                "Excess D Mean Squared",
                "Excess D Covariance Squared",
                "Excess D DL Squared",
                "Relative DeepRV D DL Squared",
                "Relative Exact GP D DL Squared",
                "Relative Excess D DL Squared",
            )
            if metric in current
        )
        for metric in metrics:
            values = current[metric]
            row[f"Mean {metric}"] = values.mean()
            row[f"Median {metric}"] = values.median()
            row[f"{metric} 5th Percentile"] = values.quantile(0.05)
            row[f"{metric} 95th Percentile"] = values.quantile(0.95)
        rows.append(row)
    return pd.DataFrame(rows)


def write_json(path, contents):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(contents, indent=2))
    temporary_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=1)
    parser.add_argument("--realizations", type=int, default=50)
    parser.add_argument("--decoder-batch-size", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--exact-gp-benchmark", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.realizations != 50:
        raise ValueError("This evaluation requires exactly 50 realizations per parameter draw")
    if args.decoder_batch_size < 1 or args.realizations % args.decoder_batch_size:
        raise ValueError("decoder-batch-size must be a positive divisor of 50")
    if args.save_every < 1:
        raise ValueError("save-every must be positive")

    output_directory = args.output_dir.resolve()
    results_directory = output_directory / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = output_directory / "run_metadata.json"
    if metadata_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite an existing run in {output_directory}")
    if metadata_path.exists() and args.resume:
        previous_metadata = json.loads(metadata_path.read_text())
        if bool(previous_metadata.get("exact_gp_samples_generated", False)) != args.exact_gp_benchmark:
            raise ValueError("The resume setting for --exact-gp-benchmark does not match the existing run")

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
        "realizations_per_parameter_draw": args.realizations,
        "decoder_batch_size": args.decoder_batch_size,
        "save_every": args.save_every,
        "seed": args.seed,
        "jitter": JITTER,
        "maximum_decoder_locations": decoder.max_locations,
        "location_counts": {name: len(values) for name, values in locations.items()},
        "jax_devices": [str(device) for device in jax.devices()],
        "versions": {"python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "target": "Zero-mean RBF GP with posterior variance and lengthscale",
        "exact_gp_samples_generated": args.exact_gp_benchmark,
        "exact_gp_realizations_per_parameter_draw": args.realizations if args.exact_gp_benchmark else 0,
        "exact_gp_random_samples_independent_of_deeprv": args.exact_gp_benchmark,
    }
    write_json(metadata_path, metadata)
    print("JAX Devices:", jax.devices(), flush=True)
    print("Output Directory:", output_directory, flush=True)
    print("Location Counts:", metadata["location_counts"], flush=True)

    rate_frames = []
    for rate_index, name in enumerate(VITAL_RATES):
        print(f"\nEvaluating {name} At {len(locations[name])} First-Year Locations", flush=True)
        frame = evaluate_vital_rate(
            name, locations[name], parameter_draws[name], decoder, args, output_directory, rate_index
        )
        rate_frames.append(frame)
        combined = pd.concat(rate_frames, ignore_index=True)
        combined.to_csv(results_directory / "dowson_landau_by_parameter.csv", index=False)
        create_summary(combined).to_csv(results_directory / "dowson_landau_summary.csv", index=False)

    results = pd.concat(rate_frames, ignore_index=True)
    summary = create_summary(results)
    results.to_csv(results_directory / "dowson_landau_by_parameter.csv", index=False)
    summary.to_csv(results_directory / "dowson_landau_summary.csv", index=False)
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
