"""Evaluate age-size DeepRV covariances with the Dowson--Landau distance."""

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

from experiments.soay.age_size.abc.deeprv.deeprv_decoder_agesize import get_runtime  # noqa: E402, I001


POOLED_YEARS = (2, 3, 4)
REQUIRED_PARAMETER_DRAWS = 5000
REQUIRED_REALIZATIONS = 500
VITAL_RATES = {
    "Survival": {"prefix": "surv", "observation": "Surv", "filename": "survival", "minimum_age": None},
    "Growth": {"prefix": "growth", "observation": "z1", "filename": "growth", "minimum_age": None},
    "Reproduction": {"prefix": "repr", "observation": "Repr", "filename": "reproduction", "minimum_age": 1},
}
METRIC_COLUMNS = [
    "D Covariance Squared",
    "D Covariance Squared Raw",
    "D Covariance",
    "Relative D Covariance Squared",
    "Empirical Covariance Trace",
    "Target Covariance Trace",
    "Cross Square Root Trace",
    "Empirical Covariance Rank",
    "Empirical Covariance Rank Tolerance",
    "Minimum Raw Empirical Eigenvalue",
    "Minimum Raw Target Eigenvalue",
    "Minimum Raw Cross Eigenvalue",
    "Minimum Clipped Empirical Eigenvalue",
    "Minimum Clipped Target Eigenvalue",
    "Minimum Clipped Cross Eigenvalue",
]


def write_json(path, contents):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(contents, indent=2))
    temporary_path.replace(path)


def write_csv(frame, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def load_inputs(data_path, samples_path):
    data = pd.read_csv(data_path)
    required_columns = {"z", "age", "yr", "Surv", "z1", "Repr"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing data columns: {missing_columns}")

    locations = {}
    location_frames = {}
    for name, settings in VITAL_RATES.items():
        mask = data[["z", "age", settings["observation"]]].notna().all(axis=1)
        mask &= data["yr"].isin(POOLED_YEARS)
        if settings["minimum_age"] is not None:
            mask &= data["age"].ge(settings["minimum_age"])
        selected = data.loc[mask, ["z", "age", "yr"]].copy()
        if selected.empty:
            raise ValueError(f"{name} has no eligible locations in pooled years {POOLED_YEARS}")
        selected.insert(0, "Dataset Row", selected.index.to_numpy())
        selected = selected.rename(columns={"z": "Size", "age": "Age", "yr": "Year"}).reset_index(drop=True)
        locations[name] = selected[["Size", "Age"]].to_numpy(dtype=np.float64)
        location_frames[name] = selected

    parameter_draws = {}
    with np.load(samples_path) as saved:
        for name, settings in VITAL_RATES.items():
            variance_key = f"{settings['prefix']}_var"
            lengthscale_key = f"{settings['prefix']}_ls"
            if variance_key not in saved.files or lengthscale_key not in saved.files:
                raise ValueError(f"Missing posterior arrays for {name}: {variance_key}, {lengthscale_key}")
            variance = np.asarray(saved[variance_key], dtype=np.float64).reshape(-1)
            lengthscales = np.asarray(saved[lengthscale_key], dtype=np.float64)
            if variance.shape != (REQUIRED_PARAMETER_DRAWS,):
                raise ValueError(f"{variance_key} must have shape ({REQUIRED_PARAMETER_DRAWS},), received {variance.shape}")
            if lengthscales.shape != (REQUIRED_PARAMETER_DRAWS, 2):
                raise ValueError(f"{lengthscale_key} must have shape ({REQUIRED_PARAMETER_DRAWS}, 2), received {lengthscales.shape}")
            if not np.all(np.isfinite(variance)) or not np.all(np.isfinite(lengthscales)):
                raise ValueError(f"{name} contains non-finite posterior GP parameters")
            if np.any(variance <= 0) or np.any(lengthscales <= 0):
                raise ValueError(f"{name} contains a non-positive variance or lengthscale")
            parameter_draws[name] = {"variance": variance, "lengthscales": lengthscales}

    return locations, location_frames, parameter_draws


def symmetric_psd(matrix, name):
    matrix = np.asarray(matrix, dtype=np.float64)
    symmetric = (matrix + matrix.T) / 2
    raw_eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(raw_eigenvalues))))
    negative_tolerance = 1e-8 * scale
    if raw_eigenvalues[0] < -negative_tolerance:
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


def dowson_landau_covariance_metrics(draws, target_covariance):
    draws = np.asarray(draws, dtype=np.float64)
    centered_draws = draws - draws.mean(axis=0)
    empirical_covariance = centered_draws.T @ centered_draws / (len(draws) - 1)
    empirical_covariance, empirical_eigenvalues, _, minimum_empirical_eigenvalue = symmetric_psd(
        empirical_covariance, "Empirical covariance"
    )
    target_covariance, target_square_root, target_eigenvalues, minimum_target_eigenvalue = psd_square_root(
        target_covariance, "Target covariance"
    )
    cross_argument = target_square_root @ empirical_covariance @ target_square_root
    _, cross_square_root, cross_eigenvalues, minimum_cross_eigenvalue = psd_square_root(
        cross_argument, "Covariance cross-square-root argument"
    )

    empirical_trace = float(np.trace(empirical_covariance))
    target_trace = float(np.trace(target_covariance))
    cross_trace = float(np.trace(cross_square_root))
    distance_squared_raw = empirical_trace + target_trace - 2 * cross_trace
    distance_tolerance = 1e-8 * max(1.0, empirical_trace + target_trace)
    if distance_squared_raw < -distance_tolerance:
        raise FloatingPointError(f"Dowson--Landau covariance term is negative: {distance_squared_raw:.3e}")
    distance_squared = max(0.0, distance_squared_raw)
    rank_tolerance = np.finfo(np.float64).eps * len(empirical_eigenvalues) * max(1.0, empirical_eigenvalues[-1])

    return np.array([
        distance_squared,
        distance_squared_raw,
        np.sqrt(distance_squared),
        distance_squared / target_trace,
        empirical_trace,
        target_trace,
        cross_trace,
        np.count_nonzero(empirical_eigenvalues > rank_tolerance),
        rank_tolerance,
        minimum_empirical_eigenvalue,
        minimum_target_eigenvalue,
        minimum_cross_eigenvalue,
        empirical_eigenvalues[0],
        target_eigenvalues[0],
        cross_eigenvalues[0],
    ], dtype=np.float64)


def prepare_decoder_inputs(decoder, locations):
    num_locations = len(locations)
    if num_locations > decoder.max_locations:
        raise ValueError(f"The decoder supports {decoder.max_locations} locations, received {num_locations}")
    padded_locations = np.zeros((decoder.max_locations, 2), dtype=np.float32)
    padded_locations[:num_locations] = locations.astype(np.float32)
    mask = np.arange(decoder.max_locations) < num_locations
    squared_differences = (padded_locations[:, None, :] - padded_locations[None, :, :]) ** 2
    return padded_locations, mask, squared_differences


def generate_deeprv_draws(decoder, prepared_inputs, variance, lengthscales, realizations, batch_size, rng):
    padded_locations, mask, squared_differences = prepared_inputs
    num_locations = int(mask.sum())
    inverse_squared_lengthscales = 1.0 / np.asarray(lengthscales, dtype=np.float32) ** 2
    exponent = np.sum(squared_differences * inverse_squared_lengthscales, axis=-1)
    unit_covariance = np.exp(-0.5 * exponent).astype(np.float32)
    unit_covariance += decoder.jitter * np.eye(decoder.max_locations, dtype=np.float32)
    batches = []

    for start in range(0, realizations, batch_size):
        current_batch_size = min(batch_size, realizations - start)
        latent = rng.standard_normal((current_batch_size, num_locations)).astype(np.float32)
        padded_latent = np.zeros((current_batch_size, decoder.max_locations), dtype=np.float32)
        padded_latent[:, :num_locations] = latent
        batched_mask = np.repeat(mask[None], current_batch_size, axis=0)
        predicted = decoder.decode(
            jnp.asarray(padded_latent),
            jnp.asarray(lengthscales, dtype=jnp.float32),
            jnp.asarray(padded_locations),
            jnp.asarray(unit_covariance),
            jnp.asarray(batched_mask),
        )[:, :num_locations, 0]
        predicted.block_until_ready()
        batches.append(np.asarray(jax.device_get(predicted), dtype=np.float32))

    return np.sqrt(np.float32(variance)) * np.concatenate(batches, axis=0)


def target_covariance(locations, variance, lengthscales, jitter):
    differences = locations[:, None, :] - locations[None, :, :]
    exponent = np.sum((differences / np.asarray(lengthscales, dtype=np.float64)) ** 2, axis=-1)
    correlation = np.exp(-0.5 * exponent)
    return variance * (correlation + jitter * np.eye(len(locations), dtype=np.float64))


def open_saved_array(path, shape, dtype, resume, fill):
    if path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {path}")
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"Stored array {path} does not match expected shape or dtype")
        return array
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[...] = fill
    array.flush()
    return array


def save_static_inputs(name, location_frame, parameters, output_directory, resume):
    slug = VITAL_RATES[name]["filename"]
    location_path = output_directory / "inputs" / f"{slug}_pooled_locations.csv"
    parameter_path = output_directory / "inputs" / f"{slug}_posterior_parameters.csv"
    parameter_frame = pd.DataFrame({
        "Parameter Draw": np.arange(REQUIRED_PARAMETER_DRAWS),
        "Variance": parameters["variance"],
        "Size Lengthscale": parameters["lengthscales"][:, 0],
        "Age Lengthscale": parameters["lengthscales"][:, 1],
    })
    if location_path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {location_path}")
        saved_locations = pd.read_csv(location_path)
        if list(saved_locations.columns) != list(location_frame.columns) or not np.allclose(saved_locations.to_numpy(), location_frame.to_numpy()):
            raise ValueError(f"Saved {name} locations do not match the current locations")
    else:
        write_csv(location_frame, location_path)
    if parameter_path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {parameter_path}")
        saved_parameters = pd.read_csv(parameter_path)
        if list(saved_parameters.columns) != list(parameter_frame.columns) or not np.allclose(saved_parameters.to_numpy(), parameter_frame.to_numpy()):
            raise ValueError(f"Saved {name} posterior parameters do not match the current parameters")
    else:
        write_csv(parameter_frame, parameter_path)


def prepare_storage(name, output_directory, resume):
    slug = VITAL_RATES[name]["filename"]
    return {
        "metrics": open_saved_array(
            output_directory / "progress" / f"{slug}_scalar_metrics.npy",
            (REQUIRED_PARAMETER_DRAWS, len(METRIC_COLUMNS)),
            np.float64,
            resume,
            np.nan,
        ),
        "completed": open_saved_array(
            output_directory / "progress" / f"{slug}_completed.npy",
            (REQUIRED_PARAMETER_DRAWS,),
            np.bool_,
            resume,
            False,
        ),
    }


def parameter_frame(name, parameters, num_locations, arrays):
    indices = np.flatnonzero(np.asarray(arrays["completed"]))
    frame = pd.DataFrame({
        "Vital Rate": name,
        "Parameter Draw": indices,
        "Variance": parameters["variance"][indices],
        "Size Lengthscale": parameters["lengthscales"][indices, 0],
        "Age Lengthscale": parameters["lengthscales"][indices, 1],
        "Number Of Locations": num_locations,
        "Number Of Realizations": REQUIRED_REALIZATIONS,
    })
    for column_index, column in enumerate(METRIC_COLUMNS):
        frame[column] = np.asarray(arrays["metrics"])[indices, column_index]
    return frame


def save_rate_results(name, parameters, num_locations, arrays, output_directory):
    frame = parameter_frame(name, parameters, num_locations, arrays)
    result_path = output_directory / "results" / f"dowson_landau_{VITAL_RATES[name]['filename']}.csv"
    write_csv(frame, result_path)
    return frame


def create_summary(results):
    rows = []
    for name, current in results.groupby("Vital Rate", sort=False):
        values = current["Relative D Covariance Squared"]
        rows.append({
            "Vital Rate": name,
            "Completed Parameter Draws": len(current),
            "Number Of Locations": int(current["Number Of Locations"].iloc[0]),
            "Number Of Realizations": int(current["Number Of Realizations"].iloc[0]),
            "Mean Relative D Covariance Squared": values.mean(),
            "Median Relative D Covariance Squared": values.median(),
            "Relative D Covariance Squared 5th Percentile": values.quantile(0.05),
            "Relative D Covariance Squared 95th Percentile": values.quantile(0.95),
            "Relative D Covariance Squared 99th Percentile": values.quantile(0.99),
        })
    return pd.DataFrame(rows)


def save_combined_results(frames, output_directory):
    completed_frames = [frame for frame in frames if not frame.empty]
    if not completed_frames:
        return
    results = pd.concat(completed_frames, ignore_index=True)
    write_csv(results, output_directory / "results" / "dowson_landau_age_size_by_parameter.csv")
    write_csv(create_summary(results), output_directory / "results" / "dowson_landau_age_size_summary.csv")


def evaluate_vital_rate(name, locations, parameters, decoder, args, output_directory, rate_index):
    num_locations = len(locations)
    prepared_inputs = prepare_decoder_inputs(decoder, locations)
    arrays = prepare_storage(name, output_directory, args.resume)
    pending_indices = []
    started = time.perf_counter()

    for parameter_index, (variance, lengthscales) in enumerate(zip(parameters["variance"], parameters["lengthscales"])):
        if arrays["completed"][parameter_index]:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, rate_index, parameter_index]))
        draws = generate_deeprv_draws(
            decoder,
            prepared_inputs,
            variance,
            lengthscales,
            REQUIRED_REALIZATIONS,
            args.decoder_batch_size,
            rng,
        )
        covariance = target_covariance(locations, variance, lengthscales, decoder.jitter)
        arrays["metrics"][parameter_index] = dowson_landau_covariance_metrics(draws, covariance)
        pending_indices.append(parameter_index)

        if len(pending_indices) == args.save_every or parameter_index + 1 == REQUIRED_PARAMETER_DRAWS:
            arrays["metrics"].flush()
            arrays["completed"][pending_indices] = True
            arrays["completed"].flush()
            pending_indices.clear()
            frame = save_rate_results(name, parameters, num_locations, arrays, output_directory)
            elapsed_hours = (time.perf_counter() - started) / 3600
            print(f"{name}: {len(frame):,}/{REQUIRED_PARAMETER_DRAWS:,} Parameter Draws ({elapsed_hours:.2f} Hours)", flush=True)

    if pending_indices:
        arrays["metrics"].flush()
        arrays["completed"][pending_indices] = True
        arrays["completed"].flush()
    return save_rate_results(name, parameters, num_locations, arrays, output_directory)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--realizations", type=int, default=REQUIRED_REALIZATIONS)
    parser.add_argument("--decoder-batch-size", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.realizations != REQUIRED_REALIZATIONS:
        raise ValueError(f"This evaluation requires exactly {REQUIRED_REALIZATIONS} realizations per parameter draw")
    if args.decoder_batch_size < 1 or REQUIRED_REALIZATIONS % args.decoder_batch_size:
        raise ValueError(f"decoder-batch-size must be a positive divisor of {REQUIRED_REALIZATIONS}")
    if args.save_every < 1:
        raise ValueError("save-every must be positive")

    checkpoint_path = args.checkpoint.resolve()
    data_path = args.data.resolve()
    samples_path = args.samples.resolve()
    output_directory = args.output_dir.resolve()
    for path in (checkpoint_path, data_path, samples_path):
        if not path.exists():
            raise FileNotFoundError(path)
    for directory in ("inputs", "progress", "results", "figures"):
        (output_directory / directory).mkdir(parents=True, exist_ok=True)

    decoder = get_runtime(checkpoint_path)
    locations, location_frames, parameter_draws = load_inputs(data_path, samples_path)
    for name in VITAL_RATES:
        if len(locations[name]) > decoder.max_locations:
            raise ValueError(f"{name} has {len(locations[name])} locations but the decoder supports {decoder.max_locations}")
        save_static_inputs(name, location_frames[name], parameter_draws[name], output_directory, args.resume)

    metadata_path = output_directory / "run_metadata.json"
    configuration = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "samples": str(samples_path),
        "pooled_years": list(POOLED_YEARS),
        "coordinate_order": ["size", "age"],
        "parameter_draws_per_vital_rate": REQUIRED_PARAMETER_DRAWS,
        "realizations_per_parameter_draw": REQUIRED_REALIZATIONS,
        "decoder_batch_size": args.decoder_batch_size,
        "save_every": args.save_every,
        "seed": args.seed,
        "jitter": decoder.jitter,
        "maximum_decoder_locations": decoder.max_locations,
        "location_counts": {name: len(values) for name, values in locations.items()},
        "target_covariance": "variance * (ARD RBF correlation + jitter * identity)",
        "empirical_covariance_denominator": REQUIRED_REALIZATIONS - 1,
        "exact_gp_draws_computed": False,
    }
    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite an existing run in {output_directory}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("configuration") != configuration:
            raise ValueError("The current configuration does not match the saved run metadata")
        metadata.setdefault("resumed_utc", []).append(datetime.now(timezone.utc).isoformat())
    else:
        metadata = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration,
        }
    metadata.update({
        "status": "running",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "jax_devices": [str(device) for device in jax.devices()],
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    })
    write_json(metadata_path, metadata)

    print("JAX Devices:", jax.devices(), flush=True)
    print("Output Directory:", output_directory, flush=True)
    print("Pooled Years:", POOLED_YEARS, flush=True)
    print("Location Counts:", configuration["location_counts"], flush=True)
    print("Target Covariance Convention:", configuration["target_covariance"], flush=True)
    started = time.perf_counter()

    try:
        frames = []
        for rate_index, name in enumerate(VITAL_RATES):
            print(f"\nEvaluating {name} At {len(locations[name])} Pooled Locations", flush=True)
            frame = evaluate_vital_rate(
                name,
                locations[name],
                parameter_draws[name],
                decoder,
                args,
                output_directory,
                rate_index,
            )
            frames.append(frame)
            save_combined_results(frames, output_directory)

        results = pd.concat(frames, ignore_index=True)
        save_combined_results(frames, output_directory)
        metadata.update({
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.perf_counter() - started,
            "result_rows": len(results),
        })
        write_json(metadata_path, metadata)
        print("\nEvaluation Complete", flush=True)
        print(create_summary(results).to_string(index=False), flush=True)
    except BaseException as error:
        metadata.update({
            "status": "failed",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        })
        write_json(metadata_path, metadata)
        raise


if __name__ == "__main__":
    main()
