"""Evaluate whether DeepRV draws follow the corresponding exact GP distributions."""

from experiments.paths import GENERATED, PROJECT_ROOT

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.special import ndtr
from scipy.stats import kstwo

REPO_ROOT = PROJECT_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.soay.size_only.training.deeprv_decoder import DeepRVDecoder, JITTER


VITAL_RATES = {
    "Survival": {"prefix": "surv", "observation": "Surv", "filename": "survival"},
    "Growth": {"prefix": "growth", "observation": "z1", "filename": "growth"},
    "Reproduction": {"prefix": "repr", "observation": "Repr", "filename": "reproduction"},
    "Recruit Size": {"prefix": "rcsz", "observation": "Rcsz", "filename": "recruit_size"},
}


def standard_normal_ks(samples):
    """Vectorized two-sided one-sample KS tests against the fixed N(0, 1) CDF."""
    ordered = np.sort(np.asarray(samples, dtype=np.float64), axis=1)
    num_values = ordered.shape[1]
    cdf = ndtr(ordered)
    upper_ranks = np.arange(1, num_values + 1, dtype=np.float64) / num_values
    lower_ranks = np.arange(num_values, dtype=np.float64) / num_values
    d_plus = np.max(upper_ranks - cdf, axis=1)
    d_minus = np.max(cdf - lower_ranks, axis=1)
    statistics = np.maximum(d_plus, d_minus)
    p_values = np.clip(kstwo.sf(statistics, num_values), 0.0, 1.0)
    return statistics, p_values


def load_inputs(data_path, samples_path):
    data = pd.read_csv(data_path)
    locations = {
        name: data.loc[data[settings["observation"]].notna(), "z"].to_numpy(dtype=np.float64)
        for name, settings in VITAL_RATES.items()
    }

    parameter_draws = {}
    with np.load(samples_path) as saved:
        for name, settings in VITAL_RATES.items():
            prefix = settings["prefix"]
            parameter_draws[name] = {
                "lengthscale": np.asarray(saved[f"{prefix}_ls"], dtype=np.float64).reshape(-1),
            }

    for name, draws in parameter_draws.items():
        lengths = {len(values) for values in draws.values()}
        if lengths != {5000}:
            raise ValueError(f"{name} must have exactly 5,000 parameter draws, found {sorted(lengths)}")
        if not all(np.all(np.isfinite(values)) for values in draws.values()):
            raise ValueError(f"{name} contains non-finite GP parameters")
        if np.any(draws["lengthscale"] <= 0):
            raise ValueError(f"{name} contains a non-positive lengthscale")

    return locations, parameter_draws


def evaluate_vital_rate(name, locations, draws, decoder, realizations, rng, output_directory):
    num_parameters = len(draws["lengthscale"])
    num_locations = len(locations)
    if num_locations > decoder.max_locations:
        raise ValueError(f"{name} has {num_locations} locations but the decoder supports {decoder.max_locations}")

    padded_locations = np.zeros(decoder.max_locations, dtype=np.float32)
    padded_locations[:num_locations] = locations.astype(np.float32)
    mask = np.arange(decoder.max_locations) < num_locations
    batched_mask = np.repeat(mask[None], realizations, axis=0)
    active_squared_distances = (locations[:, None] - locations[None, :]) ** 2
    padded_squared_distances = (padded_locations[:, None] - padded_locations[None, :]) ** 2
    active_identity = np.eye(num_locations, dtype=np.float64)
    padded_identity = np.eye(decoder.max_locations, dtype=np.float32)
    statistics = np.empty((num_parameters, realizations), dtype=np.float64)
    p_values = np.empty((num_parameters, realizations), dtype=np.float64)

    started = time.perf_counter()
    for parameter_index in range(num_parameters):
        lengthscale = draws["lengthscale"][parameter_index]

        latent = rng.standard_normal((realizations, num_locations)).astype(np.float32)
        padded_latent = np.zeros((realizations, decoder.max_locations), dtype=np.float32)
        padded_latent[:, :num_locations] = latent
        unit_covariance = np.exp(-0.5 * padded_squared_distances / lengthscale**2).astype(np.float32) + JITTER * padded_identity

        unit_draws = decoder.decode(
            jnp.asarray(padded_latent),
            jnp.asarray(lengthscale, dtype=jnp.float32),
            jnp.asarray(padded_locations)[:, None],
            jnp.asarray(unit_covariance),
            jnp.asarray(batched_mask),
        )[:, :num_locations, 0]
        unit_draws.block_until_ready()
        deeprv_draws = np.asarray(jax.device_get(unit_draws), dtype=np.float64)

        exact_covariance = np.exp(-0.5 * active_squared_distances / lengthscale**2) + JITTER * active_identity
        lower_cholesky = cholesky(exact_covariance, lower=True, check_finite=False)
        whitened = solve_triangular(lower_cholesky, deeprv_draws.T, lower=True, check_finite=False).T
        statistics[parameter_index], p_values[parameter_index] = standard_normal_ks(whitened)

        if (parameter_index + 1) % 100 == 0 or parameter_index + 1 == num_parameters:
            elapsed_hours = (time.perf_counter() - started) / 3600
            print(f"{name}: {parameter_index + 1:,}/{num_parameters:,} Parameter Draws ({elapsed_hours:.2f} Hours)", flush=True)

    detailed = pd.DataFrame({
        "Parameter Draw": np.repeat(np.arange(num_parameters), realizations),
        "Realization": np.tile(np.arange(realizations), num_parameters),
        "KS Statistic": statistics.reshape(-1),
        "KS P-Value": p_values.reshape(-1),
        "Reject At 0.05": (p_values < 0.05).reshape(-1),
        "Reject At 0.01": (p_values < 0.01).reshape(-1),
    })
    detailed_path = output_directory / f"ks_{VITAL_RATES[name]['filename']}.csv.gz"
    detailed.to_csv(detailed_path, index=False, compression="gzip")

    parameter_summary = pd.DataFrame({
        "Vital Rate": name,
        "Parameter Draw": np.arange(num_parameters),
        "Lengthscale": draws["lengthscale"],
        "Number Of Locations": num_locations,
        "Realizations": realizations,
        "Mean KS Statistic": statistics.mean(axis=1),
        "Median KS Statistic": np.median(statistics, axis=1),
        "Mean KS P-Value": p_values.mean(axis=1),
        "Median KS P-Value": np.median(p_values, axis=1),
        "Rejection Rate At 0.05": (p_values < 0.05).mean(axis=1),
        "Rejection Rate At 0.01": (p_values < 0.01).mean(axis=1),
    })

    flat_statistics = statistics.reshape(-1)
    flat_p_values = p_values.reshape(-1)
    vital_rate_summary = {
        "Vital Rate": name,
        "Parameter Draws": num_parameters,
        "Realizations Per Parameter Draw": realizations,
        "Total KS Tests": len(flat_statistics),
        "Number Of Locations": num_locations,
        "Mean KS Statistic": flat_statistics.mean(),
        "Median KS Statistic": np.median(flat_statistics),
        "KS Statistic 5th Percentile": np.quantile(flat_statistics, 0.05),
        "KS Statistic 95th Percentile": np.quantile(flat_statistics, 0.95),
        "Mean KS P-Value": flat_p_values.mean(),
        "Median KS P-Value": np.median(flat_p_values),
        "KS P-Value 5th Percentile": np.quantile(flat_p_values, 0.05),
        "KS P-Value 95th Percentile": np.quantile(flat_p_values, 0.95),
        "Rejection Rate At 0.05": np.mean(flat_p_values < 0.05),
        "Rejection Rate At 0.01": np.mean(flat_p_values < 0.01),
    }
    print(f"Saved {detailed_path}", flush=True)
    return parameter_summary, vital_rate_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=GENERATED / 'soay/size_only/training/transformer/model.ckpt')
    parser.add_argument("--data", type=Path, default=GENERATED / 'soay/size_only/data/gp_ibm_dataset.csv')
    parser.add_argument("--samples", type=Path, default=GENERATED / 'soay/size_only/inference/gpdf_gp_samples.npz')
    parser.add_argument("--output-dir", type=Path, default=GENERATED / 'soay/size_only/validation/ks/raw')
    parser.add_argument("--realizations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.realizations != 50:
        raise ValueError("This evaluation requires exactly 50 realizations per parameter draw")
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    expected_outputs = [output_directory / f"ks_{settings['filename']}.csv.gz" for settings in VITAL_RATES.values()]
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"Refusing to overwrite existing KS results in {output_directory}")

    print("JAX Devices:", jax.devices(), flush=True)
    print("Output Directory:", output_directory, flush=True)
    locations, parameter_draws = load_inputs(args.data.resolve(), args.samples.resolve())
    decoder = DeepRVDecoder(args.checkpoint.resolve())
    parameter_summaries = []
    vital_rate_summaries = []

    for rate_index, name in enumerate(VITAL_RATES):
        print(f"\nEvaluating {name} At {len(locations[name])} Locations", flush=True)
        rate_rng = np.random.default_rng(np.random.SeedSequence([args.seed, rate_index]))
        parameter_summary, vital_rate_summary = evaluate_vital_rate(
            name, locations[name], parameter_draws[name], decoder, args.realizations, rate_rng, output_directory
        )
        parameter_summaries.append(parameter_summary)
        vital_rate_summaries.append(vital_rate_summary)
        pd.concat(parameter_summaries, ignore_index=True).to_csv(output_directory / "ks_parameter_draw_summary.csv", index=False)
        pd.DataFrame(vital_rate_summaries).to_csv(output_directory / "ks_vital_rate_summary.csv", index=False)

    print("\nEvaluation Complete", flush=True)
    print(pd.DataFrame(vital_rate_summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
