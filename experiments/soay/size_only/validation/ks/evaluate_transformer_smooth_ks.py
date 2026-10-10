"""Evaluate DeepRV in GP covariance directions whose eigenvalues exceed jitter."""

from experiments.paths import GENERATED, PROJECT_ROOT

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, eigh

REPO_ROOT = PROJECT_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.soay.size_only.training.deeprv_decoder import DeepRVDecoder, JITTER
from experiments.soay.size_only.validation.ks.evaluate_transformer_ks import VITAL_RATES, load_inputs, standard_normal_ks


def calculate_whitened_diagnostics(whitened):
    direction_means = whitened.mean(axis=0)
    direction_variances = whitened.var(axis=0, ddof=1)
    covariance = np.atleast_2d(np.cov(whitened, rowvar=False, ddof=1))
    direction_standard_deviations = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(direction_standard_deviations, direction_standard_deviations)
    pair_indices = np.triu_indices(whitened.shape[1], k=1)
    return {
        "direction_means": direction_means,
        "direction_variances": direction_variances,
        "pair_indices": pair_indices,
        "pair_covariances": covariance[pair_indices],
        "pair_correlations": correlation[pair_indices],
    }


def evaluate_vital_rate(name, locations, draws, decoder, realizations, deeprv_rng, exact_rng, output_directory):
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
    rows = []
    direction_rows = []
    direction_pair_rows = []

    started = time.perf_counter()
    for parameter_index, lengthscale in enumerate(draws["lengthscale"]):
        smooth_covariance = np.exp(-0.5 * active_squared_distances / lengthscale**2)
        sampling_covariance = smooth_covariance + JITTER * active_identity
        eigenvalues, eigenvectors = eigh(smooth_covariance, overwrite_a=True, check_finite=False)
        eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
        eigenvectors = eigenvectors[:, ::-1]
        smooth_mask = eigenvalues > JITTER
        smooth_eigenvalues = eigenvalues[smooth_mask]
        smooth_eigenvectors = eigenvectors[:, smooth_mask]
        num_smooth_modes = len(smooth_eigenvalues)
        if num_smooth_modes == 0:
            raise ValueError(f"{name} parameter draw {parameter_index} has no eigenvalue above jitter")

        deeprv_latent = deeprv_rng.standard_normal((realizations, num_locations)).astype(np.float32)
        padded_latent = np.zeros((realizations, decoder.max_locations), dtype=np.float32)
        padded_latent[:, :num_locations] = deeprv_latent
        padded_covariance = np.exp(-0.5 * padded_squared_distances / lengthscale**2).astype(np.float32) + JITTER * padded_identity

        unit_draws = decoder.decode(
            jnp.asarray(padded_latent),
            jnp.asarray(lengthscale, dtype=jnp.float32),
            jnp.asarray(padded_locations)[:, None],
            jnp.asarray(padded_covariance),
            jnp.asarray(batched_mask),
        )[:, :num_locations, 0]
        unit_draws.block_until_ready()
        deeprv_draws = np.asarray(jax.device_get(unit_draws), dtype=np.float64)

        lower_cholesky = cholesky(sampling_covariance, lower=True, overwrite_a=True, check_finite=False)
        exact_latent = exact_rng.standard_normal((realizations, num_locations))
        exact_draws = exact_latent @ lower_cholesky.T

        standard_deviations = np.sqrt(smooth_eigenvalues + JITTER)[None]
        deeprv_whitened = (deeprv_draws @ smooth_eigenvectors) / standard_deviations
        exact_whitened = (exact_draws @ smooth_eigenvectors) / standard_deviations
        deeprv_statistic, deeprv_p_value = standard_normal_ks(deeprv_whitened.reshape(1, -1))
        exact_statistic, exact_p_value = standard_normal_ks(exact_whitened.reshape(1, -1))
        deeprv_diagnostics = calculate_whitened_diagnostics(deeprv_whitened)
        exact_diagnostics = calculate_whitened_diagnostics(exact_whitened)
        excluded_eigenvalues = eigenvalues[~smooth_mask]

        parameter_row = {
            "Vital Rate": name,
            "Parameter Draw": parameter_index,
            "Lengthscale": float(lengthscale),
            "Number Of Locations": num_locations,
            "Smooth Modes": num_smooth_modes,
            "Jitter-Dominated Modes": num_locations - num_smooth_modes,
            "Pooled Coefficients": realizations * num_smooth_modes,
            "Smooth Variance Explained": smooth_eigenvalues.sum() / eigenvalues.sum(),
            "Smallest Retained Eigenvalue": smooth_eigenvalues[-1],
            "Smallest Retained Eigenvalue To Jitter": smooth_eigenvalues[-1] / JITTER,
            "Largest Excluded Eigenvalue": excluded_eigenvalues[0] if len(excluded_eigenvalues) else np.nan,
            "DeepRV KS Statistic": deeprv_statistic[0],
            "DeepRV KS P-Value": deeprv_p_value[0],
            "DeepRV Reject At 0.05": deeprv_p_value[0] < 0.05,
            "DeepRV Reject At 0.01": deeprv_p_value[0] < 0.01,
            "Exact GP KS Statistic": exact_statistic[0],
            "Exact GP KS P-Value": exact_p_value[0],
            "Exact GP Reject At 0.05": exact_p_value[0] < 0.05,
            "Exact GP Reject At 0.01": exact_p_value[0] < 0.01,
        }

        for method, whitened, diagnostics in (
            ("DeepRV", deeprv_whitened, deeprv_diagnostics),
            ("Exact GP", exact_whitened, exact_diagnostics),
        ):
            direction_means = diagnostics["direction_means"]
            direction_variances = diagnostics["direction_variances"]
            pair_covariances = diagnostics["pair_covariances"]
            pair_correlations = diagnostics["pair_correlations"]
            parameter_row.update({
                f"{method} Pooled Whitened Mean": whitened.mean(),
                f"{method} Pooled Whitened Variance": whitened.var(ddof=1),
                f"{method} Mean Absolute Direction Mean": np.mean(np.abs(direction_means)),
                f"{method} Mean Direction Variance": direction_variances.mean(),
                f"{method} Mean Absolute Direction Variance Error": np.mean(np.abs(direction_variances - 1.0)),
                f"{method} Mean Absolute Between-Direction Covariance": np.mean(np.abs(pair_covariances)) if len(pair_covariances) else np.nan,
                f"{method} Mean Absolute Between-Direction Correlation": np.mean(np.abs(pair_correlations)) if len(pair_correlations) else np.nan,
            })

        rows.append(parameter_row)

        for direction_index, eigenvalue in enumerate(smooth_eigenvalues):
            direction_rows.append({
                "Vital Rate": name,
                "Parameter Draw": parameter_index,
                "Lengthscale": float(lengthscale),
                "Realizations": realizations,
                "Direction Index": direction_index + 1,
                "Eigenvalue": eigenvalue,
                "Eigenvalue To Jitter Ratio": eigenvalue / JITTER,
                "DeepRV Whitened Mean": deeprv_diagnostics["direction_means"][direction_index],
                "DeepRV Whitened Variance": deeprv_diagnostics["direction_variances"][direction_index],
                "Exact GP Whitened Mean": exact_diagnostics["direction_means"][direction_index],
                "Exact GP Whitened Variance": exact_diagnostics["direction_variances"][direction_index],
            })

        first_indices, second_indices = deeprv_diagnostics["pair_indices"]
        for pair_index, (first_index, second_index) in enumerate(zip(first_indices, second_indices)):
            direction_pair_rows.append({
                "Vital Rate": name,
                "Parameter Draw": parameter_index,
                "Lengthscale": float(lengthscale),
                "Realizations": realizations,
                "First Direction Index": first_index + 1,
                "Second Direction Index": second_index + 1,
                "DeepRV Whitened Covariance": deeprv_diagnostics["pair_covariances"][pair_index],
                "DeepRV Whitened Correlation": deeprv_diagnostics["pair_correlations"][pair_index],
                "Exact GP Whitened Covariance": exact_diagnostics["pair_covariances"][pair_index],
                "Exact GP Whitened Correlation": exact_diagnostics["pair_correlations"][pair_index],
            })

        if (parameter_index + 1) % 100 == 0 or parameter_index + 1 == num_parameters:
            elapsed_hours = (time.perf_counter() - started) / 3600
            print(f"{name}: {parameter_index + 1:,}/{num_parameters:,} Parameter Draws ({elapsed_hours:.2f} Hours)", flush=True)

    results = pd.DataFrame(rows)
    direction_results = pd.DataFrame(direction_rows)
    direction_pair_results = pd.DataFrame(direction_pair_rows)
    output_path = output_directory / f"ks_smooth_control_{VITAL_RATES[name]['filename']}.csv.gz"
    direction_output_path = output_directory / f"ks_smooth_control_direction_diagnostics_{VITAL_RATES[name]['filename']}.csv.gz"
    direction_pair_output_path = output_directory / f"ks_smooth_control_direction_pair_diagnostics_{VITAL_RATES[name]['filename']}.csv.gz"
    results.to_csv(output_path, index=False, compression="gzip")
    direction_results.to_csv(direction_output_path, index=False, compression="gzip")
    direction_pair_results.to_csv(direction_pair_output_path, index=False, compression="gzip")

    vital_rate_summary = {
        "Vital Rate": name,
        "Parameter Draws": num_parameters,
        "Realizations Per Parameter Draw": realizations,
        "Total KS Tests": num_parameters,
        "Number Of Locations": num_locations,
        "Mean Smooth Modes": results["Smooth Modes"].mean(),
        "Median Smooth Modes": results["Smooth Modes"].median(),
        "Minimum Smooth Modes": results["Smooth Modes"].min(),
        "Maximum Smooth Modes": results["Smooth Modes"].max(),
        "Mean Smooth Variance Explained": results["Smooth Variance Explained"].mean(),
    }
    for method in ("DeepRV", "Exact GP"):
        vital_rate_summary.update({
            f"{method} Mean KS Statistic": results[f"{method} KS Statistic"].mean(),
            f"{method} Median KS Statistic": results[f"{method} KS Statistic"].median(),
            f"{method} KS Statistic 5th Percentile": results[f"{method} KS Statistic"].quantile(0.05),
            f"{method} KS Statistic 95th Percentile": results[f"{method} KS Statistic"].quantile(0.95),
            f"{method} Mean KS P-Value": results[f"{method} KS P-Value"].mean(),
            f"{method} Median KS P-Value": results[f"{method} KS P-Value"].median(),
            f"{method} KS P-Value 5th Percentile": results[f"{method} KS P-Value"].quantile(0.05),
            f"{method} KS P-Value 95th Percentile": results[f"{method} KS P-Value"].quantile(0.95),
            f"{method} Rejection Rate At 0.05": results[f"{method} Reject At 0.05"].mean(),
            f"{method} Rejection Rate At 0.01": results[f"{method} Reject At 0.01"].mean(),
            f"{method} Mean Absolute Direction Mean": direction_results[f"{method} Whitened Mean"].abs().mean(),
            f"{method} Mean Direction Variance": direction_results[f"{method} Whitened Variance"].mean(),
            f"{method} Mean Absolute Direction Variance Error": (direction_results[f"{method} Whitened Variance"] - 1.0).abs().mean(),
            f"{method} Mean Absolute Between-Direction Covariance": direction_pair_results[f"{method} Whitened Covariance"].abs().mean(),
            f"{method} Mean Absolute Between-Direction Correlation": direction_pair_results[f"{method} Whitened Correlation"].abs().mean(),
        })
    print(f"Saved {output_path}", flush=True)
    print(f"Saved {direction_output_path}", flush=True)
    print(f"Saved {direction_pair_output_path}", flush=True)
    return results, direction_results, direction_pair_results, vital_rate_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=GENERATED / 'soay/size_only/training/transformer/model.ckpt')
    parser.add_argument("--data", type=Path, default=GENERATED / 'soay/size_only/data/gp_ibm_dataset.csv')
    parser.add_argument("--samples", type=Path, default=GENERATED / 'soay/size_only/inference/gpdf_gp_samples.npz')
    parser.add_argument("--output-dir", type=Path, default=GENERATED / 'soay/size_only/validation/ks/smooth_control')
    parser.add_argument("--realizations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.realizations != 50:
        raise ValueError("This evaluation requires exactly 50 realizations per parameter draw")
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    expected_outputs = [output_directory / f"ks_smooth_control_{settings['filename']}.csv.gz" for settings in VITAL_RATES.values()]
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"Refusing to overwrite existing smooth-subspace results in {output_directory}")

    print("JAX Devices:", jax.devices(), flush=True)
    print("Output Directory:", output_directory, flush=True)
    print("Smooth Mode Rule: kernel eigenvalue > jitter", flush=True)
    print("Jitter:", JITTER, flush=True)
    locations, parameter_draws = load_inputs(args.data.resolve(), args.samples.resolve())
    decoder = DeepRVDecoder(args.checkpoint.resolve())
    all_results = []
    all_direction_results = []
    all_direction_pair_results = []
    vital_rate_summaries = []

    for rate_index, name in enumerate(VITAL_RATES):
        print(f"\nEvaluating {name} At {len(locations[name])} Locations", flush=True)
        deeprv_seed, exact_seed = np.random.SeedSequence([args.seed, rate_index]).spawn(2)
        deeprv_rng = np.random.default_rng(deeprv_seed)
        exact_rng = np.random.default_rng(exact_seed)
        results, direction_results, direction_pair_results, vital_rate_summary = evaluate_vital_rate(
            name, locations[name], parameter_draws[name], decoder, args.realizations, deeprv_rng, exact_rng, output_directory
        )
        all_results.append(results)
        all_direction_results.append(direction_results)
        all_direction_pair_results.append(direction_pair_results)
        vital_rate_summaries.append(vital_rate_summary)
        pd.concat(all_results, ignore_index=True).to_csv(output_directory / "ks_smooth_control_parameter_draw_results.csv", index=False)
        pd.concat(all_direction_results, ignore_index=True).to_csv(output_directory / "ks_smooth_control_direction_diagnostics.csv.gz", index=False, compression="gzip")
        pd.concat(all_direction_pair_results, ignore_index=True).to_csv(output_directory / "ks_smooth_control_direction_pair_diagnostics.csv.gz", index=False, compression="gzip")
        pd.DataFrame(vital_rate_summaries).to_csv(output_directory / "ks_smooth_control_vital_rate_summary.csv", index=False)

    print("\nEvaluation Complete", flush=True)
    print(pd.DataFrame(vital_rate_summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
