"""Evaluate the age-size DeepRV Transformer in smooth GP covariance directions."""

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
from scipy.special import ndtr
from scipy.stats import kstwo

REPO_ROOT = PROJECT_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.soay.age_size.abc.deeprv.deeprv_decoder_agesize import get_runtime


VITAL_RATES = {
    "Survival": {"prefix": "surv", "filename": "survival"},
    "Reproduction": {"prefix": "repr", "filename": "reproduction"},
    "Growth": {"prefix": "growth", "filename": "growth"},
}


def standard_normal_ks(samples):
    """Perform one two-sided KS test against the fixed standard-normal CDF."""

    ordered = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    num_values = len(ordered)
    cdf = ndtr(ordered)
    upper_ranks = np.arange(1, num_values + 1, dtype=np.float64) / num_values
    lower_ranks = np.arange(num_values, dtype=np.float64) / num_values
    statistic = max(np.max(upper_ranks - cdf), np.max(cdf - lower_ranks))
    p_value = np.clip(kstwo.sf(statistic, num_values), 0.0, 1.0)
    return float(statistic), float(p_value)


def load_inputs(data_path, samples_path):
    """Load year-specific locations and paired posterior ARD lengthscales."""

    data = pd.read_csv(data_path)
    required_columns = {"z", "age", "Surv", "Repr", "z1", "yr"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing data columns: {missing_columns}")

    masks = {
        "Survival": data[["z", "age", "Surv"]].notna().all(axis=1),
        "Reproduction": data[["z", "age", "Repr"]].notna().all(axis=1) & data["age"].ge(1),
        "Growth": data[["z", "age", "z1"]].notna().all(axis=1),
    }
    locations = {}
    for name, mask in masks.items():
        locations[name] = {}
        for year in sorted(data.loc[mask, "yr"].dropna().unique()):
            year_mask = mask & data["yr"].eq(year)
            values = data.loc[year_mask, ["z", "age"]].to_numpy(dtype=np.float64)
            if len(values):
                locations[name][int(year)] = values

    parameter_draws = {}
    with np.load(samples_path) as saved:
        for name, settings in VITAL_RATES.items():
            key = f"{settings['prefix']}_ls"
            if key not in saved.files:
                raise ValueError(f"Missing posterior array: {key}")
            lengthscales = np.asarray(saved[key], dtype=np.float64)
            if lengthscales.shape != (5000, 2):
                raise ValueError(f"{key} must have shape (5000, 2), received {lengthscales.shape}")
            if not np.all(np.isfinite(lengthscales)) or np.any(lengthscales <= 0):
                raise ValueError(f"{key} must contain finite positive lengthscales")
            parameter_draws[name] = lengthscales

    return locations, parameter_draws


def prepare_year_context(year, locations, max_locations, realizations):
    """Precompute location-only quantities reused across parameter draws."""

    num_locations = len(locations)
    if num_locations > max_locations:
        raise ValueError(f"Year {year} has {num_locations} locations but the decoder supports {max_locations}")

    padded_locations = np.zeros((max_locations, 2), dtype=np.float32)
    padded_locations[:num_locations] = locations.astype(np.float32)
    active_differences = locations[:, None, :] - locations[None, :, :]
    padded_differences = padded_locations[:, None, :] - padded_locations[None, :, :]
    mask = np.arange(max_locations) < num_locations
    return {
        "year": year,
        "locations": locations,
        "num_locations": num_locations,
        "padded_locations": padded_locations,
        "active_squared_differences": active_differences**2,
        "padded_squared_differences": padded_differences**2,
        "active_identity": np.eye(num_locations, dtype=np.float64),
        "padded_identity": np.eye(max_locations, dtype=np.float32),
        "batched_mask": np.repeat(mask[None], realizations, axis=0),
    }


def exact_result_columns(statistic=np.nan, p_value=np.nan):
    """Return exact-control fields while preserving a fixed output schema."""

    return {
        "Exact GP KS Statistic": statistic,
        "Exact GP KS P-Value": p_value,
        "Exact GP Reject At 0.05": p_value < 0.05 if np.isfinite(p_value) else np.nan,
        "Exact GP Reject At 0.01": p_value < 0.01 if np.isfinite(p_value) else np.nan,
    }


def summarize_vital_rate(name, results, num_parameters, realizations, compute_exact):
    """Summarize pooled KS results for one vital rate."""

    summary = {
        "Vital Rate": name,
        "Parameter Draws": num_parameters,
        "Available Years": results["Year"].nunique(),
        "Realizations Per Parameter-Year": realizations,
        "Total KS Tests": len(results),
        "Minimum Number Of Locations": results["Number Of Locations"].min(),
        "Maximum Number Of Locations": results["Number Of Locations"].max(),
        "Mean Smooth Directions": results["Smooth Directions"].mean(),
        "Median Smooth Directions": results["Smooth Directions"].median(),
        "Mean Smooth Variance Explained": results["Smooth Variance Explained"].mean(),
        "Exact GP Computed": compute_exact,
    }
    for method in ("DeepRV", "Exact GP"):
        statistics = pd.to_numeric(results[f"{method} KS Statistic"], errors="coerce")
        p_values = pd.to_numeric(results[f"{method} KS P-Value"], errors="coerce")
        if not p_values.notna().any():
            summary.update({
                f"{method} Mean KS Statistic": np.nan,
                f"{method} Median KS Statistic": np.nan,
                f"{method} Mean KS P-Value": np.nan,
                f"{method} Median KS P-Value": np.nan,
                f"{method} Rejection Rate At 0.05": np.nan,
                f"{method} Rejection Rate At 0.01": np.nan,
            })
            continue
        summary.update({
            f"{method} Mean KS Statistic": statistics.mean(),
            f"{method} Median KS Statistic": statistics.median(),
            f"{method} Mean KS P-Value": p_values.mean(),
            f"{method} Median KS P-Value": p_values.median(),
            f"{method} Rejection Rate At 0.05": (p_values < 0.05).mean(),
            f"{method} Rejection Rate At 0.01": (p_values < 0.01).mean(),
        })
    return summary


def evaluate_vital_rate(name, yearly_locations, lengthscales, decoder, realizations, deeprv_rng, exact_rng, output_directory, compute_exact):
    """Evaluate all paired posterior lengthscales for every available year."""

    contexts = [
        prepare_year_context(year, locations, decoder.max_locations, realizations)
        for year, locations in yearly_locations.items()
    ]
    if not contexts:
        raise ValueError(f"{name} has no year-specific location sets")

    rows = []
    started = time.perf_counter()
    for parameter_index, lengthscale_pair in enumerate(lengthscales):
        size_lengthscale, age_lengthscale = lengthscale_pair
        inverse_squared_lengthscales = 1.0 / lengthscale_pair**2

        for context in contexts:
            active_exponent = np.sum(context["active_squared_differences"] * inverse_squared_lengthscales, axis=-1)
            smooth_covariance = np.exp(-0.5 * active_exponent)
            sampling_covariance = smooth_covariance + decoder.jitter * context["active_identity"]
            eigenvalues, eigenvectors = eigh(smooth_covariance, overwrite_a=True, check_finite=False)
            eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
            eigenvectors = eigenvectors[:, ::-1]
            smooth_mask = eigenvalues > decoder.jitter
            smooth_eigenvalues = eigenvalues[smooth_mask]
            smooth_eigenvectors = eigenvectors[:, smooth_mask]
            num_smooth_directions = len(smooth_eigenvalues)
            if num_smooth_directions == 0:
                raise ValueError(f"{name} parameter draw {parameter_index}, year {context['year']} has no eigenvalue above jitter")

            num_locations = context["num_locations"]
            deeprv_latent = deeprv_rng.standard_normal((realizations, num_locations)).astype(np.float32)
            padded_latent = np.zeros((realizations, decoder.max_locations), dtype=np.float32)
            padded_latent[:, :num_locations] = deeprv_latent
            padded_exponent = np.sum(context["padded_squared_differences"] * inverse_squared_lengthscales.astype(np.float32), axis=-1)
            padded_covariance = np.exp(-0.5 * padded_exponent).astype(np.float32)
            padded_covariance += decoder.jitter * context["padded_identity"]

            unit_draws = decoder.decode(
                jnp.asarray(padded_latent),
                jnp.asarray(lengthscale_pair, dtype=jnp.float32),
                jnp.asarray(context["padded_locations"]),
                jnp.asarray(padded_covariance),
                jnp.asarray(context["batched_mask"]),
            )[:, :num_locations, 0]
            unit_draws.block_until_ready()
            deeprv_draws = np.asarray(jax.device_get(unit_draws), dtype=np.float64)

            standard_deviations = np.sqrt(smooth_eigenvalues + decoder.jitter)[None]
            deeprv_whitened = (deeprv_draws @ smooth_eigenvectors) / standard_deviations
            deeprv_statistic, deeprv_p_value = standard_normal_ks(deeprv_whitened)

            exact_fields = exact_result_columns()
            if compute_exact:
                lower_cholesky = cholesky(sampling_covariance, lower=True, overwrite_a=True, check_finite=False)
                exact_latent = exact_rng.standard_normal((realizations, num_locations))
                exact_draws = exact_latent @ lower_cholesky.T
                exact_whitened = (exact_draws @ smooth_eigenvectors) / standard_deviations
                exact_statistic, exact_p_value = standard_normal_ks(exact_whitened)
                exact_fields = exact_result_columns(exact_statistic, exact_p_value)

            rows.append({
                "Vital Rate": name,
                "Parameter Draw": parameter_index,
                "Year": context["year"],
                "Size Lengthscale": float(size_lengthscale),
                "Age Lengthscale": float(age_lengthscale),
                "Number Of Locations": num_locations,
                "Smooth Directions": num_smooth_directions,
                "Pooled Coefficients": realizations * num_smooth_directions,
                "Smooth Variance Explained": smooth_eigenvalues.sum() / eigenvalues.sum(),
                "DeepRV KS Statistic": deeprv_statistic,
                "DeepRV KS P-Value": deeprv_p_value,
                "DeepRV Reject At 0.05": deeprv_p_value < 0.05,
                "DeepRV Reject At 0.01": deeprv_p_value < 0.01,
                **exact_fields,
            })

        if (parameter_index + 1) % 100 == 0 or parameter_index + 1 == len(lengthscales):
            elapsed_hours = (time.perf_counter() - started) / 3600
            print(f"{name}: {parameter_index + 1:,}/{len(lengthscales):,} Parameter Draws ({elapsed_hours:.2f} Hours)", flush=True)

    results = pd.DataFrame(rows)
    output_path = output_directory / f"ks_smooth_control_{VITAL_RATES[name]['filename']}.csv.gz"
    results.to_csv(output_path, index=False, compression="gzip")
    summary = summarize_vital_rate(name, results, len(lengthscales), realizations, compute_exact)
    print(f"Saved {output_path}", flush=True)
    return results, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=GENERATED / 'soay/age_size/training/transformer/model.ckpt')
    parser.add_argument("--data", type=Path, default=GENERATED / 'soay/age_size/data/gp_ibm_dataset_agesize.csv')
    parser.add_argument("--samples", type=Path, default=GENERATED / 'soay/age_size/inference/gpdf_agesize_gp_samples.npz')
    parser.add_argument("--output-dir", type=Path, default=GENERATED / 'soay/age_size/validation/ks/smooth_control')
    parser.add_argument("--realizations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--skip-exact-control", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.realizations != 50:
        raise ValueError("This evaluation requires exactly 50 realizations per parameter-year test")

    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    expected_outputs = [output_directory / f"ks_smooth_control_{settings['filename']}.csv.gz" for settings in VITAL_RATES.values()]
    expected_outputs.extend([
        output_directory / "ks_smooth_control_parameter_year_results.csv",
        output_directory / "ks_smooth_control_vital_rate_summary.csv",
    ])
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"Refusing to overwrite existing age-size smooth-subspace results in {output_directory}")

    compute_exact = not args.skip_exact_control
    print("JAX Devices:", jax.devices(), flush=True)
    print("Output Directory:", output_directory, flush=True)
    print("Smooth Direction Rule: kernel eigenvalue > checkpoint jitter", flush=True)
    print("Exact GP Control:", "Enabled" if compute_exact else "Skipped", flush=True)

    yearly_locations, parameter_draws = load_inputs(args.data.resolve(), args.samples.resolve())
    decoder = get_runtime(args.checkpoint.resolve())
    print("Maximum Locations:", decoder.max_locations, flush=True)
    print("Jitter:", decoder.jitter, flush=True)

    all_results = []
    summaries = []
    for rate_index, name in enumerate(VITAL_RATES):
        years = yearly_locations[name]
        counts = {year: len(values) for year, values in years.items()}
        print(f"\nEvaluating {name} Across Years: {counts}", flush=True)
        deeprv_seed, exact_seed = np.random.SeedSequence([args.seed, rate_index]).spawn(2)
        deeprv_rng = np.random.default_rng(deeprv_seed)
        exact_rng = np.random.default_rng(exact_seed)
        results, summary = evaluate_vital_rate(
            name,
            years,
            parameter_draws[name],
            decoder,
            args.realizations,
            deeprv_rng,
            exact_rng,
            output_directory,
            compute_exact,
        )
        all_results.append(results)
        summaries.append(summary)
        pd.concat(all_results, ignore_index=True).to_csv(output_directory / "ks_smooth_control_parameter_year_results.csv", index=False)
        pd.DataFrame(summaries).to_csv(output_directory / "ks_smooth_control_vital_rate_summary.csv", index=False)

    print("\nEvaluation Complete", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
