"""JAX experiment comparing exact-GP and DeepRV rejection ABC.

The experiment infers an RBF lengthscale from a Bernoulli observation generated
on 64 fixed locations. DeepRV is trained on paired ``(z, L_ell @ z)`` draws and
then replaces the exact GP draw inside rejection ABC.
"""

from __future__ import annotations

from experiments.paths import GENERATED, PROJECT_ROOT

import argparse
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import optax
import wandb
from dl4bi_sps.kernels import rbf
from jax import Array, jit, random
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import (
    Callback,
    TrainState,
    cosine_annealing_lr,
    evaluate,
    load_ckpt,
    save_ckpt,
    train,
)
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the complete DeepRV ABC experiment."""

    seed: int = 42
    num_locations: int = 64
    domain_start: float = 0.0
    domain_stop: float = 10.0
    variance: float = 1.0
    jitter: float = 5e-4
    min_lengthscale: float = 0.0
    max_lengthscale: float = 1.0
    true_lengthscale: float = 0.5
    num_blocks: int = 2
    train_batch_size: int = 32
    train_steps: int = 100_000
    checkpoint_steps: tuple[int, ...] = (
        1_000,
        5_000,
        10_000,
        25_000,
        50_000,
        100_000,
    )
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    gradient_clip: float = 3.0
    valid_interval: int = 10_000
    valid_steps: int = 5_000
    test_steps: int = 5_000
    calibration_datasets: int = 10_000
    threshold_quantile: float = 0.10
    abc_batch_size: int = 512
    target_accepted: int = 1_000
    max_proposals: int = 1_000_000
    output_dir: str = str(GENERATED / 'bernoulli_abc/job_68372_dense_checkpoints')


@dataclass
class ABCResult:
    """Outputs from one rejection ABC run."""

    accepted: np.ndarray
    proposals: np.ndarray
    distances: np.ndarray
    proposals_processed: int
    accepted_in_processed_batches: int
    runtime_seconds: float


def validate_config(config: ExperimentConfig) -> None:
    """Reject invalid configurations before compilation begins."""
    if config.num_locations < 2:
        raise ValueError("num_locations must be at least two")
    if not config.min_lengthscale <= config.true_lengthscale <= config.max_lengthscale:
        raise ValueError("true_lengthscale must lie inside the ABC prior")
    if not 0.0 < config.threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must lie between zero and one")
    positive_values = {
        "num_blocks": config.num_blocks,
        "train_batch_size": config.train_batch_size,
        "train_steps": config.train_steps,
        "valid_interval": config.valid_interval,
        "valid_steps": config.valid_steps,
        "test_steps": config.test_steps,
        "calibration_datasets": config.calibration_datasets,
        "abc_batch_size": config.abc_batch_size,
        "target_accepted": config.target_accepted,
        "max_proposals": config.max_proposals,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if any(step <= 0 for step in config.checkpoint_steps):
        raise ValueError("checkpoint_steps must contain positive values")


def active_checkpoint_steps(config: ExperimentConfig) -> tuple[int, ...]:
    """Return requested checkpoints up to and including the final train step."""
    steps = {step for step in config.checkpoint_steps if step <= config.train_steps}
    steps.add(config.train_steps)
    return tuple(sorted(steps))


def build_locations(config: ExperimentConfig) -> Array:
    """Return the fixed locations with shape ``(num_locations, 1)``."""
    return jnp.linspace(
        config.domain_start,
        config.domain_stop,
        config.num_locations,
        dtype=jnp.float32,
    )[:, None]


def rbf_covariance(
    locations: Array,
    lengthscale: Array | float,
    variance: float = 1.0,
    jitter: float = 5e-4,
) -> Array:
    """Build a numerically safe RBF covariance matrix."""
    values = locations.squeeze(-1)
    squared_distance = (values[:, None] - values[None, :]) ** 2
    safe_lengthscale = jnp.maximum(
        jnp.asarray(lengthscale, dtype=values.dtype), jnp.finfo(values.dtype).eps
    )
    covariance = variance * jnp.exp(-0.5 * squared_distance / safe_lengthscale**2)
    return covariance + jitter * jnp.eye(values.size, dtype=values.dtype)


def sample_observed_data(
    rng: Array, locations: Array, config: ExperimentConfig
) -> dict[str, Array]:
    """Generate the fixed observed Bernoulli dataset from the exact GP."""
    rng_z, rng_u = random.split(rng)
    covariance = rbf_covariance(
        locations, config.true_lengthscale, config.variance, config.jitter
    )
    cholesky = jnp.linalg.cholesky(covariance)
    z = random.normal(rng_z, (config.num_locations,))
    latent = cholesky @ z
    probability = jax.nn.sigmoid(latent)
    uniforms = random.uniform(rng_u, (config.num_locations,))
    observations = (uniforms < probability).astype(jnp.int8)
    return {
        "locations": locations.squeeze(-1),
        "z": z,
        "latent": latent,
        "probability": probability,
        "observations": observations,
    }


def transition_statistic(observations: Array) -> Array:
    """Calculate the proportion of adjacent Bernoulli values that differ."""
    values = observations.astype(jnp.float32)
    return jnp.mean(jnp.diff(values, axis=-1) ** 2, axis=-1)


def summary_statistic(observations: Array) -> Array:
    """Return proportions of ones and adjacent transitions."""
    values = observations.astype(jnp.float32)
    return jnp.stack((jnp.mean(values, axis=-1), transition_statistic(values)), axis=-1)


def summary_distance(simulated: Array, observed_statistic: Array) -> Array:
    """Calculate Euclidean distance between two-component summary statistics."""
    difference = summary_statistic(simulated) - observed_statistic
    return jnp.linalg.norm(difference, axis=-1)


def make_gp_dataloader(
    locations: Array, config: ExperimentConfig
) -> Callable[[Array], Iterator[dict[str, Array]]]:
    """Create the on-the-fly paired GP dataloader used by the paper scheme."""
    jitter_matrix = config.jitter * jnp.eye(config.num_locations)
    kernel = jit(
        lambda s, variance, lengthscale: rbf(s, s, variance, lengthscale)
        + jitter_matrix
    )
    sample_latent = jit(lambda cholesky, z: jnp.einsum("ij,bj->bi", cholesky, z))
    lengthscale_prior = dist.Uniform(config.min_lengthscale, config.max_lengthscale)
    standard_normal = dist.Normal()

    def dataloader(rng: Array) -> Iterator[dict[str, Array]]:
        while True:
            rng, rng_lengthscale, rng_z = random.split(rng, 3)
            lengthscale = lengthscale_prior.sample(rng_lengthscale)
            z = standard_normal.sample(
                rng_z,
                sample_shape=(config.train_batch_size, config.num_locations),
            )
            covariance = kernel(locations, config.variance, lengthscale)
            latent = sample_latent(jnp.linalg.cholesky(covariance), z)
            yield {
                "s": locations,
                "z": z,
                "conditionals": jnp.array([lengthscale]),
                "f": latent,
            }

    return dataloader


@jit
def validation_step(rng: Array, state: TrainState, batch: dict[str, Array]) -> dict:
    """Evaluate DeepRV using the paper's normalized-MSE criterion."""
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        rngs={"extra": rng},
    )
    return {"norm MSE": output.metrics(batch["f"], 1.0)["MSE"]}


def build_checkpoint_config(config: ExperimentConfig) -> DictConfig:
    """Build the minimal Hydra configuration needed to reload DeepRV."""
    return OmegaConf.create(
        {
            "model": {
                "_target_": "dl4bi.vae.gMLPDeepRV",
                "num_blks": config.num_blocks,
            }
        }
    )


def checkpoint_path(output_dir: Path, step: int) -> Path:
    """Return the checkpoint directory for one training step."""
    return output_dir / "checkpoints" / f"step_{step:06d}.ckpt"


def train_deeprv(
    rng: Array,
    locations: Array,
    config: ExperimentConfig,
    output_dir: Path,
) -> tuple[dict[int, TrainState], gMLPDeepRV, dict]:
    """Train conditional DeepRV with the established project framework."""
    rng_train, rng_test = random.split(rng)
    dataloader = make_gp_dataloader(locations, config)
    model = gMLPDeepRV(num_blks=config.num_blocks)
    schedule = cosine_annealing_lr(config.train_steps, config.learning_rate)
    optimizer = optax.adamw(schedule, weight_decay=config.weight_decay)
    optimizer = optax.chain(optax.clip_by_global_norm(config.gradient_clip), optimizer)
    steps = active_checkpoint_steps(config)
    step_set = set(steps)
    states: dict[int, TrainState] = {}
    saved_config = build_checkpoint_config(config)
    checkpoint_interval = math.gcd(*steps)

    def save_training_checkpoint(
        step: int,
        rng_step: Array,
        state: TrainState,
        batch: dict[str, Array],
        extra: None,
    ) -> None:
        del rng_step, batch, extra
        if step in step_set:
            states[step] = state
            save_ckpt(state, saved_config, checkpoint_path(output_dir, step))

    wandb.init(project="deeprv-abc-test", mode="disabled")
    started = time.perf_counter()
    final_state = train(
        rng_train,
        model,
        optimizer,
        deep_rv_train_step,
        config.train_steps,
        dataloader,
        validation_step,
        config.valid_interval,
        config.valid_steps,
        dataloader,
        callbacks=[Callback(save_training_checkpoint, checkpoint_interval)],
        return_state="last",
        valid_monitor_metric="norm MSE",
    )
    training_seconds = time.perf_counter() - started
    missing_steps = step_set.difference(states)
    if missing_steps:
        raise RuntimeError(
            f"Training did not save checkpoints: {sorted(missing_steps)}"
        )
    save_ckpt(final_state, saved_config, output_dir / "model.ckpt")
    test_norm_mse = {
        str(step): float(
            evaluate(
                rng_test,
                states[step],
                validation_step,
                dataloader,
                config.test_steps,
            )["norm MSE"]
        )
        for step in steps
    }
    wandb.finish()
    metrics = {
        "training_seconds": training_seconds,
        "checkpoint_test_norm_mse": test_norm_mse,
    }
    return states, model, metrics


def load_checkpoint_states(
    output_dir: Path, config: ExperimentConfig
) -> tuple[dict[int, TrainState], gMLPDeepRV]:
    """Load every requested checkpoint for ABC without retraining."""
    states = {}
    model = gMLPDeepRV(num_blks=config.num_blocks)
    for step in active_checkpoint_steps(config):
        path = checkpoint_path(output_dir, step)
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        state, saved_config = load_ckpt(path)
        states[step] = state
        model = gMLPDeepRV(num_blks=saved_config.model.num_blks)
    return states, model


def make_exact_distance_fn(
    locations: Array, observed_statistic: Array, config: ExperimentConfig
) -> Callable[[Array, Array, Array], Array]:
    """Build the JIT-compiled exact-GP ABC distance function."""

    def distance(lengthscales: Array, z: Array, uniforms: Array) -> Array:
        covariances = jax.vmap(
            lambda lengthscale: rbf_covariance(
                locations, lengthscale, config.variance, config.jitter
            )
        )(lengthscales)
        cholesky = jnp.linalg.cholesky(covariances)
        latent = jnp.einsum("bij,bj->bi", cholesky, z)
        simulated = uniforms < jax.nn.sigmoid(latent)
        return summary_distance(simulated, observed_statistic)

    return jax.jit(distance)


def make_deeprv_distance_fn(
    state: TrainState,
    model: gMLPDeepRV,
    locations: Array,
    observed_statistic: Array,
) -> Callable[[Array, Array, Array], Array]:
    """Build the JIT-compiled DeepRV ABC distance function."""
    variables = {"params": state.params, **state.kwargs}

    def decode_one(z: Array, lengthscale: Array) -> Array:
        output = model.apply(
            variables,
            z[None, :],
            jnp.array([lengthscale]),
            s=locations,
            method="decode",
        )
        return output.squeeze()

    def distance(lengthscales: Array, z: Array, uniforms: Array) -> Array:
        latent = jax.vmap(decode_one)(z, lengthscales)
        simulated = uniforms < jax.nn.sigmoid(latent)
        return summary_distance(simulated, observed_statistic)

    return jax.jit(distance)


def calibrate_threshold(
    rng: Array,
    locations: Array,
    observed_statistic: Array,
    config: ExperimentConfig,
) -> tuple[float, np.ndarray]:
    """Calibrate epsilon from exact datasets at the true lengthscale."""
    covariance = rbf_covariance(
        locations, config.true_lengthscale, config.variance, config.jitter
    )
    cholesky = jnp.linalg.cholesky(covariance)

    @jax.jit
    def distance_batch(z: Array, uniforms: Array) -> Array:
        latent = jnp.einsum("ij,bj->bi", cholesky, z)
        simulated = uniforms < jax.nn.sigmoid(latent)
        return summary_distance(simulated, observed_statistic)

    distance_batches = []
    generated = 0
    while generated < config.calibration_datasets:
        current_batch = min(
            config.abc_batch_size, config.calibration_datasets - generated
        )
        rng, rng_z, rng_u = random.split(rng, 3)
        z = random.normal(rng_z, (current_batch, config.num_locations))
        uniforms = random.uniform(rng_u, (current_batch, config.num_locations))
        distances = distance_batch(z, uniforms)
        distance_batches.append(np.asarray(jax.device_get(distances)))
        generated += current_batch
    calibration_distances = np.concatenate(distance_batches)
    threshold = float(
        jnp.quantile(jnp.asarray(calibration_distances), config.threshold_quantile)
    )
    return threshold, calibration_distances


def sample_abc_inputs(
    rng: Array, batch_size: int, config: ExperimentConfig
) -> tuple[Array, Array, Array]:
    """Draw one JAX batch from the ABC and DeepRV input priors."""
    rng_lengthscale, rng_z, rng_u = random.split(rng, 3)
    lengthscales = random.uniform(
        rng_lengthscale,
        (batch_size,),
        minval=config.min_lengthscale,
        maxval=config.max_lengthscale,
    )
    z = random.normal(rng_z, (batch_size, config.num_locations))
    uniforms = random.uniform(rng_u, (batch_size, config.num_locations))
    return lengthscales, z, uniforms


def run_rejection_abc(
    rng: Array,
    threshold: float,
    distance_fn: Callable[[Array, Array, Array], Array],
    config: ExperimentConfig,
) -> ABCResult:
    """Run batched rejection ABC until the accepted-sample target is met."""
    warmup_rng = random.fold_in(rng, 2_147_483_647)
    warmup_inputs = sample_abc_inputs(
        warmup_rng, min(config.abc_batch_size, config.max_proposals), config
    )
    jax.block_until_ready(distance_fn(*warmup_inputs))

    accepted_batches = []
    proposal_batches = []
    distance_batches = []
    accepted_total = 0
    proposals_processed = 0
    started = time.perf_counter()
    while (
        accepted_total < config.target_accepted
        and proposals_processed < config.max_proposals
    ):
        current_batch = min(
            config.abc_batch_size, config.max_proposals - proposals_processed
        )
        rng, rng_batch = random.split(rng)
        lengthscales, z, uniforms = sample_abc_inputs(rng_batch, current_batch, config)
        distances = distance_fn(lengthscales, z, uniforms)
        jax.block_until_ready(distances)
        lengthscales_host = np.asarray(jax.device_get(lengthscales))
        distances_host = np.asarray(jax.device_get(distances))
        accepted_mask = distances_host <= threshold
        accepted_batches.append(lengthscales_host[accepted_mask])
        proposal_batches.append(lengthscales_host)
        distance_batches.append(distances_host)
        accepted_total += int(accepted_mask.sum())
        proposals_processed += current_batch
    runtime_seconds = time.perf_counter() - started
    accepted = (
        np.concatenate(accepted_batches)
        if accepted_batches
        else np.empty(0, dtype=np.float32)
    )
    if accepted.size < config.target_accepted:
        raise RuntimeError(
            f"ABC accepted {accepted.size} samples after "
            f"{proposals_processed} proposals; increase max_proposals"
        )
    return ABCResult(
        accepted=accepted[: config.target_accepted],
        proposals=np.concatenate(proposal_batches),
        distances=np.concatenate(distance_batches),
        proposals_processed=proposals_processed,
        accepted_in_processed_batches=accepted_total,
        runtime_seconds=runtime_seconds,
    )


def posterior_summary(samples: Array, true_lengthscale: float) -> dict[str, float]:
    """Summarize one equally weighted ABC posterior using JAX."""
    values = jnp.asarray(samples)
    quantiles = jnp.quantile(values, jnp.array([0.025, 0.5, 0.975]))
    return {
        "mean": float(jnp.mean(values)),
        "median": float(quantiles[1]),
        "standard_deviation": float(jnp.std(values)),
        "lower_95": float(quantiles[0]),
        "upper_95": float(quantiles[2]),
        "rmse": float(jnp.sqrt(jnp.mean((values - true_lengthscale) ** 2))),
    }


def wasserstein_distance_equal_samples(first: Array, second: Array) -> float:
    """Calculate one-dimensional Wasserstein distance for equal sample counts."""
    first_sorted = jnp.sort(jnp.asarray(first))
    second_sorted = jnp.sort(jnp.asarray(second))
    if first_sorted.shape != second_sorted.shape:
        raise ValueError("Wasserstein inputs must contain the same number of samples")
    return float(jnp.mean(jnp.abs(first_sorted - second_sorted)))


def abc_result_summary(
    result: ABCResult, true_lengthscale: float
) -> dict[str, float | int]:
    """Combine posterior and rejection-sampler summaries."""
    return {
        **posterior_summary(result.accepted, true_lengthscale),
        "proposals_processed": result.proposals_processed,
        "accepted_in_processed_batches": result.accepted_in_processed_batches,
        "acceptance_rate": (
            result.accepted_in_processed_batches / result.proposals_processed
        ),
        "runtime_seconds": result.runtime_seconds,
    }


def checkpoint_abc_summary(
    exact_result: ABCResult,
    deeprv_result: ABCResult,
    threshold: float,
    true_lengthscale: float,
    test_norm_mse: float,
) -> dict[str, float | int]:
    """Summarize one DeepRV checkpoint relative to exact-GP ABC."""
    exact_summary = posterior_summary(exact_result.accepted, true_lengthscale)
    deeprv_summary = abc_result_summary(deeprv_result, true_lengthscale)
    shared = min(exact_result.distances.size, deeprv_result.distances.size)
    exact_distances = exact_result.distances[:shared]
    deeprv_distances = deeprv_result.distances[:shared]
    acceptance_disagreement = np.mean(
        (exact_distances <= threshold) != (deeprv_distances <= threshold)
    )
    return {
        "test_norm_mse": test_norm_mse,
        **deeprv_summary,
        "posterior_mean_difference_from_exact": (
            deeprv_summary["mean"] - exact_summary["mean"]
        ),
        "posterior_median_difference_from_exact": (
            deeprv_summary["median"] - exact_summary["median"]
        ),
        "posterior_wasserstein_distance_from_exact": (
            wasserstein_distance_equal_samples(
                exact_result.accepted, deeprv_result.accepted
            )
        ),
        "acceptance_rate_difference_from_exact": (
            deeprv_summary["acceptance_rate"]
            - exact_result.accepted_in_processed_batches
            / exact_result.proposals_processed
        ),
        "acceptance_disagreement_rate": float(acceptance_disagreement),
        "mean_absolute_distance_error": float(
            np.mean(np.abs(deeprv_distances - exact_distances))
        ),
        "shared_proposals_compared": shared,
    }


def plot_posteriors(
    exact_samples: np.ndarray,
    deeprv_samples: dict[int, np.ndarray],
    true_lengthscale: float,
    output_dir: Path,
) -> None:
    """Save exact-versus-checkpoint ABC posterior comparisons."""
    plt.style.use("seaborn-v0_8-white")
    steps = sorted(deeprv_samples)
    columns = min(3, len(steps))
    rows = math.ceil(len(steps) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.6 * columns, 3.0 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    bins = np.linspace(0.0, 1.0, 31)
    for axis, step in zip(axes.flat, steps):
        axis.hist(
            exact_samples,
            bins=bins,
            density=True,
            histtype="step",
            color="#0072B2",
            linewidth=1.4,
            alpha=0.85,
            label="Exact GP ABC",
        )
        axis.hist(
            deeprv_samples[step],
            bins=bins,
            density=True,
            histtype="step",
            color="#D55E00",
            linewidth=1.4,
            alpha=0.85,
            label="DeepRV ABC",
        )
        axis.axvline(
            true_lengthscale,
            color="#009E73",
            linewidth=1.4,
            linestyle="--",
            label="True Lengthscale",
        )
        axis.set_title(f"{step:,} Training Steps", fontsize=11)
        axis.grid(False)
        axis.tick_params(labelsize=9, direction="out")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for axis in axes.flat[len(steps) :]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.supxlabel("Lengthscale", fontsize=14)
    fig.supylabel("Density", fontsize=14)
    fig.suptitle("ABC Posterior Comparison By DeepRV Checkpoint", fontsize=16)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"posterior_comparison.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_checkpoint_diagnostics(
    checkpoint_summaries: dict[int, dict[str, float | int]], output_dir: Path
) -> None:
    """Plot DeepRV fidelity against checkpoint-level ABC discrepancies."""
    plt.style.use("seaborn-v0_8-white")
    steps = np.asarray(sorted(checkpoint_summaries))
    metrics = (
        ("test_norm_mse", "Normalized MSE", True),
        (
            "posterior_wasserstein_distance_from_exact",
            "Posterior Wasserstein Distance",
            True,
        ),
        (
            "posterior_mean_difference_from_exact",
            "Absolute Posterior Mean Difference",
            False,
        ),
        ("acceptance_disagreement_rate", "Acceptance Disagreement Rate", False),
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 6.0), constrained_layout=True, squeeze=False
    )
    for axis, (key, label, log_y) in zip(axes.flat, metrics):
        values = np.asarray([float(checkpoint_summaries[step][key]) for step in steps])
        if key == "posterior_mean_difference_from_exact":
            values = np.abs(values)
        axis.plot(
            steps,
            values,
            color="#0072B2",
            linewidth=1.4,
            alpha=0.85,
            marker="o",
            markersize=4,
        )
        axis.set_xscale("log")
        if log_y and np.all(values > 0):
            axis.set_yscale("log")
        axis.set_xlabel("Training Steps")
        axis.set_ylabel(label)
        axis.grid(False)
        axis.tick_params(labelsize=9, direction="out")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("DeepRV Approximation And ABC Discrepancy", fontsize=16)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"checkpoint_diagnostics.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_json(path: Path, values: dict) -> None:
    """Write an indented JSON file."""
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def run_experiment(config: ExperimentConfig, skip_training: bool = False) -> dict:
    """Run data generation, training, calibration, and checkpoint ABC analyses."""
    validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", asdict(config))
    root_rng = random.key(config.seed)
    rng_observed, rng_train, rng_calibration, rng_abc = random.split(root_rng, 4)
    locations = build_locations(config)
    observed = sample_observed_data(rng_observed, locations, config)
    observed_statistic = summary_statistic(observed["observations"])
    np.savez_compressed(
        output_dir / "observed_data.npz",
        **{name: np.asarray(jax.device_get(value)) for name, value in observed.items()},
        true_lengthscale=config.true_lengthscale,
        summary_statistic=np.asarray(jax.device_get(observed_statistic)),
        proportion_ones=float(observed_statistic[0]),
        proportion_transitions=float(observed_statistic[1]),
    )

    if skip_training:
        states, model = load_checkpoint_states(output_dir, config)
        metrics_path = output_dir / "training_metrics.json"
        training_history = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        states, model, training_history = train_deeprv(
            rng_train, locations, config, output_dir
        )
        write_json(output_dir / "training_metrics.json", training_history)

    threshold, calibration_distances = calibrate_threshold(
        rng_calibration, locations, observed_statistic, config
    )
    np.savez_compressed(
        output_dir / "threshold_calibration.npz",
        distances=calibration_distances,
        threshold=threshold,
        quantile=config.threshold_quantile,
        true_lengthscale=config.true_lengthscale,
        observed_statistic=np.asarray(jax.device_get(observed_statistic)),
    )

    exact_distance_fn = make_exact_distance_fn(locations, observed_statistic, config)
    exact_result = run_rejection_abc(rng_abc, threshold, exact_distance_fn, config)
    deeprv_results = {}
    for step in active_checkpoint_steps(config):
        deeprv_distance_fn = make_deeprv_distance_fn(
            states[step], model, locations, observed_statistic
        )
        deeprv_results[step] = run_rejection_abc(
            rng_abc, threshold, deeprv_distance_fn, config
        )

    result_arrays = {
        "threshold": threshold,
        "exact_proposals": exact_result.proposals,
        "exact_distances": exact_result.distances,
        "exact_accepted": exact_result.accepted,
    }
    for step, result in deeprv_results.items():
        prefix = f"deeprv_step_{step:06d}"
        result_arrays[f"{prefix}_proposals"] = result.proposals
        result_arrays[f"{prefix}_distances"] = result.distances
        result_arrays[f"{prefix}_accepted"] = result.accepted
    np.savez_compressed(output_dir / "abc_results.npz", **result_arrays)
    plot_posteriors(
        exact_result.accepted,
        {step: result.accepted for step, result in deeprv_results.items()},
        config.true_lengthscale,
        output_dir,
    )

    checkpoint_summaries = {
        step: checkpoint_abc_summary(
            exact_result,
            result,
            threshold,
            config.true_lengthscale,
            float(training_history["checkpoint_test_norm_mse"][str(step)]),
        )
        for step, result in deeprv_results.items()
    }
    plot_checkpoint_diagnostics(checkpoint_summaries, output_dir)
    summary = {
        "threshold": threshold,
        "threshold_quantile": config.threshold_quantile,
        "observed_summary_statistic": {
            "proportion_ones": float(observed_statistic[0]),
            "proportion_transitions": float(observed_statistic[1]),
        },
        "training": training_history,
        "exact_gp_abc": abc_result_summary(exact_result, config.true_lengthscale),
        "deeprv_abc": {
            str(step): values for step, values in checkpoint_summaries.items()
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for long-running settings."""
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-locations", type=int, default=defaults.num_locations)
    parser.add_argument("--domain-stop", type=float, default=defaults.domain_stop)
    parser.add_argument("--num-blocks", type=int, default=defaults.num_blocks)
    parser.add_argument("--train-steps", type=int, default=defaults.train_steps)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=defaults.checkpoint_steps,
    )
    parser.add_argument("--valid-interval", type=int, default=defaults.valid_interval)
    parser.add_argument("--valid-steps", type=int, default=defaults.valid_steps)
    parser.add_argument("--test-steps", type=int, default=defaults.test_steps)
    parser.add_argument(
        "--calibration-datasets", type=int, default=defaults.calibration_datasets
    )
    parser.add_argument(
        "--threshold-quantile", type=float, default=defaults.threshold_quantile
    )
    parser.add_argument("--target-accepted", type=int, default=defaults.target_accepted)
    parser.add_argument("--max-proposals", type=int, default=defaults.max_proposals)
    parser.add_argument("--abc-batch-size", type=int, default=defaults.abc_batch_size)
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the experiment from command-line arguments."""
    args = parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        num_locations=args.num_locations,
        domain_stop=args.domain_stop,
        num_blocks=args.num_blocks,
        train_steps=args.train_steps,
        checkpoint_steps=tuple(args.checkpoint_steps),
        valid_interval=args.valid_interval,
        valid_steps=args.valid_steps,
        test_steps=args.test_steps,
        calibration_datasets=args.calibration_datasets,
        threshold_quantile=args.threshold_quantile,
        target_accepted=args.target_accepted,
        max_proposals=args.max_proposals,
        abc_batch_size=args.abc_batch_size,
        output_dir=args.output_dir,
    )
    summary = run_experiment(config, skip_training=args.skip_training)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
