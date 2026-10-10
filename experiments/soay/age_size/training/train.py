"""Train a Transformer DeepRV for the two-dimensional Soay age-size GPs."""

from experiments.paths import GENERATED, PROJECT_ROOT

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import optax
import wandb
from jax import Array, jit, random, value_and_grad
from omegaconf import DictConfig
from orbax.checkpoint import PyTreeCheckpointer

from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import Callback, TrainState, cosine_annealing_lr, evaluate, save_ckpt, train
from dl4bi.vae import KernelBiasTransformerDeepRV
from dl4bi.vae.train_utils import generate_surrogate_decoder


SOAY_DIRECTORY = PROJECT_ROOT
AGE_VALUES = tuple(range(9))
COORDINATE_ORDER = ("size", "age")


@dataclass(frozen=True)
class TrainConfig:
    data_path: str = str(GENERATED / 'soay/age_size/data/gp_ibm_dataset_agesize.csv')
    output_dir: str = str(GENERATED / 'soay/age_size/training/transformer')
    seed: int = 23
    min_locations: int = 1
    max_locations: int = 512
    age_0_size_mean: float = 2.60
    age_1_size_mean: float = 2.90
    age_2_size_mean: float = 3.00
    age_3_plus_size_mean: float = 3.15
    size_sd: float = 0.20
    size_lengthscale_log_mean: float = 0.0
    age_lengthscale_log_mean: float = 2.0
    lengthscale_log_sd: float = 1.5
    jitter: float = 5e-4
    batch_size: int = 8
    num_steps: int = 2_000_000
    checkpoint_interval: int = 100_000
    validation_interval: int = 400_000
    validation_steps: int = 10_000
    learning_rate: float = 1e-4
    clip_norm: float = 3.0
    dim: int = 128
    num_blocks: int = 4
    num_rff_features: int = 512
    wandb_mode: str = "disabled"


class RFFEmbed(nn.Module):
    """Original DeepRV RFF embedding, applied jointly to size and age."""

    num_features: int = 128
    scale: float = 1.0
    include_original: bool = True
    head: Union[Callable, nn.Module] = lambda x: x

    @nn.compact
    def __call__(self, locations: Array):
        input_dim = locations.shape[-1]
        weights = self.param("rff_W", lambda key: random.normal(key, (self.num_features, input_dim)) * self.scale)
        phases = self.param("rff_b", lambda key: random.uniform(key, (self.num_features,), minval=0.0, maxval=2 * jnp.pi))
        features = jnp.sqrt(2.0 / self.num_features) * jnp.cos(jnp.dot(locations, weights.T) + phases)
        return self.head(jnp.concatenate([locations, features], axis=-1) if self.include_original else features)


@jit
def ard_rbf(x: Array, y: Array, variance: float, lengthscales: Array):
    """Two-dimensional ARD RBF kernel with [size, age] coordinate order."""

    differences = (x[:, None, :] - y[None, :, :]) / lengthscales
    return variance * jnp.exp(-0.5 * jnp.sum(differences**2, axis=-1))


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_age_distribution(data_path):
    data_path = Path(data_path).resolve()
    data = np.genfromtxt(data_path, delimiter=",", names=True)
    ages = np.asarray(data["age"], dtype=float)
    sizes = np.asarray(data["z"], dtype=float)
    ages = ages[np.isfinite(ages) & np.isfinite(sizes)]
    rounded_ages = np.rint(ages).astype(int)
    if not np.allclose(ages, rounded_ages):
        raise ValueError("All empirical ages must be integer-valued")
    if np.any((rounded_ages < AGE_VALUES[0]) | (rounded_ages > AGE_VALUES[-1])):
        raise ValueError(f"Empirical ages must lie between {AGE_VALUES[0]} and {AGE_VALUES[-1]}")
    counts = np.bincount(rounded_ages, minlength=len(AGE_VALUES))
    if np.any(counts == 0):
        raise ValueError("Every training age must have at least one empirical observation")
    return counts, counts / counts.sum()


def size_prior_means(config: TrainConfig):
    return jnp.asarray([config.age_0_size_mean, config.age_1_size_mean, config.age_2_size_mean, *([config.age_3_plus_size_mean] * 6)])


def sampling_metadata(config: TrainConfig, age_counts, age_probabilities):
    return {
        "coordinate_order": list(COORDINATE_ORDER),
        "active_location_distribution": {"distribution": "DiscreteUniform", "minimum": config.min_locations, "maximum": config.max_locations},
        "age_distribution": {"distribution": "Empirical", "values": list(AGE_VALUES), "counts": [int(value) for value in age_counts], "probabilities": [float(value) for value in age_probabilities]},
        "conditional_size_priors": {
            "age_0": {"distribution": "Normal", "mean": config.age_0_size_mean, "sd": config.size_sd},
            "age_1": {"distribution": "Normal", "mean": config.age_1_size_mean, "sd": config.size_sd},
            "age_2": {"distribution": "Normal", "mean": config.age_2_size_mean, "sd": config.size_sd},
            "age_3_plus": {"distribution": "Normal", "mean": config.age_3_plus_size_mean, "sd": config.size_sd},
        },
        "lengthscale_priors": {
            "size": {"distribution": "LogNormal", "log_mean": config.size_lengthscale_log_mean, "log_sd": config.lengthscale_log_sd},
            "age": {"distribution": "LogNormal", "log_mean": config.age_lengthscale_log_mean, "log_sd": config.lengthscale_log_sd},
        },
        "kernel_variance": 1.0,
        "jitter": config.jitter,
        "location_ordering": "sampled_order",
        "padding": "active_prefix_mask",
    }


def gen_dataloader(config: TrainConfig, age_probabilities):
    """Generate standard DeepRV batches with empirical age-size locations."""

    grid_size = config.max_locations
    jitter_matrix = config.jitter * jnp.eye(grid_size)
    age_probabilities = jnp.asarray(age_probabilities)
    age_values = jnp.asarray(AGE_VALUES, dtype=jnp.float32)
    conditional_size_means = size_prior_means(config)
    lengthscale_prior = dist.LogNormal(jnp.asarray([config.size_lengthscale_log_mean, config.age_lengthscale_log_mean]), jnp.full(2, config.lengthscale_log_sd))
    kernel = jit(lambda locations, lengthscales: ard_rbf(locations, locations, 1.0, lengthscales) + jitter_matrix)
    sample_f = jit(lambda covariance, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(covariance), z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_lengthscales, rng_z, rng_age, rng_size, rng_length = random.split(rng_data, 6)
            age_indices = random.categorical(rng_age, jnp.log(age_probabilities), shape=(grid_size,))
            ages = age_values[age_indices]
            sizes = conditional_size_means[age_indices] + config.size_sd * random.normal(rng_size, (grid_size,))
            locations = jnp.column_stack([sizes, ages])
            active_length = random.randint(rng_length, (), config.min_locations, grid_size + 1) if config.min_locations < grid_size else grid_size
            mask = jnp.repeat((jnp.arange(grid_size) < active_length)[None], config.batch_size, axis=0)
            lengthscales = lengthscale_prior.sample(rng_lengthscales)
            z = dist.Normal().sample(rng_z, sample_shape=(config.batch_size, grid_size)) * mask
            covariance = kernel(locations, lengthscales)
            f = sample_f(covariance, z)
            yield {"s": locations, "f": f, "z": z, "mask": mask, "conditionals": lengthscales, "K": covariance}

    return dataloader


@partial(jit, static_argnames=("var_idx",))
def masked_deep_rv_train_step(rng, state, batch, var_idx: Optional[int] = None):
    """Original DeepRV masked MSE training step."""

    def loss_fn(params):
        conditionals, mask = batch["conditionals"], batch["mask"]
        variance = conditionals[var_idx] if var_idx is not None else 1.0
        output: VAEOutput = state.apply_fn({"params": params}, **batch, rngs={"extra": rng})
        difference = (batch["f"].squeeze() - output.f_hat.squeeze()) * mask
        return (1 / variance) * (jnp.sum(difference**2) / jnp.sum(mask))

    loss, gradients = value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=gradients), loss


@jit
def valid_step(rng, state, batch):
    """Original DeepRV masked validation step."""

    output: VAEOutput = state.apply_fn({"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng})
    difference = (batch["f"].squeeze() - output.f_hat.squeeze()) * batch["mask"]
    return {"norm MSE": jnp.sum(difference**2) / jnp.sum(batch["mask"])}


def build_model(config: TrainConfig):
    head = MLP([config.dim * 4, config.dim, config.dim // 2, 1], nn.gelu)
    return KernelBiasTransformerDeepRV(max_locations=config.max_locations, num_blks=config.num_blocks, dim=config.dim, s_embed=RFFEmbed(config.num_rff_features), head=head)


def build_optimizer(config: TrainConfig):
    optimizer = optax.adamw(cosine_annealing_lr(config.num_steps, config.learning_rate))
    return optax.chain(optax.clip_by_global_norm(config.clip_norm), optimizer)


def model_metadata(config: TrainConfig):
    return {"name": "KernelBiasTransformerDeepRV", "input_dimensions": 2, "coordinate_order": list(COORDINATE_ORDER), "dim": config.dim, "num_blocks": config.num_blocks, "num_rff_features": config.num_rff_features, "max_locations": config.max_locations}


def checkpoint_config(config: TrainConfig, metadata):
    return DictConfig({"model": model_metadata(config), "training": asdict(config), "sampling": metadata})


def load_checkpoint(path, config: TrainConfig):
    model = build_model(config)
    optimizer = build_optimizer(config)
    restored = PyTreeCheckpointer().restore(Path(path).resolve())["state"]
    state = TrainState.create(apply_fn=model.apply, params=restored["params"], kwargs=restored["kwargs"], tx=optimizer)
    state = state.replace(step=restored["step"], opt_state=restored["opt_state"])
    return model, state


def validate_config(config: TrainConfig):
    if not 1 <= config.min_locations <= config.max_locations:
        raise ValueError("min_locations must be between 1 and max_locations")
    if config.size_sd <= 0 or config.lengthscale_log_sd <= 0:
        raise ValueError("Prior standard deviations must be positive")
    if config.jitter <= 0 or config.batch_size <= 0 or config.num_steps <= 0:
        raise ValueError("Jitter, batch size, and training steps must be positive")


def train_model(config: TrainConfig):
    validate_config(config)
    data_path = Path(config.data_path).resolve()
    output_dir = Path(config.output_dir).resolve()
    age_counts, age_probabilities = load_age_distribution(data_path)
    prior_metadata = sampling_metadata(config, age_counts, age_probabilities)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng_train, rng_test = random.split(random.key(config.seed))
    loader = gen_dataloader(config, age_probabilities)
    model = build_model(config)
    optimizer = build_optimizer(config)
    saved_config = checkpoint_config(config, prior_metadata)
    checkpoint_callback = Callback(lambda step, rng, state, batch, extra: save_ckpt(state, saved_config, output_dir / "latest.ckpt"), config.checkpoint_interval)
    wandb.init(project="soay-transformer-deeprv-age-size", mode=config.wandb_mode, config={"training": asdict(config), "sampling": prior_metadata})
    started = time.perf_counter()
    best_state, final_state = train(rng_train, model, optimizer, masked_deep_rv_train_step, config.num_steps, loader, valid_step, config.validation_interval, config.validation_steps, loader, return_state="both", valid_monitor_metric="norm MSE", callbacks=[checkpoint_callback])
    training_time = time.perf_counter() - started
    evaluation_mse = float(evaluate(rng_test, best_state, valid_step, loader, config.validation_steps)["norm MSE"])
    save_ckpt(best_state, saved_config, output_dir / "model.ckpt")
    save_ckpt(final_state, saved_config, output_dir / "final.ckpt")
    metadata = {
        "model": model_metadata(config),
        "training": asdict(config),
        "sampling": prior_metadata,
        "data_sha256": file_sha256(data_path),
        "evaluation_norm_mse": evaluation_mse,
        "training_time_seconds": training_time,
        "best_checkpoint": str(output_dir / "model.ckpt"),
        "final_checkpoint": str(output_dir / "final.ckpt"),
        "latest_checkpoint": str(output_dir / "latest.ckpt"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    wandb.log({"Test Norm MSE": evaluation_mse})
    wandb.finish()
    return {"model": model, "best_state": best_state, "final_state": final_state, "surrogate_decoder": generate_surrogate_decoder(best_state, model), "metadata": metadata}


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=defaults.data_path)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval)
    parser.add_argument("--validation-interval", type=int, default=defaults.validation_interval)
    parser.add_argument("--validation-steps", type=int, default=defaults.validation_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--min-locations", type=int, default=defaults.min_locations)
    parser.add_argument("--max-locations", type=int, default=defaults.max_locations)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default=defaults.wandb_mode)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(data_path=args.data_path, output_dir=args.output_dir, num_steps=args.num_steps, checkpoint_interval=args.checkpoint_interval, validation_interval=args.validation_interval, validation_steps=args.validation_steps, batch_size=args.batch_size, min_locations=args.min_locations, max_locations=args.max_locations, seed=args.seed, wandb_mode=args.wandb_mode)
    train_model(config)


if __name__ == "__main__":
    main()
