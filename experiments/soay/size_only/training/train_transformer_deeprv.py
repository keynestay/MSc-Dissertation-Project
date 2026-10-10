
from experiments.paths import GENERATED, PROJECT_ROOT
import argparse
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import flax.linen as nn
import jax.numpy as jnp
import numpyro.distributions as dist
import optax
import wandb
from dl4bi_sps.kernels import rbf
from jax import Array, jit, random, value_and_grad
from omegaconf import DictConfig
from orbax.checkpoint import PyTreeCheckpointer

from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import (
    Callback,
    TrainState,
    cosine_annealing_lr,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.vae import KernelBiasTransformerDeepRV
from dl4bi.vae.train_utils import generate_surrogate_decoder


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str = str(GENERATED / 'soay/size_only/training/transformer')
    seed: int = 23
    min_locations: int = 64
    max_locations: int = 2048
    size_mean: float = 2.8
    size_sd: float = 0.3
    lengthscale_log_mean: float = 0.0
    lengthscale_log_sd: float = 1.0
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
    '''
    Embed input locations using trainable random Fourier features.

    For F Fourier features and an input location s, the embedding is calculated as phi(s) = sqrt(2 / F) * cos(s @ W.T + b),
    where W is initialised from Normal(0, scale²) and b is initialised from Uniform(0, 2π). 
    Both W and b become trainable Flax parameters. 
    If include_original is True, the original location is concatenated with phi(s). 
    The resulting embedding is then passed through head, which defaults to the identity function.
    '''
    num_features: int = 128
    scale: float = 1.0
    include_original: bool = True
    head: Union[Callable, nn.Module] = lambda x: x

    @nn.compact
    def __call__(self, s: Array):
        input_dim = s.shape[-1]
        weights = self.param("rff_W", lambda key: random.normal(key, (self.num_features, input_dim)) * self.scale)
        phases = self.param("rff_b", lambda key: random.uniform(key, (self.num_features,), minval=0.0, maxval=2 * jnp.pi))
        features = jnp.sqrt(2.0 / self.num_features) * jnp.cos(jnp.dot(s, weights.T) + phases)
        return self.head(jnp.concatenate([s, features], axis=-1) if self.include_original else features)


def gen_dataloader(max_loc, priors, kernel, batch_size, min_train_locs, jitter=5e-4):
    '''
    Create a dataloader function that generates synthetic unit-variance RBF GP function draws for DeepRV training.

    For each batch, the dataloader samples input locations and a kernel lengthscale from their priors, 
    chooses the number of active locations, and generates independent standard-normal vectors z. 
    The mask identifies the active locations so inactive values are set to 0 and do not affect training.

    Args:
        max_loc: Maximum number of input locations in each function draw.
        priors: Dict containing distributions for the input locations "s" and lengthscale "ls".
        kernel: Function used to calculate the GP covariance matrix.
        batch_size: Number of independent GP functions generated per batch.
        min_train_locs: Minimum number of active locations in a training batch.
        jitter: Small diagonal value added to the covariance matrix for numerical stability.

    Returns:
        Dataloader function that accepts a JAX random key and repeatedly yields dictionaries containing locations s with shape (L, 1), 
        function values f with shape (B, L), latent noise z with shape (B, L), mask with shape (B, L), 
        lengthscale conditionals with shape (1,), and covariance matrix K with shape (L, L).
    '''

    jitter_matrix = jitter * jnp.eye(max_loc)
    kernel_jit = jit(lambda s, var, lengthscale: kernel(s, s, var, lengthscale) + jitter_matrix)
    sample_f = jit(lambda covariance, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(covariance), z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_lengthscale, rng_z, rng_locations, rng_length = random.split(rng_data, 5)
            variance = 1.0
            locations = priors["s"].sample(rng_locations, sample_shape=(max_loc, 1))
            active_length = random.randint(rng_length, (), min_train_locs, max_loc + 1) if min_train_locs < max_loc else max_loc
            mask = jnp.repeat((jnp.arange(max_loc) < active_length)[None], batch_size, axis=0)
            lengthscale = priors["ls"].sample(rng_lengthscale)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, max_loc)) * mask
            covariance = kernel_jit(locations, variance, lengthscale)
            f = sample_f(covariance, z)
            yield {"s": locations, 
                   "f": f, 
                   "z": z, 
                   "mask": mask, 
                   "conditionals": jnp.array([lengthscale]), 
                   "K": covariance}
    return dataloader


@partial(jit, static_argnames=("var_idx",))
def masked_deep_rv_train_step(rng, state, batch, var_idx: Optional[int] = None):
    '''
    Perform one DeepRV optimisation step using masked MSE over active locations.

    The model reconstructs the target GP values f from the latent noise and kernel information. 
    Prediction errors at inactive locations are removed using the Boolean mask.

    Args:
        rng: JAX random key used by stochastic model operations.
        state: Current DeepRV training state containing the model parameters and optimiser state.
        batch: Dict yielded by dataloader. Contains the training inputs, targets, latent noise, covariance matrix and mask.
        var_idx: Optional index of the GP variance in the conditional parameters. If None, unit variance is assumed.

    Returns:
        Tuple containing the updated training state and scalar masked training loss.
    '''
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
    
    output: VAEOutput = state.apply_fn({"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng})
    difference = (batch["f"].squeeze() - output.f_hat.squeeze()) * batch["mask"]
    return {"norm MSE": jnp.sum(difference**2) / jnp.sum(batch["mask"])}


def build_model(config: TrainConfig):
    head = MLP([config.dim * 4, config.dim, config.dim // 2, 1], nn.gelu)
    return KernelBiasTransformerDeepRV(max_locations=config.max_locations, num_blks=config.num_blocks, dim=config.dim, s_embed=RFFEmbed(config.num_rff_features), head=head)


def build_optimizer(config: TrainConfig):
    optimizer = optax.adamw(cosine_annealing_lr(config.num_steps, config.learning_rate))
    return optax.chain(optax.clip_by_global_norm(config.clip_norm), optimizer)


def checkpoint_config(config: TrainConfig):
    return DictConfig({"model": {"name": "KernelBiasTransformerDeepRV", "dim": config.dim, "num_blocks": config.num_blocks, "num_rff_features": config.num_rff_features, "max_locations": config.max_locations}, "training": asdict(config)})


def load_checkpoint(path, config: TrainConfig):
    model = build_model(config)
    optimizer = build_optimizer(config)
    restored = PyTreeCheckpointer().restore(Path(path).resolve())["state"]
    state = TrainState.create(apply_fn=model.apply, params=restored["params"], kwargs=restored["kwargs"], tx=optimizer)
    state = state.replace(step=restored["step"], opt_state=restored["opt_state"])
    return model, state


def train_model(config: TrainConfig):
    if not 1 <= config.min_locations <= config.max_locations:
        raise ValueError("min_locations must be between 1 and max_locations")
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng_train, rng_test = random.split(random.key(config.seed))
    priors = {"s": dist.Normal(config.size_mean, config.size_sd), 
              "ls": dist.LogNormal(config.lengthscale_log_mean, config.lengthscale_log_sd)}
    loader = gen_dataloader(config.max_locations, priors, rbf, config.batch_size, config.min_locations, config.jitter)

    model = build_model(config)
    optimizer = build_optimizer(config)
    saved_config = checkpoint_config(config)
    checkpoint_callback = Callback(lambda step, rng, state, batch, extra: save_ckpt(state, saved_config, output_dir / "latest.ckpt"), config.checkpoint_interval)
    wandb.init(project="soay-transformer-deeprv", mode=config.wandb_mode, config=asdict(config))
    started = time.perf_counter()
    best_state, final_state = train(rng_train, model, optimizer, masked_deep_rv_train_step, config.num_steps, loader, valid_step, config.validation_interval, config.validation_steps, loader, return_state="both", valid_monitor_metric="norm MSE", callbacks=[checkpoint_callback])
    training_time = time.perf_counter() - started
    evaluation_mse = float(evaluate(rng_test, best_state, valid_step, loader, config.validation_steps)["norm MSE"])
    save_ckpt(best_state, saved_config, output_dir / "model.ckpt")
    save_ckpt(final_state, saved_config, output_dir / "final.ckpt")
    metadata = {"evaluation_norm_mse": evaluation_mse, "training_time_seconds": training_time, "best_checkpoint": str(output_dir / "model.ckpt"), "final_checkpoint": str(output_dir / "final.ckpt"), "latest_checkpoint": str(output_dir / "latest.ckpt")}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    wandb.log({"Test Norm MSE": evaluation_mse})
    wandb.finish()
    return {"model": model, "best_state": best_state, "final_state": final_state, "surrogate_decoder": generate_surrogate_decoder(best_state, model), "metadata": metadata}


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval)
    parser.add_argument("--validation-interval", type=int, default=defaults.validation_interval)
    parser.add_argument("--validation-steps", type=int, default=defaults.validation_steps)
    parser.add_argument("--min-locations", type=int, default=defaults.min_locations)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-locations", type=int, default=defaults.max_locations)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default=defaults.wandb_mode)
    parser.add_argument("--size-mean", type=float, default=defaults.size_mean)
    parser.add_argument("--size-sd", type=float, default=defaults.size_sd)
    parser.add_argument("--lengthscale-log-mean", type=float, default=defaults.lengthscale_log_mean)
    parser.add_argument("--lengthscale-log-sd", type=float, default=defaults.lengthscale_log_sd)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(output_dir=args.output_dir, 
                         num_steps=args.num_steps, 
                         size_mean=args.size_mean, 
                         size_sd=args.size_sd, 
                         lengthscale_log_mean=args.lengthscale_log_mean, 
                         lengthscale_log_sd=args.lengthscale_log_sd,
                         checkpoint_interval=args.checkpoint_interval, 
                         validation_interval=args.validation_interval, 
                         validation_steps=args.validation_steps, 
                         batch_size=args.batch_size,
                         min_locations=args.min_locations,
                         max_locations=args.max_locations,
                         seed=args.seed, 
                         wandb_mode=args.wandb_mode)
    train_model(config)


if __name__ == "__main__":
    main()
