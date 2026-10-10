"""Train DeepRV gMLP on the 760 fixed Soay Sheep locations."""

from experiments.paths import GENERATED, PROJECT_ROOT

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import optax
import orbax.checkpoint as ocp
import wandb
from dl4bi_sps.kernels import rbf
from jax import Array, jit, random
from omegaconf import DictConfig

from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import (
    Callback,
    TrainState,
    cosine_annealing_lr,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder


@dataclass(frozen=True)
class TrainConfig:
    data_path: str = str(GENERATED / 'soay/size_only/data/gp_ibm_dataset.csv')
    output_dir: str = str(GENERATED / 'gmlp_mcmc/training')
    seed: int = 23
    num_locations: int = 760
    lengthscale_min: float = 0.05
    lengthscale_max: float = 10.0
    jitter: float = 5e-4
    batch_size: int = 32
    num_steps: int = 200_000
    checkpoint_interval: int = 25_000
    validation_interval: int = 25_000
    validation_steps: int = 5_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    clip_norm: float = 3.0
    num_blocks: int = 2
    wandb_mode: str = "disabled"


def load_locations(config: TrainConfig):
    with Path(config.data_path).open(newline="") as file:
        values = [row["z"] for row in csv.DictReader(file)]
    if len(values) != config.num_locations or any(value == "" for value in values):
        raise ValueError(f"Expected {config.num_locations} non-missing values in the z column")
    return jnp.asarray([float(value) for value in values])[:, None]


def gen_dataloader(s: Array, priors: dict, batch_size: int = 32, jitter: float = 5e-4):
    """Original fixed-location DeepRV dataloader adapted to the Soay locations."""

    jitter_matrix = jitter * jnp.eye(s.shape[0])
    kernel_jit = jit(lambda locations, variance, lengthscale: rbf(locations, locations, variance, lengthscale) + jitter_matrix)
    sample_f = jit(lambda cholesky, z: jnp.einsum("ij,bj->bi", cholesky, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_lengthscale, rng_z = random.split(rng_data, 3)
            lengthscale = priors["ls"].sample(rng_lengthscale)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            covariance = kernel_jit(s, 1.0, lengthscale)
            f = sample_f(jnp.linalg.cholesky(covariance), z)
            yield {"s": s, "f": f, "z": z, "conditionals": jnp.array([lengthscale])}

    return dataloader


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn({"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng})
    return {"norm MSE": output.metrics(batch["f"], 1.0)["MSE"]}


def build_model(config: TrainConfig):
    return gMLPDeepRV(num_blks=config.num_blocks)


def build_optimizer(config: TrainConfig):
    schedule = cosine_annealing_lr(config.num_steps, config.learning_rate)
    return optax.chain(optax.clip_by_global_norm(config.clip_norm), optax.adamw(schedule, weight_decay=config.weight_decay))


def output_paths(config: TrainConfig):
    label = f"soay_{config.num_locations}"
    output_dir = Path(config.output_dir).resolve() / f"gmlp_deeprv_rbf_{label}"
    return {"directory": output_dir, "latest": output_dir / f"latest_{label}.ckpt", "model": output_dir / f"model_{label}.ckpt", "final": output_dir / f"final_{label}.ckpt", "metadata": output_dir / f"metadata_{label}.json"}


def checkpoint_config(config: TrainConfig, data_sha256: str):
    return DictConfig({"model": {"name": "gMLPDeepRV", "kernel": "rbf", "num_blocks": config.num_blocks, "num_locations": config.num_locations}, "training": asdict(config), "data_sha256": data_sha256})


def load_checkpoint(path, config: TrainConfig):
    """Restore a single-device checkpoint on the current CPU or GPU."""

    checkpointer = ocp.PyTreeCheckpointer()
    tree = checkpointer.metadata(Path(path).resolve()).item_metadata.tree
    metadata_types = (ocp.metadata.ArrayMetadata, ocp.metadata.ScalarMetadata, ocp.metadata.StringMetadata)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    restore_args = jax.tree.map(lambda x: ocp.ArrayRestoreArgs(sharding=sharding) if isinstance(x, ocp.metadata.ArrayMetadata) else ocp.RestoreArgs(), tree, is_leaf=lambda x: isinstance(x, metadata_types))
    restored = checkpointer.restore(Path(path).resolve(), item=tree, restore_args=restore_args)["state"]
    model = build_model(config)
    state = TrainState.create(apply_fn=model.apply, params=restored["params"], kwargs=restored["kwargs"], tx=build_optimizer(config))
    return model, state.replace(step=restored["step"], opt_state=restored["opt_state"])


def train_model(config: TrainConfig):
    if config.lengthscale_min <= 0 or config.lengthscale_min >= config.lengthscale_max:
        raise ValueError("Invalid lengthscale range")
    data_path = Path(config.data_path).resolve()
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    s = load_locations(config)
    paths = output_paths(config)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    rng_train, rng_test = random.split(random.key(config.seed))
    priors = {"ls": dist.Uniform(config.lengthscale_min, config.lengthscale_max)}
    loader = gen_dataloader(s, priors, config.batch_size, config.jitter)
    model = build_model(config)
    optimizer = build_optimizer(config)
    saved_config = checkpoint_config(config, data_sha256)
    checkpoint_callback = Callback(lambda step, rng, state, batch, extra: save_ckpt(state, saved_config, paths["latest"]), config.checkpoint_interval)
    wandb.init(project="soay-gmlp-deeprv-760", mode=config.wandb_mode, config=asdict(config))
    started = time.perf_counter()
    best_state, final_state = train(rng_train, model, optimizer, deep_rv_train_step, config.num_steps, loader, valid_step, config.validation_interval, config.validation_steps, loader, return_state="both", valid_monitor_metric="norm MSE", callbacks=[checkpoint_callback])
    training_time = time.perf_counter() - started
    evaluation_mse = float(evaluate(rng_test, best_state, valid_step, loader, config.validation_steps)["norm MSE"])
    save_ckpt(best_state, saved_config, paths["model"])
    save_ckpt(final_state, saved_config, paths["final"])
    metadata = {"num_locations": config.num_locations, "data_path": str(data_path), "data_sha256": data_sha256, "location_min": float(s.min()), "location_max": float(s.max()), "evaluation_norm_mse": evaluation_mse, "training_time_seconds": training_time, "best_checkpoint": str(paths["model"]), "final_checkpoint": str(paths["final"]), "latest_checkpoint": str(paths["latest"])}
    paths["metadata"].write_text(json.dumps(metadata, indent=2))
    wandb.log({"Test Norm MSE": evaluation_mse})
    wandb.finish()
    return {"locations": s, "model": model, "best_state": best_state, "final_state": final_state, "surrogate_decoder": generate_surrogate_decoder(best_state, model), "paths": paths, "metadata": metadata}


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
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default=defaults.wandb_mode)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(data_path=args.data_path, output_dir=args.output_dir, num_steps=args.num_steps, checkpoint_interval=args.checkpoint_interval, validation_interval=args.validation_interval, validation_steps=args.validation_steps, batch_size=args.batch_size, seed=args.seed, wandb_mode=args.wandb_mode)
    train_model(config)


if __name__ == "__main__":
    main()
