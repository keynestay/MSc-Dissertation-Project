"""Draw two-dimensional age-size GP functions with Transformer DeepRV."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax import random

from experiments.soay.age_size.training.train import (
    TrainConfig,
    ard_rbf,
    build_model,
)

COORDINATE_ORDER = ("size", "age")
_RUNTIMES = {}


def restore_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(checkpoint_path).item_metadata.tree
    metadata_types = (ocp.metadata.ArrayMetadata, ocp.metadata.ScalarMetadata, ocp.metadata.StringMetadata)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    restore_args = jax.tree.map(
        lambda value: ocp.ArrayRestoreArgs(sharding=sharding) if isinstance(value, ocp.metadata.ArrayMetadata) else ocp.RestoreArgs(),
        metadata,
        is_leaf=lambda value: isinstance(value, metadata_types),
    )
    checkpoint = checkpointer.restore(checkpoint_path, item=metadata, restore_args=restore_args)
    state = checkpoint["state"]
    return {"params": state["params"], **state["kwargs"]}, checkpoint.get("config", {})


class AgeSizeDeepRVDecoder:
    """Single-request decoder matching the tested one-dimensional workflow."""

    def __init__(self, checkpoint_path):
        self.variables, saved_config = restore_checkpoint(checkpoint_path)
        model_config = saved_config.get("model", {})
        training_config = saved_config.get("training", {})
        if int(model_config.get("input_dimensions", -1)) != 2:
            raise ValueError("The checkpoint is not a two-dimensional DeepRV model")
        if tuple(model_config.get("coordinate_order", ())) != COORDINATE_ORDER:
            raise ValueError(f"The checkpoint coordinate order must be {COORDINATE_ORDER}")
        config = TrainConfig(
            max_locations=int(model_config["max_locations"]),
            dim=int(model_config["dim"]),
            num_blocks=int(model_config["num_blocks"]),
            num_rff_features=int(model_config["num_rff_features"]),
        )
        self.max_locations = config.max_locations
        self.jitter = float(training_config.get("jitter", TrainConfig.jitter))
        self.model = build_model(config)
        self.key = random.key(int.from_bytes(os.urandom(4), "little"))
        self.kernel = jax.jit(lambda locations, lengthscales: ard_rbf(locations, locations, 1.0, lengthscales) + self.jitter * jnp.eye(self.max_locations))
        self.decode = jax.jit(
            lambda z, lengthscales, locations, covariance, mask: self.model.apply(
                self.variables,
                z,
                lengthscales,
                s=locations,
                K=covariance,
                mask=mask,
                method="decode",
            )
        )

    def sample(self, locations, intercept, variance, lengthscales):
        locations = np.asarray(locations, dtype=float)
        if locations.size == 0:
            return np.empty(0, dtype=float)
        if locations.ndim != 2 or locations.shape[1] != 2:
            raise ValueError("Age-size locations must have shape (n, 2)")
        if not np.all(np.isfinite(locations)):
            raise ValueError("Age-size locations must be finite")
        if len(locations) > self.max_locations:
            raise ValueError(f"DeepRV requires at most {self.max_locations} locations, received {len(locations)}")
        lengthscales = np.asarray(lengthscales, dtype=float).reshape(-1)
        if lengthscales.shape != (2,) or not np.all(np.isfinite(lengthscales)) or np.any(lengthscales <= 0):
            raise ValueError("Lengthscales must contain two finite positive values")
        variance = float(np.asarray(variance).squeeze())
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("Variance must be finite and nonnegative")

        active_length = len(locations)
        padded_locations = np.zeros((self.max_locations, 2), dtype=float)
        padded_locations[:active_length] = locations
        mask = np.arange(self.max_locations) < active_length
        padded_locations = jnp.asarray(padded_locations)
        mask = jnp.asarray(mask)
        covariance = self.kernel(padded_locations, jnp.asarray(lengthscales))
        self.key, key = random.split(self.key)
        z = random.normal(key, (1, self.max_locations)) * mask[None]
        unit_draw = self.decode(z, jnp.asarray(lengthscales), padded_locations, covariance, mask[None])[0, :active_length, 0]
        return np.asarray(float(np.asarray(intercept).squeeze()) + jnp.sqrt(variance) * unit_draw)


def get_runtime(checkpoint_path):
    path = str(Path(checkpoint_path).resolve())
    if path not in _RUNTIMES:
        _RUNTIMES[path] = AgeSizeDeepRVDecoder(path)
    return _RUNTIMES[path]
