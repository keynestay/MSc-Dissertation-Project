import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax import random

from experiments.soay.size_only.training.train_transformer_deeprv import TrainConfig, build_model

MAX_LOCATIONS = 2048
JITTER = 5e-4
_DECODERS = {}


def rbf(x, y, lengthscale):
    return jnp.exp(-0.5 * ((x[:, None] - y[None, :]) / lengthscale) ** 2)

def unique_locations(x):
    unique, positions, inverse = [], {}, []
    for value in np.asarray(x).reshape(-1):
        key = float(value)
        if key not in positions:
            positions[key] = len(unique)
            unique.append(value)
        inverse.append(positions[key])
    return np.asarray(unique), np.asarray(inverse, dtype=int)

def restore_checkpoint(checkpoint_path):
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(Path(checkpoint_path).resolve()).item_metadata.tree
    metadata_types = (
        ocp.metadata.ArrayMetadata,
        ocp.metadata.ScalarMetadata,
        ocp.metadata.StringMetadata,
    )
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    restore_args = jax.tree.map(
        lambda value: ocp.ArrayRestoreArgs(sharding=sharding)
        if isinstance(value, ocp.metadata.ArrayMetadata)
        else ocp.RestoreArgs(),
        metadata,
        is_leaf=lambda value: isinstance(value, metadata_types),
    )
    checkpoint = checkpointer.restore(
        Path(checkpoint_path).resolve(), item=metadata, restore_args=restore_args
    )
    state = checkpoint["state"]
    variables = {"params": state["params"], **state["kwargs"]}
    return variables, checkpoint.get("config", {}).get("model", {})

def restore_variables(checkpoint_path):
    """Restore model variables while preserving the original public interface."""
    variables, _ = restore_checkpoint(checkpoint_path)
    return variables

class DeepRVDecoder:
    def __init__(self, checkpoint_path):
        self.variables, saved_config = restore_checkpoint(checkpoint_path)
        config = TrainConfig(
            max_locations=int(saved_config.get("max_locations", MAX_LOCATIONS)),
            dim=int(saved_config.get("dim", TrainConfig.dim)),
            num_blocks=int(saved_config.get("num_blocks", TrainConfig.num_blocks)),
            num_rff_features=int(
                saved_config.get("num_rff_features", TrainConfig.num_rff_features)
            ),
        )
        self.max_locations = config.max_locations
        self.model = build_model(config)
        self.key = random.key(int.from_bytes(os.urandom(4), "little"))
        self.decode = jax.jit(
            lambda z, lengthscale, locations, covariance, mask: self.model.apply(
                self.variables,
                z,
                jnp.asarray([lengthscale]),
                s=locations,
                K=covariance,
                mask=mask,
                method="decode",
            )
        )
        self.decode_batch = jax.jit(
            jax.vmap(
                lambda z, lengthscale, locations, covariance, mask: self.model.apply(
                    self.variables,
                    z[None],
                    jnp.asarray([lengthscale]),
                    s=locations,
                    K=covariance,
                    mask=mask[None],
                    method="decode",
                )[0],
            )
        )

    def sample(self, x, intercept, variance, lengthscale):
        if np.asarray(x).size == 0:
            return np.empty(0, dtype=float)
        unique_x, inverse = unique_locations(x)
        if len(unique_x) > self.max_locations:
            raise ValueError(
                f"DeepRV requires at most {self.max_locations} unique locations, received {len(unique_x)}"
            )
        padded_x = np.zeros(self.max_locations)
        padded_x[: len(unique_x)] = unique_x
        mask = np.arange(self.max_locations) < len(unique_x)
        self.key, key = random.split(self.key)
        z = random.normal(key, (1, self.max_locations)) * jnp.asarray(mask)[None]
        locations = jnp.asarray(padded_x)[:, None]
        covariance = rbf(jnp.asarray(padded_x), jnp.asarray(padded_x), lengthscale) + JITTER * jnp.eye(self.max_locations)
        unit_draw = self.decode(
            z, lengthscale, locations, covariance, jnp.asarray(mask)[None]
        )[0, : len(unique_x), 0]
        return np.asarray(intercept + jnp.sqrt(variance) * unit_draw[inverse])

def get_decoder(checkpoint_path):
    path = str(Path(checkpoint_path).resolve())
    if path not in _DECODERS:
        _DECODERS[path] = DeepRVDecoder(path)
    return _DECODERS[path]
