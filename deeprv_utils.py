import sys

sys.path.append("benchmarks/vae")
from pathlib import Path
from typing import Callable, Optional, Union
import jax
import time
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
from functools import partial
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from dl4bi_sps.kernels import rbf
from scipy.stats import wasserstein_distance
import orbax
from orbax.checkpoint import PyTreeCheckpointer
import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, train, TrainState, load_ckpt
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder
import json
from hydra.utils import instantiate

def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    surrogate_decoder: Optional[Callable] = None,
):
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=2, num_samples=4_000, num_warmup=2_000,)

    total_start = time.perf_counter()
    mcmc_start = time.perf_counter()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs,)
    mcmc_time = time.perf_counter() - mcmc_start
    mcmc.print_summary()

    sample_start = time.perf_counter()
    samples = mcmc.get_samples()
    samples = jax.block_until_ready(samples)
    sample_time = time.perf_counter() - sample_start

    predictive_start = time.perf_counter()
    post = Predictive(model, samples)(k2, surrogate_decoder=surrogate_decoder,)
    predictive_time = time.perf_counter() - predictive_start
    total_time = time.perf_counter() - total_start

    timings = {
        "mcmc_time": mcmc_time,
        "sample_time": sample_time,
        "predictive_time": predictive_time,
        "total_time": total_time,
    }

    return samples, mcmc, post["obs"], timings

def gen_train_dataloader(
    s: Array,
    priors: dict, 
    batch_size=32
    ):
    jitter = 5e-4* jnp.eye(s.shape[0])
    kernel_jit = jit(lambda s, var, ls: rbf(s, s, var, ls) + jitter)
    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = kernel_jit(s, var, ls)
            L = jnp.linalg.cholesky(K)
            yield {"s": s,
                    "z": z,
                    "conditionals": jnp.array([ls]), 
                    "f": f_jit(L, z)}

    return dataloader

@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}

def train_deeprv(
    rng_train,
    s,
    priors,
    num_steps=100_000,
):
    nn_model = gMLPDeepRV(num_blks=2)
    optimizer = optax.adamw(cosine_annealing_lr(num_steps, 1e-3), weight_decay=1e-2,)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optimizer,)
    loader = gen_train_dataloader(s, priors,)
    start = time.perf_counter()
    state = train(
        rng_train,
        nn_model,
        optimizer,
        deep_rv_train_step,
        num_steps,
        loader,
        valid_step,
        25_000,
        5_000,
        loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
    )
    training_time = time.perf_counter() - start
    surrogate_decoder = generate_surrogate_decoder(state, nn_model,)
    return surrogate_decoder, training_time, state

def run_gp_inference(
    rng,
    s,
    y,
    priors,
    inference_model,
    obs_mask=True
):
    model = inference_model(s,priors,)
    samples, mcmc, posterior, timings = hmc(rng, model, y, obs_mask,)
    return (samples, mcmc, posterior, timings,)

def run_deeprv_inference(
    rng,
    s,
    y,
    priors,
    surrogate_decoder,
    inference_model,
    obs_mask=True
):
    model = inference_model(s,priors,)
    samples, mcmc, posterior, timings = hmc(rng, model, y, obs_mask, surrogate_decoder,)
    return (samples, mcmc, posterior, timings,)

def compute_metrics(
    gp_samples,
    deeprv_samples,
    gp_time,
    deeprv_time,
    deeprv_train_time,
):
    """
    Compute comparison metrics between full GP inference and DeepRV inference.

    Parameters
    ----------
    gp_samples : dict
        Posterior samples from the full GP MCMC run. Expected keys include
        ``"mu"``, ``"ls"``, and ``"r"``.

    deeprv_samples : dict
        Posterior samples from the DeepRV MCMC run. Expected keys include
        ``"mu"``, ``"ls"``, and ``"r"``.

    gp_time : float
        Wall-clock inference time for the full GP.

    deeprv_time : float
        Wall-clock inference time for DeepRV.

    deeprv_train_time : float
        Wall-clock training time for the DeepRV surrogate.

    Returns
    -------
    dict
        Dictionary of timing and posterior-comparison metrics.

    Notes
    -----
    ``mu`` has shape ``(num_samples, N)``. Since GP and DeepRV MCMC samples are
    not paired sample-by-sample, we compare their posterior summaries:

    - posterior mean of ``mu``;
    - posterior standard deviation of ``mu``.

    For scalar hyperparameters ``ls`` and ``r``, we compare the full posterior
    sample distributions using the one-dimensional Wasserstein distance.
    """

    # Move arrays from JAX device memory to ordinary NumPy arrays.
    gp_mu = np.asarray(jax.device_get(gp_samples["mu"]))
    deeprv_mu = np.asarray(jax.device_get(deeprv_samples["mu"]))

    gp_ls = np.asarray(jax.device_get(gp_samples["ls"])).reshape(-1)
    deeprv_ls = np.asarray(jax.device_get(deeprv_samples["ls"])).reshape(-1)

    gp_r = np.asarray(jax.device_get(gp_samples["r"])).reshape(-1)
    deeprv_r = np.asarray(jax.device_get(deeprv_samples["r"])).reshape(-1)

    # Compare posterior mean latent fields: E_GP[mu] vs E_DeepRV[mu].
    gp_mu_mean = gp_mu.mean(axis=0)
    deeprv_mu_mean = deeprv_mu.mean(axis=0)

    posterior_mu_mean_mse = float(
        np.mean((gp_mu_mean - deeprv_mu_mean) ** 2)
    )

    # Compare posterior uncertainty in the latent fields.
    gp_mu_std = gp_mu.std(axis=0)
    deeprv_mu_std = deeprv_mu.std(axis=0)

    posterior_mu_std_mse = float(
        np.mean((gp_mu_std - deeprv_mu_std) ** 2)
    )

    # Compare scalar posterior distributions for kernel hyperparameters.
    ls_wasserstein = float(
        wasserstein_distance(gp_ls, deeprv_ls)
    )

    r_wasserstein = float(
        wasserstein_distance(gp_r, deeprv_r)
    )

    return {
        "gp_time": float(gp_time),
        "deeprv_time": float(deeprv_time),
        "deeprv_train_time": float(deeprv_train_time),
        "posterior_mu_mean_mse": posterior_mu_mean_mse,
        "posterior_mu_std_mse": posterior_mu_std_mse,
        "ls_wasserstein": ls_wasserstein,
        "r_wasserstein": r_wasserstein,
    }

def load_saved_deeprv(save_dir):
    save_dir = Path(save_dir)
    checkpoint_path = save_dir / "model.ckpt"

    checkpointer = PyTreeCheckpointer()

    # Read the checkpoint structure without loading its arrays.
    metadata = checkpointer.metadata(
        checkpoint_path.absolute()
    )
    metadata_tree = metadata.item_metadata.tree

    # Restore every JAX array onto the current device.
    device = jax.local_devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)

    def is_metadata(value):
        return isinstance(
            value,
            orbax.checkpoint.metadata.Metadata,
        )

    restore_args = jax.tree.map(
        lambda value: (
            orbax.checkpoint.ArrayRestoreArgs(
                sharding=sharding,
            )
            if isinstance(
                value,
                orbax.checkpoint.metadata.ArrayMetadata,
            )
            else orbax.checkpoint.RestoreArgs()
        ),
        metadata_tree,
        is_leaf=is_metadata,
    )

    checkpoint = checkpointer.restore(
        checkpoint_path.absolute(),
        args=orbax.checkpoint.args.PyTreeRestore(
            restore_args=restore_args,
        ),
    )

    with (save_dir / "metadata.json").open() as f:
        run_metadata = json.load(f)

    model = gMLPDeepRV(
        num_blks=int(run_metadata["num_blks"])
    )

    # For prediction, the original optimizer state is unnecessary.
    state = TrainState.create(
        apply_fn=model.apply,
        params=checkpoint["state"]["params"],
        kwargs=checkpoint["state"]["kwargs"],
        tx=optax.identity(),
    )

    surrogate_decoder = generate_surrogate_decoder(
        state,
        model,
    )

    s = jnp.asarray(
        np.load(save_dir / "s.npy")
    )

    return {
        "decoder": surrogate_decoder,
        "state": state,
        "model": model,
        "s": s,
        "config": checkpoint["config"],
        "metadata": run_metadata,
    }


