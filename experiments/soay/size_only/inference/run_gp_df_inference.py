
from experiments.paths import GENERATED, PROJECT_ROOT
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from dl4bi_sps.kernels import rbf
from jax import random
from numpyro.infer import MCMC, NUTS

NUM_WARMUP = 1000
NUM_SAMPLES = 2500
NUM_CHAINS = 2


def linear_binary(s, y):
    intercept = numpyro.sample("int", dist.Normal(0, 5))
    slope = numpyro.sample("slope", dist.Normal(0, 5))
    numpyro.sample("y", dist.Bernoulli(logits=intercept + slope * s), obs=y)


def linear_gaussian(s, y):
    intercept = numpyro.sample("int", dist.Normal(0, 5))
    slope = numpyro.sample("slope", dist.Normal(0, 5))
    noise = numpyro.sample("noise", dist.HalfNormal(1))
    numpyro.sample("y", dist.Normal(intercept + slope * s, noise), obs=y)


def intercept_binary(y):
    intercept = numpyro.sample("int", dist.Normal(0, 5))
    numpyro.sample("y", dist.Bernoulli(logits=intercept), obs=y)


def gp_binary(s, y):
    intercept = numpyro.sample("int", dist.Normal(0, 5))
    var = numpyro.sample("var", dist.HalfNormal(2))
    ls = numpyro.sample("ls", dist.LogNormal(0, 1))
    latent = numpyro.sample("latent", dist.Normal(0, 1).expand([len(s)]).to_event(1))
    chol = jnp.linalg.cholesky(rbf(s, s, var, ls) + 5e-4 * jnp.eye(len(s)))
    mu = numpyro.deterministic("mu", chol @ latent + intercept)
    numpyro.sample("y", dist.Bernoulli(logits=mu), obs=y)


def gp_gaussian(s, y):
    intercept = numpyro.sample("int", dist.Normal(0, 5))
    var = numpyro.sample("var", dist.HalfNormal(2))
    ls = numpyro.sample("ls", dist.LogNormal(0, 1))
    noise = numpyro.sample("noise", dist.HalfNormal(1))
    latent = numpyro.sample("latent", dist.Normal(0, 1).expand([len(s)]).to_event(1))
    chol = jnp.linalg.cholesky(rbf(s, s, var, ls) + 5e-4 * jnp.eye(len(s)))
    mu = numpyro.deterministic("mu", chol @ latent + intercept)
    numpyro.sample("y", dist.Normal(mu, noise), obs=y)


def fit(model, key, **kwargs):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
    )
    mcmc.run(key, **kwargs)
    mcmc.print_summary(exclude_deterministic=True)
    return mcmc.get_samples()


def save_samples(path, samples):
    np.savez_compressed(
        path,
        **{name: np.asarray(jax.device_get(value)) for name, value in samples.items()},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"JAX backend: {jax.default_backend()}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("JAX did not detect the allocated GPU")

    project_dir = PROJECT_ROOT
    dataset_path = GENERATED / 'soay/size_only/data/gp_ibm_dataset.csv'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    glm_path = args.output_dir / "gpdf_glm_samples.npz"
    gp_path = args.output_dir / "gpdf_gp_samples.npz"
    if glm_path.exists() or gp_path.exists():
        raise FileExistsError(f"Refusing to overwrite samples in {args.output_dir}")

    df = pd.read_csv(dataset_path)
    s_surv = df["z"].to_numpy()
    y_surv = df["Surv"].to_numpy()
    repr_data = df[["z", "Repr"]].dropna()
    s_repr = repr_data["z"].to_numpy()
    y_repr = repr_data["Repr"].to_numpy()
    growth_data = df[["z", "z1"]].dropna()
    s_growth = growth_data["z"].to_numpy()
    y_growth = growth_data["z1"].to_numpy()
    y_recr = df["Recr"].dropna().to_numpy()
    rcsz_data = df[["z", "Rcsz"]].dropna()
    s_rcsz = rcsz_data["z"].to_numpy()
    y_rcsz = rcsz_data["Rcsz"].to_numpy()

    print(f"Survival: {len(y_surv)}")
    print(f"Reproduction: {len(y_repr)}")
    print(f"Growth: {len(y_growth)}")
    print(f"Recruitment: {len(y_recr)}")
    print(f"Recruit Size: {len(y_rcsz)}")

    keys = random.split(random.key(0), 9)
    surv_glm = fit(linear_binary, keys[0], s=s_surv, y=y_surv)
    repr_glm = fit(linear_binary, keys[1], s=s_repr, y=y_repr)
    growth_glm = fit(linear_gaussian, keys[2], s=s_growth, y=y_growth)
    recr_samples = fit(intercept_binary, keys[3], y=y_recr)
    rcsz_glm = fit(linear_gaussian, keys[4], s=s_rcsz, y=y_rcsz)
    surv_gp = fit(gp_binary, keys[5], s=s_surv, y=y_surv)
    repr_gp = fit(gp_binary, keys[6], s=s_repr, y=y_repr)
    growth_gp = fit(gp_gaussian, keys[7], s=s_growth, y=y_growth)
    rcsz_gp = fit(gp_gaussian, keys[8], s=s_rcsz, y=y_rcsz)

    glm_samples = {
        "surv_int": surv_glm["int"],
        "surv_slope": surv_glm["slope"],
        "repr_int": repr_glm["int"],
        "repr_slope": repr_glm["slope"],
        "growth_int": growth_glm["int"],
        "growth_slope": growth_glm["slope"],
        "growth_noise": growth_glm["noise"],
        "recr_int": recr_samples["int"],
        "rcsz_int": rcsz_glm["int"],
        "rcsz_slope": rcsz_glm["slope"],
        "rcsz_noise": rcsz_glm["noise"],
    }
    gp_samples = {
        "surv_int": surv_gp["int"],
        "surv_var": surv_gp["var"],
        "surv_ls": surv_gp["ls"],
        "surv_mu": surv_gp["mu"],
        "surv_latent": surv_gp["latent"],
        "repr_int": repr_gp["int"],
        "repr_var": repr_gp["var"],
        "repr_ls": repr_gp["ls"],
        "repr_mu": repr_gp["mu"],
        "repr_latent": repr_gp["latent"],
        "growth_int": growth_gp["int"],
        "growth_var": growth_gp["var"],
        "growth_ls": growth_gp["ls"],
        "growth_noise": growth_gp["noise"],
        "growth_mu": growth_gp["mu"],
        "growth_latent": growth_gp["latent"],
        "recr_int": recr_samples["int"],
        "rcsz_int": rcsz_gp["int"],
        "rcsz_var": rcsz_gp["var"],
        "rcsz_ls": rcsz_gp["ls"],
        "rcsz_noise": rcsz_gp["noise"],
        "rcsz_mu": rcsz_gp["mu"],
        "rcsz_latent": rcsz_gp["latent"],
    }

    save_samples(glm_path, glm_samples)
    save_samples(gp_path, gp_samples)
    print(f"Saved GLM samples to {glm_path}")
    print(f"Saved GP samples to {gp_path}")


if __name__ == "__main__":
    main()
