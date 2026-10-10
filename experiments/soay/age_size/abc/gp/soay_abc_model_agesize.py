#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Age-size Soay GP models, IBM, summary selection, and tracked ABC-PMC."""

from experiments.paths import GENERATED, PROJECT_ROOT

import json
import os
import time
from pathlib import Path

PROJECT_DIRECTORY = PROJECT_ROOT
os.environ["PYTHONPATH"] = str(PROJECT_DIRECTORY) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gpflow
import numpy as np
import pandas as pd
import ray
import scipy.special
import tensorflow as tf
import tensorflow_probability as tfp
from ABC_GP_IPM.calculate_ss import Perted_IPM
from ABC_GP_IPM.gp_ipm import GP_IPM
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

from experiments.soay.age_size.abc.gp.tracked_abc_gp_ipm_agesize import TrackedABC_GP_IPM_AgeSize


N_DRAWS = 5000
N_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
SELECTION_REPETITIONS = 300
OPTIMUM_PERCENTAGE = 6
N_SELECTED_PER_RATE = 1
AUC_THRESHOLD = 0.7
ABC_QUANTILES = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.1])
ABC_PARTICLES = np.array([50_000, 10_000, 5_000, 1_000, 250, 250])
ABC_BATCH_SIZE = 378


FEMALE_PROBABILITY = 0.5
MAX_AGE = 7
GREYBEARD_AGE = MAX_AGE + 1
MIN_REPRODUCTIVE_AGE = 1

gpflow.config.set_default_jitter(5e-4)

zero = tf.constant(0.0, gpflow.default_float())
intercept_prior = tfp.distributions.Normal(zero, tf.constant(5.0, gpflow.default_float()))
variance_prior = tfp.distributions.HalfNormal(tf.constant(2.0, gpflow.default_float()))
lengthscale_prior = tfp.distributions.LogNormal(zero, tf.constant(1.0, gpflow.default_float()))
noise_variance_prior = tfp.distributions.Gamma(tf.constant(0.5, gpflow.default_float()), tf.constant(0.5, gpflow.default_float()))


def XY_m_grw_compu(data: pd.DataFrame):
    keep = data[["z", "age", "z1"]].notna().all(axis=1)
    return data.loc[keep, ["z", "age"]].to_numpy(float), data.loc[keep, ["z1"]].to_numpy(float)


def XY_m_surv_compu(data: pd.DataFrame):
    keep = data[["z", "age", "Surv"]].notna().all(axis=1)
    return data.loc[keep, ["z", "age"]].to_numpy(float), data.loc[keep, ["Surv"]].to_numpy(float)


def XY_m_repr_compu(data: pd.DataFrame):
    keep = (data["age"] >= MIN_REPRODUCTIVE_AGE) & data[["z", "age", "Repr"]].notna().all(axis=1)
    return data.loc[keep, ["z", "age"]].to_numpy(float), data.loc[keep, ["Repr"]].to_numpy(float)


def XY_m_recr_compu(data: pd.DataFrame):
    keep = data[["age", "Recr"]].notna().all(axis=1)
    return data.loc[keep, ["age"]].to_numpy(float), data.loc[keep, ["Recr"]].to_numpy(float)


def XY_m_rcsz_compu(data: pd.DataFrame):
    keep = data[["z", "Rcsz"]].notna().all(axis=1)
    return data.loc[keep, ["z"]].to_numpy(float), data.loc[keep, ["Rcsz"]].to_numpy(float)


def set_priors(model):
    model.mean_function.c.prior = intercept_prior
    model.kernel.variance.prior = variance_prior
    model.kernel.lengthscales.prior = lengthscale_prior


def build_gaussian(data, xy_function):
    x, y = xy_function(data)
    model = gpflow.models.GPR(data=(x, y), kernel=gpflow.kernels.SquaredExponential(lengthscales=np.ones(x.shape[1])), mean_function=gpflow.mean_functions.Constant(c=0.0))
    set_priors(model)
    model.likelihood.variance.prior = noise_variance_prior
    return model


def build_binary(data, xy_function):
    x, y = xy_function(data)
    model = gpflow.models.GPMC(data=(x, y), kernel=gpflow.kernels.SquaredExponential(lengthscales=np.ones(x.shape[1])), mean_function=gpflow.mean_functions.Constant(c=0.0), likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid))
    set_priors(model)
    return model


def build_new_m_grw(data: pd.DataFrame):
    return build_gaussian(data, XY_m_grw_compu)


def build_new_m_surv(data: pd.DataFrame):
    return build_binary(data, XY_m_surv_compu)


def build_new_m_repr(data: pd.DataFrame):
    return build_binary(data, XY_m_repr_compu)


def build_new_m_recr(data: pd.DataFrame):
    return build_binary(data, XY_m_recr_compu)


def build_new_m_rcsz(data: pd.DataFrame):
    return build_gaussian(data, XY_m_rcsz_compu)


def to_gpflow_samples(model, values):
    return [parameter.transform.inverse(tf.convert_to_tensor(values[id(parameter)], parameter.dtype)) for parameter in model.trainable_parameters]


def gpr_samples(model, samples, name):
    return to_gpflow_samples(model, {
        id(model.mean_function.c): samples[f"{name}_int"],
        id(model.kernel.variance): samples[f"{name}_var"],
        id(model.kernel.lengthscales): samples[f"{name}_ls"],
        id(model.likelihood.variance): samples[f"{name}_noise"] ** 2,
    })


def gpmc_samples(model, samples, name):
    latent = samples[f"{name}_latent"]
    if latent.ndim == 2:
        latent = latent[..., None]
    return to_gpflow_samples(model, {
        id(model.mean_function.c): samples[f"{name}_int"],
        id(model.kernel.variance): samples[f"{name}_var"],
        id(model.kernel.lengthscales): samples[f"{name}_ls"],
        id(model.V): latent,
    })


def draw_f_prior(model, inputs):
    """Draw one GP prior function at the requested input locations."""
    inputs = np.asarray(inputs, dtype=float)
    if inputs.size == 0:
        return np.empty(0, dtype=float)
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    unique_inputs, inverse = np.unique(inputs, axis=0, return_inverse=True)
    locations = tf.convert_to_tensor(unique_inputs, gpflow.default_float())
    mean = tf.reshape(model.mean_function(locations), [-1])
    covariance = model.kernel(locations)
    covariance += gpflow.config.default_jitter() * tf.eye(len(unique_inputs), dtype=gpflow.default_float())
    tf.random.set_seed(int(np.random.randint(0, np.iinfo(np.int32).max)))
    noise = tf.random.normal([len(unique_inputs)], dtype=gpflow.default_float())
    draw = mean + tf.linalg.matvec(tf.linalg.cholesky(covariance), noise)
    return draw.numpy()[inverse]


def simulate_one_year(data, models):
    size = data["z"].to_numpy(float)
    age = data["age"].to_numpy(int)
    age_size = np.column_stack([size, age])
    number = len(data)
    columns = ["z", "age", "Surv", "z1", "Repr", "Sex", "Recr", "Rcsz"]
    if number == 0:
        return pd.DataFrame(columns=columns)

    survival_probability = np.clip(scipy.special.expit(draw_f_prior(models["m_surv"], age_size)), 0, 1)
    survived = np.random.binomial(1, survival_probability)
    survivor = survived == 1

    next_size = np.full(number, np.nan)
    if survivor.any():
        growth_mean = draw_f_prior(models["m_grw"], age_size[survivor])
        growth_sd = np.sqrt(float(models["m_grw"].likelihood.variance.numpy()))
        next_size[survivor] = growth_mean + np.random.normal(0, growth_sd, survivor.sum())

    reproduced = np.full(number, np.nan)
    reproduced[survivor] = 0
    eligible = survivor & (age >= MIN_REPRODUCTIVE_AGE)
    if eligible.any():
        reproduction_probability = np.clip(scipy.special.expit(draw_f_prior(models["m_repr"], age_size[eligible])), 0, 1)
        reproduced[eligible] = np.random.binomial(1, reproduction_probability)
    breeder = eligible & (reproduced == 1)

    sex = np.full(number, np.nan)
    if breeder.any():
        sex[breeder] = np.random.binomial(1, FEMALE_PROBABILITY, breeder.sum())
    female = breeder & (sex == 1)

    recruited = np.full(number, np.nan)
    if female.any():
        recruitment_probability = np.clip(scipy.special.expit(draw_f_prior(models["m_recr"], age[female, None])), 0, 1)
        recruited[female] = np.random.binomial(1, recruitment_probability)
    recruit = female & (recruited == 1)

    recruit_size = np.full(number, np.nan)
    if recruit.any():
        recruit_mean = draw_f_prior(models["m_rcsz"], size[recruit, None])
        recruit_sd = np.sqrt(float(models["m_rcsz"].likelihood.variance.numpy()))
        recruit_size[recruit] = recruit_mean + np.random.normal(0, recruit_sd, recruit.sum())

    return pd.DataFrame({"z": size, "age": age, "Surv": survived, "z1": next_size, "Repr": reproduced, "Sex": sex, "Recr": recruited, "Rcsz": recruit_size}, columns=columns)


def make_ibm_function():
    def IBM_1step(dataset, models):
        return simulate_one_year(dataset, models)
    return IBM_1step


def population_structure(data_simu, time_step):
    survivor = (data_simu["Surv"] == 1) & data_simu["z1"].notna()
    recruit = (data_simu["Recr"] == 1) & data_simu["Rcsz"].notna()
    survivor_size = data_simu.loc[survivor, "z1"].to_numpy(float)
    survivor_age = np.minimum(data_simu.loc[survivor, "age"].to_numpy(int) + 1, GREYBEARD_AGE)
    recruit_size = data_simu.loc[recruit, "Rcsz"].to_numpy(float)
    return pd.DataFrame({"z": np.concatenate([recruit_size, survivor_size]), "age": np.concatenate([np.zeros(len(recruit_size), dtype=int), survivor_age])})


DISTANCE_NAMES = ["bhattacharyya", "hellinger", "emd", "kl", "hilbert"]
DISTRIBUTION_GROUPS = [
    "next_size",
    "next_age",
    "next_size_age",
    "breeder_next_size",
    "breeder_next_age",
    "nonbreeder_next_size",
    "nonbreeder_next_age",
    "breeder_nonbreeder_survival",
]
TOTAL_GROUPS = [
    "total_individuals_next_size",
    "total_individuals_next_age",
    "total_individuals_next_size_age",
    "total_recruits_by_size",
    "total_recruits_by_age",
    "total_recruits_by_size_age",
    "total_breeders_by_size",
    "total_breeders_by_age",
    "total_breeders_by_size_age",
    "total_survivors_by_size",
    "total_survivors_by_age",
    "total_survivors_by_size_age",
]
AVERAGE_GROUPS = [
    "average_recruits_by_size",
    "average_recruits_by_age",
    "average_recruits_by_size_age",
    "average_breeders_by_size",
    "average_breeders_by_age",
    "average_breeders_by_size_age",
    "average_survivors_by_size",
    "average_survivors_by_age",
    "average_survivors_by_size_age",
]
SUMMARY_NAMES = [f"{distance}_{group}_distribution" for group in DISTRIBUTION_GROUPS for distance in DISTANCE_NAMES]
SUMMARY_NAMES += [f"{distance}_{group}" for group in TOTAL_GROUPS for distance in ["chi2", "ssd"]]
SUMMARY_NAMES += [f"ssd_{group}" for group in AVERAGE_GROUPS]


def numeric_column(data, name):
    if name not in data:
        return np.full(len(data), np.nan)
    return pd.to_numeric(data[name], errors="coerce").to_numpy(float)


def age_group(age):
    age = np.asarray(age, dtype=float)
    group = np.full(age.shape, -1, dtype=int)
    group[(age >= 0) & (age < 1)] = 0
    group[(age >= 1) & (age <= 4)] = 1
    group[(age >= 5) & (age <= MAX_AGE)] = 2
    group[age >= GREYBEARD_AGE] = 3
    return group


def grouped_counts(size, age, size_edges):
    size = np.asarray(size, dtype=float)
    age = np.asarray(age, dtype=float)
    age_index = age_group(age)
    valid_size = np.isfinite(size)
    valid_age = age_index >= 0
    valid_joint = valid_size & valid_age
    size_counts = np.histogram(size[valid_size], size_edges)[0].astype(float)
    age_counts = np.bincount(age_index[valid_age], minlength=4).astype(float)
    joint_counts = np.histogram2d(size[valid_joint], age_index[valid_joint], bins=[size_edges, np.arange(5) - 0.5])[0].ravel()
    return {"size": size_counts, "age": age_counts, "size_age": joint_counts}


def safe_average(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def grouped_statistics(data, size_edges):
    size = numeric_column(data, "z")
    age = numeric_column(data, "age")
    survival = numeric_column(data, "Surv")
    reproduction = numeric_column(data, "Repr")
    recruitment = numeric_column(data, "Recr")
    next_size = numeric_column(data, "z1")
    recruit_size = numeric_column(data, "Rcsz")

    current = grouped_counts(size, age, size_edges)

    def event_counts(indicator):
        selected = indicator == 1
        return grouped_counts(size[selected], age[selected], size_edges)

    survivors = event_counts(survival)
    breeders = event_counts(reproduction)
    recruits = event_counts(recruitment)
    average_survivors = {group: safe_average(survivors[group], current[group]) for group in current}
    average_breeders = {group: safe_average(breeders[group], current[group]) for group in current}
    average_recruits = {group: safe_average(recruits[group], current[group]) for group in current}

    survivor = (survival == 1) & np.isfinite(next_size) & np.isfinite(age)
    recruited = (recruitment == 1) & np.isfinite(recruit_size)
    population_next_size = np.concatenate([next_size[survivor], recruit_size[recruited]])
    population_next_age = np.concatenate([np.minimum(age[survivor] + 1, GREYBEARD_AGE), np.zeros(recruited.sum())])
    population_next = grouped_counts(population_next_size, population_next_age, size_edges)

    breeder = (reproduction == 1) & survivor
    nonbreeder = (reproduction != 1) & survivor
    breeder_next = grouped_counts(next_size[breeder], np.minimum(age[breeder] + 1, GREYBEARD_AGE), size_edges)
    nonbreeder_next = grouped_counts(next_size[nonbreeder], np.minimum(age[nonbreeder] + 1, GREYBEARD_AGE), size_edges)
    breeder_indicator = (reproduction == 1).astype(int)
    survival_indicator = (survival == 1).astype(int)
    breeder_survival = np.bincount(2 * breeder_indicator + survival_indicator, minlength=4).astype(float)

    return {
        "next": population_next,
        "survivors": survivors,
        "breeders": breeders,
        "recruits": recruits,
        "average_survivors": average_survivors,
        "average_breeders": average_breeders,
        "average_recruits": average_recruits,
        "breeder_next": breeder_next,
        "nonbreeder_next": nonbreeder_next,
        "breeder_survival": breeder_survival,
    }


def ssd(observed, simulated):
    return float(np.sum((np.asarray(observed) - np.asarray(simulated)) ** 2))


def chi2(observed, expected):
    observed = np.asarray(observed, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    merged_observed = []
    merged_expected = []
    current_observed = 0.0
    current_expected = 0.0
    for observed_value, expected_value in zip(observed, expected):
        current_observed += observed_value
        current_expected += expected_value
        if current_expected >= 5:
            merged_observed.append(current_observed)
            merged_expected.append(current_expected)
            current_observed = 0.0
            current_expected = 0.0
    if current_observed or current_expected:
        if merged_expected:
            merged_observed[-1] += current_observed
            merged_expected[-1] += current_expected
        else:
            merged_observed.append(current_observed)
            merged_expected.append(current_expected)
    merged_observed = np.asarray(merged_observed)
    merged_expected = np.asarray(merged_expected)
    positive = merged_expected > 0
    return ssd(observed, expected) if not positive.any() else float(np.sum((merged_observed[positive] - merged_expected[positive]) ** 2 / merged_expected[positive]))


def distribution_distances(observed, simulated):
    observed = np.asarray(observed, dtype=float).ravel() + 1e-10
    simulated = np.asarray(simulated, dtype=float).ravel() + 1e-10
    observed /= observed.sum()
    simulated /= simulated.sum()
    positions = np.arange(len(observed), dtype=float)
    coefficient = np.clip(np.sum(np.sqrt(observed * simulated)), 1e-15, 1.0)
    return {
        "bhattacharyya": float(-np.log(coefficient)),
        "hellinger": float(np.sqrt(0.5 * np.sum((np.sqrt(observed) - np.sqrt(simulated)) ** 2))),
        "emd": float(wasserstein_distance(positions, positions, u_weights=observed, v_weights=simulated)),
        "kl": float(np.sum(observed * np.log(observed / simulated))),
        "hilbert": float(np.log(np.max(observed / simulated)) + np.log(np.max(simulated / observed))),
    }


def candidate_summary_values(exp_dataset, simu_dataset, size_edges):
    observed = grouped_statistics(exp_dataset, size_edges)
    simulated = grouped_statistics(simu_dataset, size_edges)
    distribution_sources = {
        "next_size": (observed["next"]["size"], simulated["next"]["size"]),
        "next_age": (observed["next"]["age"], simulated["next"]["age"]),
        "next_size_age": (observed["next"]["size_age"], simulated["next"]["size_age"]),
        "breeder_next_size": (observed["breeder_next"]["size"], simulated["breeder_next"]["size"]),
        "breeder_next_age": (observed["breeder_next"]["age"], simulated["breeder_next"]["age"]),
        "nonbreeder_next_size": (observed["nonbreeder_next"]["size"], simulated["nonbreeder_next"]["size"]),
        "nonbreeder_next_age": (observed["nonbreeder_next"]["age"], simulated["nonbreeder_next"]["age"]),
        "breeder_nonbreeder_survival": (observed["breeder_survival"], simulated["breeder_survival"]),
    }
    total_sources = {
        "total_individuals_next_size": (observed["next"]["size"], simulated["next"]["size"]),
        "total_individuals_next_age": (observed["next"]["age"], simulated["next"]["age"]),
        "total_individuals_next_size_age": (observed["next"]["size_age"], simulated["next"]["size_age"]),
        "total_recruits_by_size": (observed["recruits"]["size"], simulated["recruits"]["size"]),
        "total_recruits_by_age": (observed["recruits"]["age"], simulated["recruits"]["age"]),
        "total_recruits_by_size_age": (observed["recruits"]["size_age"], simulated["recruits"]["size_age"]),
        "total_breeders_by_size": (observed["breeders"]["size"], simulated["breeders"]["size"]),
        "total_breeders_by_age": (observed["breeders"]["age"], simulated["breeders"]["age"]),
        "total_breeders_by_size_age": (observed["breeders"]["size_age"], simulated["breeders"]["size_age"]),
        "total_survivors_by_size": (observed["survivors"]["size"], simulated["survivors"]["size"]),
        "total_survivors_by_age": (observed["survivors"]["age"], simulated["survivors"]["age"]),
        "total_survivors_by_size_age": (observed["survivors"]["size_age"], simulated["survivors"]["size_age"]),
    }
    average_sources = {
        "average_recruits_by_size": (observed["average_recruits"]["size"], simulated["average_recruits"]["size"]),
        "average_recruits_by_age": (observed["average_recruits"]["age"], simulated["average_recruits"]["age"]),
        "average_recruits_by_size_age": (observed["average_recruits"]["size_age"], simulated["average_recruits"]["size_age"]),
        "average_breeders_by_size": (observed["average_breeders"]["size"], simulated["average_breeders"]["size"]),
        "average_breeders_by_age": (observed["average_breeders"]["age"], simulated["average_breeders"]["age"]),
        "average_breeders_by_size_age": (observed["average_breeders"]["size_age"], simulated["average_breeders"]["size_age"]),
        "average_survivors_by_size": (observed["average_survivors"]["size"], simulated["average_survivors"]["size"]),
        "average_survivors_by_age": (observed["average_survivors"]["age"], simulated["average_survivors"]["age"]),
        "average_survivors_by_size_age": (observed["average_survivors"]["size_age"], simulated["average_survivors"]["size_age"]),
    }

    values = {}
    for group, pair in distribution_sources.items():
        for distance, value in distribution_distances(*pair).items():
            values[f"{distance}_{group}_distribution"] = value
    for group, pair in total_sources.items():
        values[f"chi2_{group}"] = chi2(*pair)
        values[f"ssd_{group}"] = ssd(*pair)
    for group, pair in average_sources.items():
        values[f"ssd_{group}"] = ssd(*pair)
    return values


def candidate_summaries(exp_dataset, simu_dataset, size_edges):
    values = candidate_summary_values(exp_dataset, simu_dataset, size_edges)
    return np.asarray([values[name] for name in SUMMARY_NAMES])


def make_candidate_summary_function(size_edges):
    size_edges = np.asarray(size_edges, dtype=float).copy()

    def summary_function(exp_dataset, simu_dataset):
        return candidate_summaries(exp_dataset, simu_dataset, size_edges)

    return summary_function


INFORMATIVE = {
    "m_grw": [name for name in SUMMARY_NAMES if "next_size" in name],
    "m_surv": SUMMARY_NAMES,
    "m_repr": [name for name in SUMMARY_NAMES if "breeder" in name or "recruit" in name],
    "m_recr": [name for name in SUMMARY_NAMES if "recruit" in name or "individuals_next" in name],
    "m_rcsz": [name for name in SUMMARY_NAMES if "next_size" in name],
}


def direction_adjusted_auc_diagnostics(summary_opt, summary_near):
    rows = []
    sample_indices = []
    for sample_index, group in summary_near.groupby("i", sort=False):
        labels = np.concatenate([np.ones(len(summary_opt)), np.zeros(len(group))])
        row = {}
        for name in SUMMARY_NAMES:
            scores = np.concatenate([summary_opt[name].to_numpy(), group[name].to_numpy()])
            finite = np.isfinite(scores)
            finite_labels = labels[finite]
            if finite_labels.size == 0 or np.unique(finite_labels).size < 2:
                row[name] = 0.5
            else:
                auc = roc_auc_score(finite_labels, scores[finite])
                row[name] = max(auc, 1 - auc)
        rows.append(row)
        sample_indices.append(int(sample_index))
    auc_by_sample = pd.DataFrame(rows, index=sample_indices)
    diagnostics = pd.DataFrame({"median_adjusted_auc": auc_by_sample.median(), "maximum_adjusted_auc": auc_by_sample.max()})
    return auc_by_sample, diagnostics


def select_summaries(ipm, target, selection_population, candidate_summary_function, ibm):
    selector = Perted_IPM(GP_IPM_instance=ipm, target=target, fun_ss=candidate_summary_function, name_ss=SUMMARY_NAMES, opt_percentage=OPTIMUM_PERCENTAGE, fun_IBM=ibm)
    negative_log_posterior = -np.asarray(selector.log_posterior_density())
    optimum = selector.ss_optVSopt(SELECTION_REPETITIONS, 123, negative_log_posterior, selection_population, min(N_CORES, SELECTION_REPETITIONS))
    number_near = int(np.sum(negative_log_posterior <= np.percentile(negative_log_posterior, OPTIMUM_PERCENTAGE)))
    near = selector.ss_optVSmcmc(SELECTION_REPETITIONS, 456, negative_log_posterior, selection_population, min(N_CORES, max(1, number_near)))
    selector.nlog_post = negative_log_posterior
    frequencies = selector.top_ss(optimum, near, N_SELECTED_PER_RATE, threshold=AUC_THRESHOLD, return_full=True).sum()
    auc_by_sample, diagnostics = direction_adjusted_auc_diagnostics(optimum, near)
    eligible_frequencies = frequencies.loc[INFORMATIVE[target]].sort_values(ascending=False, kind="stable")
    selected = eligible_frequencies[eligible_frequencies > 0].index[:N_SELECTED_PER_RATE].tolist()

    if selected:
        selection_method = "auc_threshold"
    else:
        eligible_diagnostics = diagnostics.loc[INFORMATIVE[target]].sort_values(["median_adjusted_auc", "maximum_adjusted_auc"], ascending=False, kind="stable")
        selected = [eligible_diagnostics.index[0]]
        selection_method = "fallback_best_median_auc"

    selected_name = selected[0]
    comparisons = max(1, len(auc_by_sample))
    selected_summary = {
        "statistic": selected_name,
        "selection_method": selection_method,
        "threshold_pass_count": int(frequencies[selected_name]),
        "near_optimum_samples": int(len(auc_by_sample)),
        "threshold_pass_fraction": float(frequencies[selected_name] / comparisons),
        "median_adjusted_auc": float(diagnostics.loc[selected_name, "median_adjusted_auc"]),
        "maximum_adjusted_auc": float(diagnostics.loc[selected_name, "maximum_adjusted_auc"]),
    }
    print(f"{target}: selected {selected} via {selection_method}", flush=True)
    return {
        "selected": selected,
        "selection_method": selection_method,
        "selected_summary": selected_summary,
        "auc_by_sample": auc_by_sample,
        "auc_diagnostics": diagnostics,
        "threshold_passes": frequencies,
    }


def fit_mle_models(data):
    specifications = {
        "m_grw": (XY_m_grw_compu, False),
        "m_surv": (XY_m_surv_compu, True),
        "m_repr": (XY_m_repr_compu, True),
        "m_recr": (XY_m_recr_compu, True),
        "m_rcsz": (XY_m_rcsz_compu, False),
    }
    models = {}
    for name, (xy_function, binary) in specifications.items():
        x, y = xy_function(data)
        kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones(x.shape[1]))
        mean = gpflow.mean_functions.Constant(c=0.0)
        models[name] = gpflow.models.VGP((x, y), kernel, gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid), mean) if binary else gpflow.models.GPR((x, y), kernel, mean)
        gpflow.optimizers.Scipy().minimize(models[name].training_loss, models[name].trainable_variables, options={"maxiter": 20})
    return models


def write_auc_diagnostics(selection_results, output_directory):
    raw_frames = []
    summary_frames = []
    for vital_rate, result in selection_results.items():
        raw = result["auc_by_sample"].rename_axis("posterior_sample_index").reset_index().melt(id_vars="posterior_sample_index", var_name="statistic", value_name="adjusted_auc")
        raw.insert(0, "vital_rate", vital_rate)
        raw_frames.append(raw)
        summary = result["auc_diagnostics"].copy()
        summary["threshold_pass_count"] = result["threshold_passes"]
        summary["threshold_pass_fraction"] = result["threshold_passes"] / max(1, len(result["auc_by_sample"]))
        summary["selected"] = summary.index.isin(result["selected"])
        summary.insert(0, "vital_rate", vital_rate)
        summary_frames.append(summary.rename_axis("statistic").reset_index())
    pd.concat(raw_frames, ignore_index=True).to_csv(output_directory / "summary_selection_auc_by_sample.csv", index=False)
    pd.concat(summary_frames, ignore_index=True).to_csv(output_directory / "summary_selection_auc_summary.csv", index=False)


def validate_inputs(data, samples):
    required_columns = {"z", "age", "Surv", "z1", "Repr", "Sex", "Recr", "Rcsz", "yr"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"The dataset is missing columns: {sorted(missing_columns)}")
    required_samples = {
        "surv_int", "surv_var", "surv_ls", "surv_latent",
        "growth_int", "growth_var", "growth_ls", "growth_noise",
        "repr_int", "repr_var", "repr_ls", "repr_latent",
        "recr_int", "recr_var", "recr_ls", "recr_latent",
        "rcsz_int", "rcsz_var", "rcsz_ls", "rcsz_noise",
    }
    missing_samples = required_samples - set(samples)
    if missing_samples:
        raise ValueError(f"The posterior archive is missing arrays: {sorted(missing_samples)}")
    if min(len(samples[name]) for name in required_samples) < N_DRAWS:
        raise ValueError(f"The posterior archive contains fewer than {N_DRAWS} draws")
    model_data = {"growth": XY_m_grw_compu(data), "surv": XY_m_surv_compu(data), "repr": XY_m_repr_compu(data), "recr": XY_m_recr_compu(data), "rcsz": XY_m_rcsz_compu(data)}
    for name, (inputs, _) in model_data.items():
        latent = samples.get(f"{name}_latent")
        if name in {"surv", "repr", "recr"} and (latent.ndim not in {2, 3} or latent.shape[1] != len(inputs)):
            raise ValueError(f"{name}_latent does not match the {len(inputs)} training observations")
        if samples[f"{name}_ls"].ndim != 2 or samples[f"{name}_ls"].shape[1] != inputs.shape[1]:
            raise ValueError(f"{name}_ls does not match the {inputs.shape[1]} GP input dimensions")


def run(data_file, samples_file, output_directory):
    data_file = Path(data_file).resolve()
    samples_file = Path(samples_file).resolve()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    work_directory = Path(os.environ.get("TMPDIR", output_directory / "work"))
    work_directory.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    data = pd.read_csv(data_file)
    with np.load(samples_file) as archive:
        all_samples = {name: archive[name] for name in archive.files}
    validate_inputs(data, all_samples)
    samples = {name: values[:N_DRAWS] for name, values in all_samples.items()}

    pooled_size = pd.concat([data["z"], data["z1"], data["Rcsz"]]).dropna()
    size_edges = np.r_[-np.inf, np.quantile(pooled_size, [0.2, 0.4, 0.6, 0.8]), np.inf]
    if np.unique(size_edges).size != len(size_edges):
        raise ValueError("The data do not produce five distinct size groups")
    candidate_summary_function = make_candidate_summary_function(size_edges)
    ibm = make_ibm_function()

    years = sorted(data["yr"].unique())
    if not years:
        raise ValueError("The dataset contains no years")
    observed_by_year = {index: data.loc[data["yr"] == year].copy() for index, year in enumerate(years)}
    initial_population = data.loc[data["yr"] == years[0], ["z", "age"]].dropna().copy()
    selection_population = data.loc[data["yr"] == years[-1], ["z", "age"]].dropna().copy()

    mle_models = fit_mle_models(data)
    models = {
        "m_grw": build_new_m_grw(data),
        "m_surv": build_new_m_surv(data),
        "m_repr": build_new_m_repr(data),
        "m_recr": build_new_m_recr(data),
        "m_rcsz": build_new_m_rcsz(data),
    }
    mcmc_samples = {
        "m_grw": gpr_samples(models["m_grw"], samples, "growth"),
        "m_surv": gpmc_samples(models["m_surv"], samples, "surv"),
        "m_repr": gpmc_samples(models["m_repr"], samples, "repr"),
        "m_recr": gpmc_samples(models["m_recr"], samples, "recr"),
        "m_rcsz": gpr_samples(models["m_rcsz"], samples, "rcsz"),
    }
    ipm = GP_IPM(data, observed_by_year, mle_models, models, mcmc_samples)
    for function in [XY_m_grw_compu, XY_m_surv_compu, XY_m_repr_compu, XY_m_recr_compu, XY_m_rcsz_compu]:
        ipm.add_XY_compu(function)
    for function in [build_new_m_grw, build_new_m_surv, build_new_m_repr, build_new_m_recr, build_new_m_rcsz]:
        ipm.add_model_build(function)

    ray.init(num_cpus=N_CORES, include_dashboard=False, log_to_driver=False, _temp_dir=f"/tmp/ray-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    try:
        selection_start = time.perf_counter()
        targets = ["m_grw", "m_surv", "m_repr", "m_recr", "m_rcsz"]
        selection_results = {target: select_summaries(ipm, target, selection_population, candidate_summary_function, ibm) for target in targets}
        selection_seconds = time.perf_counter() - selection_start
        write_auc_diagnostics(selection_results, output_directory)
        selected = {target: result["selected"] for target, result in selection_results.items()}
        selected_names = [name for name in SUMMARY_NAMES if any(name in names for names in selected.values())]

        def abc_summaries(exp_dataset, simu_dataset):
            values = candidate_summary_values(exp_dataset, simu_dataset, size_edges)
            return np.asarray([values[name] for name in selected_names])

        abc = TrackedABC_GP_IPM_AgeSize(ipm, abc_summaries, selected_names, initial_population, ibm, population_structure)
        details_directory = work_directory / "ABC_details"
        details_directory.mkdir(exist_ok=True)
        abc_start = time.perf_counter()
        indices, threshold = abc.ABC_PMC(ABC_QUANTILES, ABC_PARTICLES, N_CORES, details=False, smallest_unit=ABC_BATCH_SIZE, file_path2store=str(details_directory))
        abc_seconds = time.perf_counter() - abc_start
    finally:
        ray.shutdown()

    np.savez_compressed(output_directory / "abc_samples.npz", indices=np.asarray(indices), weights=np.asarray(abc.weight), threshold=np.asarray(threshold))
    final_diagnostics = abc.stage_diagnostics[-1]
    run_info = {
        "gp_sampling": "prior",
        "conditioning": "unconditional",
        "posterior_draws_per_vital_rate": N_DRAWS,
        "vital_rate_order": list(models),
        "female_probability": FEMALE_PROBABILITY,
        "maximum_natural_age": MAX_AGE,
        "greybeard_age": GREYBEARD_AGE,
        "minimum_reproductive_age": MIN_REPRODUCTIVE_AGE,
        "summary_selection_year": int(years[-1]),
        "summary_selection_population_size": int(len(selection_population)),
        "age_groups": {"newborn": [0], "young": [1, 4], "middle_aged": [5, 7], "greybeard": [8]},
        "size_quantile_boundaries": size_edges[1:-1].tolist(),
        "candidate_summary_statistics": len(SUMMARY_NAMES),
        "summary_selection_repetitions": SELECTION_REPETITIONS,
        "summary_selection_optimum_percentage": OPTIMUM_PERCENTAGE,
        "summary_selection_auc_threshold": AUC_THRESHOLD,
        "selected_summary_statistics": selected,
        "summary_selection_methods": {target: result["selection_method"] for target, result in selection_results.items()},
        "summary_selection_auc": {target: result["selected_summary"] for target, result in selection_results.items()},
        "abc_stage_diagnostics": abc.stage_diagnostics,
        "total_abc_proposals": final_diagnostics["cumulative_proposed"],
        "total_abc_acceptances": final_diagnostics["cumulative_accepted"],
        "total_abc_rejections": final_diagnostics["cumulative_rejected"],
        "final_particle_count": int(len(indices)),
        "selection_seconds": selection_seconds,
        "cache_seconds": 0.0,
        "abc_sampling_seconds": abc_seconds,
        "total_seconds": time.perf_counter() - total_start,
    }
    (output_directory / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    (output_directory / "abc_stage_diagnostics.json").write_text(json.dumps(abc.stage_diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(run_info, indent=2))
