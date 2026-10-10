import json
import os
import time
from pathlib import Path

from experiments.paths import PROJECT_ROOT

MODULE_DIRECTORY = PROJECT_ROOT
os.environ["PYTHONPATH"] = str(MODULE_DIRECTORY) + os.pathsep + os.environ.get("PYTHONPATH", "")

# One numerical thread per Ray worker prevents CPU oversubscription.
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
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance

from ABC_GP_IPM.abc_gp_ipm import ABC_GP_IPM
from ABC_GP_IPM.calculate_ss import Perted_IPM
from ABC_GP_IPM.gp_ipm import GP_IPM

N_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
N_DRAWS = 5000
SELECTION_REPETITIONS = 300
OPTIMUM_PERCENTAGE = 6
N_SELECTED_PER_RATE = 1
AUC_THRESHOLD = 0.7
ABC_QUANTILES = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.1])
ABC_PARTICLES = np.array([50_000, 10_000, 5_000, 1_000, 250, 250])
ABC_BATCH_SIZE = 384

gpflow.config.set_default_jitter(5e-4)
FEMALE_PROBABILITY = 0.5


# Set priors to match those used to create gpdf_gp_samples.npz.
zero = tf.constant(0.0, gpflow.default_float())
intercept_prior = tfp.distributions.Normal(zero, tf.constant(5.0, gpflow.default_float()))
variance_prior = tfp.distributions.HalfNormal(tf.constant(2.0, gpflow.default_float()))
lengthscale_prior = tfp.distributions.LogNormal(zero, tf.constant(1.0, gpflow.default_float()))

# GPflow uses the noise variance instead of SD for GPR. But in Numpyro we put priors on noise_sd ~ HalfNormal(1.0) 
# which is equivalent to noise_variance ~ Gamma(0.5,0.5)
noise_variance_prior = tfp.distributions.Gamma(tf.constant(0.5, gpflow.default_float()), tf.constant(0.5, gpflow.default_float()),)


# Define functions to extract respective vital rate columns 
def XY_m_grw_compu(data: pd.DataFrame):
    keep = (data["Surv"] == 1) & data["z"].notna() & data["z1"].notna()
    return data.loc[keep, ["z"]].to_numpy(float), data.loc[keep, ["z1"]].to_numpy(float)
def XY_m_surv_compu(data: pd.DataFrame):
    keep = data["z"].notna() & data["Surv"].notna()
    return data.loc[keep, ["z"]].to_numpy(float), data.loc[keep, ["Surv"]].to_numpy(float)
def XY_m_repr_compu(data: pd.DataFrame):
    keep = data["z"].notna() & data["Repr"].notna()
    return data.loc[keep, ["z"]].to_numpy(float), data.loc[keep, ["Repr"]].to_numpy(float)
def XY_m_rcsz_compu(data: pd.DataFrame):
    keep = (data["Recr"] == 1) & data["z"].notna() & data["Rcsz"].notna()
    return data.loc[keep, ["z"]].to_numpy(float), data.loc[keep, ["Rcsz"]].to_numpy(float)

def set_priors(model):
    model.mean_function.c.prior = intercept_prior
    model.kernel.variance.prior = variance_prior
    model.kernel.lengthscales.prior = lengthscale_prior

# Define functions to build GP models
def build_gaussian(data, xy_function):
    model = gpflow.models.GPR(
        data=xy_function(data),
        kernel=gpflow.kernels.SquaredExponential(),
        mean_function=gpflow.mean_functions.Constant(c=0.0),
    )
    set_priors(model)
    model.likelihood.variance.prior = noise_variance_prior
    return model

def build_binary(data, xy_function):
    model = gpflow.models.GPMC(
        data=xy_function(data),
        kernel=gpflow.kernels.SquaredExponential(),
        mean_function=gpflow.mean_functions.Constant(c=0.0),
        likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
    )
    set_priors(model)
    return model

def build_new_m_grw(data: pd.DataFrame):
    return build_gaussian(data, XY_m_grw_compu)

def build_new_m_surv(data: pd.DataFrame):
    return build_binary(data, XY_m_surv_compu)

def build_new_m_repr(data: pd.DataFrame):
    return build_binary(data, XY_m_repr_compu)

def build_new_m_rcsz(data: pd.DataFrame):
    return build_gaussian(data, XY_m_rcsz_compu)


def to_gpflow_samples(model, values):
    '''
    Convert Numpyro generated posterior draws to GPflow's unconstrained internal parameterisation
    and parameter order so SamplingHelper can later assign them correctly during ABC.

    Args:
        model: GPflow model.
        values: Dict mapping each GPflow parameter object to its posterior samples

    Returns:
        A list of transformed posterior samples
    '''

    return [
        parameter.transform.inverse(tf.convert_to_tensor(values[id(parameter)], parameter.dtype))
        for parameter in model.trainable_parameters
    ]

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


def draw_f_prior(model, x):
    """Draw one GP prior function at locations x"""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return np.empty(0, dtype=float)
    unique_x, inverse = np.unique(x, return_inverse=True)
    locations = tf.convert_to_tensor(unique_x[:, None], gpflow.default_float())
    mean = tf.reshape(model.mean_function(locations), [-1])                     # a      
    covariance = model.kernel(locations)                                        # K      
    covariance += gpflow.config.default_jitter() * tf.eye(len(unique_x), dtype=gpflow.default_float())
    tf.random.set_seed(int(np.random.randint(0, np.iinfo(np.int32).max)))
    noise = tf.random.normal([len(unique_x)], dtype=gpflow.default_float())     #z
    draw = mean + tf.linalg.matvec(tf.linalg.cholesky(covariance), noise)       # f = a + Chol(K) @ z
    return draw.numpy()[inverse]


def simulate_one_year(data, models, recruit_probability):
    z = data["z"].to_numpy(float)
    x = z[:, None]
    n = len(z)

    survival_probability = scipy.special.expit(draw_f_prior(models["m_surv"], x))
    survived = np.random.binomial(1, survival_probability)
    survivor = survived == 1

    z1 = np.full(n, np.nan)
    if survivor.any():
        mean = draw_f_prior(models["m_grw"], x[survivor])
        sd = np.sqrt(float(models["m_grw"].likelihood.variance.numpy()))
        z1[survivor] = mean + np.random.normal(0, sd, survivor.sum())

    reproduced = np.full(n, np.nan)
    if survivor.any():
        probability = scipy.special.expit(draw_f_prior(models["m_repr"], x[survivor]))
        reproduced[survivor] = np.random.binomial(1, probability)
    breeder = survivor & (reproduced == 1)

    sex = np.full(n, np.nan)
    if breeder.any():
        sex[breeder] = np.random.binomial(1, FEMALE_PROBABILITY, breeder.sum())
    female = breeder & (sex == 1)

    recruited = np.full(n, np.nan)
    if female.any():
        recruited[female] = np.random.binomial(1, recruit_probability, female.sum())
    recruit = female & (recruited == 1)

    recruit_size = np.full(n, np.nan)
    if recruit.any():
        mean = draw_f_prior(models["m_rcsz"], x[recruit])
        sd = np.sqrt(float(models["m_rcsz"].likelihood.variance.numpy()))
        recruit_size[recruit] = mean + np.random.normal(0, sd, recruit.sum())

    return pd.DataFrame({
        "z": z, 
        "Surv": survived, 
        "z1": z1, 
        "Repr": reproduced,
        "Sex": sex, 
        "Recr": recruited, 
        "Rcsz": recruit_size,
    })


def make_ibm_function(recruit_probability):
    """Return the two-argument IBM function required by ABC_GP_IPM"""
    recruit_probability = float(recruit_probability)

    def IBM_1step(dataset, models):
        return simulate_one_year(
            dataset, models, recruit_probability=recruit_probability
        )

    return IBM_1step

class PriorABC_GP_IPM(ABC_GP_IPM):
    """
    Override the package’s cache-based particle evaluation so ABC-PMC uses uncached, 
    unconditional GP prior draws. This is done to match the DeepRV sampling procedure.
    """

    def IPM_whole_givenindex(self, index):
        '''
        Evaluates one proposed ABC particle by simulating the population through all observed years
        and calculates summary statistics for ABC.

        Args:
            index: List of indices of MCMC samples, one index for each vital rate.
        Returns:
            index: List of indices of MCMC samples, one index for each vital rate.
            [summaries]: List of lists - each list contains a Numpy array of statistics for each year.
        '''
        models = self.GPmodel.copy()

        # Assign the selected ABC parameters to the corresponding GPflow model for that vital rate.
        for position, name in enumerate(self.MCMC_samples):
            helper = gpflow.optimizers.SamplingHelper(
                models[name].log_posterior_density,
                models[name].trainable_parameters,
            )
            for variable, samples in zip(
                helper.current_state, self.MCMC_samples[name]
            ):
                variable.assign(samples[int(index[position])])

        summaries = []

        # Simulate first year 
        simulated = self.fun_IBM_cache(dataset=self.ini_popu, models=models, **self.fun_IBM_kwargs) 

        # Iteratively simulate datasets and compare them to the observed dataset
        for time_step in range(len(self.POPUdata_dict)):
            if len(simulated) <= 1:             
                # If the population has gone extinct, set the statistics to inf, ensuring ABC rejection                    
                failed = [np.full(len(self.name_ss), np.inf) for _ in self.POPUdata_dict]
                return index, [failed]
            observed = self.POPUdata_dict[time_step]
            # Compute summary statistics between simulated and observed dataset for this year
            summaries.append(self.fun_ss(exp_dataset=observed, simu_dataset=simulated)) 
            if time_step + 1 < len(self.POPUdata_dict):
                population = self.fun_structure(
                    simulated,
                    time_step=time_step + 1,
                    **self.fun_structure_kwargs,
                )
                # Simulate one more year if needed
                simulated = self.fun_IBM_cache(
                    dataset=population, models=models, **self.fun_IBM_kwargs
                )
        return index, [summaries]


def population_structure(data_simu, time_step):
    survivors = data_simu.loc[data_simu["Surv"] == 1, "z1"].dropna().to_numpy()
    recruits = data_simu.loc[data_simu["Recr"] == 1, "Rcsz"].dropna().to_numpy()
    return pd.DataFrame({"z": np.concatenate([survivors, recruits])})


DISTANCE_NAMES = ["bhattacharyya", "hellinger", "emd", "kl", "hilbert"]
DISTRIBUTION_GROUPS = ["next_size", "breeder_next_size", "nonbreeder_next_size", "breeder_nonbreeder_survival",]
SUMMARY_NAMES = [
    f"{distance}_{group}_distribution"
    for group in DISTRIBUTION_GROUPS
    for distance in DISTANCE_NAMES
] + [
    "chi2_total_individuals_next_size",
    "ssd_total_individuals_next_size",
    "chi2_total_recruits_by_maternal_size",
    "ssd_total_recruits_by_maternal_size",
    "chi2_total_breeders_by_size",
    "ssd_total_breeders_by_size",
    "chi2_total_survivors_by_size",
    "ssd_total_survivors_by_size",
    "ssd_average_recruits_by_maternal_size",
    "ssd_average_breeders_by_size",
    "ssd_average_survivors_by_size",
]


def grouped_counts(data, size_edges):
    '''
    Group individual records into size-based counts and event proportions for ABC comparison.

    Args:
        data: DataFrame containing size, survival, reproduction and recruitment records for one year.
        size_edges: Numpy array of shape (M+1,) defining the boundaries used to divide individuals into M size groups.

    Returns:
        Dictionary containing next-year size distributions and size-grouped counts and proportions for survival, reproduction and recruitment.
    '''
    def column(name):
        if name not in data:
            return np.full(len(data), np.nan)
        return pd.to_numeric(data[name], errors="coerce").to_numpy(float)

    z, z1, recruit_size, survival, reproduction  = column("z"), column("z1"), column("Rcsz"),  column("Surv"), column("Repr")
    valid = np.isfinite(z)
    current = np.histogram(z[valid], size_edges)[0].astype(float)
    next_sizes = np.concatenate([z1[np.isfinite(z1)], recruit_size[np.isfinite(recruit_size)]])
    survivor = (survival == 1) & np.isfinite(z1)
    breeder = survivor & (reproduction == 1)
    nonbreeder = survivor & (reproduction != 1)
    breeder_survival = np.bincount(
        2 * (reproduction == 1).astype(int) + (survival == 1).astype(int),
        minlength=4,
    ).astype(float)

    def events(name):
        indicator = np.where(column(name) == 1, 1.0, 0.0)
        return np.histogram(z[valid], size_edges, weights=indicator[valid])[0]

    survivors, breeders, recruits = events("Surv"), events("Repr"), events("Recr")

    def average(events_by_size):
        return np.divide(events_by_size, current, out=np.zeros_like(current), where=current > 0)

    return {
        "next": np.histogram(next_sizes, size_edges)[0].astype(float),
        "breeder_next": np.histogram(z1[breeder], size_edges)[0].astype(float),
        "nonbreeder_next": np.histogram(z1[nonbreeder], size_edges)[0].astype(float),
        "breeder_survival": breeder_survival,
        "survivors": survivors, 
        "breeders": breeders, 
        "recruits": recruits,
        "average_survivors": average(survivors),
        "average_breeders": average(breeders),
        "average_recruits": average(recruits),
    }


def ssd(observed, simulated):
    return np.sum((np.asarray(observed) - np.asarray(simulated)) ** 2)

def chi2(observed, expected):
    '''
    Calculate the chi-squared distance, sum((observed - expected)² / expected), 
    which measures disagreement between observed and simulated counts. 
    Small expected counts can produce unstable values and zero expected counts cause division by zero, 
    so neighbouring groups are merged until expected counts reach five.
    SSD is used if all expected counts are zero.
    '''
    observed, expected = np.asarray(observed), np.asarray(expected)
    merged_observed, merged_expected = [], []
    current_observed = current_expected = 0.0
    for obs, exp in zip(observed, expected):
        current_observed += obs
        current_expected += exp
        if current_expected >= 5:
            merged_observed.append(current_observed)
            merged_expected.append(current_expected)
            current_observed = current_expected = 0.0
    if current_observed or current_expected:
        if merged_expected:
            merged_observed[-1] += current_observed
            merged_expected[-1] += current_expected
        else:
            merged_observed.append(current_observed)
            merged_expected.append(current_expected)
    merged_observed, merged_expected = np.asarray(merged_observed), np.asarray(merged_expected)
    positive = merged_expected > 0
    return ssd(observed, expected) if not positive.any() else np.sum(
        (merged_observed[positive] - merged_expected[positive]) ** 2
        / merged_expected[positive]
    )

def distribution_distances(observed, simulated, size_positions):
    observed = np.asarray(observed, dtype=float).ravel() + 1e-10
    simulated = np.asarray(simulated, dtype=float).ravel() + 1e-10
    observed /= observed.sum()
    simulated /= simulated.sum()
    coefficient = np.clip(np.sum(np.sqrt(observed * simulated)), 1e-15, 1.0)
    return [
        -np.log(coefficient),                                                           # Bhattacharyya
        np.sqrt(0.5 * np.sum((np.sqrt(observed) - np.sqrt(simulated)) ** 2)),           # Hellinger
        wasserstein_distance(size_positions, size_positions,                            # EMD/ Wasserstein
                             u_weights=observed, v_weights=simulated),      
        np.sum(observed * np.log(observed / simulated)),                                # KL
        np.log(np.max(observed / simulated)) + np.log(np.max(simulated / observed)),    # Hilbert
    ]

def candidate_summaries(exp_dataset, simu_dataset, size_edges, size_positions):
    observed = grouped_counts(exp_dataset, size_edges)
    simulated = grouped_counts(simu_dataset, size_edges)
    return np.asarray(distribution_distances(
        observed["next"], simulated["next"], size_positions
    ) + distribution_distances(
        observed["breeder_next"], simulated["breeder_next"], size_positions
    ) + distribution_distances(
        observed["nonbreeder_next"], simulated["nonbreeder_next"], size_positions
    ) + distribution_distances(
        observed["breeder_survival"], simulated["breeder_survival"], np.arange(4)
    ) + [
        chi2(observed["next"], simulated["next"]),
        ssd(observed["next"], simulated["next"]),
        chi2(observed["recruits"], simulated["recruits"]),
        ssd(observed["recruits"], simulated["recruits"]),
        chi2(observed["breeders"], simulated["breeders"]),
        ssd(observed["breeders"], simulated["breeders"]),
        chi2(observed["survivors"], simulated["survivors"]),
        ssd(observed["survivors"], simulated["survivors"]),
        ssd(observed["average_recruits"], simulated["average_recruits"]),
        ssd(observed["average_breeders"], simulated["average_breeders"]),
        ssd(observed["average_survivors"], simulated["average_survivors"]),
    ])

def make_candidate_summary_function(size_edges, size_positions):
    '''
    size_edges: The six boundaries defining the five size groups.
    size_positions: The numerical positions representing those five groups.
    '''
    size_edges = np.asarray(size_edges, dtype=float).copy()
    size_positions = np.asarray(size_positions, dtype=float).copy()

    def summary_function(exp_dataset, simu_dataset):
        return candidate_summaries(exp_dataset, simu_dataset, size_edges, size_positions)
    return summary_function

# Restricts the candidate set of summary statistics considered for each vital rate.
INFORMATIVE = {
    "m_grw": [name for name in SUMMARY_NAMES if "next_size" in name],
    "m_surv": SUMMARY_NAMES,
    "m_repr": [
        name for name in SUMMARY_NAMES
        if "next_size_distribution" in name
        or "total_individuals_next_size" in name
        or "recruit" in name
        or "breeder" in name
    ],
    "m_rcsz": [name for name in SUMMARY_NAMES if "next_size" in name],
}

def direction_adjusted_auc_diagnostics(summary_opt, summary_near):
    '''
    Measure how well each summary statistic distinguishes optimum simulations from nearby 
    posterior simulations using direction-adjusted ROC AUC.

    N_opt:  Number of simulated datasets using optimum parameter
    N_near: Total number of simulated datasets using all 'nearby' perturbed parameters.
    N_s:    Number of statistics used

    Args:
        summary_opt: DataFrame of summary statistics generated using the optimum parameters.    (N_opt, N_s)
        summary_near: DataFrame of summary statistics generated using nearby posterior samples, (N_near, N_s+1)
                      grouped by the "i" column. 

    Returns:
        DataFrame indexed by summary-statistic name containing the 
        median and maximum direction-adjusted AUC across nearby sample groups. (N_s, 2)
    '''
    auc_rows = []
    sample_indices = []
    for sample_index, group in summary_near.groupby("i", sort=False):
        labels = np.concatenate([np.ones(len(summary_opt)), np.zeros(len(group)),])
        row = {}
        for name in SUMMARY_NAMES:
            scores = np.concatenate([
                summary_opt[name].to_numpy(),
                group[name].to_numpy(),
            ])
            finite = np.isfinite(scores)
            finite_labels = labels[finite]
            if finite_labels.size == 0 or np.unique(finite_labels).size < 2:
                row[name] = 0.5
                continue
            auc = roc_auc_score(finite_labels, scores[finite])
            row[name] = max(auc, 1.0 - auc)
        auc_rows.append(row)
        sample_indices.append(int(sample_index))

    auc_by_sample = pd.DataFrame(auc_rows, index=sample_indices)
    return pd.DataFrame({
        "median_auc": auc_by_sample.median(),
        "maximum_auc": auc_by_sample.max(),
    })


def select_summaries(
    ipm, target, initial_population, candidate_summary_function, ibm,
):
    selector = Perted_IPM(
        GP_IPM_instance=ipm,
        target=target,
        fun_ss=candidate_summary_function,
        name_ss=SUMMARY_NAMES,
        opt_percentage=OPTIMUM_PERCENTAGE,
        fun_IBM=ibm,
    )
    negative_log_posterior = -np.asarray(selector.log_posterior_density())
    optimum = selector.ss_optVSopt(
        SELECTION_REPETITIONS, 123, negative_log_posterior,
        initial_population, min(N_CORES, SELECTION_REPETITIONS),
    )
    number_near = np.sum(
        negative_log_posterior
        <= np.percentile(negative_log_posterior, OPTIMUM_PERCENTAGE)
    )
    near = selector.ss_optVSmcmc(
        SELECTION_REPETITIONS, 456, negative_log_posterior,
        initial_population, min(N_CORES, int(number_near)),
    )
    selector.nlog_post = negative_log_posterior
    frequencies = selector.top_ss(
        optimum, near, N_SELECTED_PER_RATE,
        threshold=AUC_THRESHOLD, return_full=True,
    ).sum()
    diagnostics = direction_adjusted_auc_diagnostics(optimum, near)
    eligible_frequencies = frequencies.loc[INFORMATIVE[target]].sort_values(
        ascending=False, kind="stable"
    )
    selected = eligible_frequencies[eligible_frequencies > 0].index[
        :N_SELECTED_PER_RATE
    ].tolist()

    if selected:
        selection_method = "auc_threshold"
    else:
        eligible_diagnostics = diagnostics.loc[INFORMATIVE[target]].sort_values(
            ["median_auc", "maximum_auc"],
            ascending=False,
            kind="stable",
        )
        selected = [eligible_diagnostics.index[0]]
        selection_method = "fallback_best_median_auc"

    selected_name = selected[0]
    pass_count = int(frequencies[selected_name])
    near_count = int(number_near)
    print(
        f"{target}: selected {selected} via {selection_method}",
        flush=True,
    )
    return {
        "selected": selected,
        "selection_method": selection_method,
        "auc_summary": {
            "statistic": selected_name,
            "selection_method": selection_method,
            "threshold_pass_count": pass_count,
            "near_optimum_samples": near_count,
            "threshold_pass_fraction": pass_count / near_count,
            "median_adjusted_auc": float(
                diagnostics.loc[selected_name, "median_auc"]
            ),
            "maximum_adjusted_auc": float(
                diagnostics.loc[selected_name, "maximum_auc"]
            ),
        },
    }


def fit_mle_models(data):
    models = {
        "m_grw": gpflow.models.GPR(
            XY_m_grw_compu(data), gpflow.kernels.SquaredExponential(),
            gpflow.mean_functions.Constant(c=0.0)),
        "m_surv": gpflow.models.VGP(
            XY_m_surv_compu(data), gpflow.kernels.SquaredExponential(),
            gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
            gpflow.mean_functions.Constant(c=0.0)),
        "m_repr": gpflow.models.VGP(
            XY_m_repr_compu(data), gpflow.kernels.SquaredExponential(),
            gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
            gpflow.mean_functions.Constant(c=0.0)),
        "m_rcsz": gpflow.models.GPR(
            XY_m_rcsz_compu(data), gpflow.kernels.SquaredExponential(),
            gpflow.mean_functions.Constant(c=0.0)),
    }
    for model in models.values():
        gpflow.optimizers.Scipy().minimize(
            model.training_loss, model.trainable_variables,
            options={"maxiter": 20},
        )
    return models


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

    if min(map(len, all_samples.values())) < N_DRAWS:
        raise ValueError(f"The posterior archive contains fewer than {N_DRAWS} draws")
    
    samples = {name: value[:N_DRAWS] for name, value in all_samples.items()}
    recruit_probability = float(scipy.special.expit(all_samples["recr_int"]).mean())
    print(
        f"Fixed recruitment probability: {recruit_probability:.8f}",
        flush=True,
    )

    pooled_size = pd.concat([data["z"], data["z1"], data["Rcsz"]]).dropna()
    size_edges = np.r_[-np.inf, np.quantile(pooled_size, [.2, .4, .6, .8]), np.inf,]    # Splits sizes into 5 bins based on quantiles
    size_positions = np.arange(5, dtype=float)                                          # Creates the representative positions used to calculate EMD.
    candidate_summary_function = make_candidate_summary_function(
        size_edges, size_positions
    )
    ibm = make_ibm_function(recruit_probability)

    years = sorted(data["yr"].unique())
    observed_by_year = {i: data.loc[data["yr"] == year].copy() for i, year in enumerate(years)}
    initial_population = data.loc[data["yr"] == years[0], ["z"]].copy()
    mle_models = fit_mle_models(data)
    models = {
        "m_grw": build_new_m_grw(data),
        "m_surv": build_new_m_surv(data),
        "m_repr": build_new_m_repr(data),
        "m_rcsz": build_new_m_rcsz(data),
    }
    mcmc_samples = {
        "m_grw": gpr_samples(models["m_grw"], samples, "growth"),
        "m_surv": gpmc_samples(models["m_surv"], samples, "surv"),
        "m_repr": gpmc_samples(models["m_repr"], samples, "repr"),
        "m_rcsz": gpr_samples(models["m_rcsz"], samples, "rcsz"),
    }
    ipm = GP_IPM(data, observed_by_year, mle_models, models, mcmc_samples)
    for function in [XY_m_grw_compu, XY_m_surv_compu, XY_m_repr_compu, XY_m_rcsz_compu]:
        ipm.add_XY_compu(function)
    for function in [build_new_m_grw, build_new_m_surv, build_new_m_repr, build_new_m_rcsz]:
        ipm.add_model_build(function)

    # Begin SS selection and ABC-PMC
    ray_temp_directory = f"/tmp/ray-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    ray.init(num_cpus=N_CORES, include_dashboard=False, log_to_driver=False, _temp_dir=ray_temp_directory,)
    try:
        selection_start = time.perf_counter()

        # Select summary statistics
        selection_results = {
            target: select_summaries(ipm, target, initial_population, candidate_summary_function, ibm,)
            for target in ["m_grw", "m_surv", "m_repr", "m_rcsz"]
        }
        selected = {
            target: result["selected"]
            for target, result in selection_results.items()
        }
        selection_seconds = time.perf_counter() - selection_start

        selected_names = [
            name for name in SUMMARY_NAMES
            if any(name in names for names in selected.values())
        ]

        def abc_summaries(exp_dataset, simu_dataset):
            values = dict(zip(
                SUMMARY_NAMES,
                candidate_summary_function(exp_dataset, simu_dataset),
            ))
            return np.asarray([values[name] for name in selected_names])

        # Use our defined object for GP prior sampling during IBMs
        abc = PriorABC_GP_IPM(
            ipm, abc_summaries, selected_names, initial_population,
            ibm, population_structure,
        )
        details_directory = work_directory / "ABC_details"
        details_directory.mkdir(exist_ok=True)
        abc_start = time.perf_counter()

        # Perform ABC-PMC
        indices, threshold = abc.ABC_PMC(ABC_QUANTILES, 
                                         ABC_PARTICLES, 
                                         N_CORES, details=False,
                                         smallest_unit=ABC_BATCH_SIZE, 
                                         file_path2store=str(details_directory),)
        abc_seconds = time.perf_counter() - abc_start
    finally:
        ray.shutdown()

    np.savez_compressed(
        output_directory / "abc_samples.npz",
        indices=np.asarray(indices),
        weights=np.asarray(abc.weight),
        threshold=np.asarray(threshold),
    )
    run_info = {
        "gp_sampling": "prior",
        "conditioning": "unconditional",
        "recruit_probability": recruit_probability,
        "female_probability": FEMALE_PROBABILITY,
        "size_quantile_boundaries": size_edges[1:-1].tolist(),
        "selected_summary_statistics": selected,
        "summary_selection_methods": {target: result["selection_method"] for target, result in selection_results.items()},
        "summary_selection_auc": {target: result["auc_summary"] for target, result in selection_results.items()},
        "selection_seconds": selection_seconds,
        "cache_seconds": 0.0,
        "abc_sampling_seconds": abc_seconds,
        "total_seconds": time.perf_counter() - total_start,
    }
    (output_directory / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_info, indent=2))
