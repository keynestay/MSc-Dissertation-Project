import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import scipy.special
from ABC_GP_IPM.abc_gp_ipm import ABC_GP_IPM
from ABC_GP_IPM.gp_ipm import GP_IPM

from experiments.soay.size_only.abc import soay_abc_model as base
from experiments.soay.size_only.training.deeprv_decoder import get_decoder

PREFIXES = {"m_grw": "growth", 
            "m_surv": "surv", 
            "m_repr": "repr", 
            "m_rcsz": "rcsz"}

N_DEEPRV_WORKERS = int(os.environ.get("DEEPRV_WORKERS", "1"))


def load_summary_selection(selection_file):
    '''
    Load and validate the summary statistics and size boundaries saved by the standard GP ABC run.

    Args:
        selection_file: Path to the run_info.json file containing the selected summary statistics and size boundaries.

    Returns:
        Tuple containing the resolved file path, complete run-information dictionary, 
        statistics selected for each vital rate, combined ordered list of selected statistics 
        and six size-bin edges.
    '''
    selection_file = Path(selection_file).resolve()
    run_info = json.loads(selection_file.read_text(encoding="utf-8"))
    selected = run_info.get("selected_summary_statistics")
    if not isinstance(selected, dict):
        raise ValueError(
            f"{selection_file} does not contain selected_summary_statistics"
        )
    for target in PREFIXES:
        names = selected.get(target)
        if not isinstance(names, list) or not names:
            raise ValueError(f"No summary statistics were saved for {target}")
        unknown = set(names) - set(base.SUMMARY_NAMES)
        if unknown:
            raise ValueError(f"Unknown summary statistics for {target}: {sorted(unknown)}")
    boundaries = np.asarray(run_info.get("size_quantile_boundaries"), dtype=float)
    if boundaries.shape != (4,) or not np.all(np.isfinite(boundaries)):
        raise ValueError("size_quantile_boundaries must contain four finite values")
    if not np.all(np.diff(boundaries) > 0):
        raise ValueError("size_quantile_boundaries must be strictly increasing")
    selected_names = [
        name
        for name in base.SUMMARY_NAMES
        if any(name in selected[target] for target in PREFIXES)
    ]
    size_edges = np.r_[-np.inf, boundaries, np.inf]
    return selection_file, run_info, selected, selected_names, size_edges

def simulate_one_year_deeprv(data, params, checkpoint_path, recruit_probability):
    '''
    Simulate one annual population transition using DeepRV vital-rate draws.
    Args:
        data: DataFrame containing the current size of each individual in the "z" column.
        params: Dict containing the selected GP parameters for each vital rate and observation noise for growth and recruit size.
        checkpoint_path: Path to the trained DeepRV decoder checkpoint.
        recruit_probability: Probability that a surviving female breeder successfully recruits.

    Returns:
        DataFrame with one row per initial individual containing current size, survival, next size, reproduction, sex, recruitment and recruit size.
    '''
    runtime = get_runtime(checkpoint_path)
    def draw(vital_rate, x):
        model_params = params[vital_rate]
        return runtime.sample(
            x,
            model_params["intercept"],
            model_params["variance"],
            model_params["lengthscale"],
        )

    z = data["z"].to_numpy(float)
    x = z[:, None]
    n = len(z)
    survival_probability = scipy.special.expit(draw("m_surv", x))
    survived = np.random.binomial(1, survival_probability)
    survivor = survived == 1
    z1 = np.full(n, np.nan)
    if survivor.any():
        z1[survivor] = draw("m_grw", x[survivor]) + np.random.normal(
            0, params["m_grw"]["noise"], survivor.sum()
        )
    reproduced = np.full(n, np.nan)
    if survivor.any():
        reproduced[survivor] = np.random.binomial(
            1, scipy.special.expit(draw("m_repr", x[survivor]))
        )
    breeder = survivor & (reproduced == 1)
    sex = np.full(n, np.nan)
    if breeder.any():
        sex[breeder] = np.random.binomial(1, base.FEMALE_PROBABILITY, breeder.sum())
    female = breeder & (sex == 1)
    recruited = np.full(n, np.nan)
    if female.any():
        recruited[female] = np.random.binomial(1, recruit_probability, female.sum())
    recruit = female & (recruited == 1)
    recruit_size = np.full(n, np.nan)
    if recruit.any():
        recruit_size[recruit] = draw("m_rcsz", x[recruit]) + np.random.normal(
            0, params["m_rcsz"]["noise"], recruit.sum()
        )
    return pd.DataFrame(
        {
            "z": z,
            "Surv": survived,
            "z1": z1,
            "Repr": reproduced,
            "Sex": sex,
            "Recr": recruited,
            "Rcsz": recruit_size,
        }
    )


def make_deeprv_ibm_function(checkpoint_path, recruit_probability):
    checkpoint_path = str(Path(checkpoint_path).resolve())

    def IBM_1step(dataset, models):
        return simulate_one_year_deeprv(
            dataset, models, checkpoint_path, float(recruit_probability)
        )

    return IBM_1step


class DeepRVABC_GP_IPM(ABC_GP_IPM):
    '''
    Adapts ABC_GP_IPM to use DeepRV for vital-rate function draws.
    The class retains the package's ABC-PMC sampling, weighting and acceptance procedure, 
    but converts each particle's posterior-sample indices into GP parameter values and 
    evaluates that particle using the DeepRV IBM instead of GPflow.
    '''

    def __init__(self, *args, posterior_samples, **kwargs):
        super().__init__(*args, **kwargs)
        self.posterior_samples = posterior_samples

    def select_models(self, index):
        '''
        Select the GP parameters represented by one ABC particle.

        Args:
            index: Array containing one posterior-sample index for each vital-rate model.
        Returns:
            Dictionary containing the selected intercept, variance and lengthscale for each vital rate, 
            including observation noise for growth and recruit size.
        '''
        selected_params = {}
        for k, name in enumerate(self.MCMC_samples):
            prefix, i = PREFIXES[name], int(index[k])
            selected_params[name] = {
                "intercept": self.posterior_samples[f"{prefix}_int"][i],
                "variance": self.posterior_samples[f"{prefix}_var"][i],
                "lengthscale": self.posterior_samples[f"{prefix}_ls"][i],
            }
            if prefix in ("growth", "rcsz"):
                selected_params[name]["noise"] = self.posterior_samples[f"{prefix}_noise"][i]
        return selected_params

    def IPM_whole_givenindex(self, index):
        models = self.selected_models(index)
        summaries, data_simu = ([], self.fun_IBM_cache( dataset=self.ini_popu, models=models, **self.fun_IBM_kwargs),)
        for t in range(len(self.POPUdata_dict)):
            if len(data_simu) <= 1:
                return index, [
                    [np.full(len(self.name_ss), np.inf) for _ in self.POPUdata_dict]
                ]
            observed = self.POPUdata_dict[t]
            simulated = (
                self.fun_capture(
                    data_simu=data_simu, time_step=t, **self.fun_capture_kwargs
                )
                if self.capture
                else data_simu
            )
            summaries.append(self.fun_ss(exp_dataset=observed, simu_dataset=simulated))
            if t + 1 < len(self.POPUdata_dict):
                population = self.fun_structure(
                    data_simu, time_step=t + 1, **self.fun_structure_kwargs
                )
                data_simu = self.fun_IBM_cache(
                    dataset=population, models=models, **self.fun_IBM_kwargs
                )
        return index, [summaries]


def run(
    data_file,
    samples_file,
    output_directory,
    use_deeprv=False,
    checkpoint_path=None,
    selection_file=None,
):
    if not use_deeprv:
        return base.run(data_file, samples_file, output_directory)
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required when use_deeprv=True")
    if selection_file is None:
        raise ValueError("selection_file is required when use_deeprv=True")
    selection_file, selection_run_info, selected, selected_names, size_edges = (
        load_summary_selection(selection_file)
    )
    data_file, samples_file, output_directory = (
        Path(data_file).resolve(),
        Path(samples_file).resolve(),
        Path(output_directory).resolve(),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    work_directory = Path(os.environ.get("TMPDIR", output_directory / "work"))
    work_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = pd.read_csv(data_file)
    with np.load(samples_file) as archive:
        recruit_probability = float(scipy.special.expit(archive["recr_int"]).mean())
        samples = {
            name: archive[name][: base.N_DRAWS]
            for name in archive.files
            if archive[name].ndim > 0
        }
    if min(len(value) for value in samples.values()) < base.N_DRAWS:
        raise ValueError(
            f"The posterior archive contains fewer than {base.N_DRAWS} draws"
        )
    candidate_summary = base.make_candidate_summary_function(
        size_edges, np.arange(5, dtype=float)
    )
    years = sorted(data["yr"].unique())
    observed_by_year = {
        i: data.loc[data["yr"] == year].copy() for i, year in enumerate(years)
    }
    initial_population = data.loc[data["yr"] == years[0], ["z"]].copy()
    mle_models = base.fit_mle_models(data)
    models = {
        "m_grw": base.build_new_m_grw(data),
        "m_surv": base.build_new_m_surv(data),
        "m_repr": base.build_new_m_repr(data),
        "m_rcsz": base.build_new_m_rcsz(data),
    }
    mcmc_samples = {
        "m_grw": base.gpr_samples(models["m_grw"], samples, "growth"),
        "m_surv": base.gpmc_samples(models["m_surv"], samples, "surv"),
        "m_repr": base.gpmc_samples(models["m_repr"], samples, "repr"),
        "m_rcsz": base.gpr_samples(models["m_rcsz"], samples, "rcsz"),
    }
    ipm = GP_IPM(data, observed_by_year, mle_models, models, mcmc_samples)
    for function in [
        base.XY_m_grw_compu,
        base.XY_m_surv_compu,
        base.XY_m_repr_compu,
        base.XY_m_rcsz_compu,
    ]:
        ipm.add_XY_compu(function)
    for function in [
        base.build_new_m_grw,
        base.build_new_m_surv,
        base.build_new_m_repr,
        base.build_new_m_rcsz,
    ]:
        ipm.add_model_build(function)
    ray.init(
        num_cpus=base.N_CORES,
        include_dashboard=False,
        log_to_driver=False,
        _temp_dir=f"/tmp/ray-{os.environ.get('SLURM_JOB_ID', os.getpid())}",
    )
    try:
        def abc_summaries(exp_dataset, simu_dataset):
            values = dict(
                zip(base.SUMMARY_NAMES, candidate_summary(exp_dataset, simu_dataset))
            )
            return np.asarray([values[name] for name in selected_names])

        abc = DeepRVABC_GP_IPM(
            ipm,
            abc_summaries,
            selected_names,
            initial_population,
            make_deeprv_ibm_function(checkpoint_path, recruit_probability),
            base.population_structure,
            posterior_samples=samples,
        )
        indices, threshold = abc.ABC_PMC(
            base.ABC_QUANTILES,
            base.ABC_PARTICLES,
            N_DEEPRV_WORKERS,
            details=False,
            smallest_unit=base.ABC_BATCH_SIZE,
            file_path2store=str(work_directory / "ABC_details"),
        )
    finally:
        ray.shutdown()
    np.savez_compressed(
        output_directory / "abc_samples.npz",
        indices=np.asarray(indices),
        weights=np.asarray(abc.weight),
        threshold=np.asarray(threshold),
    )
    run_info = {
        "use_deeprv": True,
        "deeprv_sampling": "raw_decoder",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "summary_selection_file": str(selection_file),
        "deeprv_workers": N_DEEPRV_WORKERS,
        "recruit_probability": recruit_probability,
        "female_probability": base.FEMALE_PROBABILITY,
        "size_quantile_boundaries": size_edges[1:-1].tolist(),
        "selected_summary_statistics": selected,
        "summary_selection_methods": selection_run_info.get("summary_selection_methods", {}),
        "total_seconds": time.perf_counter() - started,
    }
    (output_directory / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_info, indent=2))
