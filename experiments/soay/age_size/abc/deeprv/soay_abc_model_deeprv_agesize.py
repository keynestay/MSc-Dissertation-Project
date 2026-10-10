#!/usr/bin/env python3
"""Tracked age-size Soay ABC with a Transformer DeepRV IBM."""

import json
import os
import time
from pathlib import Path

os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gpflow
import numpy as np
import pandas as pd
import ray
import scipy.special
from ABC_GP_IPM.gp_ipm import GP_IPM

from experiments.soay.age_size.abc.gp import soay_abc_model_agesize as base
from experiments.soay.age_size.abc.gp.tracked_abc_gp_ipm_agesize import TrackedABC_GP_IPM_AgeSize
from experiments.soay.age_size.abc.deeprv.deeprv_decoder_agesize import (
    COORDINATE_ORDER,
    get_runtime,
)

PREFIXES = {"m_grw": "growth", "m_surv": "surv", "m_repr": "repr", "m_recr": "recr", "m_rcsz": "rcsz"}
DEEPRV_RATES = {"m_grw", "m_surv", "m_repr"}
N_DEEPRV_WORKERS = int(os.environ.get("DEEPRV_WORKERS", "1"))


class PopulationTooLargeError(RuntimeError):
    pass


def load_summary_selection(selection_file):
    selection_file = Path(selection_file).resolve()
    run_info = json.loads(selection_file.read_text(encoding="utf-8"))
    if run_info.get("gp_sampling") != "prior" or run_info.get("conditioning") != "unconditional":
        raise ValueError("Summary selection must come from unconditional prior GP sampling")
    selected = run_info.get("selected_summary_statistics")
    if not isinstance(selected, dict):
        raise ValueError(f"{selection_file} does not contain selected_summary_statistics")
    for target in PREFIXES:
        names = selected.get(target)
        if not isinstance(names, list) or not names:
            raise ValueError(f"No summary statistics were saved for {target}")
        unknown = set(names) - set(base.SUMMARY_NAMES)
        if unknown:
            raise ValueError(f"Unknown summary statistics for {target}: {sorted(unknown)}")
    boundaries = np.asarray(run_info.get("size_quantile_boundaries"), dtype=float)
    if boundaries.shape != (4,) or not np.all(np.isfinite(boundaries)) or not np.all(np.diff(boundaries) > 0):
        raise ValueError("size_quantile_boundaries must contain four increasing finite values")
    selected_names = [name for name in base.SUMMARY_NAMES if any(name in selected[target] for target in PREFIXES)]
    return selection_file, run_info, selected, selected_names, np.r_[-np.inf, boundaries, np.inf]


def load_checkpoint_metadata(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    metadata_path = checkpoint_path.parent / "metadata.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = metadata.get("model", {})
    if int(model.get("input_dimensions", -1)) != 2 or tuple(model.get("coordinate_order", ())) != COORDINATE_ORDER:
        raise ValueError("Checkpoint metadata does not describe a [size, age] model")
    if int(model.get("max_locations", -1)) != 512:
        raise ValueError("The age-size DeepRV checkpoint must support 512 locations")
    return checkpoint_path, metadata


def record_population_overflow(path):
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, b"1\n")
    finally:
        os.close(descriptor)


def simulate_one_year_deeprv(data, models, checkpoint_path, overflow_file):
    size = data["z"].to_numpy(float)
    age = data["age"].to_numpy(int)
    number = len(data)
    columns = ["z", "age", "Surv", "z1", "Repr", "Sex", "Recr", "Rcsz"]
    if number == 0:
        return pd.DataFrame(columns=columns)

    runtime = get_runtime(checkpoint_path)
    if number > runtime.max_locations:
        record_population_overflow(overflow_file)
        raise PopulationTooLargeError(f"Population has {number} locations; DeepRV supports {runtime.max_locations}")

    def draw(name, locations):
        model = models[name]
        return runtime.sample(locations, model["intercept"], model["variance"], model["lengthscales"])

    age_size = np.column_stack([size, age])
    survival_probability = np.clip(scipy.special.expit(draw("m_surv", age_size)), 0, 1)
    survived = np.random.binomial(1, survival_probability)
    survivor = survived == 1

    next_size = np.full(number, np.nan)
    if survivor.any():
        growth_mean = draw("m_grw", age_size[survivor])
        next_size[survivor] = growth_mean + np.random.normal(0, models["m_grw"]["noise"], survivor.sum())

    reproduced = np.full(number, np.nan)
    reproduced[survivor] = 0
    eligible = survivor & (age >= base.MIN_REPRODUCTIVE_AGE)
    if eligible.any():
        reproduction_probability = np.clip(scipy.special.expit(draw("m_repr", age_size[eligible])), 0, 1)
        reproduced[eligible] = np.random.binomial(1, reproduction_probability)
    breeder = eligible & (reproduced == 1)

    sex = np.full(number, np.nan)
    if breeder.any():
        sex[breeder] = np.random.binomial(1, base.FEMALE_PROBABILITY, breeder.sum())
    female = breeder & (sex == 1)

    recruited = np.full(number, np.nan)
    if female.any():
        recruitment_probability = np.clip(scipy.special.expit(base.draw_f_prior(models["m_recr"], age[female, None])), 0, 1)
        recruited[female] = np.random.binomial(1, recruitment_probability)
    recruit = female & (recruited == 1)

    recruit_size = np.full(number, np.nan)
    if recruit.any():
        recruit_mean = base.draw_f_prior(models["m_rcsz"], size[recruit, None])
        recruit_sd = np.sqrt(float(models["m_rcsz"].likelihood.variance.numpy()))
        recruit_size[recruit] = recruit_mean + np.random.normal(0, recruit_sd, recruit.sum())

    return pd.DataFrame({"z": size, "age": age, "Surv": survived, "z1": next_size, "Repr": reproduced, "Sex": sex, "Recr": recruited, "Rcsz": recruit_size}, columns=columns)


def make_deeprv_ibm_function(checkpoint_path, overflow_file):
    checkpoint_path = str(Path(checkpoint_path).resolve())
    overflow_file = str(Path(overflow_file).resolve())

    def IBM_1step(dataset, models):
        return simulate_one_year_deeprv(dataset, models, checkpoint_path, overflow_file)

    return IBM_1step


class DeepRVABC_GP_IPM_AgeSize(TrackedABC_GP_IPM_AgeSize):
    def __init__(self, *args, posterior_samples, **kwargs):
        super().__init__(*args, **kwargs)
        self.posterior_samples = posterior_samples

    def selected_models(self, index):
        models = self.GPmodel.copy()
        for position, name in enumerate(self.MCMC_samples):
            sample_index = int(index[position])
            if name in DEEPRV_RATES:
                prefix = PREFIXES[name]
                models[name] = {
                    "intercept": self.posterior_samples[f"{prefix}_int"][sample_index],
                    "variance": self.posterior_samples[f"{prefix}_var"][sample_index],
                    "lengthscales": self.posterior_samples[f"{prefix}_ls"][sample_index],
                }
                if name == "m_grw":
                    models[name]["noise"] = self.posterior_samples["growth_noise"][sample_index]
            else:
                helper = gpflow.optimizers.SamplingHelper(models[name].log_posterior_density, models[name].trainable_parameters)
                for variable, samples in zip(helper.current_state, self.MCMC_samples[name]):
                    variable.assign(samples[sample_index])
        return models

    def IPM_whole_givenindex(self, index):
        try:
            models = self.selected_models(index)
            summaries = []
            simulated = self.fun_IBM_cache(dataset=self.ini_popu, models=models, **self.fun_IBM_kwargs)
            if len(simulated) <= 1:
                return self._infinite_result(index)
            compared = simulated if not self.capture else self.fun_capture(data_simu=simulated, time_step=0, **self.fun_capture_kwargs)
            summaries.append(self.fun_ss(exp_dataset=self.POPUdata_dict[0], simu_dataset=compared))
            for time_step in range(1, len(self.POPUdata_dict)):
                population = self.fun_structure(data_simu=simulated, time_step=time_step, **self.fun_structure_kwargs)
                simulated = self.fun_IBM_cache(dataset=population, models=models, **self.fun_IBM_kwargs)
                if len(simulated) <= 1:
                    return self._infinite_result(index)
                compared = simulated if not self.capture else self.fun_capture(data_simu=simulated, time_step=time_step, **self.fun_capture_kwargs)
                summaries.append(self.fun_ss(exp_dataset=self.POPUdata_dict[time_step], simu_dataset=compared))
            return index, [summaries]
        except PopulationTooLargeError:
            return self._infinite_result(index)


def run(data_file, samples_file, checkpoint_path, output_directory, selection_file):
    if N_DEEPRV_WORKERS < 1:
        raise ValueError("DEEPRV_WORKERS must be positive")
    data_file = Path(data_file).resolve()
    samples_file = Path(samples_file).resolve()
    checkpoint_path, checkpoint_metadata = load_checkpoint_metadata(checkpoint_path)
    selection_file, selection_run_info, selected, selected_names, size_edges = load_summary_selection(selection_file)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    work_directory = Path(os.environ.get("TMPDIR", output_directory / "work"))
    work_directory.mkdir(parents=True, exist_ok=True)
    overflow_file = work_directory / "population_overflows.txt"
    overflow_file.write_text("", encoding="utf-8")
    started = time.perf_counter()

    data = pd.read_csv(data_file)
    with np.load(samples_file) as archive:
        all_samples = {name: archive[name] for name in archive.files}
    base.validate_inputs(data, all_samples)
    samples = {name: values[: base.N_DRAWS] for name, values in all_samples.items()}

    years = sorted(data["yr"].unique())
    observed_by_year = {index: data.loc[data["yr"] == year].copy() for index, year in enumerate(years)}
    initial_population = data.loc[data["yr"] == years[0], ["z", "age"]].dropna().copy()
    mle_models = base.fit_mle_models(data)
    models = {
        "m_grw": base.build_new_m_grw(data),
        "m_surv": base.build_new_m_surv(data),
        "m_repr": base.build_new_m_repr(data),
        "m_recr": base.build_new_m_recr(data),
        "m_rcsz": base.build_new_m_rcsz(data),
    }
    mcmc_samples = {
        "m_grw": base.gpr_samples(models["m_grw"], samples, "growth"),
        "m_surv": base.gpmc_samples(models["m_surv"], samples, "surv"),
        "m_repr": base.gpmc_samples(models["m_repr"], samples, "repr"),
        "m_recr": base.gpmc_samples(models["m_recr"], samples, "recr"),
        "m_rcsz": base.gpr_samples(models["m_rcsz"], samples, "rcsz"),
    }
    ipm = GP_IPM(data, observed_by_year, mle_models, models, mcmc_samples)
    for function in [base.XY_m_grw_compu, base.XY_m_surv_compu, base.XY_m_repr_compu, base.XY_m_recr_compu, base.XY_m_rcsz_compu]:
        ipm.add_XY_compu(function)
    for function in [base.build_new_m_grw, base.build_new_m_surv, base.build_new_m_repr, base.build_new_m_recr, base.build_new_m_rcsz]:
        ipm.add_model_build(function)

    ray.init(num_cpus=base.N_CORES, include_dashboard=False, log_to_driver=False, _temp_dir=f"/tmp/ray-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    try:
        def abc_summaries(exp_dataset, simu_dataset):
            values = base.candidate_summary_values(exp_dataset, simu_dataset, size_edges)
            return np.asarray([values[name] for name in selected_names])

        abc = DeepRVABC_GP_IPM_AgeSize(
            ipm,
            abc_summaries,
            selected_names,
            initial_population,
            make_deeprv_ibm_function(checkpoint_path, overflow_file),
            base.population_structure,
            posterior_samples=samples,
        )
        abc_started = time.perf_counter()
        indices, threshold = abc.ABC_PMC(
            base.ABC_QUANTILES,
            base.ABC_PARTICLES,
            N_DEEPRV_WORKERS,
            details=False,
            smallest_unit=base.ABC_BATCH_SIZE,
            file_path2store=str(work_directory / "ABC_details"),
        )
        abc_seconds = time.perf_counter() - abc_started
    finally:
        ray.shutdown()

    overflow_count = sum(1 for line in overflow_file.read_text(encoding="utf-8").splitlines() if line)
    np.savez_compressed(output_directory / "abc_samples.npz", indices=np.asarray(indices), weights=np.asarray(abc.weight), threshold=np.asarray(threshold))
    final_diagnostics = abc.stage_diagnostics[-1]
    run_info = {
        "use_deeprv": True,
        "deeprv_sampling": "raw_two_dimensional_decoder",
        "gp_sampling": "prior",
        "conditioning": "unconditional",
        "deeprv_rates": sorted(DEEPRV_RATES),
        "exact_gp_rates": sorted(set(PREFIXES) - DEEPRV_RATES),
        "checkpoint": str(checkpoint_path),
        "checkpoint_evaluation_norm_mse": checkpoint_metadata.get("evaluation_norm_mse"),
        "coordinate_order": list(COORDINATE_ORDER),
        "max_locations": 512,
        "population_overflows": overflow_count,
        "fresh_gp_draw_per_rate_and_year": True,
        "posterior_draws_per_vital_rate": base.N_DRAWS,
        "vital_rate_order": list(models),
        "summary_selection_file": str(selection_file),
        "deeprv_workers": N_DEEPRV_WORKERS,
        "female_probability": base.FEMALE_PROBABILITY,
        "maximum_natural_age": base.MAX_AGE,
        "greybeard_age": base.GREYBEARD_AGE,
        "minimum_reproductive_age": base.MIN_REPRODUCTIVE_AGE,
        "size_quantile_boundaries": size_edges[1:-1].tolist(),
        "selected_summary_statistics": selected,
        "summary_selection_methods": selection_run_info.get("summary_selection_methods", {}),
        "abc_stage_diagnostics": abc.stage_diagnostics,
        "total_abc_proposals": final_diagnostics["cumulative_proposed"],
        "total_abc_acceptances": final_diagnostics["cumulative_accepted"],
        "total_abc_rejections": final_diagnostics["cumulative_rejected"],
        "final_particle_count": int(len(indices)),
        "abc_sampling_seconds": abc_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    (output_directory / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    (output_directory / "abc_stage_diagnostics.json").write_text(json.dumps(abc.stage_diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(run_info, indent=2), flush=True)
