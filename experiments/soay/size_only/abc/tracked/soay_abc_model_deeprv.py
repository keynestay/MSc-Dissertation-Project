#!/usr/bin/env python3
"""DeepRV Soay ABC workflow with per-stage rejection tracking."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import scipy.special
from ABC_GP_IPM.gp_ipm import GP_IPM

from experiments.soay.size_only.abc import soay_abc_model as base
from experiments.soay.size_only.abc import soay_abc_model_deeprv as deeprv_base
from experiments.soay.size_only.abc.tracked.soay_abc_model import run as run_normal
from experiments.soay.size_only.abc.tracked.tracked_abc_gp_ipm import TrackedABC_GP_IPM


class TrackedDeepRVABC_GP_IPM(
    deeprv_base.DeepRVABC_GP_IPM, TrackedABC_GP_IPM
):
    pass


def run(
    data_file,
    samples_file,
    output_directory,
    use_deeprv=False,
    checkpoint_path=None,
    selection_file=None,
):
    if not use_deeprv:
        return run_normal(data_file, samples_file, output_directory)
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required when use_deeprv=True")
    if selection_file is None:
        raise ValueError("selection_file is required when use_deeprv=True")
    selection_file, selection_run_info, selected, selected_names, size_edges = (
        deeprv_base.load_summary_selection(selection_file)
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

        abc = TrackedDeepRVABC_GP_IPM(
            ipm,
            abc_summaries,
            selected_names,
            initial_population,
            deeprv_base.make_deeprv_ibm_function(
                checkpoint_path, recruit_probability
            ),
            base.population_structure,
            posterior_samples=samples,
        )
        abc_start = time.perf_counter()
        indices, threshold = abc.ABC_PMC(
            base.ABC_QUANTILES,
            base.ABC_PARTICLES,
            deeprv_base.N_DEEPRV_WORKERS,
            details=False,
            smallest_unit=base.ABC_BATCH_SIZE,
            file_path2store=str(work_directory / "ABC_details"),
        )
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
        "use_deeprv": True,
        "deeprv_sampling": "raw_decoder",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "summary_selection_file": str(selection_file),
        "deeprv_workers": deeprv_base.N_DEEPRV_WORKERS,
        "recruit_probability": recruit_probability,
        "female_probability": base.FEMALE_PROBABILITY,
        "size_quantile_boundaries": size_edges[1:-1].tolist(),
        "selected_summary_statistics": selected,
        "summary_selection_methods": selection_run_info.get(
            "summary_selection_methods", {}
        ),
        "abc_stage_diagnostics": abc.stage_diagnostics,
        "abc_sampling_seconds": abc_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    (output_directory / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    (output_directory / "abc_stage_diagnostics.json").write_text(
        json.dumps(abc.stage_diagnostics, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_info, indent=2))
