#!/usr/bin/env python3
"""Normal Soay ABC workflow with per-stage rejection tracking."""

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
from experiments.soay.size_only.abc.tracked.tracked_abc_gp_ipm import TrackedABC_GP_IPM


class TrackedPriorABC_GP_IPM(base.PriorABC_GP_IPM, TrackedABC_GP_IPM):
    """Exact-prior ABC with per-stage proposal diagnostics."""


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
    if min(map(len, all_samples.values())) < base.N_DRAWS:
        raise ValueError(
            f"The posterior archive contains fewer than {base.N_DRAWS} draws"
        )
    samples = {
        name: value[: base.N_DRAWS] for name, value in all_samples.items()
    }
    recruit_probability = float(
        scipy.special.expit(all_samples["recr_int"]).mean()
    )
    if not 0.0 <= recruit_probability <= 1.0:
        raise ValueError("Recruitment probability must lie between zero and one")
    print(f"Fixed recruitment probability: {recruit_probability:.8f}", flush=True)

    pooled_size = pd.concat([data["z"], data["z1"], data["Rcsz"]]).dropna()
    size_edges = np.r_[
        -np.inf,
        np.quantile(pooled_size, [0.2, 0.4, 0.6, 0.8]),
        np.inf,
    ]
    candidate_summary = base.make_candidate_summary_function(
        size_edges, np.arange(5, dtype=float)
    )
    ibm = base.make_ibm_function(recruit_probability)

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
        selection_start = time.perf_counter()
        selection_results = {
            target: base.select_summaries(
                ipm,
                target,
                initial_population,
                candidate_summary,
                ibm,
            )
            for target in ["m_grw", "m_surv", "m_repr", "m_rcsz"]
        }
        selected = {
            target: result["selected"]
            for target, result in selection_results.items()
        }
        selection_seconds = time.perf_counter() - selection_start
        selected_names = [
            name
            for name in base.SUMMARY_NAMES
            if any(name in names for names in selected.values())
        ]

        def abc_summaries(exp_dataset, simu_dataset):
            values = dict(
                zip(
                    base.SUMMARY_NAMES,
                    candidate_summary(exp_dataset, simu_dataset),
                )
            )
            return np.asarray([values[name] for name in selected_names])

        abc = TrackedPriorABC_GP_IPM(
            ipm,
            abc_summaries,
            selected_names,
            initial_population,
            ibm,
            base.population_structure,
        )
        details_directory = work_directory / "ABC_details"
        details_directory.mkdir(exist_ok=True)
        abc_start = time.perf_counter()
        indices, threshold = abc.ABC_PMC(
            base.ABC_QUANTILES,
            base.ABC_PARTICLES,
            base.N_CORES,
            details=False,
            smallest_unit=base.ABC_BATCH_SIZE,
            file_path2store=str(details_directory),
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
        "gp_sampling": "prior",
        "conditioning": "unconditional",
        "recruit_probability": recruit_probability,
        "female_probability": base.FEMALE_PROBABILITY,
        "size_quantile_boundaries": size_edges[1:-1].tolist(),
        "selected_summary_statistics": selected,
        "summary_selection_methods": {
            target: result["selection_method"]
            for target, result in selection_results.items()
        },
        "summary_selection_auc": {
            target: result["auc_summary"]
            for target, result in selection_results.items()
        },
        "abc_stage_diagnostics": abc.stage_diagnostics,
        "selection_seconds": selection_seconds,
        "cache_seconds": 0.0,
        "abc_sampling_seconds": abc_seconds,
        "total_seconds": time.perf_counter() - total_start,
    }
    (output_directory / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    (output_directory / "abc_stage_diagnostics.json").write_text(
        json.dumps(abc.stage_diagnostics, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_info, indent=2))
