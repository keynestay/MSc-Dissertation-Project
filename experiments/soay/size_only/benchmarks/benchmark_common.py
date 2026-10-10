"""Shared design and output helpers for the IBM speed benchmark."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAMES = ("m_grw", "m_surv", "m_repr", "m_rcsz")
PARAMETER_FIELDS = {
    "m_grw": ("growth_int", "growth_var", "growth_ls", "growth_noise"),
    "m_surv": ("surv_int", "surv_var", "surv_ls"),
    "m_repr": ("repr_int", "repr_var", "repr_ls"),
    "m_rcsz": ("rcsz_int", "rcsz_var", "rcsz_ls", "rcsz_noise"),
}
PARAMETER_NAMES = tuple(
    field for model_name in MODEL_NAMES for field in PARAMETER_FIELDS[model_name]
)
FEMALE_PROBABILITY = 0.5
JITTER = 5e-4


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def load_samples(samples_path):
    required = list(PARAMETER_NAMES) + ["recr_int"]
    with np.load(samples_path) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"Missing posterior arrays: {missing}")
        samples = {name: np.asarray(archive[name]) for name in required}
    for name, values in samples.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, received {values.shape}")
    return samples


def sample_counts(samples):
    counts = []
    for model_name in MODEL_NAMES:
        lengths = {len(samples[field]) for field in PARAMETER_FIELDS[model_name]}
        if len(lengths) != 1:
            raise ValueError(
                f"Posterior arrays for {model_name} have different lengths"
            )
        counts.append(lengths.pop())
    return np.asarray(counts, dtype=np.int64)


def parameter_values(samples, indices):
    columns = []
    for model_column, model_name in enumerate(MODEL_NAMES):
        selected = indices[:, model_column]
        for field in PARAMETER_FIELDS[model_name]:
            columns.append(samples[field][selected])
    values = np.column_stack(columns)
    if not np.isfinite(values).all():
        raise ValueError("Selected parameter values must be finite")
    for name in ("growth_var", "surv_var", "repr_var", "rcsz_var"):
        if np.any(values[:, PARAMETER_NAMES.index(name)] < 0):
            raise ValueError(f"{name} must be non-negative")
    for name in ("growth_ls", "surv_ls", "repr_ls", "rcsz_ls"):
        if np.any(values[:, PARAMETER_NAMES.index(name)] <= 0):
            raise ValueError(f"{name} must be positive")
    for name in ("growth_noise", "rcsz_noise"):
        if np.any(values[:, PARAMETER_NAMES.index(name)] < 0):
            raise ValueError(f"{name} must be non-negative")
    return values


def load_benchmark(data_path, design_path):
    data_path = Path(data_path).resolve()
    design_path = Path(design_path).resolve()
    with np.load(design_path) as archive:
        design = {name: np.asarray(archive[name]) for name in archive.files}

    if file_sha256(data_path) != str(design["data_sha256"].item()):
        raise ValueError("The dataset does not match the benchmark design")
    saved_names = tuple(str(name) for name in design["parameter_names"])
    if saved_names != PARAMETER_NAMES:
        raise ValueError("The parameter columns do not match this benchmark version")

    data = pd.read_csv(data_path)
    if "initial_years" in design:
        initial_years = np.atleast_1d(design["initial_years"])
    else:
        initial_years = np.atleast_1d(design["initial_year"])
    available_years = data["yr"].dropna().unique()
    if not np.isin(initial_years, available_years).all():
        raise ValueError("The benchmark design contains unknown initial years")
    if "initial_sizes" in design:
        initial_sizes = np.asarray(design["initial_sizes"], dtype=float).ravel()
    else:
        initial_sizes = (
            data.loc[data["yr"].isin(initial_years), "z"].dropna().to_numpy(float)
        )
    if len(initial_sizes) <= 1:
        raise ValueError("The initial population must contain at least two individuals")
    if not np.isfinite(initial_sizes).all():
        raise ValueError("The initial population sizes must be finite")
    initial_unique_locations = int(np.unique(initial_sizes).size)
    if "initial_population_size" in design:
        if int(design["initial_population_size"].item()) != len(initial_sizes):
            raise ValueError("The initial population does not match the design")
    if "initial_unique_locations" in design:
        if int(design["initial_unique_locations"].item()) != initial_unique_locations:
            raise ValueError(
                "The initial unique-location count does not match the design"
            )

    values = np.asarray(design["parameter_values"], dtype=float)
    seeds = np.asarray(design["task_seeds"], dtype=np.uint32)
    number_datasets = int(design["number_datasets"].item())
    if values.shape != (number_datasets, len(PARAMETER_NAMES)):
        raise ValueError("Unexpected parameter matrix shape")
    if seeds.shape != (number_datasets,):
        raise ValueError("Unexpected task seed shape")

    return {
        "parameter_indices": np.asarray(design["parameter_indices"], dtype=np.int64),
        "parameter_values": values,
        "task_seeds": seeds,
        "number_datasets": number_datasets,
        "number_years": int(design["number_years"].item()),
        "experiment_seed": int(design["experiment_seed"].item()),
        "initial_year": initial_years[0].item(),
        "initial_years": initial_years.tolist(),
        "initial_sizes": initial_sizes,
        "initial_population_method": str(
            design.get("initial_population_method", np.asarray("observed")).item()
        ),
        "initial_unique_locations": initial_unique_locations,
        "recruit_probability": float(design["recruit_probability"].item()),
        "data_path": str(data_path),
        "design_path": str(design_path),
        "design_sha256": file_sha256(design_path),
        "samples_sha256": str(design["samples_sha256"].item()),
        "data": data,
    }


def combine_outputs(parts):
    combined = {
        name: np.concatenate([part[name] for part in parts], axis=0)
        for name in parts[0]
    }
    order = np.argsort(combined["task_id"])
    return {name: values[order] for name, values in combined.items()}


def output_summary(outputs):
    return {
        "extinctions": int(np.sum(outputs["extinction_year"] >= 0)),
        "overflows": int(np.sum(outputs["overflow"])),
        "total_requested_locations": int(np.sum(outputs["requested_locations"])),
        "total_unique_requested_locations": int(
            np.sum(outputs["unique_requested_locations"])
        ),
        "total_population_sizes": int(np.sum(outputs["population_sizes"])),
        "mean_final_population_size": float(np.mean(outputs["final_population_size"])),
        "mean_final_size_total": float(np.mean(outputs["final_size_total"])),
    }
