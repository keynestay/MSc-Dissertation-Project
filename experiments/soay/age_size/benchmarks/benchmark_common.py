"""Shared design and output helpers for the age-size IBM speed benchmark."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAMES = ("m_grw", "m_surv", "m_repr", "m_recr", "m_rcsz")
PARAMETER_FIELDS = {
    "m_grw": ("growth_int", "growth_var", "growth_ls", "growth_noise"),
    "m_surv": ("surv_int", "surv_var", "surv_ls"),
    "m_repr": ("repr_int", "repr_var", "repr_ls"),
    "m_recr": ("recr_int", "recr_var", "recr_ls"),
    "m_rcsz": ("rcsz_int", "rcsz_var", "rcsz_ls", "rcsz_noise"),
}
PARAMETER_NAMES = (
    "growth_int",
    "growth_var",
    "growth_ls_size",
    "growth_ls_age",
    "growth_noise",
    "surv_int",
    "surv_var",
    "surv_ls_size",
    "surv_ls_age",
    "repr_int",
    "repr_var",
    "repr_ls_size",
    "repr_ls_age",
    "recr_int",
    "recr_var",
    "recr_ls_age",
    "rcsz_int",
    "rcsz_var",
    "rcsz_ls_size",
    "rcsz_noise",
)
FEMALE_PROBABILITY = 0.5
GREYBEARD_AGE = 8
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
    required = [field for fields in PARAMETER_FIELDS.values() for field in fields]
    with np.load(samples_path) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"Missing posterior arrays: {missing}")
        samples = {name: np.asarray(archive[name]) for name in required}

    scalar_fields = [name for name in required if not name.endswith("_ls")]
    for name in scalar_fields:
        if samples[name].ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional, received {samples[name].shape}"
            )

    dimensions = {
        "growth_ls": 2,
        "surv_ls": 2,
        "repr_ls": 2,
        "recr_ls": 1,
        "rcsz_ls": 1,
    }
    for name, dimension in dimensions.items():
        values = samples[name]
        if values.ndim == 1 and dimension == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != dimension:
            raise ValueError(
                f"{name} must have shape (n, {dimension}), received {values.shape}"
            )
        samples[name] = values
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
        fields = PARAMETER_FIELDS[model_name]
        columns.extend([samples[fields[0]][selected], samples[fields[1]][selected]])
        lengthscales = samples[fields[2]][selected]
        columns.extend(
            lengthscales[:, dimension] for dimension in range(lengthscales.shape[1])
        )
        if len(fields) == 4:
            columns.append(samples[fields[3]][selected])

    values = np.column_stack(columns)
    if values.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("Unexpected parameter matrix width")
    if not np.isfinite(values).all():
        raise ValueError("Selected parameter values must be finite")
    for name in ("growth_var", "surv_var", "repr_var", "recr_var", "rcsz_var"):
        if np.any(values[:, PARAMETER_NAMES.index(name)] < 0):
            raise ValueError(f"{name} must be non-negative")
    for name in (
        "growth_ls_size",
        "growth_ls_age",
        "surv_ls_size",
        "surv_ls_age",
        "repr_ls_size",
        "repr_ls_age",
        "recr_ls_age",
        "rcsz_ls_size",
    ):
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
    initial_year = design["initial_year"].item()
    initial_population = (
        data.loc[data["yr"] == initial_year, ["z", "age"]].dropna().copy()
    )
    if len(initial_population) <= 1:
        raise ValueError("The initial population must contain at least two individuals")
    if not np.allclose(initial_population["age"], np.round(initial_population["age"])):
        raise ValueError("Initial ages must be integers")
    initial_population["age"] = initial_population["age"].astype(np.int32)

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
        "initial_year": initial_year,
        "initial_sizes": initial_population["z"].to_numpy(float),
        "initial_ages": initial_population["age"].to_numpy(np.int32),
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
