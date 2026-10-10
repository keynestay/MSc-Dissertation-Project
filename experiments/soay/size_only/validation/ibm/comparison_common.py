"""Shared Soay IBM and discrepancy code for simulator validation."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special
from scipy.stats import wasserstein_distance


JITTER = 5e-4
FEMALE_PROBABILITY = 0.5
MODEL_NAMES = ("m_grw", "m_surv", "m_repr", "m_rcsz")
METHOD_EXACT_GP = 1
METHOD_DEEPRV = 2
STATUS_SUCCESS = 0
STATUS_EXTINCTION = 1
STATUS_MAX_LOCATIONS = 2
STATUS_ERROR = 3

SUMMARY_NAMES = [
    "bhattacharyya_next_size_distribution",
    "hellinger_next_size_distribution",
    "emd_next_size_distribution",
    "kl_next_size_distribution",
    "hilbert_next_size_distribution",
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

PARAMETER_FIELDS = {
    "m_grw": ("growth_int", "growth_var", "growth_ls", "growth_noise"),
    "m_surv": ("surv_int", "surv_var", "surv_ls"),
    "m_repr": ("repr_int", "repr_var", "repr_ls"),
    "m_rcsz": ("rcsz_int", "rcsz_var", "rcsz_ls", "rcsz_noise"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_existing_chunks(method_directory, design_hash, number_chunks):
    method_directory = Path(method_directory)
    if not list(method_directory.glob("chunk_*.npz")):
        return
    run_info_path = method_directory / "run_info.json"
    if not run_info_path.exists():
        raise ValueError(f"Existing chunks in {method_directory} have no run_info.json")
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    if run_info.get("design_sha256") != design_hash:
        raise ValueError(f"Existing chunks in {method_directory} use another design")
    if run_info.get("number_chunks") != number_chunks:
        raise ValueError(
            f"Existing chunks in {method_directory} use a different chunk layout"
        )


def load_compact_samples(samples_path):
    required = [field for fields in PARAMETER_FIELDS.values() for field in fields]
    required.append("recr_int")
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
        fields = PARAMETER_FIELDS[model_name]
        field_counts = {len(samples[field]) for field in fields}
        if len(field_counts) != 1:
            raise ValueError(f"Posterior arrays for {model_name} have different lengths")
        counts.append(field_counts.pop())
    return np.asarray(counts, dtype=np.int64)


def parameter_names():
    return np.asarray(
        [field for model_name in MODEL_NAMES for field in PARAMETER_FIELDS[model_name]]
    )


def parameter_values(samples, indices):
    columns = []
    for model_column, model_name in enumerate(MODEL_NAMES):
        selected = indices[:, model_column]
        for field in PARAMETER_FIELDS[model_name]:
            columns.append(samples[field][selected])
    return np.column_stack(columns)


def selected_models(samples, index):
    index = np.asarray(index, dtype=int)
    models = {}
    for model_column, model_name in enumerate(MODEL_NAMES):
        sample_index = index[model_column]
        fields = PARAMETER_FIELDS[model_name]
        model = {
            "intercept": float(samples[fields[0]][sample_index]),
            "variance": float(samples[fields[1]][sample_index]),
            "lengthscale": float(samples[fields[2]][sample_index]),
        }
        if len(fields) == 4:
            model["noise"] = float(samples[fields[3]][sample_index])
        if model["variance"] < 0 or model["lengthscale"] <= 0:
            raise ValueError(f"Invalid kernel parameters for {model_name}: {model}")
        if model.get("noise", 0.0) < 0:
            raise ValueError(f"Invalid observation noise for {model_name}: {model}")
        models[model_name] = model
    return models


def make_task_rng(experiment_seed, method, parameter_id, replicate_id):
    seed = np.random.SeedSequence(
        [int(experiment_seed), int(method), int(parameter_id), int(replicate_id)]
    )
    return np.random.default_rng(seed)


def make_task_seed(experiment_seed, method, parameter_id, replicate_id):
    seed = np.random.SeedSequence(
        [int(experiment_seed), int(method), int(parameter_id), int(replicate_id), 1]
    )
    return int(seed.generate_state(1, dtype=np.uint32)[0])


def ordered_unique(values):
    unique, positions, inverse = [], {}, []
    for value in np.asarray(values, dtype=float).reshape(-1):
        key = float(value)
        if key not in positions:
            positions[key] = len(unique)
            unique.append(value)
        inverse.append(positions[key])
    return np.asarray(unique), np.asarray(inverse, dtype=int)


def population_structure(simulated):
    survivors = simulated.loc[simulated["Surv"] == 1, "z1"].dropna().to_numpy()
    recruits = simulated.loc[simulated["Recr"] == 1, "Rcsz"].dropna().to_numpy()
    return pd.DataFrame({"z": np.concatenate([survivors, recruits])})


def simulate_one_year(population, models, draw_function, recruit_probability, rng):
    sizes = population["z"].to_numpy(float)
    number_individuals = len(sizes)

    survival_probability = scipy.special.expit(
        draw_function("m_surv", sizes, models["m_surv"])
    )
    survived = rng.binomial(1, survival_probability)
    survivor = survived == 1

    next_size = np.full(number_individuals, np.nan)
    if survivor.any():
        growth = draw_function("m_grw", sizes[survivor], models["m_grw"])
        next_size[survivor] = growth + rng.normal(
            0, models["m_grw"]["noise"], survivor.sum()
        )

    reproduced = np.full(number_individuals, np.nan)
    if survivor.any():
        reproduction_probability = scipy.special.expit(
            draw_function("m_repr", sizes[survivor], models["m_repr"])
        )
        reproduced[survivor] = rng.binomial(1, reproduction_probability)
    breeder = survivor & (reproduced == 1)

    sex = np.full(number_individuals, np.nan)
    if breeder.any():
        sex[breeder] = rng.binomial(1, FEMALE_PROBABILITY, breeder.sum())
    female = breeder & (sex == 1)

    recruited = np.full(number_individuals, np.nan)
    if female.any():
        recruited[female] = rng.binomial(1, recruit_probability, female.sum())
    recruit = female & (recruited == 1)

    recruit_size = np.full(number_individuals, np.nan)
    if recruit.any():
        expected_size = draw_function("m_rcsz", sizes[recruit], models["m_rcsz"])
        recruit_size[recruit] = expected_size + rng.normal(
            0, models["m_rcsz"]["noise"], recruit.sum()
        )

    return pd.DataFrame(
        {
            "z": sizes,
            "Surv": survived,
            "z1": next_size,
            "Repr": reproduced,
            "Sex": sex,
            "Recr": recruited,
            "Rcsz": recruit_size,
        }
    )


def grouped_counts(data, size_edges):
    def column(name):
        if name not in data:
            return np.full(len(data), np.nan)
        return pd.to_numeric(data[name], errors="coerce").to_numpy(float)

    size = column("z")
    next_size = column("z1")
    recruit_size = column("Rcsz")
    valid = np.isfinite(size)
    current = np.histogram(size[valid], size_edges)[0].astype(float)
    next_sizes = np.concatenate(
        [next_size[np.isfinite(next_size)], recruit_size[np.isfinite(recruit_size)]]
    )

    def events(name):
        indicator = np.where(column(name) == 1, 1.0, 0.0)
        return np.histogram(size[valid], size_edges, weights=indicator[valid])[0]

    survivors = events("Surv")
    breeders = events("Repr")
    recruits = events("Recr")

    def average(events_by_size):
        return np.divide(
            events_by_size,
            current,
            out=np.zeros_like(current),
            where=current > 0,
        )

    return {
        "next": np.histogram(next_sizes, size_edges)[0].astype(float),
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
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    merged_observed, merged_expected = [], []
    current_observed = current_expected = 0.0
    for observed_value, expected_value in zip(observed, expected):
        current_observed += observed_value
        current_expected += expected_value
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
    merged_observed = np.asarray(merged_observed)
    merged_expected = np.asarray(merged_expected)
    positive = merged_expected > 0
    if not positive.any():
        return ssd(observed, expected)
    return np.sum(
        (merged_observed[positive] - merged_expected[positive]) ** 2
        / merged_expected[positive]
    )


def distribution_distances(observed, simulated, size_positions):
    observed = (np.asarray(observed) + 1e-10) / (np.sum(observed) + 5e-10)
    simulated = (np.asarray(simulated) + 1e-10) / (np.sum(simulated) + 5e-10)
    coefficient = np.clip(np.sum(np.sqrt(observed * simulated)), 1e-15, 1.0)
    return [
        -np.log(coefficient),
        np.sqrt(0.5 * np.sum((np.sqrt(observed) - np.sqrt(simulated)) ** 2)),
        wasserstein_distance(
            size_positions, size_positions, u_weights=observed, v_weights=simulated
        ),
        np.sum(observed * np.log(observed / simulated)),
        np.log(np.max(observed / simulated))
        + np.log(np.max(simulated / observed)),
    ]


def candidate_summaries(observed_data, simulated_data, size_edges):
    observed = grouped_counts(observed_data, size_edges)
    simulated = grouped_counts(simulated_data, size_edges)
    size_positions = np.arange(5, dtype=float)
    return np.asarray(
        distribution_distances(observed["next"], simulated["next"], size_positions)
        + [
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
        ],
        dtype=float,
    )


def simulate_dataset(
    models,
    draw_function,
    initial_population,
    observed_by_year,
    size_edges,
    recruit_probability,
    rng,
):
    number_years = len(observed_by_year)
    distances = np.full((number_years, len(SUMMARY_NAMES)), np.nan)
    population_sizes = np.full(number_years, np.nan)
    population = initial_population.copy()
    status = STATUS_SUCCESS
    extinction_year = -1

    for year_index, observed in enumerate(observed_by_year):
        if len(population) <= 1:
            distances[year_index:] = np.inf
            population_sizes[year_index:] = len(population)
            status = STATUS_EXTINCTION
            extinction_year = year_index
            break
        simulated = simulate_one_year(
            population, models, draw_function, recruit_probability, rng
        )
        distances[year_index] = candidate_summaries(
            observed, simulated, size_edges
        )
        population = population_structure(simulated)
        population_sizes[year_index] = len(population)
        if len(population) <= 1 and year_index == number_years - 1:
            status = STATUS_EXTINCTION
            extinction_year = number_years

    return distances, population_sizes, status, extinction_year


def make_task_chunks(number_parameters, number_replicates, target_chunk_size=100, minimum_chunks=1):
    tasks = np.column_stack(
        [
            np.repeat(np.arange(number_parameters), number_replicates),
            np.tile(np.arange(number_replicates), number_parameters),
        ]
    )
    if len(tasks) == 0:
        return []
    number_chunks = max(minimum_chunks, math.ceil(len(tasks) / target_chunk_size))
    number_chunks = min(len(tasks), number_chunks)
    return [chunk for chunk in np.array_split(tasks, number_chunks) if len(chunk)]


def load_experiment(data_path, samples_path, design_path):
    data_path = Path(data_path).resolve()
    samples_path = Path(samples_path).resolve()
    design_path = Path(design_path).resolve()
    with np.load(design_path) as design_file:
        design = {name: np.asarray(design_file[name]) for name in design_file.files}

    expected_data_hash = str(design["data_sha256"].item())
    expected_samples_hash = str(design["samples_sha256"].item())
    if file_sha256(data_path) != expected_data_hash:
        raise ValueError("The dataset does not match the benchmark design")
    if file_sha256(samples_path) != expected_samples_hash:
        raise ValueError("The MCMC archive does not match the benchmark design")

    data = pd.read_csv(data_path)
    years = design["years"]
    observed_by_year = [data.loc[data["yr"] == year].copy() for year in years]
    if any(len(observed) == 0 for observed in observed_by_year):
        raise ValueError("At least one design year is absent from the dataset")
    initial_population = data.loc[data["yr"] == years[0], ["z"]].copy()

    return {
        "samples": load_compact_samples(samples_path),
        "parameter_indices": design["parameter_indices"].astype(np.int64),
        "parameter_values": design["parameter_values"],
        "parameter_names": design["parameter_names"],
        "number_parameters": int(design["number_parameters"].item()),
        "number_replicates": int(design["number_replicates"].item()),
        "experiment_seed": int(design["experiment_seed"].item()),
        "years": years,
        "size_edges": design["size_edges"],
        "recruit_probability": float(design["recruit_probability"].item()),
        "initial_population": initial_population,
        "observed_by_year": observed_by_year,
        "design_path": str(design_path),
        "design_sha256": file_sha256(design_path),
    }


def empty_task_result(parameter_id, replicate_id, number_years):
    return {
        "parameter_id": int(parameter_id),
        "replicate_id": int(replicate_id),
        "distances": np.full((number_years, len(SUMMARY_NAMES)), np.nan),
        "population_sizes": np.full(number_years, np.nan),
        "status": STATUS_ERROR,
        "extinction_year": -1,
        "seconds": np.nan,
        "error": "",
    }


def combine_task_results(results):
    return {
        "parameter_id": np.asarray([result["parameter_id"] for result in results]),
        "replicate_id": np.asarray([result["replicate_id"] for result in results]),
        "distances": np.asarray([result["distances"] for result in results]),
        "population_sizes": np.asarray(
            [result["population_sizes"] for result in results]
        ),
        "status": np.asarray([result["status"] for result in results]),
        "extinction_year": np.asarray(
            [result["extinction_year"] for result in results]
        ),
        "seconds": np.asarray([result["seconds"] for result in results]),
        "error": np.asarray([result["error"] for result in results], dtype="U500"),
    }


def save_chunk(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **combine_task_results(results))


def status_counts(status):
    status = np.asarray(status)
    return {
        "success": int(np.sum(status == STATUS_SUCCESS)),
        "extinction": int(np.sum(status == STATUS_EXTINCTION)),
        "max_locations": int(np.sum(status == STATUS_MAX_LOCATIONS)),
        "error": int(np.sum(status == STATUS_ERROR)),
    }
