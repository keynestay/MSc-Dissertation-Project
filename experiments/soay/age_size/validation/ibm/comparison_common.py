"""Shared design, IBM, discrepancy, and output helpers for age-size validation."""

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
MIN_REPRODUCTIVE_AGE = 1
GREYBEARD_AGE = 8
MAX_LOCATIONS = 512
METHOD_EXACT_GP = 1
METHOD_DEEPRV = 2
STATUS_SUCCESS = 0
STATUS_EXTINCTION = 1
STATUS_MAX_LOCATIONS = 2
STATUS_ERROR = 3

MODEL_NAMES = ("m_grw", "m_surv", "m_repr", "m_recr", "m_rcsz")
SUMMARY_NAMES = (
    "emd_breeder_next_size_distribution",
    "chi2_total_survivors_by_size_age",
    "chi2_total_breeders_by_size",
    "chi2_total_recruits_by_age",
    "bhattacharyya_nonbreeder_next_size_distribution",
)
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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_sha256(path):
    path = Path(path).resolve()
    if path.is_file():
        return file_sha256(path)
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        with file.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_existing_chunks(
    method_directory, design_hash, number_chunks, checkpoint_hash=None
):
    method_directory = Path(method_directory)
    if not list(method_directory.glob("chunk_*.npz")):
        return
    run_info_path = method_directory / "run_info.json"
    if not run_info_path.exists():
        raise ValueError(f"Existing chunks in {method_directory} have no run_info.json")
    run_info = read_json(run_info_path)
    if run_info.get("design_sha256") != design_hash:
        raise ValueError(f"Existing chunks in {method_directory} use another design")
    if run_info.get("number_chunks") != number_chunks:
        raise ValueError(
            f"Existing chunks in {method_directory} use a different chunk layout"
        )
    if (
        checkpoint_hash is not None
        and run_info.get("checkpoint_sha256") != checkpoint_hash
    ):
        raise ValueError(
            f"Existing chunks in {method_directory} use another checkpoint"
        )


def load_compact_samples(samples_path):
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
        field_counts = {len(samples[field]) for field in PARAMETER_FIELDS[model_name]}
        if len(field_counts) != 1:
            raise ValueError(
                f"Posterior arrays for {model_name} have different lengths"
            )
        counts.append(field_counts.pop())
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
    return np.column_stack(columns)


def selected_models(samples, indices):
    models = {}
    for model_column, model_name in enumerate(MODEL_NAMES):
        sample_index = int(indices[model_column])
        fields = PARAMETER_FIELDS[model_name]
        model = {
            "intercept": float(samples[fields[0]][sample_index]),
            "variance": float(samples[fields[1]][sample_index]),
            "lengthscales": np.asarray(samples[fields[2]][sample_index], dtype=float),
        }
        if len(fields) == 4:
            model["noise"] = float(samples[fields[3]][sample_index])
        if model["variance"] < 0 or np.any(model["lengthscales"] <= 0):
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


def ordered_unique_rows(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    unique, positions, inverse = [], {}, []
    for row in values:
        key = tuple(row)
        if key not in positions:
            positions[key] = len(unique)
            unique.append(row)
        inverse.append(positions[key])
    return np.asarray(unique, dtype=float), np.asarray(inverse, dtype=int)


def make_exact_draw_function(rng):
    def draw(locations, model):
        unique_locations, inverse = ordered_unique_rows(locations)
        if len(unique_locations) == 0:
            return np.empty(0)
        differences = (
            unique_locations[:, None, :] - unique_locations[None, :, :]
        ) / model["lengthscales"]
        covariance = np.exp(-0.5 * np.sum(differences**2, axis=-1))
        covariance.flat[:: len(unique_locations) + 1] += JITTER
        unit_draw = np.linalg.cholesky(covariance) @ rng.standard_normal(
            len(unique_locations)
        )
        return model["intercept"] + np.sqrt(model["variance"]) * unit_draw[inverse]

    return draw


def age_groups(ages):
    ages = np.asarray(ages, dtype=float)
    groups = np.full(ages.shape, -1, dtype=int)
    groups[(ages >= 0) & (ages < 1)] = 0
    groups[(ages >= 1) & (ages <= 4)] = 1
    groups[(ages >= 5) & (ages <= 7)] = 2
    groups[ages >= GREYBEARD_AGE] = 3
    return groups


def grouped_counts(sizes, ages, size_edges):
    sizes = np.asarray(sizes, dtype=float)
    groups = age_groups(ages)
    valid_size = np.isfinite(sizes)
    valid_age = groups >= 0
    size = np.histogram(sizes[valid_size], size_edges)[0].astype(float)
    age = np.bincount(groups[valid_age], minlength=4).astype(float)
    joint = np.histogram2d(
        sizes[valid_size & valid_age],
        groups[valid_size & valid_age],
        bins=[size_edges, np.arange(5) - 0.5],
    )[0].ravel()
    return size, age, joint


def selected_components(data, size_edges):
    def column(name):
        return pd.to_numeric(data[name], errors="coerce").to_numpy(float)

    size, age = column("z"), column("age")
    survival, reproduction, recruitment = column("Surv"), column("Repr"), column("Recr")
    next_size = column("z1")
    survivor = (survival == 1) & np.isfinite(next_size) & np.isfinite(age)
    breeder = (reproduction == 1) & survivor
    nonbreeder = (reproduction != 1) & survivor
    survivors = grouped_counts(size[survival == 1], age[survival == 1], size_edges)[2]
    breeders = grouped_counts(
        size[reproduction == 1], age[reproduction == 1], size_edges
    )[0]
    recruits = grouped_counts(
        size[recruitment == 1], age[recruitment == 1], size_edges
    )[1]
    breeder_next = grouped_counts(
        next_size[breeder], np.minimum(age[breeder] + 1, GREYBEARD_AGE), size_edges
    )[0]
    nonbreeder_next = grouped_counts(
        next_size[nonbreeder],
        np.minimum(age[nonbreeder] + 1, GREYBEARD_AGE),
        size_edges,
    )[0]
    return {
        "breeder_next_size": breeder_next,
        "survivors_size_age": survivors,
        "breeders_size": breeders,
        "recruits_age": recruits,
        "nonbreeder_next_size": nonbreeder_next,
    }


def ssd(observed, simulated):
    return float(np.sum((np.asarray(observed) - np.asarray(simulated)) ** 2))


def chi2(observed, expected):
    observed = np.asarray(observed, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
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
    merged_observed, merged_expected = (
        np.asarray(merged_observed),
        np.asarray(merged_expected),
    )
    positive = merged_expected > 0
    return (
        ssd(observed, expected)
        if not positive.any()
        else float(
            np.sum(
                (merged_observed[positive] - merged_expected[positive]) ** 2
                / merged_expected[positive]
            )
        )
    )


def normalized_distribution(values):
    values = np.asarray(values, dtype=float) + 1e-10
    return values / values.sum()


def selected_distances(observed, simulated):
    observed_breeder = normalized_distribution(observed["breeder_next_size"])
    simulated_breeder = normalized_distribution(simulated["breeder_next_size"])
    observed_nonbreeder = normalized_distribution(observed["nonbreeder_next_size"])
    simulated_nonbreeder = normalized_distribution(simulated["nonbreeder_next_size"])
    coefficient = np.clip(
        np.sum(np.sqrt(observed_nonbreeder * simulated_nonbreeder)), 1e-15, 1.0
    )
    emd = wasserstein_distance(
        np.arange(5),
        np.arange(5),
        u_weights=observed_breeder,
        v_weights=simulated_breeder,
    )
    return np.asarray(
        [
            emd,
            chi2(observed["survivors_size_age"], simulated["survivors_size_age"]),
            chi2(observed["breeders_size"], simulated["breeders_size"]),
            chi2(observed["recruits_age"], simulated["recruits_age"]),
            -np.log(coefficient),
        ]
    )


def simulate_one_year(population, models, draw, rng):
    size = population["z"].to_numpy(float)
    age = population["age"].to_numpy(int)
    locations = np.column_stack([size, age])
    number = len(population)

    survival_probability = scipy.special.expit(draw(locations, models["m_surv"]))
    survived = rng.binomial(1, survival_probability)
    survivor = survived == 1

    next_size = np.full(number, np.nan)
    if survivor.any():
        growth = draw(locations[survivor], models["m_grw"])
        next_size[survivor] = growth + rng.normal(
            0, models["m_grw"]["noise"], survivor.sum()
        )

    reproduced = np.full(number, np.nan)
    reproduced[survivor] = 0
    eligible = survivor & (age >= MIN_REPRODUCTIVE_AGE)
    if eligible.any():
        reproduction_probability = scipy.special.expit(
            draw(locations[eligible], models["m_repr"])
        )
        reproduced[eligible] = rng.binomial(1, reproduction_probability)
    breeder = eligible & (reproduced == 1)

    female = np.zeros(number, dtype=bool)
    if breeder.any():
        female[breeder] = rng.binomial(1, FEMALE_PROBABILITY, breeder.sum()).astype(
            bool
        )

    recruited = np.full(number, np.nan)
    if female.any():
        recruitment_probability = scipy.special.expit(
            draw(age[female, None], models["m_recr"])
        )
        recruited[female] = rng.binomial(1, recruitment_probability)
    recruit = female & (recruited == 1)

    recruit_size = np.full(number, np.nan)
    if recruit.any():
        expected_size = draw(size[recruit, None], models["m_rcsz"])
        recruit_size[recruit] = expected_size + rng.normal(
            0, models["m_rcsz"]["noise"], recruit.sum()
        )

    return pd.DataFrame(
        {
            "z": size,
            "age": age,
            "Surv": survived,
            "z1": next_size,
            "Repr": reproduced,
            "Recr": recruited,
            "Rcsz": recruit_size,
        }
    )


def population_structure(simulated):
    survivor = (simulated["Surv"] == 1) & simulated["z1"].notna()
    recruit = (simulated["Recr"] == 1) & simulated["Rcsz"].notna()
    survivor_size = simulated.loc[survivor, "z1"].to_numpy(float)
    survivor_age = np.minimum(
        simulated.loc[survivor, "age"].to_numpy(int) + 1, GREYBEARD_AGE
    )
    recruit_size = simulated.loc[recruit, "Rcsz"].to_numpy(float)
    return pd.DataFrame(
        {
            "z": np.concatenate([recruit_size, survivor_size]),
            "age": np.concatenate(
                [np.zeros(len(recruit_size), dtype=int), survivor_age]
            ),
        }
    )


def simulate_dataset(models, initial_population, observed_components, size_edges, rng):
    number_years = len(observed_components)
    distances = np.full((number_years, len(SUMMARY_NAMES)), np.nan)
    population_sizes = np.full(number_years, np.nan)
    population = initial_population.copy()
    status, extinction_year = STATUS_SUCCESS, -1
    draw = make_exact_draw_function(rng)
    for year_index, observed in enumerate(observed_components):
        if len(population) <= 1:
            distances[year_index:] = np.inf
            population_sizes[year_index:] = len(population)
            status, extinction_year = STATUS_EXTINCTION, year_index
            break
        simulated = simulate_one_year(population, models, draw, rng)
        distances[year_index] = selected_distances(
            observed, selected_components(simulated, size_edges)
        )
        population = population_structure(simulated)
        population_sizes[year_index] = len(population)
        if len(population) <= 1 and year_index == number_years - 1:
            status, extinction_year = STATUS_EXTINCTION, number_years
    return distances, population_sizes, status, extinction_year


def make_task_chunks(
    number_parameters, number_replicates, target_chunk_size=100, minimum_chunks=1
):
    tasks = np.column_stack(
        [
            np.repeat(np.arange(number_parameters), number_replicates),
            np.tile(np.arange(number_replicates), number_parameters),
        ]
    )
    if len(tasks) == 0:
        return []
    number_chunks = min(
        len(tasks), max(minimum_chunks, math.ceil(len(tasks) / target_chunk_size))
    )
    return [chunk for chunk in np.array_split(tasks, number_chunks) if len(chunk)]


def load_experiment(data_path, samples_path, design_path):
    data_path, samples_path, design_path = map(
        lambda path: Path(path).resolve(), (data_path, samples_path, design_path)
    )
    with np.load(design_path) as archive:
        design = {name: np.asarray(archive[name]) for name in archive.files}
    if file_sha256(data_path) != str(design["data_sha256"].item()):
        raise ValueError("The dataset does not match the validation design")
    if file_sha256(samples_path) != str(design["samples_sha256"].item()):
        raise ValueError("The MCMC archive does not match the validation design")
    if tuple(design["summary_names"].tolist()) != SUMMARY_NAMES:
        raise ValueError("The design does not contain the expected five summaries")
    if tuple(design["parameter_names"].tolist()) != PARAMETER_NAMES:
        raise ValueError(
            "The design parameter columns do not match this implementation"
        )

    data = pd.read_csv(data_path)
    years = design["years"]
    observed_by_year = [data.loc[data["yr"] == year].copy() for year in years]
    initial_population = observed_by_year[0][["z", "age"]].dropna().copy()
    initial_population["age"] = initial_population["age"].astype(int)
    return {
        "samples": load_compact_samples(samples_path),
        "parameter_indices": design["parameter_indices"].astype(np.int64),
        "parameter_values": design["parameter_values"],
        "number_parameters": int(design["number_parameters"].item()),
        "number_replicates": int(design["number_replicates"].item()),
        "experiment_seed": int(design["experiment_seed"].item()),
        "years": years,
        "size_edges": design["size_edges"],
        "initial_population": initial_population,
        "observed_components": [
            selected_components(data, design["size_edges"]) for data in observed_by_year
        ],
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
