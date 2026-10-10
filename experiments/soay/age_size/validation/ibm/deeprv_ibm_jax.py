"""Fully batched JAX age-size IBM for DeepRV accuracy validation."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from experiments.soay.age_size.validation.ibm.comparison_common import (
    FEMALE_PROBABILITY,
    GREYBEARD_AGE,
    JITTER,
    PARAMETER_NAMES,
    STATUS_ERROR,
    STATUS_EXTINCTION,
    STATUS_MAX_LOCATIONS,
    STATUS_SUCCESS,
)

COLUMN = {name: position for position, name in enumerate(PARAMETER_NAMES)}
RECRUIT_SIZE_BUCKETS = (8, 16, 32, 64, 128, 256, 512)


def chi2_jax(observed, expected):
    """Match the sequential small-expected-count merging used by the ABC code."""
    observed, expected = observed.ravel(), expected.ravel()
    total = last_observed = last_expected = pending_observed = pending_expected = (
        jnp.float32(0)
    )
    has_last = jnp.bool_(False)
    for index in range(observed.shape[0]):
        pending_observed += observed[index]
        pending_expected += expected[index]
        close = pending_expected >= 5
        previous = jnp.where(
            has_last & close,
            (last_observed - last_expected) ** 2 / jnp.maximum(last_expected, 1e-30),
            0,
        )
        total += previous
        last_observed = jnp.where(close, pending_observed, last_observed)
        last_expected = jnp.where(close, pending_expected, last_expected)
        has_last = has_last | close
        pending_observed = jnp.where(close, 0, pending_observed)
        pending_expected = jnp.where(close, 0, pending_expected)
    has_pending = (pending_observed != 0) | (pending_expected != 0)
    last_observed = jnp.where(
        has_last & has_pending,
        last_observed + pending_observed,
        jnp.where(has_pending, pending_observed, last_observed),
    )
    last_expected = jnp.where(
        has_last & has_pending,
        last_expected + pending_expected,
        jnp.where(has_pending, pending_expected, last_expected),
    )
    has_last = has_last | has_pending
    merged = total + jnp.where(
        has_last & (last_expected > 0),
        (last_observed - last_expected) ** 2 / jnp.maximum(last_expected, 1e-30),
        0,
    )
    return jnp.where(
        has_last & (last_expected > 0), merged, jnp.sum((observed - expected) ** 2)
    )


def normalized_distribution_jax(values):
    values = values + 1e-10
    return values / jnp.sum(values, axis=-1, keepdims=True)


def selected_distances_jax(observed, simulated):
    observed_breeder = normalized_distribution_jax(observed["breeder_next_size"])
    simulated_breeder = normalized_distribution_jax(simulated["breeder_next_size"])
    breeder_cdf_difference = jnp.cumsum(observed_breeder - simulated_breeder, axis=-1)
    emd = jnp.sum(jnp.abs(breeder_cdf_difference[..., :-1]), axis=-1)

    observed_nonbreeder = normalized_distribution_jax(observed["nonbreeder_next_size"])
    simulated_nonbreeder = normalized_distribution_jax(
        simulated["nonbreeder_next_size"]
    )
    coefficient = jnp.clip(
        jnp.sum(jnp.sqrt(observed_nonbreeder * simulated_nonbreeder), axis=-1), 1e-15, 1
    )
    bhattacharyya = -jnp.log(coefficient)

    def chi2_batch(observed_values, simulated_values):
        return jax.vmap(lambda row: chi2_jax(observed_values, row))(simulated_values)

    return jnp.column_stack(
        [
            emd,
            chi2_batch(observed["survivors_size_age"], simulated["survivors_size_age"]),
            chi2_batch(observed["breeders_size"], simulated["breeders_size"]),
            chi2_batch(observed["recruits_age"], simulated["recruits_age"]),
            bhattacharyya,
        ]
    )


def make_deeprv_simulator(
    runtime, number_years, initial_population, observed_components, size_edges
):
    """Build one compiled function for a static batch of complete 10-year datasets."""
    capacity = runtime.max_locations
    if capacity != RECRUIT_SIZE_BUCKETS[-1]:
        raise ValueError(
            f"This validation expects a {RECRUIT_SIZE_BUCKETS[-1]}-location checkpoint"
        )
    if not np.isclose(runtime.jitter, JITTER):
        raise ValueError(f"The checkpoint jitter must be {JITTER}")
    positions = jnp.arange(capacity)
    eye = jnp.eye(capacity, dtype=jnp.float32)
    age_grid = jnp.arange(GREYBEARD_AGE + 1, dtype=jnp.float32)
    age_eye = jnp.eye(GREYBEARD_AGE + 1, dtype=jnp.float32)
    size_boundaries = jnp.asarray(np.asarray(size_edges)[1:-1], dtype=jnp.float32)

    initial_sizes = np.zeros(capacity, dtype=np.float32)
    initial_ages = np.zeros(capacity, dtype=np.int32)
    initial_count = len(initial_population)
    if initial_count > capacity:
        raise ValueError("The initial population exceeds the Transformer capacity")
    initial_sizes[:initial_count] = initial_population["z"].to_numpy(np.float32)
    initial_ages[:initial_count] = initial_population["age"].to_numpy(np.int32)
    initial_sizes, initial_ages = jnp.asarray(initial_sizes), jnp.asarray(initial_ages)
    initial_active = positions < initial_count
    observed = {
        name: jnp.asarray(
            np.stack([year[name] for year in observed_components]), dtype=jnp.float32
        )
        for name in observed_components[0]
    }
    growth_lengthscale_columns = jnp.asarray(
        [COLUMN["growth_ls_size"], COLUMN["growth_ls_age"]]
    )
    survival_lengthscale_columns = jnp.asarray(
        [COLUMN["surv_ls_size"], COLUMN["surv_ls_age"]]
    )
    reproduction_lengthscale_columns = jnp.asarray(
        [COLUMN["repr_ls_size"], COLUMN["repr_ls_age"]]
    )

    def fold_keys(keys, tag):
        return jax.vmap(lambda key: random.fold_in(key, tag))(keys)

    def normal_rows(keys, length=capacity):
        return jax.vmap(lambda key: random.normal(key, (length,), dtype=jnp.float32))(
            keys
        )

    def bernoulli_rows(keys, probabilities):
        return jax.vmap(random.bernoulli)(keys, probabilities)

    def decode_one(z, lengthscales, locations, covariance, mask):
        output = runtime.model.apply(
            runtime.variables,
            z[None],
            lengthscales,
            s=locations,
            K=covariance,
            mask=mask[None],
            method="decode",
        )
        return output[0, :, 0]

    decode_batch = jax.vmap(decode_one)

    def draw_deeprv(locations, active, keys, intercept, variance, lengthscales):
        has_locations = jnp.any(active, axis=1)
        safe_active = active.at[:, 0].set(active[:, 0] | ~has_locations)
        safe_locations = jnp.where(safe_active[..., None], locations, 0)
        differences = (
            safe_locations[:, :, None, :] - safe_locations[:, None, :, :]
        ) / lengthscales[:, None, None, :]
        covariance = (
            jnp.exp(-0.5 * jnp.sum(differences**2, axis=-1))
            + runtime.jitter * eye[None]
        )
        z = normal_rows(keys) * safe_active
        unit_draw = decode_batch(
            z, lengthscales, safe_locations, covariance, safe_active
        )
        draws = intercept[:, None] + jnp.sqrt(variance[:, None]) * unit_draw
        return jnp.where(active, draws, 0)

    def draw_recruitment_age(ages, active, keys, intercept, variance, lengthscale):
        differences = age_grid[None, :, None] - age_grid[None, None, :]
        covariance = (
            jnp.exp(-0.5 * (differences / lengthscale[:, None, None]) ** 2)
            + JITTER * age_eye[None]
        )
        unit_grid = jnp.einsum(
            "bij,bj->bi",
            jnp.linalg.cholesky(covariance),
            normal_rows(keys, GREYBEARD_AGE + 1),
        )
        unit_values = jnp.take_along_axis(
            unit_grid, jnp.clip(ages, 0, GREYBEARD_AGE), axis=1
        )
        draws = intercept[:, None] + jnp.sqrt(variance[:, None]) * unit_values
        return jnp.where(active, draws, 0)

    def recruit_size_branch(bucket):
        bucket_positions = jnp.arange(bucket)
        bucket_eye = jnp.eye(bucket, dtype=jnp.float32)

        def draw(operand):
            sizes, active, key, intercept, variance, lengthscale = operand
            count = jnp.sum(active, dtype=jnp.int32)
            indices = jnp.nonzero(active, size=bucket, fill_value=0)[0]
            selected_active = bucket_positions < count
            locations = sizes[indices]
            differences = locations[:, None] - locations[None, :]
            covariance = jnp.exp(-0.5 * (differences / lengthscale) ** 2)
            pair_active = selected_active[:, None] & selected_active[None, :]
            covariance = jnp.where(pair_active, covariance, 0) + bucket_eye * jnp.where(
                selected_active, JITTER, 1
            )
            unit_draw = jnp.linalg.cholesky(covariance) @ random.normal(
                key, (bucket,), dtype=jnp.float32
            )
            draws = jnp.where(
                selected_active, intercept + jnp.sqrt(variance) * unit_draw, 0
            )
            return jnp.zeros(capacity, dtype=jnp.float32).at[indices].add(draws)

        return draw

    recruit_size_branches = tuple(
        recruit_size_branch(bucket) for bucket in RECRUIT_SIZE_BUCKETS
    )

    def draw_recruit_size(sizes, active, keys, intercept, variance, lengthscale):
        def draw_row(
            row_sizes, row_active, key, row_intercept, row_variance, row_lengthscale
        ):
            count = jnp.sum(row_active, dtype=jnp.int32)
            branch = jnp.sum(count > jnp.asarray(RECRUIT_SIZE_BUCKETS), dtype=jnp.int32)
            return jax.lax.switch(
                branch,
                recruit_size_branches,
                (
                    row_sizes,
                    row_active,
                    key,
                    row_intercept,
                    row_variance,
                    row_lengthscale,
                ),
            )

        return jax.vmap(draw_row)(sizes, active, keys, intercept, variance, lengthscale)

    def age_group(ages):
        return jnp.where(
            ages < 1, 0, jnp.where(ages <= 4, 1, jnp.where(ages <= 7, 2, 3))
        )

    def histogram(groups, active, number_groups):
        safe_groups = jnp.where(active, groups, 0)
        return jnp.sum(
            jax.nn.one_hot(safe_groups, number_groups, dtype=jnp.float32)
            * active[..., None],
            axis=1,
        )

    def components(size, age, survived, next_size, reproduced, recruited):
        size_group = jnp.digitize(size, size_boundaries)
        next_size_group = jnp.digitize(next_size, size_boundaries)
        age_group_value = age_group(age)
        joint_group = size_group * 4 + age_group_value
        breeder = reproduced
        nonbreeder = survived & ~reproduced
        return {
            "breeder_next_size": histogram(next_size_group, breeder, 5),
            "survivors_size_age": histogram(joint_group, survived, 20),
            "breeders_size": histogram(size_group, breeder, 5),
            "recruits_age": histogram(age_group_value, recruited, 4),
            "nonbreeder_next_size": histogram(next_size_group, nonbreeder, 5),
        }

    def pack_population(candidate_size, candidate_age, candidate_active):
        def pack_row(row_size, row_age, row_active):
            indices = jnp.nonzero(row_active, size=capacity, fill_value=0)[0]
            count = jnp.sum(row_active, dtype=jnp.int32)
            packed_active = positions < jnp.minimum(count, capacity)
            return (
                jnp.where(packed_active, row_size[indices], 0),
                jnp.where(packed_active, row_age[indices], 0),
                packed_active,
                count,
            )

        return jax.vmap(pack_row)(candidate_size, candidate_age, candidate_active)

    @jax.jit
    def simulate(parameters, task_seeds, valid_tasks):
        batch_size = parameters.shape[0]
        keys = jax.vmap(random.key)(task_seeds)
        size = jnp.broadcast_to(initial_sizes, (batch_size, capacity))
        age = jnp.broadcast_to(initial_ages, (batch_size, capacity))
        active = (
            jnp.broadcast_to(initial_active, (batch_size, capacity))
            & valid_tasks[:, None]
        )
        running = valid_tasks & (initial_count > 1)
        overflow = jnp.zeros(batch_size, dtype=bool)
        extinction_year = jnp.where(valid_tasks & ~running, 0, -1).astype(jnp.int32)
        population_count = jnp.where(valid_tasks, initial_count, -1).astype(jnp.int32)
        distances, population_sizes = [], []

        for year in range(number_years):
            tag = year * 16
            evaluation_active = active & running[:, None]
            locations = jnp.stack([size, age.astype(jnp.float32)], axis=-1)

            survival_draw = draw_deeprv(
                locations,
                evaluation_active,
                fold_keys(keys, tag),
                parameters[:, COLUMN["surv_int"]],
                parameters[:, COLUMN["surv_var"]],
                parameters[:, survival_lengthscale_columns],
            )
            survived = evaluation_active & bernoulli_rows(
                fold_keys(keys, tag + 1), jax.nn.sigmoid(survival_draw)
            )
            eligible = survived & (age >= 1)

            joint_locations = jnp.concatenate([locations, locations], axis=0)
            joint_active = jnp.concatenate([survived, eligible], axis=0)
            joint_keys = jnp.concatenate(
                [fold_keys(keys, tag + 2), fold_keys(keys, tag + 3)], axis=0
            )
            joint_intercept = jnp.concatenate(
                [parameters[:, COLUMN["growth_int"]], parameters[:, COLUMN["repr_int"]]]
            )
            joint_variance = jnp.concatenate(
                [parameters[:, COLUMN["growth_var"]], parameters[:, COLUMN["repr_var"]]]
            )
            growth_lengthscales = parameters[:, growth_lengthscale_columns]
            reproduction_lengthscales = parameters[:, reproduction_lengthscale_columns]
            joint_lengthscales = jnp.concatenate(
                [growth_lengthscales, reproduction_lengthscales], axis=0
            )
            joint_draw = draw_deeprv(
                joint_locations,
                joint_active,
                joint_keys,
                joint_intercept,
                joint_variance,
                joint_lengthscales,
            )
            growth_draw, reproduction_draw = (
                joint_draw[:batch_size],
                joint_draw[batch_size:],
            )

            next_size = growth_draw + parameters[:, COLUMN["growth_noise"]][
                :, None
            ] * normal_rows(fold_keys(keys, tag + 4))
            next_size = jnp.where(survived, next_size, 0)
            reproduced = eligible & bernoulli_rows(
                fold_keys(keys, tag + 5), jax.nn.sigmoid(reproduction_draw)
            )
            female = reproduced & bernoulli_rows(
                fold_keys(keys, tag + 6),
                jnp.full_like(reproduction_draw, FEMALE_PROBABILITY),
            )

            recruitment_draw = draw_recruitment_age(
                age,
                female,
                fold_keys(keys, tag + 7),
                parameters[:, COLUMN["recr_int"]],
                parameters[:, COLUMN["recr_var"]],
                parameters[:, COLUMN["recr_ls_age"]],
            )
            recruited = female & bernoulli_rows(
                fold_keys(keys, tag + 8), jax.nn.sigmoid(recruitment_draw)
            )
            recruit_size = draw_recruit_size(
                size,
                recruited,
                fold_keys(keys, tag + 9),
                parameters[:, COLUMN["rcsz_int"]],
                parameters[:, COLUMN["rcsz_var"]],
                parameters[:, COLUMN["rcsz_ls_size"]],
            )
            recruit_size += parameters[:, COLUMN["rcsz_noise"]][:, None] * normal_rows(
                fold_keys(keys, tag + 10)
            )
            recruit_size = jnp.where(recruited, recruit_size, 0)

            simulated = components(
                size, age, survived, next_size, reproduced, recruited
            )
            observed_year = {name: values[year] for name, values in observed.items()}
            year_distances = selected_distances_jax(observed_year, simulated)
            year_distances = jnp.where(
                running[:, None],
                year_distances,
                jnp.where(valid_tasks[:, None], jnp.inf, jnp.nan),
            )

            candidate_size = jnp.concatenate([recruit_size, next_size], axis=1)
            candidate_age = jnp.concatenate(
                [jnp.zeros_like(age), jnp.minimum(age + 1, GREYBEARD_AGE)], axis=1
            )
            candidate_active = jnp.concatenate([recruited, survived], axis=1)
            packed_size, packed_age, packed_active, next_count = pack_population(
                candidate_size, candidate_age, candidate_active
            )
            new_overflow = running & (next_count > capacity)
            newly_extinct = running & (next_count <= 1)
            extinction_year = jnp.where(
                (extinction_year < 0) & newly_extinct, year + 1, extinction_year
            )
            population_count = jnp.where(running, next_count, population_count)
            size = jnp.where(running[:, None], packed_size, size)
            age = jnp.where(running[:, None], packed_age, age)
            active = jnp.where(running[:, None], packed_active, active)
            overflow |= new_overflow
            running &= ~newly_extinct & ~new_overflow
            distances.append(year_distances)
            population_sizes.append(population_count)

        distance_array = jnp.stack(distances, axis=1)
        distance_array = jnp.where(overflow[:, None, None], jnp.inf, distance_array)
        status = jnp.where(
            ~valid_tasks,
            STATUS_ERROR,
            jnp.where(
                overflow,
                STATUS_MAX_LOCATIONS,
                jnp.where(extinction_year >= 0, STATUS_EXTINCTION, STATUS_SUCCESS),
            ),
        )
        return {
            "distances": distance_array,
            "population_sizes": jnp.stack(population_sizes, axis=1),
            "status": status,
            "extinction_year": extinction_year,
        }

    return simulate
