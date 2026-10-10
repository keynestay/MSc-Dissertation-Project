"""JAX-only batched age-size IBM using the Transformer DeepRV decoder."""

import jax
import jax.numpy as jnp
from jax import random

from experiments.soay.age_size.benchmarks.benchmark_common import (
    FEMALE_PROBABILITY,
    GREYBEARD_AGE,
    JITTER,
    PARAMETER_NAMES,
)

COLUMN = {name: position for position, name in enumerate(PARAMETER_NAMES)}
RECRUIT_SIZE_BUCKETS = (8, 16, 32, 64, 128, 256, 512)


def make_deeprv_simulator(runtime, number_years):
    """Build one compiled function that simulates a batch for the requested years."""
    capacity = runtime.max_locations
    if capacity != RECRUIT_SIZE_BUCKETS[-1]:
        raise ValueError(
            f"This benchmark expects a {RECRUIT_SIZE_BUCKETS[-1]}-location checkpoint"
        )
    if not jnp.isclose(runtime.jitter, JITTER):
        raise ValueError(f"The checkpoint jitter must be {JITTER}")

    positions = jnp.arange(capacity)
    eye = jnp.eye(capacity, dtype=jnp.float32)
    age_grid = jnp.arange(GREYBEARD_AGE + 1, dtype=jnp.float32)
    age_eye = jnp.eye(GREYBEARD_AGE + 1, dtype=jnp.float32)
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

    def unique_active_age_count(ages, active):
        groups = jax.nn.one_hot(
            jnp.clip(ages, 0, GREYBEARD_AGE), GREYBEARD_AGE + 1, dtype=jnp.int32
        )
        return jnp.sum(
            jnp.any((groups * active[..., None]) > 0, axis=1), axis=1, dtype=jnp.int32
        )

    @jax.jit
    def simulate(
        parameters, task_seeds, task_ids, valid_tasks, initial_sizes, initial_ages
    ):
        batch_size = parameters.shape[0]
        keys = jax.vmap(random.key)(task_seeds)
        initial_count = jnp.sum(jnp.isfinite(initial_sizes), dtype=jnp.int32)
        size = jnp.broadcast_to(jnp.nan_to_num(initial_sizes), (batch_size, capacity))
        age = jnp.broadcast_to(initial_ages, (batch_size, capacity))
        active = (
            jnp.broadcast_to(positions < initial_count, (batch_size, capacity))
            & valid_tasks[:, None]
        )
        running = valid_tasks & (initial_count > 1)
        overflow = jnp.zeros(batch_size, dtype=bool)
        extinction_year = jnp.full(batch_size, -1, dtype=jnp.int32)

        population_sizes = []
        survivor_counts = []
        breeder_counts = []
        recruit_counts = []
        requested_locations = []
        unique_requested_locations = []

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
            joint_lengthscales = jnp.concatenate(
                [
                    parameters[:, growth_lengthscale_columns],
                    parameters[:, reproduction_lengthscale_columns],
                ],
                axis=0,
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

            current_count = jnp.sum(active, axis=1, dtype=jnp.int32)
            survivor_count = jnp.sum(survived, axis=1, dtype=jnp.int32)
            eligible_count = jnp.sum(eligible, axis=1, dtype=jnp.int32)
            breeder_count = jnp.sum(reproduced, axis=1, dtype=jnp.int32)
            female_count = jnp.sum(female, axis=1, dtype=jnp.int32)
            recruit_count = jnp.sum(recruited, axis=1, dtype=jnp.int32)
            requested = (
                current_count
                + survivor_count
                + eligible_count
                + female_count
                + recruit_count
            )
            unique_requested = (
                current_count
                + survivor_count
                + eligible_count
                + unique_active_age_count(age, female)
                + recruit_count
            )
            requested = jnp.where(running, requested, 0)
            unique_requested = jnp.where(running, unique_requested, 0)

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
            reported_count = jnp.where(running, next_count, current_count)
            size = jnp.where(running[:, None], packed_size, size)
            age = jnp.where(running[:, None], packed_age, age)
            active = jnp.where(running[:, None], packed_active, active)
            overflow |= new_overflow
            running &= ~newly_extinct & ~new_overflow

            population_sizes.append(reported_count)
            survivor_counts.append(survivor_count)
            breeder_counts.append(breeder_count)
            recruit_counts.append(recruit_count)
            requested_locations.append(requested)
            unique_requested_locations.append(unique_requested)

        final_population_size = jnp.sum(active, axis=1, dtype=jnp.int32)
        final_size_total = jnp.sum(jnp.where(active, size, 0), axis=1)
        return {
            "task_id": task_ids,
            "population_sizes": jnp.stack(population_sizes, axis=1),
            "survivor_counts": jnp.stack(survivor_counts, axis=1),
            "breeder_counts": jnp.stack(breeder_counts, axis=1),
            "recruit_counts": jnp.stack(recruit_counts, axis=1),
            "requested_locations": jnp.stack(requested_locations, axis=1),
            "unique_requested_locations": jnp.stack(unique_requested_locations, axis=1),
            "final_population_size": final_population_size,
            "final_size_total": final_size_total,
            "extinction_year": extinction_year,
            "overflow": overflow,
        }

    return simulate
