"""JAX-only batched Soay IBM using the Transformer DeepRV decoder."""

import jax
import jax.numpy as jnp
from jax import random

from experiments.soay.size_only.benchmarks.benchmark_common import (
    FEMALE_PROBABILITY,
    JITTER,
    PARAMETER_NAMES,
)

COLUMN = {name: position for position, name in enumerate(PARAMETER_NAMES)}


def make_deeprv_simulator(runtime, number_years, recruit_probability):
    """Build one compiled function that simulates a batch for the requested years."""
    capacity = runtime.max_locations
    positions = jnp.arange(capacity)
    eye = jnp.eye(capacity, dtype=jnp.float32)

    def fold_keys(keys, tag):
        return jax.vmap(lambda key: random.fold_in(key, tag))(keys)

    def normal_rows(keys):
        return jax.vmap(lambda key: random.normal(key, (capacity,), dtype=jnp.float32))(
            keys
        )

    def bernoulli_rows(keys, probabilities):
        return jax.vmap(random.bernoulli)(keys, probabilities)

    def ordered_unique(values, active):
        """Return fixed-size ordered unique values and an inverse map."""
        masked_values = jnp.where(active, values, jnp.inf)
        unique_sorted, first_indices, inverse_sorted = jnp.unique(
            masked_values,
            return_index=True,
            return_inverse=True,
            size=capacity,
            fill_value=jnp.inf,
        )
        valid_sorted = jnp.isfinite(unique_sorted)
        order_key = jnp.where(
            valid_sorted, first_indices, capacity + jnp.arange(capacity)
        )
        order = jnp.argsort(order_key)
        ordered_values = unique_sorted[order]
        ordered_active = valid_sorted[order]
        sorted_to_ordered = (
            jnp.zeros(capacity, dtype=jnp.int32)
            .at[order]
            .set(jnp.arange(capacity, dtype=jnp.int32))
        )
        inverse = sorted_to_ordered[inverse_sorted]
        inverse = jnp.where(active, inverse, 0)
        return jnp.where(ordered_active, ordered_values, 0.0), ordered_active, inverse

    ordered_unique_batch = jax.vmap(ordered_unique)

    def draw_gp(locations, active, keys, intercept, variance, lengthscale):
        unique_locations, unique_active, inverse = ordered_unique_batch(
            locations, active
        )
        has_locations = jnp.any(unique_active, axis=1)
        safe_active = unique_active.at[:, 0].set(unique_active[:, 0] | ~has_locations)
        differences = unique_locations[:, :, None] - unique_locations[:, None, :]
        covariance = (
            jnp.exp(-0.5 * (differences / lengthscale[:, None, None]) ** 2)
            + JITTER * eye[None]
        )
        z = normal_rows(keys) * safe_active
        unit_unique = runtime.decode_batch(
            z,
            lengthscale,
            unique_locations[..., None],
            covariance,
            safe_active,
        )[..., 0]
        unit_values = jnp.take_along_axis(unit_unique, inverse, axis=1)
        draws = intercept[:, None] + jnp.sqrt(variance[:, None]) * unit_values
        unique_count = jnp.sum(unique_active, axis=1, dtype=jnp.int32)
        return jnp.where(active, draws, 0.0), unique_count

    def pack_population(values, active):
        def pack_row(row_values, row_active):
            indices = jnp.nonzero(row_active, size=capacity, fill_value=0)[0]
            count = jnp.sum(row_active, dtype=jnp.int32)
            packed_active = positions < jnp.minimum(count, capacity)
            packed_values = jnp.where(packed_active, row_values[indices], 0.0)
            return packed_values, packed_active, count

        return jax.vmap(pack_row)(values, active)

    @jax.jit
    def simulate(parameters, task_seeds, task_ids, valid_tasks, initial_sizes):
        batch_size = parameters.shape[0]
        keys = jax.vmap(random.key)(task_seeds)
        sizes = jnp.broadcast_to(initial_sizes, (batch_size, capacity))
        initial_count = jnp.sum(jnp.isfinite(initial_sizes), dtype=jnp.int32)
        active = jnp.broadcast_to(positions < initial_count, (batch_size, capacity))
        active = active & valid_tasks[:, None]
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

            survival_draw, survival_unique_count = draw_gp(
                sizes,
                evaluation_active,
                fold_keys(keys, tag),
                parameters[:, COLUMN["surv_int"]],
                parameters[:, COLUMN["surv_var"]],
                parameters[:, COLUMN["surv_ls"]],
            )
            survival_probability = jax.nn.sigmoid(survival_draw)
            survived = evaluation_active & bernoulli_rows(
                fold_keys(keys, tag + 1), survival_probability
            )

            survivor_locations = jnp.concatenate([sizes, sizes], axis=0)
            survivor_active = jnp.concatenate([survived, survived], axis=0)
            survivor_keys = jnp.concatenate(
                [fold_keys(keys, tag + 2), fold_keys(keys, tag + 3)], axis=0
            )
            survivor_intercepts = jnp.concatenate(
                [
                    parameters[:, COLUMN["growth_int"]],
                    parameters[:, COLUMN["repr_int"]],
                ]
            )
            survivor_variances = jnp.concatenate(
                [
                    parameters[:, COLUMN["growth_var"]],
                    parameters[:, COLUMN["repr_var"]],
                ]
            )
            survivor_lengthscales = jnp.concatenate(
                [
                    parameters[:, COLUMN["growth_ls"]],
                    parameters[:, COLUMN["repr_ls"]],
                ]
            )
            growth_and_reproduction, survivor_unique_counts = draw_gp(
                survivor_locations,
                survivor_active,
                survivor_keys,
                survivor_intercepts,
                survivor_variances,
                survivor_lengthscales,
            )
            growth_draw = growth_and_reproduction[:batch_size]
            reproduction_draw = growth_and_reproduction[batch_size:]

            growth_noise = normal_rows(fold_keys(keys, tag + 4))
            next_size = (
                growth_draw
                + parameters[:, COLUMN["growth_noise"]][:, None] * growth_noise
            )
            next_size = jnp.where(survived, next_size, 0.0)

            reproduction_probability = jax.nn.sigmoid(reproduction_draw)
            reproduced = survived & bernoulli_rows(
                fold_keys(keys, tag + 5), reproduction_probability
            )
            female = reproduced & bernoulli_rows(
                fold_keys(keys, tag + 6),
                jnp.full_like(reproduction_probability, FEMALE_PROBABILITY),
            )
            recruited = female & bernoulli_rows(
                fold_keys(keys, tag + 7),
                jnp.full_like(reproduction_probability, recruit_probability),
            )

            recruit_size_draw, recruit_unique_count = draw_gp(
                sizes,
                recruited,
                fold_keys(keys, tag + 8),
                parameters[:, COLUMN["rcsz_int"]],
                parameters[:, COLUMN["rcsz_var"]],
                parameters[:, COLUMN["rcsz_ls"]],
            )
            recruit_noise = normal_rows(fold_keys(keys, tag + 9))
            recruit_size = (
                recruit_size_draw
                + parameters[:, COLUMN["rcsz_noise"]][:, None] * recruit_noise
            )
            recruit_size = jnp.where(recruited, recruit_size, 0.0)

            candidates = jnp.concatenate([next_size, recruit_size], axis=1)
            candidate_active = jnp.concatenate([survived, recruited], axis=1)
            packed_sizes, packed_active, next_count = pack_population(
                candidates, candidate_active
            )
            new_overflow = running & (next_count > capacity)
            newly_extinct = running & (next_count <= 1)
            extinction_year = jnp.where(
                (extinction_year < 0) & newly_extinct, year + 1, extinction_year
            )

            current_count = jnp.sum(active, axis=1, dtype=jnp.int32)
            reported_count = jnp.where(running, next_count, current_count)
            sizes = jnp.where(running[:, None], packed_sizes, sizes)
            active = jnp.where(running[:, None], packed_active, active)
            overflow = overflow | new_overflow
            running = running & ~newly_extinct & ~new_overflow

            survivor_count = jnp.sum(survived, axis=1, dtype=jnp.int32)
            breeder_count = jnp.sum(reproduced, axis=1, dtype=jnp.int32)
            recruit_count = jnp.sum(recruited, axis=1, dtype=jnp.int32)
            requested = current_count + 2 * survivor_count + recruit_count
            requested = jnp.where(evaluation_active.any(axis=1), requested, 0)
            unique_requested = (
                survival_unique_count
                + survivor_unique_counts[:batch_size]
                + survivor_unique_counts[batch_size:]
                + recruit_unique_count
            )

            population_sizes.append(reported_count)
            survivor_counts.append(survivor_count)
            breeder_counts.append(breeder_count)
            recruit_counts.append(recruit_count)
            requested_locations.append(requested)
            unique_requested_locations.append(unique_requested)

        final_population_size = jnp.sum(active, axis=1, dtype=jnp.int32)
        final_size_total = jnp.sum(jnp.where(active, sizes, 0.0), axis=1)
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
