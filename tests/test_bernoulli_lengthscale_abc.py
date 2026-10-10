from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from experiments.bernoulli_abc.bernoulli_lengthscale_abc import (
    ABCResult,
    ExperimentConfig,
    active_checkpoint_steps,
    build_locations,
    calibrate_threshold,
    checkpoint_abc_summary,
    make_exact_distance_fn,
    plot_checkpoint_diagnostics,
    plot_posteriors,
    rbf_covariance,
    run_rejection_abc,
    sample_observed_data,
    summary_distance,
    summary_statistic,
    transition_statistic,
)


def test_rbf_covariance_is_finite_at_zero_lengthscale():
    config = replace(ExperimentConfig(), num_locations=8)
    covariance = rbf_covariance(
        build_locations(config), 0.0, config.variance, config.jitter
    )
    assert covariance.shape == (8, 8)
    assert jnp.isfinite(covariance).all()
    assert jnp.allclose(jnp.diag(covariance), 1.0 + config.jitter)


def test_build_locations_uses_requested_size_and_domain():
    config = replace(ExperimentConfig(), num_locations=256, domain_stop=100.0)
    locations = build_locations(config)
    assert locations.shape == (256, 1)
    assert jnp.isclose(locations[0, 0], 0.0)
    assert jnp.isclose(locations[-1, 0], 100.0)


def test_summary_statistic_contains_ones_and_adjacent_changes():
    observations = jnp.array([0, 0, 1, 1, 0])
    assert jnp.isclose(transition_statistic(observations), 0.5)
    assert jnp.allclose(summary_statistic(observations), jnp.array([0.4, 0.5]))


def test_summary_distance_is_two_component_euclidean_distance():
    observed_statistic = jnp.array([0.4, 0.5])
    simulated = jnp.array([[0, 0, 1, 1, 0], [1, 1, 1, 1, 1]])
    distances = summary_distance(simulated, observed_statistic)
    assert jnp.allclose(distances, jnp.array([0.0, jnp.sqrt(0.61)]))


def test_active_checkpoints_include_requested_and_final_steps():
    config = replace(
        ExperimentConfig(),
        train_steps=12_000,
        checkpoint_steps=(1_000, 5_000, 10_000, 25_000),
    )
    assert active_checkpoint_steps(config) == (1_000, 5_000, 10_000, 12_000)


def test_observed_data_and_exact_distances_have_expected_shapes():
    config = replace(ExperimentConfig(), num_locations=8)
    locations = build_locations(config)
    observed = sample_observed_data(random.key(1), locations, config)
    observed_statistic = summary_statistic(observed["observations"])
    distance_fn = make_exact_distance_fn(locations, observed_statistic, config)
    lengthscales = jnp.array([0.0, 0.5, 0.9])
    z = random.normal(random.key(2), (3, 8))
    uniforms = random.uniform(random.key(3), (3, 8))
    distances = distance_fn(lengthscales, z, uniforms)
    assert observed["observations"].shape == (8,)
    assert set(np.asarray(observed["observations"]).tolist()) <= {0, 1}
    assert distances.shape == (3,)
    assert jnp.isfinite(distances).all()


def test_threshold_calibration_returns_empirical_quantile():
    config = replace(
        ExperimentConfig(),
        num_locations=8,
        calibration_datasets=32,
        abc_batch_size=8,
    )
    locations = build_locations(config)
    observed = sample_observed_data(random.key(4), locations, config)
    observed_statistic = summary_statistic(observed["observations"])
    threshold, distances = calibrate_threshold(
        random.key(5), locations, observed_statistic, config
    )
    expected = float(jnp.quantile(jnp.asarray(distances), 0.10))
    assert distances.shape == (32,)
    assert np.isclose(threshold, expected)


def test_rejection_abc_stops_at_target_sample_count():
    config = replace(
        ExperimentConfig(),
        num_locations=4,
        abc_batch_size=8,
        target_accepted=10,
        max_proposals=32,
    )

    @jax.jit
    def zero_distance(lengthscales, z, uniforms):
        del z, uniforms
        return jnp.zeros_like(lengthscales)

    result = run_rejection_abc(random.key(6), 0.0, zero_distance, config)
    assert result.accepted.shape == (10,)
    assert result.proposals_processed == 16
    assert result.accepted_in_processed_batches == 16
    assert np.all(result.distances == 0.0)


def test_checkpoint_summary_compares_paired_abc_results():
    exact = ABCResult(
        accepted=np.array([0.4, 0.5]),
        proposals=np.array([0.4, 0.5, 0.6]),
        distances=np.array([0.0, 0.2, 0.1]),
        proposals_processed=3,
        accepted_in_processed_batches=2,
        runtime_seconds=1.0,
    )
    deeprv = ABCResult(
        accepted=np.array([0.5, 0.6]),
        proposals=np.array([0.4, 0.5, 0.6]),
        distances=np.array([0.2, 0.2, 0.0]),
        proposals_processed=3,
        accepted_in_processed_batches=1,
        runtime_seconds=0.5,
    )
    summary = checkpoint_abc_summary(exact, deeprv, 0.1, 0.5, 0.01)
    assert np.isclose(summary["posterior_mean_difference_from_exact"], 0.1)
    assert np.isclose(summary["acceptance_disagreement_rate"], 1 / 3)
    assert summary["shared_proposals_compared"] == 3


def test_checkpoint_plots_are_saved(tmp_path):
    exact = np.linspace(0.0, 1.0, 20)
    deeprv = {1_000: exact**2, 10_000: exact}
    summaries = {
        1_000: {
            "test_norm_mse": 0.1,
            "posterior_wasserstein_distance_from_exact": 0.2,
            "posterior_mean_difference_from_exact": -0.1,
            "acceptance_disagreement_rate": 0.3,
        },
        10_000: {
            "test_norm_mse": 0.01,
            "posterior_wasserstein_distance_from_exact": 0.02,
            "posterior_mean_difference_from_exact": -0.01,
            "acceptance_disagreement_rate": 0.03,
        },
    }
    plot_posteriors(exact, deeprv, 0.5, tmp_path)
    plot_checkpoint_diagnostics(summaries, tmp_path)
    for name in ("posterior_comparison", "checkpoint_diagnostics"):
        assert (tmp_path / f"{name}.png").exists()
        assert (tmp_path / f"{name}.pdf").exists()
