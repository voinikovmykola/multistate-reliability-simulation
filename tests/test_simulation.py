"""Invariants of the simulation core."""

from __future__ import annotations

import numpy as np
import pytest

from multistate_reliability.data_loading import (
    load_architectures, load_transition_probability_matrix)
from multistate_reliability.simulation import (
    ADMISSIBLE_TRANSITION_MATRIX, Architecture, N_STATES, SystemState,
    constant_business_criticality, simulate_architecture,
    validate_transition_probability_matrix)

COST_RATES = np.array([0.0, 25.0, 130.0, 550.0, 100.0])


def make_architecture(**overrides) -> Architecture:
    matrix = np.array([
        [0.90, 0.05, 0.03, 0.02, 0.00],
        [0.40, 0.40, 0.10, 0.05, 0.05],
        [0.00, 0.30, 0.40, 0.10, 0.20],
        [0.00, 0.10, 0.10, 0.40, 0.40],
        [0.50, 0.20, 0.10, 0.10, 0.10],
    ])
    defaults = {"name": "A0 - Test architecture",
                "transition_probability_matrix": matrix,
                "state_cost_rates": COST_RATES}
    return Architecture(**{**defaults, **overrides})


def test_rejects_probability_on_forbidden_transition():
    matrix = np.eye(N_STATES)
    matrix[0] = [0.9, 0.0, 0.0, 0.0, 0.1]        # S1 -> S5 is not admissible
    with pytest.raises(ValueError, match="forbidden transition"):
        validate_transition_probability_matrix(matrix)


def test_rejects_rows_that_do_not_sum_to_one():
    matrix = np.full((N_STATES, N_STATES), 0.1) * ADMISSIBLE_TRANSITION_MATRIX
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_transition_probability_matrix(matrix)


def test_rejects_probability_outside_unit_interval():
    matrix = np.zeros((N_STATES, N_STATES))
    matrix[:, 1] = 1.5
    matrix[:, 2] = -0.5
    with pytest.raises(ValueError, match=r"0 <= p_ij <= 1"):
        validate_transition_probability_matrix(matrix)


def test_rejects_wrong_number_of_cost_rates():
    with pytest.raises(ValueError, match="state_cost_rates"):
        make_architecture(state_cost_rates=np.zeros(N_STATES + 1))


def test_architecture_id_comes_from_the_name():
    assert make_architecture().architecture_id == "A0"


def test_same_seed_reproduces_the_same_losses():
    kwargs = {"horizon_hours": 720, "num_runs": 20}
    first = simulate_architecture(make_architecture(), random_seed=7, **kwargs)
    second = simulate_architecture(make_architecture(), random_seed=7, **kwargs)
    third = simulate_architecture(make_architecture(), random_seed=8, **kwargs)

    np.testing.assert_array_equal(first.loss_per_run, second.loss_per_run)
    assert not np.array_equal(first.loss_per_run, third.loss_per_run)


def test_occupancy_and_loss_breakdown_are_exact_decompositions():
    horizon = 1000
    result = simulate_architecture(make_architecture(), horizon_hours=horizon,
                                   num_runs=15, random_seed=3)

    np.testing.assert_allclose(result.state_occupancy_time.sum(axis=1), horizon)
    np.testing.assert_allclose(result.state_loss_contribution.sum(axis=1),
                               result.loss_per_run)
    np.testing.assert_allclose(result.mean_state_loss_contribution.sum(),
                               result.expected_annual_loss)


def test_constant_criticality_reduces_loss_to_sum_of_occupancy_times_cost():
    """With q(t) = 1, Eq. 10 must collapse to Eq. 5."""
    result = simulate_architecture(make_architecture(), horizon_hours=800,
                                   num_runs=10, random_seed=11,
                                   business_criticality=constant_business_criticality)
    np.testing.assert_allclose(result.loss_per_run,
                               result.state_occupancy_time @ COST_RATES)


def test_availability_matches_time_outside_failed_states():
    result = simulate_architecture(make_architecture(), horizon_hours=600,
                                   num_runs=10, random_seed=5)
    available_hours = result.state_occupancy_time[:, :2].sum(axis=1)

    np.testing.assert_allclose(result.generalized_availability,
                               available_hours / 600)
    assert 0.0 <= result.mean_generalized_availability <= 1.0


def test_trajectory_never_takes_a_forbidden_transition():
    result = simulate_architecture(make_architecture(), horizon_hours=5000,
                                   num_runs=1, random_seed=1)
    trajectory = result.example_trajectory
    for source, destination in zip(trajectory[:-1], trajectory[1:]):
        assert ADMISSIBLE_TRANSITION_MATRIX[source, destination] == 1


def test_state_enum_matches_the_documented_order():
    assert (int(SystemState.NORMAL_OPERATION), int(SystemState.RECOVERY)) == (0, 4)
    assert len(SystemState) == N_STATES


def stationary_availability(P: np.ndarray) -> float:
    """sum of pi_i over W = {S1, S2}, with pi the stationary distribution of P."""
    values, vectors = np.linalg.eig(P.T)
    pi = np.abs(np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))]))
    return float(pi[:2].sum() / pi.sum())


@pytest.mark.parametrize("architecture_id", ("A0", "A1", "A2"))
def test_simulated_availability_matches_the_stationary_value(architecture_id):
    architecture = next(a for a in load_architectures()
                        if a.architecture_id == architecture_id)
    simulated = simulate_architecture(architecture, horizon_hours=8760,
                                      num_runs=60, random_seed=17)
    expected = stationary_availability(architecture.transition_probability_matrix)

    assert simulated.mean_generalized_availability == pytest.approx(expected,
                                                                   abs=5e-3)


def test_reliability_ordering_of_the_published_architectures():
    availability = [stationary_availability(load_transition_probability_matrix(key))
                    for key in ("A0", "A1", "A2")]
    assert availability[0] < availability[1] < availability[2]
