"""The dataset and the configuration must agree with each other and with the code."""

from __future__ import annotations

import numpy as np
import pytest

from multistate_reliability.config import (INPUT_DIR, PARAMETERS_FILE,
                                           load_experiment_config)
from multistate_reliability.data_loading import (
    functionally_available_state_indices, load_admissible_transitions,
    load_architectures, load_business_criticality_profile, load_state_names,
    load_states, load_transition_probability_matrix, state_code_to_index)
from multistate_reliability.plot_style import ARCHITECTURE_IDS
from multistate_reliability.simulation import (ADMISSIBLE_TRANSITION_MATRIX,
                                               N_STATES, STATE_CODES)

EXPECTED_STATE_NAMES = ["Normal operating state", "Degraded operating state",
                        "Partial failure state", "Critical failure state",
                        "Recovery state"]


def test_input_files_are_present():
    assert PARAMETERS_FILE.exists()
    for name in ("states.csv", "architectures.csv", "admissible_transitions.csv",
                 "business_criticality_profile.csv"):
        assert (INPUT_DIR / name).exists(), name


def test_state_names_match_the_article():
    assert load_state_names() == EXPECTED_STATE_NAMES


def test_state_table_is_ordered_and_complete():
    states = load_states()
    assert states["state_code"].tolist() == list(STATE_CODES)
    assert states["state_cost_rate_mu_per_hour"].iloc[0] == 0.0


def test_admissibility_matrix_in_the_dataset_matches_the_code():
    np.testing.assert_array_equal(load_admissible_transitions(),
                                  ADMISSIBLE_TRANSITION_MATRIX)


@pytest.mark.parametrize("architecture_id", ARCHITECTURE_IDS)
def test_transition_matrices_are_stochastic_and_admissible(architecture_id):
    P = load_transition_probability_matrix(architecture_id)
    assert P.shape == (N_STATES, N_STATES)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-8)
    assert not np.any(P[ADMISSIBLE_TRANSITION_MATRIX == 0] > 0)


def test_architectures_load_in_baseline_first_order():
    architectures = load_architectures()
    assert [a.architecture_id for a in architectures] == list(ARCHITECTURE_IDS)
    assert architectures[0].annualized_cost == 0.0
    assert all(a.annualized_cost > 0 for a in architectures[1:])


def test_criticality_profile_covers_one_week_with_positive_weights():
    q = load_business_criticality_profile()
    assert q.size == 168
    assert (q > 0).all()


def test_available_states_agree_between_dataset_and_configuration():
    config = load_experiment_config()
    from_flags = functionally_available_state_indices()
    from_config = tuple(state_code_to_index(code)
                        for code in config.functionally_available_states)

    assert from_flags == from_config


def test_initial_state_in_configuration_is_a_known_state():
    config = load_experiment_config()
    assert config.initial_state in STATE_CODES


def test_configuration_values_are_the_published_criteria():
    config = load_experiment_config()

    assert config.horizon_hours == 8760
    assert config.time_step_hours == 1
    assert config.num_runs in config.checked_run_counts
    assert config.confidence_level == 0.95
    assert config.target_relative_margin_of_error == 0.05
    assert config.stabilization_threshold == 0.01
    assert config.acceptable_annual_loss_threshold_mu == 110_000
    assert config.max_checked_run_count == max(config.checked_run_counts)
