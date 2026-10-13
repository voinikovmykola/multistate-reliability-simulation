"""Constraints the sensitivity analysis must respect."""

from __future__ import annotations

import numpy as np
import pytest

from multistate_reliability.config import load_experiment_config
from multistate_reliability.data_loading import (
    load_architectures, load_transition_probability_matrix,
    make_business_criticality_function)
from multistate_reliability.sensitivity import (SCENARIO_RANGE,
                                                SensitivityScenario,
                                                build_scenarios, compact_table,
                                                evaluate_scenarios,
                                                ranges_table,
                                                scale_admissible_transitions,
                                                scale_business_criticality,
                                                simulate_scenario_grid,
                                                summarize)
from multistate_reliability.simulation import (ADMISSIBLE_TRANSITION_MATRIX,
                                               simulate_architecture)

ARCHITECTURE_IDS = ("A0", "A1", "A2")
FACTORS = (1 - SCENARIO_RANGE, 1.0, 1 + SCENARIO_RANGE)


@pytest.fixture(scope="module")
def config():
    from dataclasses import replace
    return replace(load_experiment_config(), horizon_hours=720, num_runs=40,
                   checked_run_counts=(40,))


@pytest.mark.parametrize("architecture_id", ARCHITECTURE_IDS)
@pytest.mark.parametrize("factor", FACTORS)
def test_perturbed_matrix_keeps_structural_zeros_and_row_sums(architecture_id,
                                                             factor):
    P = load_transition_probability_matrix(architecture_id)
    perturbed = scale_admissible_transitions(P, factor)

    np.testing.assert_allclose(perturbed.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(perturbed[ADMISSIBLE_TRANSITION_MATRIX == 0] == 0.0)
    assert np.all(perturbed >= 0.0) and np.all(perturbed <= 1.0)


@pytest.mark.parametrize("architecture_id", ARCHITECTURE_IDS)
def test_higher_factor_makes_the_system_leave_its_state_more_readily(architecture_id):
    P = load_transition_probability_matrix(architecture_id)
    low = scale_admissible_transitions(P, 1 - SCENARIO_RANGE)
    high = scale_admissible_transitions(P, 1 + SCENARIO_RANGE)

    assert np.all(np.diag(high) < np.diag(P))
    assert np.all(np.diag(low) > np.diag(P))


def test_unit_factor_leaves_the_matrix_unchanged():
    P = load_transition_probability_matrix("A1")
    np.testing.assert_allclose(scale_admissible_transitions(P, 1.0), P,
                               atol=1e-15)


def test_criticality_profile_stays_positive_under_every_exponent():
    base = make_business_criticality_function()
    for exponent in (0.0, 0.5, 1.0, 1.5):
        scaled = scale_business_criticality(base, exponent)
        values = np.array([scaled(t) for t in range(168)])
        assert (values > 0).all(), exponent
        if exponent == 0.0:
            np.testing.assert_allclose(values, 1.0)
        if exponent == 1.0:
            np.testing.assert_allclose(values,
                                       [base(t) for t in range(168)])


def test_amplified_profile_widens_the_spread_around_one():
    base = make_business_criticality_function()
    published = np.array([base(t) for t in range(168)])
    attenuated = np.array([scale_business_criticality(base, 0.5)(t)
                           for t in range(168)])
    amplified = np.array([scale_business_criticality(base, 1.5)(t)
                          for t in range(168)])

    assert attenuated.std() < published.std() < amplified.std()


def test_scenario_grid_covers_every_parameter_group(config):
    scenarios = build_scenarios(config)
    groups = {scenario.parameter for scenario in scenarios}

    assert groups == {"published parameters", "transition probabilities p_ij",
                      "state cost rates C2 to C5", "business criticality q(t)",
                      "annualized costs I1, I2",
                      "acceptable loss threshold L_allow"}
    assert sum(s.scenario_id == "base" for s in scenarios) == 1
    # Only the transition and criticality groups need their own simulation.
    assert len({s.simulation_key for s in scenarios}) == 6


def test_thresholds_span_the_accepted_range(config):
    thresholds = sorted(s.acceptable_loss_threshold_mu
                        for s in build_scenarios(config)
                        if s.acceptable_loss_threshold_mu is not None)
    published = config.acceptable_annual_loss_threshold_mu

    assert thresholds[0] == pytest.approx(published * (1 - SCENARIO_RANGE))
    assert thresholds[-1] == pytest.approx(published * (1 + SCENARIO_RANGE))


def test_common_random_numbers_across_architectures():
    """Architectures sharing a matrix must produce identical trajectories."""
    baseline = load_architectures()[0]
    twin = load_architectures()[0]
    twin.state_cost_rates = twin.state_cost_rates * 3.0

    first = simulate_architecture(baseline, horizon_hours=720, num_runs=10,
                                  random_seed=42)
    second = simulate_architecture(twin, horizon_hours=720, num_runs=10,
                                   random_seed=42)

    np.testing.assert_array_equal(first.state_occupancy_time,
                                  second.state_occupancy_time)
    np.testing.assert_allclose(second.loss_per_run, first.loss_per_run * 3.0)


def test_cost_scaling_matches_a_direct_simulation():
    """Scaling C2..C5 must scale the annual loss exactly, which is why the
    sensitivity analysis rescales the sample instead of re-simulating."""
    architecture = load_architectures()[1]
    criticality = make_business_criticality_function()
    factor = 1 + SCENARIO_RANGE

    published = simulate_architecture(
        architecture, horizon_hours=720, num_runs=15,
        business_criticality=criticality, random_seed=7)

    scaled_architecture = load_architectures()[1]
    scaled_architecture.state_cost_rates = (
        scaled_architecture.state_cost_rates * factor)
    directly = simulate_architecture(
        scaled_architecture, horizon_hours=720, num_runs=15,
        business_criticality=criticality, random_seed=7)

    np.testing.assert_allclose(published.loss_per_run * factor,
                               directly.loss_per_run)


@pytest.fixture(scope="module")
def evaluated(config):
    scenarios = build_scenarios(config)
    samples = simulate_scenario_grid(config, scenarios)
    return scenarios, evaluate_scenarios(config, scenarios, samples)


def test_every_scenario_reports_every_architecture(evaluated):
    scenarios, table = evaluated
    assert len(table) == len(scenarios) * 3
    assert set(table["architecture_id"]) == set(ARCHITECTURE_IDS)


def test_exceedance_count_is_consistent_with_the_probability(evaluated):
    _, table = evaluated
    np.testing.assert_allclose(
        table["exceedance_count_k"] / table["num_runs_r"],
        table["loss_exceedance_probability"])
    assert (table["exceedance_count_k"] <= table["num_runs_r"]).all()


def test_baseline_has_no_net_economic_benefit(evaluated):
    _, table = evaluated
    baseline = table[table.architecture_id == "A0"]
    assert baseline["net_economic_benefit_mu_per_year"].isna().all()
    assert table[table.architecture_id != "A0"][
        "net_economic_benefit_mu_per_year"].notna().all()


def test_investment_scaling_only_moves_the_net_economic_benefit(evaluated):
    _, table = evaluated
    published = table[(table.scenario_id == "base")
                      & (table.architecture_id == "A2")].iloc[0]
    cheaper = table[(table.scenario_id == "investment-")
                    & (table.architecture_id == "A2")].iloc[0]

    assert cheaper["expected_annual_loss_mu"] == pytest.approx(
        published["expected_annual_loss_mu"])
    assert cheaper["net_economic_benefit_mu_per_year"] > (
        published["net_economic_benefit_mu_per_year"])


def test_ranking_and_summary_agree(evaluated):
    _, table = evaluated
    summary = summarize(table)

    assert summary["num_scenarios"] == table["scenario_id"].nunique()
    assert summary["accepted_scenario_range"] == SCENARIO_RANGE
    assert (summary["num_scenarios_with_a2_first"]
            + len(summary["scenarios_where_ranking_changes"])
            == summary["num_scenarios"])

    leaders = table[table.rank_by_net_economic_benefit == 1]
    assert len(leaders) == summary["num_scenarios"]


def test_compact_table_has_one_row_per_scenario(evaluated):
    scenarios, table = evaluated
    compact = compact_table(table)

    assert len(compact) == len(scenarios)
    assert "leading_architecture" in compact.columns
    assert "expected_annual_loss_mu_A2" in compact.columns


def test_scenario_simulation_key_groups_post_processed_variants():
    """Cost, investment and threshold changes must not trigger a simulation."""
    base = SensitivityScenario("base", "p", "p")
    for scenario in (SensitivityScenario("c", "p", "p", cost_factor=1.2),
                     SensitivityScenario("i", "p", "p", investment_factor=0.8),
                     SensitivityScenario("t", "p", "p",
                                         acceptable_loss_threshold_mu=1.0)):
        assert scenario.simulation_key == base.simulation_key


def test_break_even_is_reported_when_the_ranking_never_changes(evaluated):
    """If nothing flips the ranking, the distance to the flip must be stated."""
    _, table = evaluated
    summary = summarize(table)

    if summary["scenarios_where_ranking_changes"]:
        pytest.skip("the ranking changes in this grid, no break-even needed")

    assert summary["ranking_margin_a2_over_a1_mu_per_year"] > 0
    assert (summary["break_even_annualized_cost_of_a2_mu_per_year"]
            > table.loc[table.architecture_id == "A2",
                        "annualized_cost_mu_per_year"].max())
    assert summary["break_even_annualized_cost_factor"] > 1.0
    assert summary["break_even_expected_annual_loss_factor"] > 1.0


def test_ranges_table_spans_every_scenario(evaluated):
    scenarios, table = evaluated
    ranges = ranges_table(table).set_index("architecture_id")

    assert (ranges["num_configurations"] == len(scenarios)).all()
    for key in ARCHITECTURE_IDS:
        losses = table.loc[table.architecture_id == key, "expected_annual_loss_mu"]
        assert ranges.loc[key, "expected_annual_loss_mu_min"] == losses.min()
        assert ranges.loc[key, "expected_annual_loss_mu_max"] == losses.max()
    assert (ranges.loc["A1":, ["num_configurations_ranked_first",
                               "num_configurations_ranked_second"]].sum(axis=1)
            == len(scenarios)).all()
