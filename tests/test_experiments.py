"""Table building, the stabilization criterion and the saved sample round trip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multistate_reliability.config import load_experiment_config
from multistate_reliability.data_loading import (
    load_architectures, make_business_criticality_function)
from multistate_reliability.experiments import (build_convergence_table,
                                                build_run_length_summary,
                                                build_run_metadata,
                                                build_stabilization_table,
                                                build_state_table,
                                                build_summary_table,
                                                load_loss_samples,
                                                save_loss_samples,
                                                simulate_all_architectures)
from multistate_reliability.simulation import STATE_CODES

SMALL_RUN_COUNTS = (20, 40)


@pytest.fixture(scope="module")
def config():
    from dataclasses import replace
    return replace(load_experiment_config(), horizon_hours=720, num_runs=40,
                   pilot_num_runs=20, checked_run_counts=SMALL_RUN_COUNTS)


@pytest.fixture(scope="module")
def results(config):
    return simulate_all_architectures(config,
                                      make_business_criticality_function())


def test_simulation_covers_every_architecture_at_the_largest_count(config, results):
    assert set(results) == {"A0", "A1", "A2"}
    assert all(r.num_runs == config.max_checked_run_count
               for r in results.values())


def test_summary_table_reports_one_row_per_architecture(config, results):
    losses = {k: r.loss_per_run for k, r in results.items()}
    availability = {k: r.generalized_availability for k, r in results.items()}
    summary = build_summary_table(load_architectures(), losses, availability,
                                  config)

    assert summary["architecture_id"].tolist() == ["A0", "A1", "A2"]
    assert np.isnan(summary.loc[0, "net_economic_benefit_mu_per_year"]), (
        "the baseline is the reference point, so its benefit is undefined")
    assert summary.loc[1:, "net_economic_benefit_mu_per_year"].notna().all()
    assert (summary["loss_exceedance_probability"].between(0, 1)).all()
    assert (summary["generalized_availability_pct"].between(0, 100)).all()


def test_net_economic_benefit_equals_avoided_loss_minus_cost(config, results):
    losses = {k: r.loss_per_run for k, r in results.items()}
    availability = {k: r.generalized_availability for k, r in results.items()}
    summary = build_summary_table(load_architectures(), losses, availability,
                                  config).set_index("architecture_id")

    baseline_loss = summary.loc["A0", "expected_annual_loss_mu"]
    for key in ("A1", "A2"):
        row = summary.loc[key]
        assert row["net_economic_benefit_mu_per_year"] == pytest.approx(
            baseline_loss - row["expected_annual_loss_mu"]
            - row["annualized_cost_mu_per_year"])


def test_state_tables_are_consistent_with_the_summary(config, results):
    losses = {k: r.loss_per_run for k, r in results.items()}
    availability = {k: r.generalized_availability for k, r in results.items()}
    summary = build_summary_table(load_architectures(), losses, availability,
                                  config).set_index("architecture_id")

    occupancy = build_state_table(results, "state_occupancy_time",
                                  config.num_runs)
    contribution = build_state_table(results, "state_loss_contribution",
                                     config.num_runs)

    assert occupancy.index.tolist() == list(STATE_CODES)
    for key in results:
        assert occupancy[key].sum() == pytest.approx(config.horizon_hours)
        assert contribution[key].sum() == pytest.approx(
            summary.loc[key, "expected_annual_loss_mu"])


def test_convergence_table_covers_every_architecture_and_count(config, results):
    losses = {k: r.loss_per_run for k, r in results.items()}
    availability = {k: r.generalized_availability for k, r in results.items()}
    convergence = build_convergence_table(losses, availability, config)

    assert len(convergence) == len(results) * len(config.checked_run_counts)
    assert (convergence["relative_margin_of_error_pct"] > 0).all()
    assert (convergence["loss_quantile_099_mu"]
            >= convergence["loss_quantile_095_mu"]).all()


def test_stabilization_criterion_tracks_the_threshold(config):
    """A change just above the threshold fails, just below it passes."""
    def convergence_with_change(relative_change: float) -> pd.DataFrame:
        rows = []
        for count, factor in zip(SMALL_RUN_COUNTS, (1.0, 1.0 + relative_change)):
            rows.append({"architecture_id": "A0", "num_runs": count,
                         "expected_annual_loss_mu": 100.0 * factor,
                         "generalized_availability_pct": 99.0,
                         "loss_quantile_095_mu": 150.0 * factor,
                         "loss_quantile_099_mu": 180.0 * factor,
                         "margin_of_error_mu": 1.0,
                         "relative_margin_of_error_pct": 1.0})
        return pd.DataFrame(rows)

    threshold = config.stabilization_threshold
    assert build_stabilization_table(convergence_with_change(threshold / 2),
                                     config)["criterion_met"].all()
    assert not build_stabilization_table(convergence_with_change(threshold * 2),
                                         config)["criterion_met"].any()


def test_run_length_summary_reports_the_adopted_count(config, results):
    losses = {k: r.loss_per_run for k, r in results.items()}
    availability = {k: r.generalized_availability for k, r in results.items()}
    convergence = build_convergence_table(losses, availability, config)
    stabilization = build_stabilization_table(convergence, config)
    pilot = pd.DataFrame({"architecture_id": ["A0"], "min_required_runs": [7]})

    summary = build_run_length_summary(pilot, convergence, stabilization, config)

    assert summary["min_required_runs_from_formula"] == 7
    assert summary["adopted_num_runs"] == config.num_runs
    assert summary["acceptable_annual_loss_threshold_mu"] == 110_000


def test_loss_samples_survive_a_save_and_load_round_trip(tmp_path, results):
    path = tmp_path / "loss_samples.npz"
    save_loss_samples(results, path)
    losses, availability = load_loss_samples(path)

    assert set(losses) == set(availability) == set(results)
    for key, result in results.items():
        np.testing.assert_array_equal(losses[key], result.loss_per_run)
        np.testing.assert_array_equal(availability[key],
                                      result.generalized_availability)


def test_run_metadata_records_the_configuration_and_inputs(config):
    metadata = build_run_metadata(config, num_runs=40, random_seed=42)

    assert metadata["num_runs"] == 40
    assert metadata["random_seed"] == 42
    assert metadata["package_version"]
    assert "parameters.json" in metadata["input_file_sha256_prefixes"]
    assert "states.csv" in metadata["input_file_sha256_prefixes"]
