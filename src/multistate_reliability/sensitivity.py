"""One-factor-at-a-time sensitivity analysis (Table 5 of the article).

Each parameter group is varied while all others are held at their published
values:

* transition probabilities: the admissible off-diagonal entries of P are scaled
  by 0.8 and 1.2 and each row is renormalized, so the diagonal absorbs the
  change and structural zeros of A stay zero;
* state cost rates C2..C5: scaled by 0.8 and 1.2; with C1 = 0 every run's loss
  scales by exactly that factor, so the stored sample is rescaled instead of
  re-simulating;
* business criticality q(t): raised to the exponent 0 (uniform profile), 0.5
  (attenuated), 1 (baseline) or 1.5 (amplified); the profiles are not
  renormalized to a common mean;
* annualized costs I of A1 and A2: scaled by 0.8 and 1.2;
* acceptable loss threshold: 0.8, 0.9, 1.1 and 1.2 times the published value.

The ±20 % range is a scenario assumption, not an estimated uncertainty. Every
simulation uses the main seed, so all architectures and scenarios share the
same random-number stream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data_loading import (functionally_available_state_indices,
                           load_architectures, make_business_criticality_function,
                           state_code_to_index)
from .simulation import (ADMISSIBLE_TRANSITION_MATRIX, Architecture, N_STATES,
                         simulate_architecture,
                         validate_transition_probability_matrix)

logger = logging.getLogger(__name__)

SCENARIO_RANGE = 0.20

CRITICALITY_EXPONENTS = {0.0: "uniform profile", 0.5: "attenuated profile",
                         1.0: "baseline profile", 1.5: "amplified profile"}


@dataclass(frozen=True)
class SensitivityScenario:
    """One configuration relative to the published parameters.

    Only ``transition_factor`` and ``criticality_exponent`` require a
    simulation; the other fields are applied to the resulting loss sample.
    """

    scenario_id: str
    parameter: str
    change: str
    transition_factor: float = 1.0
    criticality_exponent: float = 1.0
    cost_factor: float = 1.0
    investment_factor: float = 1.0
    acceptable_loss_threshold_mu: float | None = None

    @property
    def simulation_key(self) -> tuple[float, float]:
        return self.transition_factor, self.criticality_exponent


def scale_admissible_transitions(
    transition_probability_matrix: np.ndarray,
    factor: float,
    admissible_transition_matrix: np.ndarray = ADMISSIBLE_TRANSITION_MATRIX,
) -> np.ndarray:
    """Scale the admissible off-diagonal probabilities of P and renormalize rows."""
    P = np.asarray(transition_probability_matrix, dtype=float).copy()
    admissible = np.asarray(admissible_transition_matrix) > 0
    off_diagonal = admissible & ~np.eye(N_STATES, dtype=bool)

    P[off_diagonal] *= factor
    P /= P.sum(axis=1, keepdims=True)

    validate_transition_probability_matrix(P, admissible_transition_matrix)
    return P


def scale_business_criticality(base_profile: Callable[[int], float],
                              exponent: float) -> Callable[[int], float]:
    """q(t) ** exponent."""
    def scaled(t: int) -> float:
        return float(base_profile(t) ** exponent)

    return scaled


def build_scenarios(config: ExperimentConfig,
                    scenario_range: float = SCENARIO_RANGE) -> list[SensitivityScenario]:
    """The scenario grid: one group per parameter, the published values as base."""
    low, high = 1 - scenario_range, 1 + scenario_range
    threshold = config.acceptable_annual_loss_threshold_mu
    percent = f"{scenario_range:.0%}"

    scenarios = [SensitivityScenario("base", "published parameters",
                                     "published values")]

    for factor, direction in ((low, "-"), (high, "+")):
        scenarios.append(SensitivityScenario(
            f"transitions{direction}", "transition probabilities p_ij",
            f"{direction}{percent}", transition_factor=factor))

    for factor, direction in ((low, "-"), (high, "+")):
        scenarios.append(SensitivityScenario(
            f"cost_rates{direction}", "state cost rates C2 to C5",
            f"{direction}{percent}", cost_factor=factor))

    for exponent, label in CRITICALITY_EXPONENTS.items():
        if exponent == 1.0:
            continue
        scenarios.append(SensitivityScenario(
            f"criticality_{exponent:g}", "business criticality q(t)", label,
            criticality_exponent=exponent))

    for factor, direction in ((low, "-"), (high, "+")):
        scenarios.append(SensitivityScenario(
            f"investment{direction}", "annualized costs I1, I2",
            f"{direction}{percent}", investment_factor=factor))

    for factor in (low, 1 - scenario_range / 2, 1 + scenario_range / 2, high):
        value = threshold * factor
        scenarios.append(SensitivityScenario(
            f"threshold_{value:.0f}", "acceptable loss threshold L_allow",
            f"{value:,.0f} MU/year".replace(",", " "),
            acceptable_loss_threshold_mu=value))

    return scenarios


def _perturbed_architectures(architectures: list[Architecture],
                             transition_factor: float) -> list[Architecture]:
    if transition_factor == 1.0:
        return architectures
    return [Architecture(
        name=architecture.name,
        transition_probability_matrix=scale_admissible_transitions(
            architecture.transition_probability_matrix, transition_factor),
        state_cost_rates=architecture.state_cost_rates,
        annualized_cost=architecture.annualized_cost,
        description=architecture.description) for architecture in architectures]


def simulate_scenario_grid(config: ExperimentConfig,
                           scenarios: list[SensitivityScenario],
                           architectures: list[Architecture] | None = None,
                           ) -> dict[tuple[float, float], dict[str, np.ndarray]]:
    """Loss samples of every architecture, once per distinct simulation key."""
    architectures = architectures or load_architectures()
    base_profile = make_business_criticality_function()
    available_states = functionally_available_state_indices()
    initial_state = state_code_to_index(config.initial_state)

    samples: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for key in dict.fromkeys(scenario.simulation_key for scenario in scenarios):
        transition_factor, exponent = key
        logger.info("simulating scenario grid point: transition factor %.2f, "
                    "criticality exponent %.2f", transition_factor, exponent)

        criticality = scale_business_criticality(base_profile, exponent)
        samples[key] = {
            architecture.architecture_id: simulate_architecture(
                architecture,
                horizon_hours=config.horizon_hours,
                num_runs=config.num_runs,
                initial_state=initial_state,
                business_criticality=criticality,
                functionally_available_states=available_states,
                random_seed=config.main_random_seed).loss_per_run
            for architecture in _perturbed_architectures(architectures,
                                                         transition_factor)
        }
    return samples


def evaluate_scenarios(config: ExperimentConfig,
                       scenarios: list[SensitivityScenario],
                       samples: dict[tuple[float, float], dict[str, np.ndarray]],
                       architectures: list[Architecture] | None = None,
                       ) -> pd.DataFrame:
    """One row per scenario and architecture, with every reported indicator."""
    architectures = architectures or load_architectures()
    baseline_id = architectures[0].architecture_id
    base_profile = make_business_criticality_function()

    rows = []
    for scenario in scenarios:
        losses = {key: sample * scenario.cost_factor
                  for key, sample in samples[scenario.simulation_key].items()}
        baseline_loss = float(losses[baseline_id].mean())
        threshold = (scenario.acceptable_loss_threshold_mu
                     if scenario.acceptable_loss_threshold_mu is not None
                     else config.acceptable_annual_loss_threshold_mu)
        mean_criticality = float(np.mean(
            [base_profile(t) ** scenario.criticality_exponent
             for t in range(168)]))

        for architecture in architectures:
            key = architecture.architecture_id
            loss = losses[key]
            annualized_cost = (architecture.annualized_cost
                               * scenario.investment_factor)
            is_baseline = key == baseline_id

            rows.append({
                "scenario_id": scenario.scenario_id,
                "parameter": scenario.parameter,
                "change": scenario.change,
                "architecture_id": key,
                "transition_factor": scenario.transition_factor,
                "criticality_exponent": scenario.criticality_exponent,
                "mean_business_criticality": mean_criticality,
                "cost_factor": scenario.cost_factor,
                "investment_factor": scenario.investment_factor,
                "acceptable_annual_loss_threshold_mu": threshold,
                "expected_annual_loss_mu": float(loss.mean()),
                "loss_quantile_095_mu": float(np.percentile(loss, 95)),
                "loss_quantile_099_mu": float(np.percentile(loss, 99)),
                "exceedance_count_k": int((loss > threshold).sum()),
                "num_runs_r": int(loss.size),
                "loss_exceedance_probability": float((loss > threshold).mean()),
                "annualized_cost_mu_per_year": annualized_cost,
                "net_economic_benefit_mu_per_year": (
                    np.nan if is_baseline
                    else baseline_loss - float(loss.mean()) - annualized_cost),
            })

    table = pd.DataFrame(rows)
    return _add_ranking(table)


def _add_ranking(table: pd.DataFrame) -> pd.DataFrame:
    """Rank the non-baseline architectures by net economic benefit, best first."""
    table = table.copy()
    table["rank_by_net_economic_benefit"] = (
        table.groupby("scenario_id")["net_economic_benefit_mu_per_year"]
        .rank(ascending=False, method="min").astype("Int64"))
    return table


def _ranking_margin(table: pd.DataFrame) -> dict:
    """Break-even values of I and Ê(L) of A2 at which A1 would overtake it."""
    base = table[table.scenario_id == "base"].set_index("architecture_id")
    if not {"A1", "A2"} <= set(base.index):
        return {}

    margin = float(base.loc["A2", "net_economic_benefit_mu_per_year"]
                   - base.loc["A1", "net_economic_benefit_mu_per_year"])
    annualized_cost = float(base.loc["A2", "annualized_cost_mu_per_year"])
    expected_loss = float(base.loc["A2", "expected_annual_loss_mu"])

    return {
        "ranking_margin_a2_over_a1_mu_per_year": margin,
        "break_even_annualized_cost_of_a2_mu_per_year": annualized_cost + margin,
        "break_even_annualized_cost_factor": (annualized_cost + margin) / annualized_cost,
        "break_even_expected_annual_loss_of_a2_mu": expected_loss + margin,
        "break_even_expected_annual_loss_factor": (expected_loss + margin) / expected_loss,
    }


def summarize(table: pd.DataFrame) -> dict:
    """Ranking stability over all scenarios and the break-even margins."""
    leaders = (table.dropna(subset=["net_economic_benefit_mu_per_year"])
               .sort_values("net_economic_benefit_mu_per_year", ascending=False)
               .groupby("scenario_id", sort=False).head(1)
               .set_index("scenario_id"))

    best = leaders["architecture_id"]
    changed = leaders.loc[best != "A2"]

    return {
        "accepted_scenario_range": SCENARIO_RANGE,
        "num_scenarios": int(len(best)),
        "num_runs_r": int(table["num_runs_r"].max()),
        "num_scenarios_with_a2_first": int((best == "A2").sum()),
        "scenarios_where_ranking_changes": [
            {"scenario_id": scenario_id,
             "parameter": row["parameter"],
             "change": row["change"],
             "leading_architecture": row["architecture_id"]}
            for scenario_id, row in changed.iterrows()],
        "min_net_economic_benefit_of_a2_mu_per_year": float(
            table.loc[table.architecture_id == "A2",
                      "net_economic_benefit_mu_per_year"].min()),
        "max_exceedance_probability_of_a2": float(
            table.loc[table.architecture_id == "A2",
                      "loss_exceedance_probability"].max()),
        "max_exceedance_probability_of_a0": float(
            table.loc[table.architecture_id == "A0",
                      "loss_exceedance_probability"].max()),
        "min_exceedance_probability_of_a0": float(
            table.loc[table.architecture_id == "A0",
                      "loss_exceedance_probability"].min()),
        **_ranking_margin(table),
    }


def compact_table(table: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario, architectures across the columns."""
    wide = table.pivot_table(
        index=["scenario_id", "parameter", "change"],
        columns="architecture_id",
        values=["expected_annual_loss_mu", "loss_quantile_095_mu",
                "exceedance_count_k", "net_economic_benefit_mu_per_year"],
        sort=False)
    wide.columns = [f"{metric}_{architecture}"
                    for metric, architecture in wide.columns]

    leaders = (table.dropna(subset=["net_economic_benefit_mu_per_year"])
               .sort_values("net_economic_benefit_mu_per_year", ascending=False)
               .groupby("scenario_id", sort=False).head(1)
               .set_index("scenario_id")["architecture_id"])
    wide["leading_architecture"] = leaders.reindex(
        wide.index.get_level_values("scenario_id")).to_numpy()

    return wide.reset_index()


def ranges_table(table: pd.DataFrame) -> pd.DataFrame:
    """Table 5: per architecture, the range of each indicator over all scenarios."""
    rows = []
    for key, group in table.groupby("architecture_id", sort=True):
        row = {"architecture_id": key, "num_configurations": len(group),
               "num_runs_r": int(group["num_runs_r"].max())}
        for column in ("expected_annual_loss_mu", "loss_quantile_095_mu",
                       "loss_quantile_099_mu", "exceedance_count_k",
                       "net_economic_benefit_mu_per_year"):
            row[f"{column}_min"] = group[column].min()
            row[f"{column}_max"] = group[column].max()
        ranks = group["rank_by_net_economic_benefit"].dropna()
        row["num_configurations_ranked_first"] = int((ranks == 1).sum())
        row["num_configurations_ranked_second"] = int((ranks == 2).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run_sensitivity_analysis(config: ExperimentConfig,
                             scenario_range: float = SCENARIO_RANGE,
                             ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the whole grid; return the detailed table, compact table and summary."""
    architectures = load_architectures()
    scenarios = build_scenarios(config, scenario_range)

    samples = simulate_scenario_grid(config, scenarios, architectures)
    detailed = evaluate_scenarios(config, scenarios, samples, architectures)

    return detailed, compact_table(detailed), summarize(detailed)
