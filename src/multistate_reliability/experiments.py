"""Experiment stages that turn the dataset into the reported tables.

The main simulation runs every architecture at the largest checked run count
(5000) and saves the per-run samples. The tables at the adopted R = 3000 and the
stabilization check at every smaller checked count are computed on the leading
runs of those samples, so all reported numbers come from one simulated sample.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (ExperimentConfig, INPUT_DIR, PARAMETERS_FILE,
                     loss_sample_file)
from .data_loading import (functionally_available_state_indices,
                           load_architectures, state_code_to_index)
from .simulation import (Architecture, STATE_CODES, SimulationResult,
                         simulate_architecture)

logger = logging.getLogger(__name__)


def simulate_all_architectures(
    config: ExperimentConfig,
    business_criticality,
    num_runs: int | None = None,
    random_seed: int | None = None,
    architectures: list[Architecture] | None = None,
) -> dict[str, SimulationResult]:
    """Simulate every architecture with the same seed (common random numbers).

    ``num_runs`` defaults to the largest checked run count and ``random_seed``
    to the main seed.
    """
    architectures = architectures or load_architectures()
    num_runs = num_runs or config.max_checked_run_count
    random_seed = config.main_random_seed if random_seed is None else random_seed
    available_states = functionally_available_state_indices()

    results = {}
    for architecture in architectures:
        logger.info("simulating %s over %d runs (seed %d)",
                    architecture.name, num_runs, random_seed)
        results[architecture.architecture_id] = simulate_architecture(
            architecture,
            horizon_hours=config.horizon_hours,
            num_runs=num_runs,
            initial_state=state_code_to_index(config.initial_state),
            business_criticality=business_criticality,
            functionally_available_states=available_states,
            random_seed=random_seed,
        )
    return results


def save_loss_samples(results: dict[str, SimulationResult], path: Path) -> None:
    """Arrays ``loss_<id>`` (MU/year) and ``availability_<id>`` (fraction)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"loss_{key}": result.loss_per_run for key, result in results.items()},
        **{f"availability_{key}": result.generalized_availability
           for key, result in results.items()},
    )
    logger.info("wrote %s", path)


def load_loss_samples(path: Path) -> tuple[dict[str, np.ndarray],
                                           dict[str, np.ndarray]]:
    with np.load(path) as archive:
        losses = {name.removeprefix("loss_"): archive[name]
                  for name in archive.files if name.startswith("loss_")}
        availability = {name.removeprefix("availability_"): archive[name]
                        for name in archive.files
                        if name.startswith("availability_")}
    return losses, availability


def build_summary_table(architectures: list[Architecture],
                        losses: dict[str, np.ndarray],
                        availability: dict[str, np.ndarray],
                        config: ExperimentConfig) -> pd.DataFrame:
    """Table 4 of the article: one row per architecture at the adopted R.

    NEB (Eq. 11) is measured against the first architecture (A0) and is left
    undefined (NaN) for A0 itself.
    """
    adopted = config.num_runs
    threshold = config.acceptable_annual_loss_threshold_mu
    baseline_id = architectures[0].architecture_id
    baseline_loss = float(losses[baseline_id][:adopted].mean())

    rows = []
    for architecture in architectures:
        key = architecture.architecture_id
        loss = losses[key][:adopted]
        available = availability[key][:adopted]
        expected_loss = float(loss.mean())

        rows.append({
            "architecture_id": key,
            "architecture_name": architecture.name,
            "generalized_availability_pct": float(available.mean() * 100),
            "downtime_hours_per_year": float((1 - available).mean()
                                             * config.horizon_hours),
            "expected_annual_loss_mu": expected_loss,
            "loss_quantile_095_mu": float(np.percentile(loss, 95)),
            "loss_quantile_099_mu": float(np.percentile(loss, 99)),
            "loss_exceedance_probability": float(np.mean(loss > threshold)),
            "annualized_cost_mu_per_year": architecture.annualized_cost,
            "net_economic_benefit_mu_per_year": (
                np.nan if key == baseline_id
                else baseline_loss - expected_loss - architecture.annualized_cost),
        })
    return pd.DataFrame(rows)


def build_state_table(results: dict[str, SimulationResult], attribute: str,
                      num_runs: int) -> pd.DataFrame:
    """Per-state means over the first ``num_runs`` runs, one column per architecture.

    ``state_loss_contribution`` gives Fig. 4 (MU/year, columns sum to Ê(L));
    ``state_occupancy_time`` gives hours per year.
    """
    table = pd.DataFrame(
        {key: getattr(result, attribute)[:num_runs].mean(axis=0)
         for key, result in results.items()},
        index=list(STATE_CODES))
    return table.rename_axis("state")


def build_pilot_table(config: ExperimentConfig,
                      business_criticality) -> pd.DataFrame:
    """Pilot series: R_min = ceil((z * s0 / e)^2) with e the target relative margin
    of error times the pilot mean loss."""
    results = simulate_all_architectures(
        config, business_criticality, num_runs=config.pilot_num_runs,
        random_seed=config.pilot_random_seed)

    rows = []
    for key, result in results.items():
        mean_loss = result.expected_annual_loss
        sample_std = result.loss_sample_std
        margin = config.target_relative_margin_of_error * mean_loss
        rows.append({
            "architecture_id": key,
            "pilot_num_runs": config.pilot_num_runs,
            "mean_loss_mu": mean_loss,
            "sample_std_loss_mu": sample_std,
            "coefficient_of_variation": sample_std / mean_loss,
            "margin_of_error_mu": margin,
            "min_required_runs": int(np.ceil((config.z_value * sample_std
                                              / margin) ** 2)),
        })
    return pd.DataFrame(rows)


def _indicators_at(loss: np.ndarray, availability: np.ndarray, num_runs: int,
                   z_value: float) -> dict:
    loss, availability = loss[:num_runs], availability[:num_runs]
    mean_loss = float(loss.mean())
    margin = float(z_value * loss.std(ddof=1) / np.sqrt(num_runs))
    return {
        "num_runs": num_runs,
        "expected_annual_loss_mu": mean_loss,
        "generalized_availability_pct": float(availability.mean() * 100),
        "loss_quantile_095_mu": float(np.percentile(loss, 95)),
        "loss_quantile_099_mu": float(np.percentile(loss, 99)),
        "margin_of_error_mu": margin,
        "relative_margin_of_error_pct": margin / mean_loss * 100,
    }


def build_convergence_table(losses: dict[str, np.ndarray],
                            availability: dict[str, np.ndarray],
                            config: ExperimentConfig) -> pd.DataFrame:
    """Indicators on the first R runs of the saved samples, for every checked R."""
    return pd.DataFrame([
        {"architecture_id": key,
         **_indicators_at(losses[key], availability[key], num_runs,
                          config.z_value)}
        for key in losses
        for num_runs in config.checked_run_counts
    ])


def build_stabilization_table(convergence: pd.DataFrame,
                              config: ExperimentConfig) -> pd.DataFrame:
    """Successive relative change of each indicator, maximized over architectures.

    The criterion is met when Ê(L), L95 and L99 each change by no more
    than the stabilization threshold. Availability is reported as an absolute
    change in percentage points.
    """
    monetary_columns = ("expected_annual_loss_mu", "loss_quantile_095_mu",
                        "loss_quantile_099_mu")
    threshold_pct = config.stabilization_threshold * 100

    rows = []
    counts = config.checked_run_counts
    for lower, upper in zip(counts[:-1], counts[1:]):
        relative_changes = {column: [] for column in monetary_columns}
        availability_changes = []

        for _, group in convergence.groupby("architecture_id"):
            group = group.set_index("num_runs")
            for column in monetary_columns:
                before, after = group.loc[lower, column], group.loc[upper, column]
                # Relative to the estimate at the larger number of runs.
                relative_changes[column].append(
                    abs(after - before) / abs(after) * 100)
            availability_changes.append(
                abs(group.loc[upper, "generalized_availability_pct"]
                    - group.loc[lower, "generalized_availability_pct"]))

        worst = {column: max(values) for column, values in relative_changes.items()}
        rows.append({
            "transition": f"{lower}->{upper}",
            "num_runs_from": lower,
            "num_runs_to": upper,
            "max_change_expected_annual_loss_pct": worst["expected_annual_loss_mu"],
            "max_change_availability_pp": max(availability_changes),
            "max_change_loss_quantile_095_pct": worst["loss_quantile_095_mu"],
            "max_change_loss_quantile_099_pct": worst["loss_quantile_099_mu"],
            "criterion_met": bool(max(worst.values()) <= threshold_pct),
        })
    return pd.DataFrame(rows)


def build_run_length_summary(pilot: pd.DataFrame, convergence: pd.DataFrame,
                             stabilization: pd.DataFrame,
                             config: ExperimentConfig) -> dict:
    """Justification of the adopted number of runs."""
    min_required = int(pilot["min_required_runs"].max())
    meeting_criterion = stabilization.loc[stabilization["criterion_met"],
                                          "num_runs_to"]
    at_adopted = convergence.loc[convergence.num_runs == config.num_runs,
                                 "relative_margin_of_error_pct"]

    return {
        "min_required_runs_from_formula": min_required,
        "first_num_runs_meeting_stabilization_criterion":
            int(meeting_criterion.iloc[0]) if len(meeting_criterion) else None,
        "adopted_num_runs": config.num_runs,
        "max_relative_margin_of_error_at_adopted_runs_pct": float(at_adopted.max()),
        "acceptable_annual_loss_threshold_mu":
            config.acceptable_annual_loss_threshold_mu,
        "units": config.monetary_unit,
    }


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_run_metadata(config: ExperimentConfig, num_runs: int,
                       random_seed: int) -> dict:
    """Software versions, seed, run count and input-file digests of a result set."""
    from . import __version__

    inputs = sorted(INPUT_DIR.glob("*.csv")) + [PARAMETERS_FILE]
    return {
        "package_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "horizon_hours": config.horizon_hours,
        "num_runs": num_runs,
        "random_seed": random_seed,
        "checked_run_counts": list(config.checked_run_counts),
        "input_file_sha256_prefixes": {path.name: _file_digest(path)
                                       for path in inputs},
    }


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("wrote %s", path)


def write_table(table: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=index, encoding="utf-8")
    logger.info("wrote %s", path)


def loss_samples_path(config: ExperimentConfig) -> Path:
    return loss_sample_file(config.max_checked_run_count)
