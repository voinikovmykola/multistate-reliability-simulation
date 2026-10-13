"""Repository paths and the experiment configuration in config/parameters.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
INPUT_DIR = REPO_ROOT / "data" / "input"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_TABLES_DIR = OUTPUT_DIR / "tables"

PARAMETERS_FILE = CONFIG_DIR / "parameters.json"


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment settings and the accepted decision criteria of the study.

    Monetary values are in MU; one simulation time step is one hour.
    """

    horizon_hours: int
    time_step_hours: int
    num_runs: int
    pilot_num_runs: int
    checked_run_counts: tuple[int, ...]
    main_random_seed: int
    pilot_random_seed: int
    initial_state: str
    functionally_available_states: tuple[str, ...]
    confidence_level: float
    z_value: float
    target_relative_margin_of_error: float
    stabilization_threshold: float
    acceptable_annual_loss_threshold_mu: float
    monetary_unit: str

    @property
    def max_checked_run_count(self) -> int:
        return max(self.checked_run_counts)


def load_experiment_config(parameters_file: Path | str = PARAMETERS_FILE) -> ExperimentConfig:
    with open(parameters_file, encoding="utf-8") as handle:
        raw = json.load(handle)

    experiment, criteria = raw["experiment"], raw["criteria"]
    return ExperimentConfig(
        horizon_hours=experiment["horizon_hours"],
        time_step_hours=experiment["time_step_hours"],
        num_runs=experiment["num_runs"],
        pilot_num_runs=experiment["pilot_num_runs"],
        checked_run_counts=tuple(sorted(experiment["checked_run_counts"])),
        main_random_seed=experiment["main_random_seed"],
        pilot_random_seed=experiment["pilot_random_seed"],
        initial_state=experiment["initial_state"],
        functionally_available_states=tuple(
            experiment["functionally_available_states"]),
        confidence_level=criteria["confidence_level"],
        z_value=criteria["z_value"],
        target_relative_margin_of_error=criteria["target_relative_margin_of_error"],
        stabilization_threshold=criteria["stabilization_threshold"],
        acceptable_annual_loss_threshold_mu=criteria[
            "acceptable_annual_loss_threshold_mu"],
        monetary_unit=raw["units"]["monetary"],
    )


def loss_sample_file(num_runs: int) -> Path:
    """Raw per-run loss and availability samples for a given run count."""
    return OUTPUT_DATA_DIR / f"loss_samples_{num_runs}_runs.npz"
