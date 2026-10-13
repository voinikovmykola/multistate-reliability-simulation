"""Pilot simulation and the stabilization check behind the choice R = 3000.

Reads the loss samples written by run_simulation.py, which must run first.

Usage:
    python scripts/run_stability_analysis.py
"""

from __future__ import annotations

import logging

from multistate_reliability.config import (OUTPUT_DATA_DIR, OUTPUT_TABLES_DIR,
                                           load_experiment_config)
from multistate_reliability.data_loading import make_business_criticality_function
from multistate_reliability.experiments import (build_convergence_table,
                                                build_pilot_table,
                                                build_run_length_summary,
                                                build_stabilization_table,
                                                load_loss_samples,
                                                loss_samples_path, write_json,
                                                write_table)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_experiment_config()
    samples_file = loss_samples_path(config)
    if not samples_file.exists():
        raise SystemExit(f"{samples_file} not found: run "
                         "scripts/run_simulation.py first")

    losses, availability = load_loss_samples(samples_file)
    business_criticality = make_business_criticality_function()

    pilot = build_pilot_table(config, business_criticality)
    write_table(pilot, OUTPUT_TABLES_DIR / "pilot_sample_size.csv")

    convergence = build_convergence_table(losses, availability, config)
    write_table(convergence, OUTPUT_TABLES_DIR / "convergence_by_runs.csv")

    stabilization = build_stabilization_table(convergence, config)
    write_table(stabilization, OUTPUT_TABLES_DIR / "stabilization.csv")

    summary = build_run_length_summary(pilot, convergence, stabilization, config)
    write_json(summary, OUTPUT_DATA_DIR / "run_length_summary.json")

    logger.info("minimum required runs from the sample-size formula: %d",
                summary["min_required_runs_from_formula"])
    logger.info("first run count meeting the stabilization criterion: %s",
                summary["first_num_runs_meeting_stabilization_criterion"])
    logger.info("adopted number of runs: %d (attained relative margin of error "
                "%.4f %%)", summary["adopted_num_runs"],
                summary["max_relative_margin_of_error_at_adopted_runs_pct"])


if __name__ == "__main__":
    main()
