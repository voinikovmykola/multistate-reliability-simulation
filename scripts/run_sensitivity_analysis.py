"""Sensitivity analysis (Table 5). Independent of the other scripts.

Usage:
    python scripts/run_sensitivity_analysis.py
"""

from __future__ import annotations

import logging

from multistate_reliability.config import (OUTPUT_DATA_DIR, OUTPUT_TABLES_DIR,
                                           load_experiment_config)
from multistate_reliability.experiments import write_json, write_table
from multistate_reliability.sensitivity import (ranges_table,
                                              run_sensitivity_analysis)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_experiment_config()
    detailed, compact, summary = run_sensitivity_analysis(config)

    write_table(detailed, OUTPUT_TABLES_DIR / "sensitivity_analysis.csv")
    write_table(compact, OUTPUT_TABLES_DIR / "sensitivity_summary.csv")
    write_table(ranges_table(detailed),
                OUTPUT_TABLES_DIR / "sensitivity_ranges.csv")
    write_json(summary, OUTPUT_DATA_DIR / "sensitivity_summary.json")

    logger.info("\n%d scenarios; A2 leads in %d of them",
                summary["num_scenarios"], summary["num_scenarios_with_a2_first"])
    for entry in summary["scenarios_where_ranking_changes"]:
        logger.info("  ranking changes under %s (%s): %s leads",
                    entry["parameter"], entry["change"],
                    entry["leading_architecture"])
    logger.info("A2 net economic benefit stays at or above %.0f MU/year",
                summary["min_net_economic_benefit_of_a2_mu_per_year"])


if __name__ == "__main__":
    main()
