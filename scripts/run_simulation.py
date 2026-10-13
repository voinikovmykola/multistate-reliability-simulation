"""Main experiment: Table 4 and the data behind Figs. 3 and 4.

Simulates every architecture over the largest checked run count (5000); the
reported tables use the first 3000 runs.

Usage:
    python scripts/run_simulation.py
"""

from __future__ import annotations

import logging

from multistate_reliability.config import (OUTPUT_DATA_DIR, OUTPUT_TABLES_DIR,
                                           load_experiment_config)
from multistate_reliability.data_loading import (
    load_architectures, make_business_criticality_function)
from multistate_reliability.experiments import (build_run_metadata,
                                                build_state_table,
                                                build_summary_table,
                                                loss_samples_path,
                                                save_loss_samples,
                                                simulate_all_architectures,
                                                write_json, write_table)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_experiment_config()
    architectures = load_architectures()
    business_criticality = make_business_criticality_function()

    results = simulate_all_architectures(config, business_criticality,
                                         architectures=architectures)

    save_loss_samples(results, loss_samples_path(config))

    losses = {key: result.loss_per_run for key, result in results.items()}
    availability = {key: result.generalized_availability
                    for key, result in results.items()}

    summary = build_summary_table(architectures, losses, availability, config)
    write_table(summary, OUTPUT_TABLES_DIR / "summary_table.csv")

    for filename, attribute in (
            ("state_occupancy_time.csv", "state_occupancy_time"),
            ("state_loss_contribution.csv", "state_loss_contribution")):
        table = build_state_table(results, attribute, config.num_runs)
        write_table(table, OUTPUT_TABLES_DIR / filename, index=True)

    write_json(build_run_metadata(config, config.max_checked_run_count,
                                  config.main_random_seed),
               OUTPUT_DATA_DIR / "run_metadata.json")

    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
