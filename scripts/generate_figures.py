"""Regenerate Figs. 2-4 from the outputs of run_simulation.py.

Usage:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import logging

import matplotlib

from multistate_reliability.config import load_experiment_config
from multistate_reliability.experiments import loss_samples_path
from multistate_reliability.plot_style import apply_style
from multistate_reliability.plotting import generate_all_figures

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    matplotlib.use("Agg")

    config = load_experiment_config()
    samples_file = loss_samples_path(config)
    if not samples_file.exists():
        raise SystemExit(f"{samples_file} not found: run "
                         "scripts/run_simulation.py first")

    apply_style()
    paths = generate_all_figures(config, samples_file)
    logger.info("wrote %d figures", len(paths))


if __name__ == "__main__":
    main()
