"""Figures 2-4 of the article, written to outputs/figures/ at 300 dpi.

    example_state_trajectory.png    Fig. 2, one A0 trajectory with q(t)
    loss_distributions.png          Fig. 3, loss distributions with L95
    loss_breakdown_by_state.png     Fig. 4, expected annual economic loss by state

Fig. 1 (state-transition graph) is a drawn diagram, see docs/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (ExperimentConfig, OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR)
from .data_loading import (functionally_available_state_indices,
                           load_architectures, load_state_names,
                           make_business_criticality_function,
                           state_code_to_index)
from .experiments import load_loss_samples
from .plot_style import (ARCHITECTURE_COLORS, ARCHITECTURE_LABELS,
                         ARCHITECTURE_TICK_LABELS, LEGEND_KWARGS,
                         SINGLE_PANEL_SIZE, STACKED_PANEL_SIZE, STATE_COLORS,
                         format_thousands, present_architecture_ids,
                         state_legend_labels)
from .simulation import STATE_CODES, simulate_architecture

logger = logging.getLogger(__name__)

THOUSAND = 1000.0


def _save(figure: plt.Figure, name: str, figures_dir: Path) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    figure.savefig(path)
    plt.close(figure)
    logger.info("wrote %s", path)
    return path


def plot_loss_distributions(losses: dict[str, np.ndarray],
                            config: ExperimentConfig,
                            figures_dir: Path = OUTPUT_FIGURES_DIR) -> Path:
    """Empirical distributions of annual economic loss, with the 0.95 quantile."""
    figure, axes = plt.subplots(figsize=SINGLE_PANEL_SIZE)

    for key in present_architecture_ids(list(losses)):
        loss = losses[key][:config.num_runs] / THOUSAND
        axes.hist(loss, bins=60, density=True, alpha=0.45,
                  color=ARCHITECTURE_COLORS[key], label=ARCHITECTURE_LABELS[key])

        quantile = np.percentile(loss, 95)
        axes.axvline(quantile, color=ARCHITECTURE_COLORS[key], ls="--", lw=1.4)
        axes.annotate(f"$L_{{0.95}}$ = {format_thousands(quantile)}",
                      xy=(quantile, max(axes.get_ylim()[1], 0.05) * 0.78),
                      xytext=(4, 0), textcoords="offset points",
                      color=ARCHITECTURE_COLORS[key], fontsize=8, rotation=90,
                      va="top")

    axes.set_xlabel("Annual economic loss, thousand MU")
    axes.set_ylabel("Probability density")

    axes.legend(**LEGEND_KWARGS)

    return _save(figure, "loss_distributions.png", figures_dir)


def plot_loss_breakdown_by_state(contribution: pd.DataFrame,
                                 state_names: list[str],
                                 figures_dir: Path = OUTPUT_FIGURES_DIR) -> Path:
    """Expected annual economic loss by state; bar totals equal Ê(L)."""
    labels = state_legend_labels(state_names)
    keys = present_architecture_ids(list(contribution.columns))
    values = np.array([[contribution.loc[code, key] for code in STATE_CODES]
                       for key in keys]) / THOUSAND

    figure, axes = plt.subplots(figsize=SINGLE_PANEL_SIZE)
    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))

    for index in range(len(STATE_CODES)):
        if values[:, index].max() <= 0:
            continue          # S1 has C1 = 0
        axes.bar(x, values[:, index], 0.55, bottom=bottom,
                 color=STATE_COLORS[index], label=labels[index])
        bottom += values[:, index]

    for position, total in zip(x, bottom):
        axes.text(position, total * 1.02, format_thousands(total, 1),
                  ha="center", va="bottom", fontsize=9)

    axes.set_xticks(x)
    axes.set_xticklabels([ARCHITECTURE_TICK_LABELS[key] for key in keys])
    axes.set_ylabel("Contribution to expected annual loss, thousand MU")
    axes.legend(fontsize=8, **LEGEND_KWARGS)
    axes.margins(y=0.12)

    return _save(figure, "loss_breakdown_by_state.png", figures_dir)


def plot_example_state_trajectory(config: ExperimentConfig, weeks: int = 3,
                                  figures_dir: Path = OUTPUT_FIGURES_DIR) -> Path:
    """First run of the main A0 experiment, re-simulated with the main seed."""
    business_criticality = make_business_criticality_function()
    result = simulate_architecture(
        load_architectures()[0],
        horizon_hours=config.horizon_hours,
        num_runs=1,
        initial_state=state_code_to_index(config.initial_state),
        business_criticality=business_criticality,
        functionally_available_states=functionally_available_state_indices(),
        random_seed=config.main_random_seed)

    hours = weeks * 168
    trajectory = result.example_trajectory[:hours]
    q = np.array([business_criticality(t) for t in range(hours)])
    time = np.arange(hours)

    figure, axes = plt.subplots(2, 1, figsize=STACKED_PANEL_SIZE, sharex=True,
                                gridspec_kw={"height_ratios": [2.1, 1]})

    axes[0].step(time, trajectory + 1, where="post", color="#37474f", lw=0.9)
    for index in range(len(STATE_CODES)):
        selected = trajectory == index
        axes[0].scatter(time[selected], np.full(selected.sum(), index + 1), s=7,
                        color=STATE_COLORS[index], zorder=3)
    axes[0].set_yticks(range(1, len(STATE_CODES) + 1))
    axes[0].set_yticklabels(STATE_CODES)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("System state")
    axes[0].set_title("Example state trajectory for the baseline architecture "
                      f"(first {weeks} weeks)")

    axes[1].fill_between(time, q, color="#1565c0", alpha=0.30, step="post")
    axes[1].step(time, q, where="post", color="#1565c0", lw=0.9)
    axes[1].set_ylabel("$q(t)$")
    axes[1].set_xlabel("Simulation time, h")
    axes[1].set_title("Business criticality profile $q(t)$")

    for day_start in range(0, hours, 24):
        for panel in axes:
            panel.axvline(day_start, color="0.85", lw=0.4, zorder=0)

    figure.tight_layout()
    return _save(figure, "example_state_trajectory.png", figures_dir)


def generate_all_figures(config: ExperimentConfig,
                         loss_samples_file: Path,
                         tables_dir: Path = OUTPUT_TABLES_DIR,
                         figures_dir: Path = OUTPUT_FIGURES_DIR) -> list[Path]:
    losses, _ = load_loss_samples(loss_samples_file)
    contribution = pd.read_csv(tables_dir / "state_loss_contribution.csv",
                               index_col=0)

    return [
        plot_loss_distributions(losses, config, figures_dir),
        plot_loss_breakdown_by_state(contribution, load_state_names(),
                                     figures_dir),
        plot_example_state_trajectory(config, figures_dir=figures_dir),
    ]
