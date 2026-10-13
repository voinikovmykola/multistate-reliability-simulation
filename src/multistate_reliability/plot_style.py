"""Shared figure styling, palettes and labels."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .simulation import STATE_CODES

ARCHITECTURE_IDS: tuple[str, ...] = ("A0", "A1", "A2")

STATE_COLORS: tuple[str, ...] = ("#2e7d32", "#f9a825", "#ef6c00", "#c62828",
                                 "#1565c0")
ARCHITECTURE_COLORS = {"A0": "#c62828", "A1": "#1565c0", "A2": "#2e7d32"}

# Short legend labels; the full A2 name in the dataset is too long for a legend.
ARCHITECTURE_LABELS = {"A0": "A0 - Baseline architecture",
                       "A1": "A1 - Hot-standby architecture",
                       "A2": "A2 - Fault-tolerant architecture"}
ARCHITECTURE_TICK_LABELS = {key: label.replace(" - ", "\n")
                            for key, label in ARCHITECTURE_LABELS.items()}

LEGEND_KWARGS = {
    "loc": "upper right",
    "frameon": True,
    "framealpha": 1.0,
    "fancybox": False,
    "facecolor": "white",
    "edgecolor": "0.6",
    "borderpad": 0.4,
    "borderaxespad": 0.15,
}

SINGLE_PANEL_SIZE = (7.2, 4.4)
STACKED_PANEL_SIZE = (9.5, 5.0)

EXPORT_DPI = 300

_RC_PARAMS = {
    "figure.dpi": 110,
    "savefig.dpi": EXPORT_DPI,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
}


def apply_style() -> None:
    plt.rcParams.update(_RC_PARAMS)


def format_thousands(value: float, decimals: int = 0) -> str:
    """Format a number with a space as the thousands separator."""
    return f"{value:,.{decimals}f}".replace(",", " ")


def state_legend_labels(state_names: list[str]) -> list[str]:
    return [f"{code} - {name}" for code, name in zip(STATE_CODES, state_names)]


def present_architecture_ids(available: list[str]) -> list[str]:
    return [key for key in ARCHITECTURE_IDS if key in available]
