"""Load the input dataset in data/input/ into model objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import INPUT_DIR
from .simulation import Architecture, STATE_CODES


def load_states(input_dir: Path | str = INPUT_DIR) -> pd.DataFrame:
    """State definitions, availability flags and cost rates C_i, ordered S1..S5."""
    return pd.read_csv(Path(input_dir) / "states.csv").sort_values("state_id")


def load_state_names(input_dir: Path | str = INPUT_DIR) -> list[str]:
    return load_states(input_dir)["state_name"].tolist()


def load_admissible_transitions(input_dir: Path | str = INPUT_DIR) -> np.ndarray:
    """Admissible transition matrix A (Eq. 4)."""
    table = pd.read_csv(Path(input_dir) / "admissible_transitions.csv",
                        index_col=0)
    return table.to_numpy(dtype=float)


def load_transition_probability_matrix(
    architecture_id: str,
    input_dir: Path | str = INPUT_DIR,
) -> np.ndarray:
    """Transition probability matrix P of one architecture (Eq. 2, Eq. 3)."""
    path = (Path(input_dir)
            / f"transition_probability_matrix_{architecture_id.lower()}.csv")
    return pd.read_csv(path, index_col=0).to_numpy(dtype=float)


def load_business_criticality_profile(input_dir: Path | str = INPUT_DIR) -> np.ndarray:
    """Business criticality coefficient q(t) over one week, 168 hourly values."""
    table = pd.read_csv(Path(input_dir) / "business_criticality_profile.csv")
    q = table.sort_values("hour_of_week")[
        "business_criticality_coefficient"].to_numpy(dtype=float)
    if q.size != 168:
        raise ValueError(f"profile must have 168 rows, got {q.size}")
    return q


def make_business_criticality_function(input_dir: Path | str = INPUT_DIR):
    """q(t) as a callable; the weekly profile repeats, so q(t) = q[t mod 168]."""
    q = load_business_criticality_profile(input_dir)

    def business_criticality(t: int) -> float:
        return float(q[t % 168])

    return business_criticality


def load_architectures(input_dir: Path | str = INPUT_DIR) -> list[Architecture]:
    """Architectures in dataset order; the first one (A0) is the NEB reference."""
    input_dir = Path(input_dir)

    cost_rates = load_states(input_dir)[
        "state_cost_rate_mu_per_hour"].to_numpy(dtype=float)
    admissible = load_admissible_transitions(input_dir)
    meta = pd.read_csv(input_dir / "architectures.csv")

    architectures = []
    for _, row in meta.iterrows():
        architecture_id = row["architecture_id"]
        P = load_transition_probability_matrix(architecture_id, input_dir)
        if np.any(P[admissible == 0] > 0):
            raise ValueError(f"{architecture_id}: P has non-zero probability on "
                             "a forbidden transition (violates Eq. 4)")
        architectures.append(Architecture(
            name=f"{architecture_id} - {row['architecture_name']}",
            transition_probability_matrix=P,
            state_cost_rates=cost_rates,
            annualized_cost=float(row["annualized_cost_mu_per_year"]),
            description=str(row["architecture_description"]),
        ))
    return architectures


def functionally_available_state_indices(
    input_dir: Path | str = INPUT_DIR,
) -> tuple[int, ...]:
    """Indices of the functionally available states W (Eq. 8)."""
    flags = load_states(input_dir)["is_functionally_available"]
    return tuple(int(i) for i, flag in enumerate(flags) if flag)


def state_code_to_index(state_code: str) -> int:
    return STATE_CODES.index(state_code)
