"""Discrete-time multi-state model of information system states.

The next state depends only on the current state (transition probability
matrix P). Losses accumulate hour by hour along a system state trajectory, with each
state-specific cost rate scaled by the time-dependent business criticality
coefficient q(t).

Equation numbers refer to the accompanying article:

    (2)  P = [p_ij], 0 <= p_ij <= 1                 transition probability matrix
    (3)  sum_j p_ij = 1                             row normalization
    (4)  A = [a_ij], a_ij in {0, 1}                 admissible transition matrix
    (5)  L = sum_i T_i * C_i                        total economic loss, q(t) = 1
    (6)  Ê(L) = (1/R) * sum_r L_r                   estimated expected economic loss
    (8)  K_av^(n) = sum_{j in W} T_j / sum_i T_i    generalized multi-state availability
    (9)  C_i(t) = C_i * q(t)                        time-dependent state-specific cost rate
    (10) L = sum_t C_{s(t)} * q(t)                  total economic loss, time-dependent

One simulation time step is one hour. Monetary values are in illustrative
monetary units (MU).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Sequence

import numpy as np


class SystemState(IntEnum):
    NORMAL_OPERATION = 0
    DEGRADED_OPERATION = 1
    PARTIAL_FAILURE = 2
    CRITICAL_FAILURE = 3
    RECOVERY = 4


N_STATES = len(SystemState)
STATE_CODES: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")

# W = {S1, S2} (Eq. 8): all defined functions remain available in the degraded
# operating state.
DEFAULT_FUNCTIONALLY_AVAILABLE_STATES = (SystemState.NORMAL_OPERATION,
                                         SystemState.DEGRADED_OPERATION)

# Table 2 of the article; rows are current states, columns next states.
# S1 -> S5 is not admissible because a normally operating system does not
# require recovery, and S3 -> S1, S4 -> S1 are not admissible because
# restoration after a failure requires operational verification.
ADMISSIBLE_TRANSITION_MATRIX = np.array([
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=int)


def validate_transition_probability_matrix(
    transition_probability_matrix: np.ndarray,
    admissible_transition_matrix: np.ndarray = ADMISSIBLE_TRANSITION_MATRIX,
    tol: float = 1e-9,
) -> None:
    """Check P against Eq. 2, Eq. 3 and the admissibility structure A (Eq. 4)."""
    P = np.asarray(transition_probability_matrix, dtype=float)
    A = np.asarray(admissible_transition_matrix, dtype=int)

    if P.shape != A.shape:
        raise ValueError(f"P has shape {P.shape}, expected {A.shape}")
    if np.any(P < -tol) or np.any(P > 1 + tol):
        raise ValueError("violates 0 <= p_ij <= 1 (Eq. 2)")

    forbidden = A == 0
    if np.any(np.abs(P[forbidden]) > tol):
        cells = np.argwhere(forbidden & (np.abs(P) > tol)).tolist()
        raise ValueError("non-zero probability on a forbidden transition, "
                         f"cells (i, j): {cells}")

    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"rows of P must sum to 1 (Eq. 3), got {row_sums}")


def constant_business_criticality(t: int) -> float:
    """q(t) = 1, under which Eq. 10 reduces to Eq. 5."""
    return 1.0


@dataclass
class Architecture:
    """One architecture variant.

    ``state_cost_rates`` are C_i in MU/h (C_1 = 0, losses are counted relative
    to normal operation); ``annualized_cost`` is I in MU/year (Eq. 11).
    """

    name: str
    transition_probability_matrix: np.ndarray
    state_cost_rates: np.ndarray
    annualized_cost: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        self.transition_probability_matrix = np.asarray(
            self.transition_probability_matrix, dtype=float)
        self.state_cost_rates = np.asarray(self.state_cost_rates, dtype=float)
        validate_transition_probability_matrix(
            self.transition_probability_matrix)
        if self.state_cost_rates.shape != (N_STATES,):
            raise ValueError(f"state_cost_rates must hold {N_STATES} values")

    @property
    def architecture_id(self) -> str:
        """Short identifier such as 'A0', taken from the architecture name."""
        return self.name.split(" ")[0]


@dataclass
class SimulationResult:
    """Per-run samples of a series of independent simulation runs."""

    architecture_name: str
    horizon_hours: int
    num_runs: int
    loss_per_run: np.ndarray
    state_occupancy_time: np.ndarray
    # sum_t C_i * q(t) over the hours spent in S_i, shape (num_runs, N_STATES).
    # Accumulated during simulation rather than reconstructed as T_i * C_i,
    # which would ignore q(t); rows sum to loss_per_run.
    state_loss_contribution: np.ndarray
    generalized_availability: np.ndarray
    example_trajectory: np.ndarray

    @property
    def expected_annual_loss(self) -> float:
        """Estimated expected economic loss Ê(L) (Eq. 6)."""
        return float(self.loss_per_run.mean())

    @property
    def loss_sample_std(self) -> float:
        return float(self.loss_per_run.std(ddof=1))

    @property
    def mean_state_loss_contribution(self) -> np.ndarray:
        return self.state_loss_contribution.mean(axis=0)

    @property
    def mean_generalized_availability(self) -> float:
        return float(self.generalized_availability.mean())


def _simulate_state_trajectory(
    transition_probability_matrix: np.ndarray,
    horizon_hours: int,
    initial_state: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one system state trajectory s(t), t = 1..H, using P.

    Each run draws exactly H uniforms, so runs with the same seed consume the
    same random stream regardless of P (common random numbers across
    architectures and sensitivity scenarios).
    """
    trajectory = np.empty(horizon_hours, dtype=np.int8)
    state = initial_state
    row_cdf = np.cumsum(transition_probability_matrix, axis=1)
    uniforms = rng.random(horizon_hours)

    for t in range(horizon_hours):
        state = int(np.searchsorted(row_cdf[state], uniforms[t], side="right"))
        state = min(state, N_STATES - 1)      # guard against float round-off
        trajectory[t] = state

    return trajectory


def simulate_architecture(
    architecture: Architecture,
    horizon_hours: int = 8760,
    num_runs: int = 2000,
    initial_state: int = SystemState.NORMAL_OPERATION,
    business_criticality: Callable[[int], float] = constant_business_criticality,
    functionally_available_states: Sequence[int] = DEFAULT_FUNCTIONALLY_AVAILABLE_STATES,
    random_seed: int | None = 42,
) -> SimulationResult:
    """Monte Carlo simulation of one architecture over ``num_runs`` runs."""
    rng = np.random.default_rng(random_seed)

    loss_per_run = np.empty(num_runs)
    availability = np.empty(num_runs)
    occupancy = np.empty((num_runs, N_STATES))
    loss_contribution = np.empty((num_runs, N_STATES))
    example_trajectory = None

    available_mask = np.zeros(N_STATES, dtype=bool)
    available_mask[list(functionally_available_states)] = True

    q = np.fromiter((business_criticality(t) for t in range(horizon_hours)),
                    dtype=float, count=horizon_hours)
    cost_rates = architecture.state_cost_rates

    for run in range(num_runs):
        trajectory = _simulate_state_trajectory(
            architecture.transition_probability_matrix, horizon_hours,
            initial_state, rng)

        hours_in_state = np.bincount(trajectory,
                                     minlength=N_STATES).astype(float)
        criticality_weighted_hours = np.bincount(trajectory, weights=q,
                                                 minlength=N_STATES)

        occupancy[run] = hours_in_state
        loss_contribution[run] = cost_rates * criticality_weighted_hours
        loss_per_run[run] = float(loss_contribution[run].sum())
        availability[run] = (hours_in_state[available_mask].sum()
                             / hours_in_state.sum())

        if run == 0:
            example_trajectory = trajectory.copy()

    return SimulationResult(
        architecture_name=architecture.name,
        horizon_hours=horizon_hours,
        num_runs=num_runs,
        loss_per_run=loss_per_run,
        state_occupancy_time=occupancy,
        state_loss_contribution=loss_contribution,
        generalized_availability=availability,
        example_trajectory=example_trajectory,
    )
