# Data Dictionary

Dataset accompanying the article _Simulation of information system states for
reliability assessment in the context of business digital transformation_.

**Units.** All monetary values are expressed in **illustrative monetary units
(MU)**. They are expert-specified parameters rather than measurements obtained
from a real organization. Because every monetary quantity shares the same unit,
all ratios, all comparisons between architectures, and the sign of the net
economic benefit are independent of the chosen scale. **Time unit: hour**; one
simulation time step equals one hour, and the simulation horizon is 8760 hours
(one year).

**Provenance.** All input files were written by the authors for this study.
The transition probabilities, state cost rates, business criticality profile,
and architecture costs are expert-specified illustrative parameters, not
estimates derived from empirical incident data. They were chosen to satisfy a
qualitative reliability ordering of the three architectures (A0 < A1 < A2).
Results derived from them illustrate the method and are not empirical findings
about a real system.

Input files are in `data/input/`, experiment settings in
`config/parameters.json`, and generated files in `outputs/`. No script writes
to the input files.

---

## `states.csv`

One row per system state (n = 5).

| Column                        | Type        | Description                                                                                                                                     |
| ----------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `state_id`                    | integer 1-5 | State number as used in the article ($S_1, \dots, S_5$).                                                                                        |
| `state_code`                  | string      | Short code, `S1`…`S5`.                                                                                                                          |
| `state_name`                  | string      | Normative state name.                                                                                                                           |
| `is_functionally_available`   | 0/1         | 1 if the state belongs to the set of functionally available states $W$ used in the generalized multi-state availability $K_{av}^{(n)}$ (Eq. 8). |
| `state_cost_rate_mu_per_hour` | float ≥ 0   | Cost rate $C_i$ of one hour spent in the state, MU/h. $C_1 = 0$ because losses are counted relative to normal operation.                        |

The five states are the normal operating state, the degraded operating state,
the partial failure state, the critical failure state, and the recovery state.
The functionally available states are $W = \lbrace S_1, S_2 \rbrace$: in the
degraded operating state, all defined functions remain available. The partial
failure state is excluded, although key business processes remain operational,
and the recovery state is excluded because availability has not yet been
operationally confirmed.

## `admissible_transitions.csv`

The admissible transition matrix $A$ (Eq. 4), $5 \times 5$; rows correspond
to current states and columns to next states, and the first column holds the
current state. $a_{ij} = 1$ means that the transition $S_i \to S_j$ is
admissible; $a_{ij} = 0$ is a structural zero and forces $p_{ij} = 0$. The
transition $S_1 \to S_5$ is not admissible because a normally operating system
does not require recovery. $S_3 \to S_1$ and $S_4 \to S_1$ are not admissible
because restoration after a failure requires operational verification through
$S_5$ or an intermediate degraded condition.

## `transition_probability_matrix_a0.csv`, `..._a1.csv`, `..._a2.csv`

The transition probability matrix $P$ (Eq. 2) of each architecture, in the same
layout as $A$. Each row sums to one (Eq. 3), and every structural zero of $A$ is
zero in $P$. The diagonal entry $p_{ii}$ is the probability of remaining in the
current state.

## `architectures.csv`

One row per architecture variant.

| Column                        | Type      | Description                                                                                                                                                          |
| ----------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `architecture_id`             | string    | `A0`, `A1`, `A2`.                                                                                                                                                    |
| `architecture_name`           | string    | Normative architecture name.                                                                                                                                         |
| `annualized_cost_mu_per_year` | float ≥ 0 | Annualized implementation and maintenance cost $I$ of the reliability improvement measure, MU/year. zero for A0, the baseline against which the others are compared. |
| `architecture_description`    | string    | What the variant consists of technically.                                                                                                                            |

Only what the model actually represents is described: A2 differs from A1 by
automatic failover, not by load balancing or backup.

## `business_criticality_profile.csv`

The time-dependent business criticality coefficient $q(t)$ (Eq. 9), tabulated
over one full week. The profile repeats weekly, so the value at model hour $t$
is the row with `hour_of_week` $= t \bmod 168$.

| Column                             | Type          | Description                                                                                                                                                |
| ---------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hour_of_week`                     | integer 0-167 | Hour index within the week; hour 0 = Monday 00:00.                                                                                                         |
| `day_of_week`                      | integer 0-6   | 0 = Monday … 5 = Saturday, 6 = Sunday.                                                                                                                     |
| `hour_of_day`                      | integer 0-23  | Hour of day.                                                                                                                                               |
| `business_criticality_coefficient` | float > 0     | Dimensionless multiplier applied to the state cost rate at that hour. Values above 1 mark peak-activity periods, values below 1 mark low-activity periods. |

## `config/parameters.json`

Experiment configuration and the accepted decision criteria.

- `units` - unit declarations and the scale note.
- `experiment` - `horizon_hours`, `time_step_hours`, the adopted `num_runs`, the
  `pilot_num_runs`, the `checked_run_counts` examined for stabilization, the
  `main_random_seed` and `pilot_random_seed`, the `initial_state`, and the set of
  `functionally_available_states` $W$.
- `criteria` - `confidence_level` (95 %), `z_value` = 1.96, the
  `target_relative_margin_of_error` = 0.05, the `stabilization_threshold` = 0.01,
  and the `acceptable_annual_loss_threshold_mu`, the acceptable loss threshold
  $L_{allow} = 110\,000$ MU/year against which the empirical exceedance
  frequency $k/R$ (`loss_exceedance_probability`) is computed.

The target relative margin of error, the 1 % stabilization threshold, and
$L_{allow}$ are the criteria chosen for this study. $L_{allow}$ is the annual loss
level an organization declares tolerable; it has to be set anew before the
exceedance frequency is interpreted for any other system.

---

## Generated results (`outputs/`)

Reproducible from `data/input/` and `config/parameters.json`; README.md maps
each file to the article and to the script that writes it.

| File                                 | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tables/summary_table.csv`           | Table 4, one row per architecture: `generalized_availability_pct`, `downtime_hours_per_year`, `expected_annual_loss_mu`, `loss_quantile_095_mu`, `loss_quantile_099_mu`, `loss_exceedance_probability`, `annualized_cost_mu_per_year`, `net_economic_benefit_mu_per_year`.                                                                                                                                                                                                                                           |
| `tables/state_occupancy_time.csv`    | Mean state occupancy times $T_i$ in hours per year, at the adopted number of runs. One column per architecture, one row per state.                                                                                                                                                                                                                                                                                                                                                                                   |
| `tables/state_loss_contribution.csv` | Mean loss attributed to each state in MU per year, at the adopted number of runs. Each column sums to that architecture's `expected_annual_loss_mu`.                                                                                                                                                                                                                                                                                                                                                                 |
| `tables/pilot_sample_size.csv`       | Pilot series per architecture: `mean_loss_mu`, `sample_std_loss_mu`, `coefficient_of_variation`, `margin_of_error_mu`, `min_required_runs`.                                                                                                                                                                                                                                                                                                                                                                          |
| `tables/convergence_by_runs.csv`     | Indicators at every checked run count, including `margin_of_error_mu` and `relative_margin_of_error_pct`.                                                                                                                                                                                                                                                                                                                                                                                                            |
| `tables/stabilization.csv`           | Successive relative change of each indicator between adjacent run counts, and `criterion_met`.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `data/loss_samples_5000_runs.npz`    | Raw per-run samples. Array names are `loss_<architecture_id>` (annual loss in MU) and `availability_<architecture_id>` (generalized availability as a fraction), each a 1-D float64 array of length 5000 ordered by run index.                                                                                                                                                                                                                                                                                       |
| `data/run_length_summary.json`       | Run-length justification: `min_required_runs_from_formula`, `first_num_runs_meeting_stabilization_criterion`, `adopted_num_runs`, `max_relative_margin_of_error_at_adopted_runs_pct`, `acceptable_annual_loss_threshold_mu`.                                                                                                                                                                                                                                                                                         |
| `tables/sensitivity_analysis.csv`    | Sensitivity analysis, one row per scenario and architecture: `parameter`, `change`, the factors applied (`transition_factor`, `criticality_exponent`, `mean_business_criticality`, `cost_factor`, `investment_factor`, `acceptable_annual_loss_threshold_mu`), then `expected_annual_loss_mu`, `loss_quantile_095_mu`, `loss_quantile_099_mu`, `exceedance_count_k`, `num_runs_r`, `loss_exceedance_probability`, `annualized_cost_mu_per_year`, `net_economic_benefit_mu_per_year`, `rank_by_net_economic_benefit`. |
| `tables/sensitivity_ranges.csv`      | Table 5 of the article: per architecture, `num_configurations` (14), the minimum and maximum of each indicator over all configurations (`<indicator>_min`, `<indicator>_max`), and how many configurations rank it first or second by net economic benefit.                                                                                                                                                                                                                                                          |
| `tables/sensitivity_summary.csv`     | The same results as the compact article table: one row per scenario, the four headline indicators spread across the architectures, and `leading_architecture`.                                                                                                                                                                                                                                                                                                                                                       |
| `data/sensitivity_summary.json`      | Number of scenarios, how many keep A2 first, the scenarios where the ranking changes, and the worst-case net economic benefit and exceedance probability of A2.                                                                                                                                                                                                                                                                                                                                                      |
| `data/run_metadata.json`             | Provenance of the current result set: package and dependency versions, seed, run count, and a digest of every input file.                                                                                                                                                                                                                                                                                                                                                                                            |
| `figures/*.png`                      | Figs. 2-4 of the article at 300 dpi.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

The net economic benefit is defined relative to the baseline and is therefore
undefined for the baseline itself, where it is reported as an empty field (NaN)
rather than zero.
