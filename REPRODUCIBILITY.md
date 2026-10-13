# Reproducibility record

Validation of the released code and outputs against the article _Simulation of
information system states for reliability assessment in the context of
business digital transformation_ (manuscript version of 2026-10-12).

## Environment

|                         |                                                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| OS                      | macOS 26.6.2, arm64 (Apple silicon)                                                                                     |
| Python                  | CPython 3.13.9                                                                                                          |
| Packages                | as pinned in `requirements.txt` (NumPy 2.5.2, pandas 3.0.5, Matplotlib 3.11.1, FreeType 2.14.3 bundled with Matplotlib) |
| Random number generator | `numpy.random.default_rng` (PCG64); seeds 42 (main, sensitivity) and 2025 (pilot)                                       |

## Commands

```bash
python scripts/run_simulation.py
python scripts/run_stability_analysis.py
python scripts/generate_figures.py
python scripts/run_sensitivity_analysis.py
python scripts/check_article_values.py
pytest
```

Run time on the machine above: about 3.5 min for the main simulation, 40 s for
the stability analysis, 10 s for the figures, and about 12 min for the sensitivity
analysis.

## Procedure

1. **Baseline.** The pre-release code (commit `416b7da`) was run in a clean
   checkout. Its outputs were byte-identical to the result files from which the
   article was prepared (all tables, JSON files, the loss samples, and Figs. 2-4),
   so they served as the baseline.
2. **Refactored code, same stabilization formula.** All numeric columns of all
   tables, the loss samples, and Figs. 2-4 were byte-identical to the baseline.
   The only differences were two removed metadata fields
   (`min_required_runs_rounded_up`, `range_is_a_modelling_choice_not_an_estimated_error`)
   and the relabeling of the $q(t)$ scenarios to the article's terms
   (`q(t) = 1 (flat)` to `uniform profile`, `weakened profile` to
   `attenuated profile`).
3. **Article check.** `scripts/check_article_values.py` compares 98 numbers
   quoted in the article (Tables 4 and 5, pilot and stabilization results, Fig. 4
   values, percentages, and break-even values in the text) at their printed
   precision, plus the SHA-256 of Figs. 2-4 against the images embedded in the
   manuscript. At step 2, 93 of 98 values and all three figures matched. The
   five mismatches were all in the stabilization check (see below).
4. **Stabilization fix.** After the correction described below, all 98 values
   and all three figures match.
5. **Repeat run.** `outputs/` was deleted and regenerated from scratch. The
   result was byte-identical to the previous run, figures included.
6. **After the repository cleanup** (licensing and metadata files,
   documentation, removal of duplicated input validation, and the threshold
   symbol renamed from `L_acc` to the article's `L_allow`), `outputs/` was
   deleted and regenerated again (16 min). Every number and all figures were
   byte-identical to step 5. The only differences were the scenario label
   `acceptable loss threshold L_allow` in the sensitivity tables and the
   `parameters.json` digest in `run_metadata.json`, whose note text changed.
   The article check and the test suite passed.

## Correction to the stabilization check

The article defines the relative change between consecutive simulation sizes as
"the absolute difference between the two estimates divided by the absolute
value of the estimate obtained with the larger number of runs". The pre-release
code divided by the estimate at the _smaller_ number of runs. The numbers
printed in the article follow the article's definition:

| Comparison                                 | Article         | Pre-release code | Released code   |
| ------------------------------------------ | --------------- | ---------------- | --------------- |
| 500 vs 1 000, max quantile change          | 2.53 %          | 2.47 %           | 2.53 %          |
| 1 000 vs 2 000, max quantile change        | 2.02 %          | 2.06 %           | 2.02 %          |
| 2 000 vs 3 000, max quantile / mean change | 0.90 % / 0.18 % | 0.91 % / 0.18 %  | 0.90 % / 0.18 % |
| 3 000 vs 5 000, max quantile / mean change | 0.99 % / 0.28 % | 1.00 % / 0.28 %  | 0.99 % / 0.28 % |
| 3 000 vs 5 000 meets the 1 % criterion     | yes             | no               | yes             |

The released code implements the article's definition
(`build_stabilization_table` in `experiments.py`). This affects only
`outputs/tables/stabilization.csv`. The first simulation size that meets the
criterion is $R = 3000$ under both formulas, and no other output changes.

## Reproduced results

All of the following match the article at the printed precision
(`python scripts/check_article_values.py`).

**Table 4** ($R = 3000$; `outputs/tables/summary_table.csv`)

|     | $K_{av}^{(n)}$, % | $S_3$-$S_5$, h/year | $\hat{E}(L)$ | $L_{95}$   | $L_{99}$   | $k/R$       | $I$    | $NEB$      |
| --- | ----------------- | ------------------- | ------------ | ---------- | ---------- | ----------- | ------ | ---------- |
| A0  | 92.71             | 638.53              | 146 468.76   | 175 678.33 | 187 196.11 | 2 977/3 000 | 0      | -          |
| A1  | 97.80             | 193.01              | 44 825.17    | 58 083.58  | 64 185.11  | 0/3 000     | 4 000  | 97 643.59  |
| A2  | 99.40             | 52.95               | 12 335.54    | 18 035.60  | 20 952.53  | 0/3 000     | 11 500 | 122 633.22 |

**Choice of $R$.** Pilot minimum runs 20 / 50 / 103 (A0 / A1 / A2), $R_{min} = 103$;
stabilization criterion first met at $R = 3000$; maximum change in generalized multi-state
availability between consecutive sizes 0.0086 percentage points.

**Fig. 4.** Contribution of S4: 74 950 / 20 743 / 4 030 MU/year.

**Table 5** (14 configurations; `outputs/tables/sensitivity_ranges.csv`). A0
$\hat{E}(L)$ 117 175-175 763, $k$ 2 051-3 000; A1 $NEB$ 77 315-117 972, second in 14/14;
A2 $NEB$ 95 807-149 460, first in 14/14; no exceedances for A1 or A2. Break-even
values of A2: $I = 36\,490$ MU/year, $\hat{E}(L) = 37\,325$ MU/year.

**Figures 2-4.** `outputs/figures/*.png` are byte-identical to the images in the
manuscript.

## Remaining limitations

- Byte identity of the PNG files depends on the Matplotlib and FreeType
  versions. With other versions, the figures show the same data but may differ
  in bytes, and `check_article_values.py` reports this as a figure mismatch.
- The numerical results were verified only on the environment above. The
  simulation uses only integer indexing, `searchsorted` and sums over the
  generated trajectories. Other platforms are expected to give identical
  numbers, but this was not tested.
- Fig. 1 is a drawn diagram (`docs/state_transition_graph.png`), not generated
  by the code. It was checked visually against `data/input/admissible_transitions.csv`.
