"""Compare the generated outputs with the values printed in the article.

Every number is compared at the precision it is printed with in the article
(e.g. 92.7 is matched by any value in [92.65, 92.75]). Figures are compared by
SHA-256 against the images embedded in the submitted manuscript; byte identity
requires the pinned versions in requirements.txt (Matplotlib and its bundled
FreeType determine the rendered pixels).

Run after the four experiment scripts. Exits with status 1 on any mismatch.

Usage:
    python scripts/check_article_values.py [--outputs DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from multistate_reliability.config import OUTPUT_DIR

# SHA-256 of Figs. 2-4 as embedded in the manuscript.
ARTICLE_FIGURES = {
    "example_state_trajectory.png":
        "3d9af17301b27cfab708b1b66ab654bba9a5aba401b4fc1ca3341b63cde0cfac",
    "loss_distributions.png":
        "8d488cb80ffe6ec49b06eb77f9dea5856d6bec1c12cace679ada1677a2969aad",
    "loss_breakdown_by_state.png":
        "6f3f99a94ae80252310dd3b059721b30fb7665b72db2fa2964b19ddc2e6b7b19",
}

K = 1000.0


def pct_change(new: float, old: float) -> float:
    return (new / old - 1) * 100


def reduction(new: float, old: float) -> float:
    return (1 - new / old) * 100


def article_values(outputs: Path) -> list[tuple[str, float, int, float]]:
    """(label, value printed in the article, decimals printed, value computed)."""
    tables, data = outputs / "tables", outputs / "data"
    summary = pd.read_csv(tables / "summary_table.csv", index_col="architecture_id")
    pilot = pd.read_csv(tables / "pilot_sample_size.csv", index_col="architecture_id")
    stab = pd.read_csv(tables / "stabilization.csv", index_col="transition")
    contribution = pd.read_csv(tables / "state_loss_contribution.csv", index_col=0)
    sens = pd.read_csv(tables / "sensitivity_analysis.csv")
    ranges = pd.read_csv(tables / "sensitivity_ranges.csv", index_col="architecture_id")
    run_length = json.loads((data / "run_length_summary.json").read_text())
    sens_summary = json.loads((data / "sensitivity_summary.json").read_text())

    values = []

    # Table 4.
    table4 = {
        "A0": (92.7, 638.5, 146469, 175678, 187196, 2977, 0, None),
        "A1": (97.8, 193.0, 44825, 58084, 64185, 0, 4000, 97644),
        "A2": (99.4, 53.0, 12336, 18036, 20953, 0, 11500, 122633),
    }
    for key, (kav, downtime, el, l95, l99, k, cost, neb) in table4.items():
        row = summary.loc[key]
        runs = run_length["adopted_num_runs"]
        values += [
            (f"Table 4 {key} K_av^(n), %", kav, 1, row.generalized_availability_pct),
            (f"Table 4 {key} mean time in S3-S5, h/year", downtime, 1,
             row.downtime_hours_per_year),
            (f"Table 4 {key} E(L), MU/year", el, 0, row.expected_annual_loss_mu),
            (f"Table 4 {key} L95, MU/year", l95, 0, row.loss_quantile_095_mu),
            (f"Table 4 {key} L99, MU/year", l99, 0, row.loss_quantile_099_mu),
            (f"Table 4 {key} k (of R = 3 000)", k, 0,
             row.loss_exceedance_probability * runs),
            (f"Table 4 {key} I, MU/year", cost, 0, row.annualized_cost_mu_per_year),
        ]
        if neb is not None:
            values.append((f"Table 4 {key} NEB, MU/year", neb, 0,
                           row.net_economic_benefit_mu_per_year))
    values.append(("Table 4 A0 k/R", 0.992, 3,
                   summary.loc["A0", "loss_exceedance_probability"]))
    values.append(("R = 3 000", 3000, 0, run_length["adopted_num_runs"]))

    # Pilot simulation and stabilization.
    for key, runs in (("A0", 20), ("A1", 50), ("A2", 103)):
        values.append((f"pilot minimum runs {key}", runs, 0,
                       pilot.loc[key, "min_required_runs"]))
    values.append(("R_min", 103, 0, run_length["min_required_runs_from_formula"]))

    def quantile_change(transition):
        return max(stab.loc[transition, "max_change_loss_quantile_095_pct"],
                   stab.loc[transition, "max_change_loss_quantile_099_pct"])

    values += [
        ("stabilization 500 vs 1 000, quantiles, %", 2.53, 2,
         quantile_change("500->1000")),
        ("stabilization 1 000 vs 2 000, quantiles, %", 2.02, 2,
         quantile_change("1000->2000")),
        ("stabilization 2 000 vs 3 000, quantiles, %", 0.90, 2,
         quantile_change("2000->3000")),
        ("stabilization 2 000 vs 3 000, mean loss, %", 0.18, 2,
         stab.loc["2000->3000", "max_change_expected_annual_loss_pct"]),
        ("stabilization 3 000 vs 5 000, quantiles, %", 0.99, 2,
         quantile_change("3000->5000")),
        ("stabilization 3 000 vs 5 000, mean loss, %", 0.28, 2,
         stab.loc["3000->5000", "max_change_expected_annual_loss_pct"]),
        ("stabilization criterion met (1 = yes) 500 vs 1 000", 0, 0,
         stab.loc["500->1000", "criterion_met"]),
        ("stabilization criterion met (1 = yes) 1 000 vs 2 000", 0, 0,
         stab.loc["1000->2000", "criterion_met"]),
        ("stabilization criterion met (1 = yes) 2 000 vs 3 000", 1, 0,
         stab.loc["2000->3000", "criterion_met"]),
        ("stabilization criterion met (1 = yes) 3 000 vs 5 000", 1, 0,
         stab.loc["3000->5000", "criterion_met"]),
        ("first R meeting the stabilization criterion", 3000, 0,
         run_length["first_num_runs_meeting_stabilization_criterion"]),
        ("max availability change <= 0.01 pp (1 = yes)", 1, 0,
         stab["max_change_availability_pp"].max() <= 0.01),
    ]

    # Results text, Fig. 4 and Discussion.
    el = summary["expected_annual_loss_mu"]
    l95 = summary["loss_quantile_095_mu"]
    l99 = summary["loss_quantile_099_mu"]
    neb = summary["net_economic_benefit_mu_per_year"]
    for key, value in (("A0", 75.0), ("A1", 20.7), ("A2", 4.0)):
        values.append((f"Fig. 4 {key} S4 contribution, thousand MU", value, 1,
                       contribution.loc["S4", key] / K))
    for key, (r_el, r_l95, r_l99) in (("A1", (69.4, 66.9, 65.7)),
                                      ("A2", (91.6, 89.7, 88.8))):
        values += [
            (f"{key} reduction of E(L) vs A0, %", r_el, 1, reduction(el[key], el["A0"])),
            (f"{key} reduction of L95 vs A0, %", r_l95, 1, reduction(l95[key], l95["A0"])),
            (f"{key} reduction of L99 vs A0, %", r_l99, 1, reduction(l99[key], l99["A0"])),
        ]
    values += [
        ("A2 vs A1 additional loss reduction, thousand MU", 32.5, 1,
         (el["A1"] - el["A2"]) / K),
        ("A2 vs A1 additional cost, thousand MU/year", 7.5, 1,
         (summary.loc["A2", "annualized_cost_mu_per_year"]
          - summary.loc["A1", "annualized_cost_mu_per_year"]) / K),
        ("A2 vs A1 NEB advantage, thousand MU/year", 25.0, 1,
         (neb["A2"] - neb["A1"]) / K),
    ]

    # Table 5 and the sensitivity text.
    table5 = {
        "A0": ((117.2, 175.8), (140.5, 210.8), (149.8, 224.6), (2051, 3000), None),
        "A1": ((35.9, 53.8), (46.5, 69.7), (51.3, 77.0), (0, 0), (77.3, 118.0)),
        "A2": ((9.9, 14.8), (14.4, 21.6), (16.8, 25.1), (0, 0), (95.8, 149.5)),
    }
    columns = ("expected_annual_loss_mu", "loss_quantile_095_mu",
               "loss_quantile_099_mu", "exceedance_count_k",
               "net_economic_benefit_mu_per_year")
    for key, expected in table5.items():
        for column, bounds in zip(columns, expected):
            if bounds is None:
                continue
            scale, decimals = (1.0, 0) if column == "exceedance_count_k" else (K, 1)
            for bound, suffix in zip(bounds, ("min", "max")):
                values.append((f"Table 5 {key} {column} {suffix}", bound, decimals,
                               ranges.loc[key, f"{column}_{suffix}"] / scale))
    values += [
        ("Table 5 number of configurations", 14, 0, sens_summary["num_scenarios"]),
        ("Table 5 A1 ranked second in", 14, 0,
         ranges.loc["A1", "num_configurations_ranked_second"]),
        ("Table 5 A2 ranked first in", 14, 0,
         ranges.loc["A2", "num_configurations_ranked_first"]),
    ]

    base = sens[sens.scenario_id == "base"].set_index("architecture_id")
    for scenario, label, expected in (
            ("criticality_0.5", "attenuated", (-5.1, -9.0, -10.1)),
            ("criticality_1.5", "amplified", (12.7, 15.7, 17.3))):
        row = sens[(sens.scenario_id == scenario)].set_index("architecture_id")
        for column, value in zip(columns[:3], expected):
            values.append((f"q(t) {label} A2 {column} change, %", value, 1,
                           pct_change(row.loc["A2", column], base.loc["A2", column])))
    for scenario, k in (("criticality_0.5", 2947), ("base", 2977),
                        ("criticality_1.5", 2997)):
        row = sens[(sens.scenario_id == scenario)].set_index("architecture_id")
        values.append((f"q(t) {scenario} A0 exceedances k", k, 0,
                       row.loc["A0", "exceedance_count_k"]))
    values += [
        ("break-even annualized cost of A2, thousand MU", 36.5, 1,
         sens_summary["break_even_annualized_cost_of_a2_mu_per_year"] / K),
        ("break-even expected annual loss of A2, thousand MU", 37.3, 1,
         sens_summary["break_even_expected_annual_loss_of_a2_mu"] / K),
    ]
    return values


def check_figures(figures_dir: Path) -> list[tuple[str, bool, str]]:
    results = []
    for name, expected in ARTICLE_FIGURES.items():
        path = figures_dir / name
        if not path.exists():
            results.append((name, False, "missing"))
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        results.append((name, digest == expected, digest[:16]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outputs", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    failures = 0
    for label, expected, decimals, computed in article_values(args.outputs):
        computed = float(computed)
        ok = abs(computed - expected) <= 0.5 * 10 ** -decimals + 1e-9
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {label}: article {expected:.{decimals}f}, "
              f"computed {computed:.{decimals + 2}f}")

    for name, ok, detail in check_figures(args.outputs / "figures"):
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} figure {name}: sha256 {detail}")

    print(f"\n{failures} mismatch(es)" if failures else "\nall values and figures match the article")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
