from pathlib import Path

import pandas as pd

from research.cli import main


def _variant_rows(family: str, *, standard_10: float, standard_50: float, alt3: float, alt7: float, strategy_dd: float = -0.40, benchmark_dd: float = -0.38) -> list[dict]:
    rows = []
    specs = [
        ("standard_5y_1y", 10.0, standard_10),
        ("standard_5y_1y", 50.0, standard_50),
        ("alternate_3y_1y", 10.0, alt3),
        ("alternate_7y_1y", 10.0, alt7),
    ]
    for wf_variant, cost, ratio in specs:
        rows.append(
            {
                "family": family,
                "category": "tqqq",
                "experiment_name": family.lower().replace(" ", "_"),
                "config_path": "",
                "strategy_name": "ma",
                "variant": f"{wf_variant}__{cost:g}bps",
                "walk_forward_variant": wf_variant,
                "transaction_cost_bps": cost,
                "status": "ok",
                "final_equity_ratio": ratio,
                "strategy_max_dd": strategy_dd,
                "benchmark_max_dd": benchmark_dd,
                "error": "",
            }
        )
    return rows


def _yearly_rows(family: str, concentrated: bool = False) -> list[dict]:
    if concentrated:
        returns = [
            (2018, 0.30, 0.10),
            (2019, 0.10, 0.09),
            (2020, 0.08, 0.07),
            (2021, 0.09, 0.08),
        ]
    else:
        returns = [
            (2018, 0.20, 0.10),
            (2019, 0.16, 0.10),
            (2020, 0.22, 0.12),
            (2021, 0.15, 0.12),
        ]
    return [
        {
            "family": family,
            "year": year,
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
        }
        for year, strategy_return, benchmark_return in returns
    ]


def test_extract_candidates_applies_hard_gates(tmp_path: Path) -> None:
    out = tmp_path / "tournament"
    out.mkdir()
    rows = []
    rows.extend(_variant_rows("Candidate", standard_10=1.20, standard_50=0.96, alt3=1.02, alt7=0.98))
    rows.extend(_variant_rows("Raw Fail", standard_10=0.98, standard_50=0.96, alt3=1.01, alt7=0.99))
    rows.extend(_variant_rows("Cost Sensitive", standard_10=1.15, standard_50=0.89, alt3=1.02, alt7=0.98))
    rows.extend(_variant_rows("Overfit Stability", standard_10=1.16, standard_50=0.95, alt3=1.03, alt7=0.99))
    rows.extend(_variant_rows("Missing Artifacts", standard_10=1.18, standard_50=0.96, alt3=1.02, alt7=0.99))
    pd.DataFrame(rows).to_csv(out / "tournament_variant_results.csv", index=False)

    stability = pd.DataFrame(
        [
            {"family": "Candidate", "metric": "test_final_equity_ratio", "neighborhood_median_final_equity_ratio": 0.98},
            {"family": "Raw Fail", "metric": "test_final_equity_ratio", "neighborhood_median_final_equity_ratio": 0.98},
            {"family": "Cost Sensitive", "metric": "test_final_equity_ratio", "neighborhood_median_final_equity_ratio": 0.98},
            {"family": "Overfit Stability", "metric": "test_final_equity_ratio", "neighborhood_median_final_equity_ratio": 0.90},
        ]
    )
    stability.to_csv(out / "parameter_stability.csv", index=False)

    yearly = []
    for family in ["Candidate", "Raw Fail", "Cost Sensitive", "Overfit Stability"]:
        yearly.extend(_yearly_rows(family))
    pd.DataFrame(yearly).to_csv(out / "yearly_returns.csv", index=False)

    assert main(["extract-candidates", str(out)]) == 0

    for filename in [
        "final_candidates.csv",
        "rejected_strategies.csv",
        "candidate_report.md",
        "candidate_extraction_config.json",
    ]:
        assert (out / filename).exists()

    candidates = pd.read_csv(out / "final_candidates.csv")
    assert list(candidates["family"]) == ["Candidate"]
    assert bool(candidates.loc[0, "accepted_final_candidate"])

    rejected = pd.read_csv(out / "rejected_strategies.csv")
    reasons = dict(zip(rejected["family"], rejected["primary_rejection_reason"]))
    assert reasons["Raw Fail"] == "failed raw outperformance"
    assert reasons["Cost Sensitive"] == "too cost-sensitive"
    assert reasons["Overfit Stability"] == "overfit"
    assert reasons["Missing Artifacts"] == "data unavailable"
