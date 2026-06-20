from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.cli import main


def _gate_dir(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "baseline_regression"
    directory.mkdir(parents=True, exist_ok=True)
    gate = directory / "BASELINE_GATE_FAILED.txt"
    gate.write_text("failed\n", encoding="utf-8")
    monkeypatch.setenv("RESEARCH_BASELINE_GATE_FILE", str(gate))
    return directory


def _write_acceptance_inputs(
    tmp_path: Path,
    *,
    tqqq_weight_diff: float = 0.0,
    qqq_weight_diff: float = 0.0,
    position_diff: float = 0.0,
    turnover_diff: float = 0.0,
    old_final: float = 6.124223906488713,
    replay_final: float = 6.1242430658682485,
    allocation_source: str = "data mismatch; return composition mismatch",
) -> dict[str, Path]:
    replay = tmp_path / "replay_compare_summary.csv"
    data_audit = tmp_path / "baseline_data_audit_summary.csv"
    allocation = tmp_path / "regime_allocation_audit_summary.csv"
    baseline = tmp_path / "baseline_regression_summary.csv"

    pd.DataFrame(
        [
            {
                "old_start": "2016-01-04",
                "old_end": "2026-04-02",
                "replay_start": "2016-01-04",
                "replay_end": "2026-04-02",
                "old_rows": 2577,
                "replay_rows": 2577,
                "common_rows": 2577,
                "old_final_equity": old_final,
                "replay_final_equity": replay_final,
                "final_equity_diff": replay_final - old_final,
                "passed": False,
            }
        ]
    ).to_csv(replay, index=False)
    pd.DataFrame(
        [
            {
                "current_data_matches_old_baseline_returns": "no",
                "max_abs_tqqq_return_diff": 8.49e-7,
                "max_abs_qqq_return_diff": 1.01e-6,
            }
        ]
    ).to_csv(data_audit, index=False)
    pd.DataFrame(
        [
            {
                "max_abs_tqqq_weight_diff": tqqq_weight_diff,
                "max_abs_qqq_weight_diff": qqq_weight_diff,
                "max_abs_position_diff": position_diff,
                "max_abs_turnover_diff": turnover_diff,
                "likely_source": allocation_source,
            }
        ]
    ).to_csv(allocation, index=False)
    pd.DataFrame(
        [
            {
                "baseline_equivalence_passed": False,
                "date_range_now_matches": True,
                "row_count_now_matches": True,
                "old_selected_params_replayed_exactly": True,
                "regression_classification": "wrong baseline comparison",
                "likely_root_cause": "Current cached TQQQ/QQQ daily returns differ from the v10 stitched raw returns.",
            }
        ]
    ).to_csv(baseline, index=False)
    return {
        "replay": replay,
        "data_audit": data_audit,
        "allocation": allocation,
        "baseline": baseline,
    }


def _accept_args(paths: dict[str, Path], threshold: str = "0.00001") -> list[str]:
    return [
        "accept-baseline-regression",
        "--reason",
        "data-vintage mismatch; old v10 price cache unavailable; allocation replay matches",
        "--max-final-equity-rel-diff",
        threshold,
        "--require-weight-match",
        "--replay-summary",
        str(paths["replay"]),
        "--baseline-data-summary",
        str(paths["data_audit"]),
        "--allocation-summary",
        str(paths["allocation"]),
        "--baseline-regression-summary",
        str(paths["baseline"]),
    ]


def test_acceptance_succeeds_when_weights_match_and_drift_below_threshold(tmp_path: Path, monkeypatch) -> None:
    gate_dir = _gate_dir(tmp_path, monkeypatch)
    paths = _write_acceptance_inputs(tmp_path)

    assert main(_accept_args(paths)) == 0

    assert (gate_dir / "BASELINE_ACCEPTED_DATA_VINTAGE.txt").exists()
    assert not (gate_dir / "BASELINE_GATE_FAILED.txt").exists()
    assert (gate_dir / "archive" / "BASELINE_GATE_FAILED_superseded.txt").exists()
    summary = pd.read_csv(gate_dir / "baseline_acceptance_summary.csv")
    assert bool(summary.loc[0, "accepted"])
    assert summary.loc[0, "final_equity_relative_diff"] < 0.00001


def test_acceptance_fails_when_weights_differ(tmp_path: Path, monkeypatch) -> None:
    gate_dir = _gate_dir(tmp_path, monkeypatch)
    paths = _write_acceptance_inputs(tmp_path, tqqq_weight_diff=0.25)

    assert main(_accept_args(paths)) == 1

    assert not (gate_dir / "BASELINE_ACCEPTED_DATA_VINTAGE.txt").exists()
    summary = pd.read_csv(gate_dir / "baseline_acceptance_summary.csv")
    assert "max_abs_tqqq_weight_diff" in summary.loc[0, "failure_reasons"]


def test_acceptance_fails_when_final_equity_drift_exceeds_threshold(tmp_path: Path, monkeypatch) -> None:
    gate_dir = _gate_dir(tmp_path, monkeypatch)
    paths = _write_acceptance_inputs(tmp_path, old_final=6.0, replay_final=6.2)

    assert main(_accept_args(paths, threshold="0.00001")) == 1

    assert not (gate_dir / "BASELINE_ACCEPTED_DATA_VINTAGE.txt").exists()
    summary = pd.read_csv(gate_dir / "baseline_acceptance_summary.csv")
    assert "final equity relative drift" in summary.loc[0, "failure_reasons"]


def test_check_blockers_passes_with_warning_after_accepted_data_vintage_baseline(tmp_path: Path, monkeypatch) -> None:
    _gate_dir(tmp_path, monkeypatch)
    paths = _write_acceptance_inputs(tmp_path)
    assert main(_accept_args(paths)) == 0

    assert main(["check-blockers"]) == 0


def _write_candidate_inputs(out: Path) -> None:
    out.mkdir()
    family = "Candidate"
    variant_rows = []
    for wf_variant, cost, ratio in [
        ("standard_5y_1y", 10.0, 1.20),
        ("standard_5y_1y", 50.0, 0.96),
        ("alternate_3y_1y", 10.0, 1.02),
        ("alternate_7y_1y", 10.0, 0.98),
    ]:
        variant_rows.append(
            {
                "family": family,
                "category": "tqqq",
                "experiment_name": "candidate",
                "config_path": "",
                "strategy_name": "ma",
                "variant": f"{wf_variant}__{cost:g}bps",
                "walk_forward_variant": wf_variant,
                "transaction_cost_bps": cost,
                "status": "ok",
                "final_equity_ratio": ratio,
                "strategy_max_dd": -0.40,
                "benchmark_max_dd": -0.38,
                "error": "",
            }
        )
    pd.DataFrame(variant_rows).to_csv(out / "tournament_variant_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "family": family,
                "category": "tqqq",
                "experiment_name": "candidate",
                "config_path": "",
                "strategy_name": "ma",
            }
        ]
    ).to_csv(out / "tournament_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "family": family,
                "metric": "test_final_equity_ratio",
                "neighborhood_median_final_equity_ratio": 0.98,
            }
        ]
    ).to_csv(out / "parameter_stability.csv", index=False)
    pd.DataFrame(
        [
            {"family": family, "year": 2018, "strategy_return": 0.20, "benchmark_return": 0.10},
            {"family": family, "year": 2019, "strategy_return": 0.16, "benchmark_return": 0.10},
            {"family": family, "year": 2020, "strategy_return": 0.22, "benchmark_return": 0.12},
            {"family": family, "year": 2021, "strategy_return": 0.15, "benchmark_return": 0.12},
        ]
    ).to_csv(out / "yearly_returns.csv", index=False)


def test_candidate_extraction_report_includes_baseline_data_vintage_warning(tmp_path: Path, monkeypatch) -> None:
    gate_dir = _gate_dir(tmp_path, monkeypatch)
    paths = _write_acceptance_inputs(tmp_path)
    assert main(_accept_args(paths)) == 0
    assert (gate_dir / "BASELINE_ACCEPTED_DATA_VINTAGE.txt").exists()
    tournament_dir = tmp_path / "tournament"
    _write_candidate_inputs(tournament_dir)

    assert main(["extract-candidates", str(tournament_dir)]) == 0

    report = (tournament_dir / "candidate_report.md").read_text(encoding="utf-8")
    assert "Baseline Compatibility Warning" in report
    assert "data-vintage mismatch" in report
