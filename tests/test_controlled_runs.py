from pathlib import Path

import pandas as pd
import yaml

from research.cli import main
from research.controlled_runs import ControlledRunSpec, summarize_controlled_runs


def _write_summary(
    output_dir: Path,
    *,
    final_equity_ratio: float,
    raw_pass: bool,
    strategy_final: float = 1.0,
    benchmark_final: float = 1.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "benchmark_symbol": "TQQQ",
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "stitched_rows": 3,
                "strategy_final_equity": strategy_final,
                "benchmark_final_equity": benchmark_final,
                "strategy_cagr": 0.10,
                "benchmark_cagr": 0.05,
                "strategy_max_dd": -0.20,
                "benchmark_max_dd": -0.30,
                "final_equity_ratio": final_equity_ratio,
                "excess_cagr": 0.05,
                "years_strategy_beats_benchmark": 1,
                "total_years": 1,
                "raw_outperformance_pass": raw_pass,
            }
        ]
    ).to_csv(output_dir / "same_period_benchmark_summary.csv", index=False)


def _write_yearly(output_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "yearly_returns.csv", index=False)


def _write_stitched(output_dir: Path) -> None:
    pd.DataFrame(
        [
            {"Date": "2020-01-01", "equity": 1.0, "execution_model": "close_to_close_shifted", "cost": 0.001},
            {"Date": "2020-01-02", "equity": 1.1, "execution_model": "close_to_close_shifted", "cost": 0.0},
            {"Date": "2020-01-03", "equity": 1.2, "execution_model": "close_to_close_shifted", "cost": 0.001},
        ]
    ).to_csv(output_dir / "stitched_equity.csv", index=False)


def test_summarize_controlled_runs_classifies_raw_failure_and_missing_output(tmp_path: Path) -> None:
    raw_fail_dir = tmp_path / "raw_fail"
    _write_summary(
        raw_fail_dir,
        final_equity_ratio=0.80,
        raw_pass=False,
        strategy_final=0.8,
        benchmark_final=1.0,
    )
    _write_yearly(
        raw_fail_dir,
        [{"year": 2020, "strategy_return": 0.05, "benchmark_return": 0.10, "strategy_beats_benchmark": False}],
    )
    _write_stitched(raw_fail_dir)
    missing_dir = tmp_path / "missing"

    summary = summarize_controlled_runs(
        output_dir=tmp_path / "controlled",
        run_specs=[
            ControlledRunSpec("raw_fail", raw_fail_dir),
            ControlledRunSpec("missing", missing_dir),
        ],
    )

    classifications = dict(zip(summary["strategy_config"], summary["classification"]))
    assert classifications["raw_fail"] == "empirical rejection"
    assert classifications["missing"] == "infrastructure failure"
    contracts = dict(zip(summary["strategy_config"], summary["output_contract_status"]))
    assert contracts["raw_fail"] == "pass"
    assert contracts["missing"] == "missing_same_period_benchmark_summary"
    assert (tmp_path / "controlled" / "controlled_run_summary.csv").exists()
    assert (tmp_path / "controlled" / "controlled_run_report.md").exists()


def test_summarize_controlled_runs_identifies_candidate_and_audits(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate"
    config_path = tmp_path / "candidate.yaml"
    _write_summary(
        candidate_dir,
        final_equity_ratio=1.20,
        raw_pass=True,
        strategy_final=1.2,
        benchmark_final=1.0,
    )
    _write_yearly(
        candidate_dir,
        [
            {"year": 2020, "strategy_return": 0.20, "benchmark_return": 0.10, "strategy_beats_benchmark": True},
            {"year": 2021, "strategy_return": 0.18, "benchmark_return": 0.10, "strategy_beats_benchmark": True},
        ],
    )
    _write_stitched(candidate_dir)
    config_path.write_text(
        yaml.safe_dump({"objective": "objective_final_ratio", "transaction_cost_bps": 10.0}),
        encoding="utf-8",
    )

    summary = summarize_controlled_runs(
        output_dir=tmp_path / "controlled",
        run_specs=[ControlledRunSpec("candidate", candidate_dir, config_path=config_path)],
    )

    row = summary.iloc[0]
    assert row["classification"] == "candidate for tournament_gate validation"
    assert row["benchmark_date_alignment"] == "pass"
    assert row["position_shift_audit"] == "shifted_execution_model"
    assert row["transaction_cost_audit"] == "costs_present"
    assert row["output_contract_status"] == "pass"
    assert row["objective_used"] == "objective_final_ratio"
    assert not bool(row["one_year_drives_excess_return"])


def test_summarize_controlled_runs_cli_writes_default_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["summarize-controlled-runs", "--output-dir", str(tmp_path / "controlled")]) == 0
    summary = pd.read_csv(tmp_path / "controlled" / "controlled_run_summary.csv")
    assert len(summary) == 5
    assert set(summary["classification"]) == {"infrastructure failure"}
    assert set(summary["output_contract_status"]) == {"missing_same_period_benchmark_summary"}
