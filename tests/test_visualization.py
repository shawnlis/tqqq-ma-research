from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research import cli
from research.visualization import (
    _resolve_candidate_output_dir,
    calculate_drawdown,
    generate_equity_visualizations,
    prepare_equity_curves,
)


def _write_stage3_fixture(tmp_path: Path, *, include_exposure: bool = True) -> tuple[Path, Path]:
    input_dir = tmp_path / "stage3"
    run_dir = input_dir / "candidate"
    run_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "family": "VolTargetTQQQStrategy",
                "category": "tqqq_voltarget",
                "is_primary_candidate": True,
                "classification": "crash_control_candidate",
                "run_deep_tournament_recommendation": "defer_deep_tournament",
            }
        ]
    ).to_csv(input_dir / "candidate_decision_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "family": "VolTargetTQQQStrategy",
                "category": "tqqq_voltarget",
                "baseline_output_dir": "candidate",
                "is_primary_candidate": True,
                "classification": "crash_control_candidate",
                "full_period_ratio_vs_tqqq": 3.5,
                "ex_2022_ratio_vs_tqqq": 2.5,
                "ex_2022_ratio_vs_simple_voltarget_no_trend": 0.7,
                "one_year_dominance_share": 0.54,
                "dominant_year": 2022,
                "run_deep_tournament_recommendation": "defer_deep_tournament",
            }
        ]
    ).to_csv(input_dir / "voltarget_stage3_summary.csv", index=False)

    stitched = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]),
            "equity": [10.0, 11.0, 9.9, 12.0],
            "daily_ret_tqqq": [0.0, 0.20, -0.25, 0.50],
        }
    )
    if include_exposure:
        stitched["target_exposure"] = [0.5, 1.0, 0.25, 1.5]
    stitched.to_csv(run_dir / "stitched_equity.csv", index=False)
    pd.DataFrame(
        [
            {
                "benchmark_symbol": "TQQQ",
                "start_date": "2020-01-01",
                "end_date": "2020-01-06",
                "strategy_final_equity": 12.0,
                "benchmark_final_equity": 1.35,
                "final_equity_ratio": 8.8888888889,
            }
        ]
    ).to_csv(run_dir / "same_period_benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {"year": 2020, "strategy_return": 0.2, "benchmark_return": 0.35, "strategy_beats_benchmark": False},
            {"year": 2022, "strategy_return": -0.1, "benchmark_return": -0.7, "strategy_beats_benchmark": True},
        ]
    ).to_csv(run_dir / "yearly_returns.csv", index=False)
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    return input_dir, run_dir


def test_equity_curves_are_normalized_to_first_common_date() -> None:
    stitched = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "equity": [10.0, 12.0, 9.0],
            "daily_ret_tqqq": [0.0, 0.1, -0.2],
        }
    )

    curves = prepare_equity_curves(stitched)

    assert curves["strategy_equity"].iloc[0] == pytest.approx(1.0)
    assert curves["benchmark_equity"].iloc[0] == pytest.approx(1.0)


def test_benchmark_uses_same_dates_as_stitched_strategy() -> None:
    stitched = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-02", "2020-01-06"]),
            "equity": [2.0, 3.0],
            "daily_ret_tqqq": [0.0, 0.5],
        }
    )

    curves = prepare_equity_curves(stitched)

    assert list(curves.index) == list(pd.to_datetime(["2020-01-02", "2020-01-06"]))


def test_drawdown_calculation_is_correct_on_fixture() -> None:
    drawdown = calculate_drawdown(pd.Series([1.0, 2.0, 1.0, 3.0, 2.4]))

    assert drawdown.tolist() == pytest.approx([0.0, 0.0, -0.5, 0.0, -0.2])


def test_relative_equity_equals_strategy_divided_by_benchmark() -> None:
    stitched = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "equity": [1.0, 2.0, 4.0],
            "daily_ret_tqqq": [0.0, 1.0, 0.0],
        }
    )

    curves = prepare_equity_curves(stitched)

    expected = curves["strategy_equity"] / curves["benchmark_equity"]
    assert curves["relative_equity"].tolist() == pytest.approx(expected.tolist())


def test_command_writes_expected_chart_and_report_files(tmp_path: Path) -> None:
    input_dir, _ = _write_stage3_fixture(tmp_path, include_exposure=True)
    output_dir = tmp_path / "visualizations"

    assert (
        cli.main(
            [
                "plot-equity-curve",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    expected_files = {
        "equity_curve_linear.png",
        "equity_curve_log.png",
        "drawdown_curve.png",
        "relative_equity_curve.png",
        "yearly_return_comparison.png",
        "exposure_over_time.png",
        "visualization_summary.csv",
        "visualization_report.md",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})
    summary = pd.read_csv(output_dir / "visualization_summary.csv")
    for column in [
        "strategy_volatility",
        "tqqq_volatility",
        "strategy_sharpe",
        "tqqq_sharpe",
        "strategy_calmar",
        "tqqq_calmar",
    ]:
        assert column in summary.columns
    report = (output_dir / "visualization_report.md").read_text(encoding="utf-8")
    assert "Strategy annualized volatility" in report
    assert "Strategy Sharpe" in report
    assert "Strategy Calmar ratio" in report


def test_missing_exposure_column_is_handled_gracefully(tmp_path: Path) -> None:
    input_dir, _ = _write_stage3_fixture(tmp_path, include_exposure=False)
    output_dir = tmp_path / "visualizations"

    result = generate_equity_visualizations(input_dir, output_dir)

    assert "exposure_over_time" not in result.chart_paths
    assert not (output_dir / "exposure_over_time.png").exists()
    assert result.summary.iloc[0]["exposure_status"] == "exposure column unavailable"
    assert "Exposure column unavailable" in (output_dir / "visualization_report.md").read_text(encoding="utf-8")


def test_repo_relative_outputs_candidate_dir_is_not_prefixed_twice(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    stage_dir = repo_root / "outputs" / "tournament_voltarget_stage3"
    candidate_dir = stage_dir / "experiments" / "stage3_vol_target_tqqq" / "standard_5y_1y__10bps"
    candidate_dir.mkdir(parents=True)
    monkeypatch.chdir(repo_root)

    resolved = _resolve_candidate_output_dir(
        stage_dir,
        pd.Series({"baseline_output_dir": "outputs/tournament_voltarget_stage3/experiments/stage3_vol_target_tqqq/standard_5y_1y__10bps"}),
    )

    assert resolved == Path("outputs/tournament_voltarget_stage3/experiments/stage3_vol_target_tqqq/standard_5y_1y__10bps")
