from pathlib import Path

import pandas as pd
import pytest

from research.fairness import (
    LEGACY_SAME_MAX_CONSTANT_RATIO,
    STRATEGY_VS_SAME_MAX_CONSTANT_RATIO,
    build_constant_leverage_benchmark_summary,
    constant_exposure_backtest,
    write_voltarget_fairness_outputs,
)
from research.metrics import max_drawdown


def _price_data(values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame(
        {
            "TQQQ": values,
            "QQQ": values,
        },
        index=dates,
    )


def test_constant_exposure_1x_matches_buy_and_hold_after_initial_cost() -> None:
    prices = _price_data([100.0, 110.0, 121.0])
    bt = constant_exposure_backtest(
        prices,
        exposure=1.0,
        transaction_cost_bps=100.0,
    )

    expected_equity = (1.0 - 0.01) * 1.10 * 1.10
    assert bt["turnover"].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert float(bt["equity"].iloc[-1]) == pytest.approx(expected_equity)

    summary = build_constant_leverage_benchmark_summary(
        stitched=bt[["ret", "equity", "turnover", "cost"]],
        price_data=prices,
        transaction_cost_bps=100.0,
        selected_max_exposure=1.0,
    )
    one_x = summary.loc[summary["benchmark_name"] == "buy_and_hold_tqqq_1.0"].iloc[0]
    assert float(one_x["constant_final_equity"]) == pytest.approx(expected_equity)
    assert float(one_x["strategy_final_equity"]) == pytest.approx(expected_equity)


def test_constant_exposure_2x_has_larger_drawdown_on_falling_fixture() -> None:
    prices = _price_data([100.0, 90.0, 81.0, 72.9])
    one_x = constant_exposure_backtest(prices, exposure=1.0, transaction_cost_bps=0.0)
    two_x = constant_exposure_backtest(prices, exposure=2.0, transaction_cost_bps=0.0)

    one_x_dd = max_drawdown(one_x["equity"])
    two_x_dd = max_drawdown(two_x["equity"])
    assert two_x_dd < one_x_dd


def test_same_max_constant_ratio_formula_uses_strategy_over_constant_final_equity() -> None:
    prices = _price_data([100.0, 110.0, 121.0])
    strategy = constant_exposure_backtest(prices, exposure=1.0, transaction_cost_bps=0.0)

    summary = build_constant_leverage_benchmark_summary(
        stitched=strategy[["ret", "equity", "turnover", "cost"]],
        price_data=prices,
        transaction_cost_bps=0.0,
        selected_max_exposure=2.0,
        exposures=(1.0, 2.0),
    )

    same_max = summary.loc[summary["is_same_max_exposure_benchmark"]].iloc[0]
    expected_ratio = float(same_max["strategy_final_equity"]) / float(same_max["constant_final_equity"])
    assert float(same_max["constant_exposure"]) == pytest.approx(2.0)
    assert float(same_max["strategy_final_equity"]) == pytest.approx(1.21)
    assert float(same_max["constant_final_equity"]) == pytest.approx(1.44)
    assert float(same_max[STRATEGY_VS_SAME_MAX_CONSTANT_RATIO]) == pytest.approx(expected_ratio)
    assert LEGACY_SAME_MAX_CONSTANT_RATIO not in summary.columns


def test_voltarget_fairness_report_includes_same_max_exposure_benchmark(tmp_path: Path) -> None:
    prices = _price_data([100.0, 102.0, 101.0, 105.0])
    stitched = constant_exposure_backtest(prices, exposure=1.25, transaction_cost_bps=10.0)
    wf_table = pd.DataFrame([{"max_exposure": 2.0}])

    summary = write_voltarget_fairness_outputs(
        output_dir=tmp_path,
        stitched=stitched,
        price_data=prices,
        transaction_cost_bps=10.0,
        wf_table=wf_table,
    )

    assert (tmp_path / "constant_leverage_benchmark_summary.csv").exists()
    report_path = tmp_path / "voltarget_fairness_report.md"
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "same-max-exposure benchmark" in report
    assert "strategy_final_equity / constant_same_max_exposure_final_equity" in report
    assert "Initial transaction cost is included" in report
    assert "Equity is not floored" in report
    assert "Leverage drag is handled through daily compounding" in report
    assert "VolTarget differs because its exposure changes over time" in report
    assert "constant_tqqq_2.0" in set(summary["benchmark_name"])
    same_max = summary.loc[summary["is_same_max_exposure_benchmark"]].iloc[0]
    assert float(same_max["constant_exposure"]) == pytest.approx(2.0)
    assert STRATEGY_VS_SAME_MAX_CONSTANT_RATIO in summary.columns
    assert LEGACY_SAME_MAX_CONSTANT_RATIO not in summary.columns
