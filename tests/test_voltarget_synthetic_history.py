from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.cli import main
from research.fairness import (
    LEGACY_SAME_MAX_CONSTANT_RATIO,
    STRATEGY_VS_SAME_MAX_CONSTANT_RATIO,
)


def _write_qqq_only_history(path: Path) -> None:
    dates = pd.bdate_range("1999-03-10", "2008-12-31")
    steps = np.arange(len(dates), dtype=float)
    returns = 0.00025 + 0.0020 * np.sin(steps / 13.0) + 0.0015 * np.cos(steps / 41.0)
    prices = pd.DataFrame(
        {
            "Date": dates,
            "QQQ": 100.0 * np.cumprod(1.0 + returns),
        }
    )
    prices.to_csv(path, index=False)


def _synthetic_config(tmp_path: Path) -> Path:
    data_csv = tmp_path / "qqq_only.csv"
    _write_qqq_only_history(data_csv)
    output_dir = tmp_path / "voltarget_synthetic_history"
    config = {
        "experiment_name": "voltarget_synthetic_history_test",
        "experiment_type": "voltarget_synthetic_history",
        "output_dir": str(output_dir),
        "symbols": ["QQQ", "TQQQ", "CASH"],
        "benchmark_symbol": "TQQQ",
        "strategy_name": "vol_target",
        "objective": "objective_final_ratio",
        "transaction_cost_bps": 0.0,
        "train_years": 5,
        "test_years": 1,
        "target_ann_vol_values": [0.60],
        "max_exposure_values": [1.0],
        "vol_target_params": {
            "realized_vol_window": 5,
            "min_exposure": 0.25,
            "trend_window": 20,
            "momentum_window": 5,
            "trend_multiplier_below_ma": 0.50,
            "momentum_boost": 1.15,
            "crash_vol_cutoff": 0.05,
            "crash_exposure": 0.25,
            "risk_off_symbol": "CASH",
            "risk_off_weight": 0.0,
        },
        "use_synthetic_leverage": True,
        "synthetic_base_symbol": "QQQ",
        "synthetic_target_symbol": "TQQQ",
        "real_leveraged_symbol": "TQQQ",
        "synthetic_leverage": 3.0,
        "synthetic_expense_ratio": 0.0,
        "synthetic_financing_spread": 0.0,
        "data_csv": str(data_csv),
        "synthetic_periods": [
            {
                "label": "synthetic_full_qqq_history",
                "start_date": "1999-03-10",
                "end_date": "2008-12-31",
                "use_synthetic": True,
                "train_years": 5,
                "test_years": 1,
            },
            {
                "label": "synthetic_2000_2002",
                "start_date": "1999-03-10",
                "end_date": "2002-12-31",
                "use_synthetic": True,
                "train_years": 1,
                "test_years": 1,
            },
            {
                "label": "synthetic_2008",
                "start_date": "2003-01-01",
                "end_date": "2008-12-31",
                "use_synthetic": True,
                "train_years": 5,
                "test_years": 1,
            },
        ],
    }
    config_path = tmp_path / "voltarget_synthetic_history.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_synthetic_voltarget_does_not_require_real_tqqq_before_inception(tmp_path: Path) -> None:
    config_path = _synthetic_config(tmp_path)

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "voltarget_synthetic_history"
    summary = pd.read_csv(output_dir / "voltarget_synthetic_history.csv")
    pre_inception = summary.loc[summary["period_label"] == "synthetic_2000_2002"]
    assert not pre_inception.empty
    assert set(pre_inception["status"]) == {"ok"}


def test_synthetic_voltarget_benchmark_is_synthetic_3x_qqq(tmp_path: Path) -> None:
    config_path = _synthetic_config(tmp_path)

    assert main(["run-config", str(config_path)]) == 0

    summary = pd.read_csv(tmp_path / "voltarget_synthetic_history" / "voltarget_synthetic_history.csv")
    assert set(summary["benchmark_symbol"]) == {"TQQQ"}
    assert set(summary["benchmark_source"]) == {"synthetic_3x_QQQ"}
    assert summary["use_synthetic_leverage"].astype(bool).all()


def test_synthetic_voltarget_report_includes_full_history_and_crash_period_rows(tmp_path: Path) -> None:
    config_path = _synthetic_config(tmp_path)

    assert main(["run-config", str(config_path)]) == 0

    report = (
        tmp_path
        / "voltarget_synthetic_history"
        / "voltarget_synthetic_history_report.md"
    ).read_text(encoding="utf-8")
    summary = pd.read_csv(tmp_path / "voltarget_synthetic_history" / "voltarget_synthetic_history.csv")
    assert STRATEGY_VS_SAME_MAX_CONSTANT_RATIO in summary.columns
    assert LEGACY_SAME_MAX_CONSTANT_RATIO not in summary.columns
    assert "synthetic_full_qqq_history" in report
    assert "synthetic_2000_2002" in report
    assert "synthetic_2008" in report
    assert "Full-History And Crash-Period Rows" in report
    assert "strategy_final_equity / constant_same_max_exposure_final_equity" in report
    assert "Initial transaction cost is included" in report
    assert "not floored" in report
    assert "Leverage drag" in report
    assert "VolTarget differs" in report
