from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.cli import main


def _write_prices(path: Path) -> None:
    dates = pd.bdate_range("2018-01-01", "2020-12-31")
    steps = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame(
        {
            "Date": dates,
            "QQQ": 100.0 + steps * 0.05,
            "TQQQ": 100.0 + steps * 0.15,
        }
    )
    prices.to_csv(path, index=False)


def _ma_config(tmp_path: Path, name: str, *, benchmark: str | None = "TQQQ", objective: str = "objective_final_ratio") -> dict:
    config = {
        "experiment_name": name,
        "symbols": ["TQQQ", "QQQ", "CASH"],
        "start_date": "2018-01-01",
        "end_date": "2020-12-31",
        "strategy_name": "ma",
        "strategy_params": {
            "short": 1,
            "long": 2,
            "ma_type": "sma",
            "signal_asset": "TQQQ",
            "threshold": 0.0,
            "cooldown_days": 0,
            "qqq_filter": False,
            "qqq_filter_window": 2,
            "risk_off_symbol": "CASH",
            "risk_off_weight": 0.0,
        },
        "transaction_cost_bps": 10.0,
        "train_years": 1,
        "test_years": 1,
        "objective": objective,
        "output_dir": str(tmp_path / name),
        "data_csv": str(tmp_path / "prices.csv"),
    }
    if benchmark is not None:
        config["benchmark_symbol"] = benchmark
    return config


def _write_tournament(tmp_path: Path, configs: list[dict]) -> Path:
    config_paths = []
    for idx, config in enumerate(configs):
        path = tmp_path / f"experiment_{idx}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        config_paths.append(path)
    tournament_path = tmp_path / "tournament.yaml"
    tournament_path.write_text(
        yaml.safe_dump(
            {
                "tournament_name": "audit_tournament",
                "benchmark_symbol": "TQQQ",
                "output_dir": str(tmp_path / "tournament_out"),
                "strategies": [
                    {"family": f"Experiment {idx}", "category": "tqqq", "config": str(path)}
                    for idx, path in enumerate(config_paths)
                ],
            }
        ),
        encoding="utf-8",
    )
    return tournament_path


def test_audit_config_catches_missing_benchmark_symbol(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    tournament_path = _write_tournament(
        tmp_path,
        [_ma_config(tmp_path, "missing_benchmark", benchmark=None)],
    )

    assert main(["audit-config", str(tournament_path)]) == 0

    summary = pd.read_csv(tmp_path / "tournament_out" / "audit_config_summary.csv")
    assert "benchmark_symbol" in summary.columns
    assert "missing benchmark_symbol" in summary.loc[0, "warnings"]


def test_audit_config_warns_on_sharpe_or_calmar_for_main_tqqq(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    tournament_path = _write_tournament(
        tmp_path,
        [_ma_config(tmp_path, "sharpe_objective", objective="sharpe")],
    )

    assert main(["audit-config", str(tournament_path)]) == 0

    summary = pd.read_csv(tmp_path / "tournament_out" / "audit_config_summary.csv")
    assert bool(summary.loc[0, "objective_prioritizes_raw_final_equity_ratio"]) is False
    assert "Sharpe/Calmar" in summary.loc[0, "warnings"]


def test_audit_data_works_on_synthetic_minimal_fixture_data(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _ma_config(tmp_path, "synthetic_audit")
    config.update(
        {
            "use_synthetic_leverage": True,
            "synthetic_base_symbol": "QQQ",
            "synthetic_leverage": 3.0,
            "synthetic_expense_ratio": 0.0,
            "synthetic_financing_spread": 0.0,
        }
    )
    tournament_path = _write_tournament(tmp_path, [config])

    assert main(["audit-data", str(tournament_path)]) == 0

    summary = pd.read_csv(tmp_path / "tournament_out" / "audit_data_summary.csv")
    assert {"TQQQ", "QQQ", "CASH"}.issubset(set(summary["symbol"]))
    assert summary["symbol_loaded_successfully"].all()
    assert summary["benchmark_available"].all()
    assert summary["synthetic_real_tqqq_overlap_rows"].max() > 0
    cash = summary[summary["symbol"] == "CASH"].iloc[0]
    assert bool(cash["cash_synthetic_series_aligned"])


def test_audit_commands_do_not_mutate_strategy_outputs(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _ma_config(tmp_path, "strategy_output")
    strategy_output = Path(config["output_dir"])
    strategy_output.mkdir()
    existing_output = strategy_output / "stitched_equity.csv"
    existing_output.write_text("original\n", encoding="utf-8")
    before = existing_output.read_text(encoding="utf-8")
    before_mtime = existing_output.stat().st_mtime_ns
    tournament_path = _write_tournament(tmp_path, [config])

    assert main(["audit-config", str(tournament_path)]) == 0
    assert main(["audit-data", str(tournament_path)]) == 0

    assert existing_output.read_text(encoding="utf-8") == before
    assert existing_output.stat().st_mtime_ns == before_mtime
