import json
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
            "TQQQ": 100.0 + steps * 0.10,
            "QQQ": 100.0 + steps * 0.05,
            "SPY": 100.0 + steps * 0.04,
            "SOXL": 80.0 + steps * 0.12,
            "SOXX": 120.0 + steps * 0.06,
            "SMH": 110.0 + steps * 0.065,
            "UPRO": 90.0 + steps * 0.08,
            "BIL": 100.0 + steps * 0.005,
            "SGOV": 100.0 + steps * 0.004,
        }
    )
    prices.to_csv(path, index=False)


def _experiment_config(tmp_path: Path, name: str, output_name: str) -> dict:
    data_csv = tmp_path / "prices.csv"
    return {
        "experiment_name": name,
        "symbols": ["TQQQ", "QQQ"],
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
        },
        "benchmark_symbol": "TQQQ",
        "transaction_cost_bps": 0.0,
        "train_years": 1,
        "test_years": 1,
        "objective": "final_equity_ratio",
        "anti_overfit_validation": {"bootstrap_iterations": 20},
        "output_dir": str(tmp_path / output_name),
        "data_csv": str(data_csv),
    }


def test_run_config_minimal_synthetic_dataset(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(
        yaml.safe_dump(_experiment_config(tmp_path, "minimal", "experiment")),
        encoding="utf-8",
    )

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "experiment"
    for filename in [
        "stitched_equity.csv",
        "walk_forward_windows.csv",
        "parameter_selection_by_window.csv",
        "same_period_benchmark_summary.csv",
        "yearly_returns.csv",
        "validation_summary.csv",
        "walk_forward_variant_summary.csv",
        "bootstrap_summary.csv",
        "subperiod_summary.csv",
        "parameter_stability.csv",
        "cost_sensitivity.csv",
        "execution_model_sensitivity.csv",
        "report.md",
        "report_summary.csv",
        "run_config.json",
    ]:
        assert (output_dir / filename).exists()

    for filename in [
        "equity_vs_benchmark_log.png",
        "drawdown.png",
        "yearly_relative_returns.png",
        "exposure_over_time.png",
        "cost_sensitivity.png",
        "validation_final_equity_ratio.png",
    ]:
        assert (output_dir / "charts" / filename).exists()

    summary = pd.read_csv(output_dir / "same_period_benchmark_summary.csv")
    assert "final_equity_ratio" in summary.columns
    assert summary.loc[0, "benchmark_symbol"] == "TQQQ"

    validation = pd.read_csv(output_dir / "validation_summary.csv")
    assert "candidate_viable" in validation.columns
    variants = pd.read_csv(output_dir / "walk_forward_variant_summary.csv")
    assert set(variants["variant"]) >= {
        "standard_5y_1y",
        "alt_3y_1y",
        "alt_7y_1y",
        "anchored_expanding_1y",
        "cost_25bps",
        "cost_50bps",
    }

    costs = pd.read_csv(output_dir / "cost_sensitivity.csv")
    assert set(costs["transaction_cost_bps"]) == {0.0, 10.0, 25.0, 50.0, 100.0}
    assert "zero_bps_only_outperformance" in costs.columns

    execution = pd.read_csv(output_dir / "execution_model_sensitivity.csv")
    assert set(execution["requested_execution_model"]) == {
        "close_to_close_shifted",
        "next_open_to_close",
        "next_open_to_next_open",
    }


def test_run_config_dry_run_prints_estimate_without_backtest_outputs(tmp_path: Path, capsys) -> None:
    _write_prices(tmp_path / "prices.csv")
    config_path = tmp_path / "dry_run.yaml"
    config_path.write_text(
        yaml.safe_dump(_experiment_config(tmp_path, "dry_run", "dry_run_output")),
        encoding="utf-8",
    )

    assert main(["run-config", str(config_path), "--dry-run"]) == 0

    captured = capsys.readouterr()
    assert "estimated_parameter_combinations" in captured.out
    assert "estimated_runtime_seconds" in captured.out
    assert not (tmp_path / "dry_run_output" / "stitched_equity.csv").exists()


def test_run_config_objective_cli_override(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _experiment_config(tmp_path, "objective_override", "objective_override")
    config_path = tmp_path / "objective_override.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(
        [
            "run-config",
            str(config_path),
            "--objective",
            "objective_excess_cagr_with_dd_guard",
        ]
    ) == 0

    run_config = json.loads(
        (tmp_path / "objective_override" / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["objective"] == "objective_excess_cagr_with_dd_guard"


def test_run_config_synthetic_leverage_writes_tracking_outputs(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _experiment_config(tmp_path, "synthetic", "synthetic_experiment")
    config.update(
        {
            "use_synthetic_leverage": True,
            "synthetic_base_symbol": "QQQ",
            "synthetic_leverage": 3.0,
            "synthetic_expense_ratio": 0.0,
            "synthetic_financing_spread": 0.0,
        }
    )
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "synthetic_experiment"
    assert (output_dir / "synthetic_tracking_summary.csv").exists()
    assert (output_dir / "synthetic_tracking_equity.csv").exists()

    tracking = pd.read_csv(output_dir / "synthetic_tracking_summary.csv")
    assert "tracking_error" in tracking.columns
    assert "cagr_difference" in tracking.columns
    assert "max_drawdown_difference" in tracking.columns


def test_run_batch_writes_rankings(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config_paths = []
    for name in ["first", "second"]:
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(
            yaml.safe_dump(_experiment_config(tmp_path, name, name)),
            encoding="utf-8",
        )
        config_paths.append(str(config_path))

    batch_path = tmp_path / "batch.yaml"
    batch_path.write_text(
        yaml.safe_dump(
            {
                "batch_name": "synthetic_batch",
                "configs": config_paths,
                "output_dir": str(tmp_path / "batch"),
            }
        ),
        encoding="utf-8",
    )

    assert main(["run-batch", str(batch_path)]) == 0

    batch_dir = tmp_path / "batch"
    for filename in [
        "batch_summary.csv",
        "ranked_by_final_equity_ratio.csv",
        "ranked_by_excess_cagr.csv",
        "ranked_by_calmar.csv",
        "risk_off_comparison.csv",
    ]:
        assert (batch_dir / filename).exists()

    ranked = pd.read_csv(batch_dir / "ranked_by_final_equity_ratio.csv")
    assert list(ranked["final_equity_ratio"]) == sorted(
        ranked["final_equity_ratio"],
        reverse=True,
    )


def test_run_batch_dry_run_respects_max_configs_without_strategy_outputs(tmp_path: Path, capsys) -> None:
    _write_prices(tmp_path / "prices.csv")
    config_paths = []
    for name in ["first", "second", "third"]:
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(
            yaml.safe_dump(_experiment_config(tmp_path, name, name)),
            encoding="utf-8",
        )
        config_paths.append(str(config_path))

    batch_path = tmp_path / "batch_dry.yaml"
    batch_path.write_text(
        yaml.safe_dump(
            {
                "batch_name": "dry_batch",
                "configs": config_paths,
                "output_dir": str(tmp_path / "batch_dry"),
            }
        ),
        encoding="utf-8",
    )

    assert main(["run-batch", str(batch_path), "--dry-run", "--max-configs", "2"]) == 0

    captured = capsys.readouterr()
    assert "dry_run" in captured.out
    dry_summary = pd.read_csv(tmp_path / "batch_dry" / "batch_dry_run_summary.csv")
    assert len(dry_summary) == 2
    assert set(dry_summary["experiment_name"]) == {"first", "second"}
    assert dry_summary["batch_total_configs"].iloc[0] == 3
    assert dry_summary["batch_selected_configs"].iloc[0] == 2
    assert not (tmp_path / "first" / "stitched_equity.csv").exists()


def test_run_config_core_overlay_minimal_dataset(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _experiment_config(tmp_path, "core_overlay", "core_overlay")
    config["strategy_name"] = "core_overlay"
    config["strategy_params"] = {
        "core_exposure": 0.50,
        "overlay_max": 0.35,
        "max_total_exposure": 1.00,
        "trend_window": 20,
        "fast_trend_window": 5,
        "momentum_window": 5,
        "vol_window": 5,
        "vol_cap": 0.03,
        "crash_cut_exposure": 0.25,
        "rebound_boost": True,
    }
    config_path = tmp_path / "core_overlay.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "core_overlay"
    assert (output_dir / "parameter_selection_by_window.csv").exists()
    selections = pd.read_csv(output_dir / "parameter_selection_by_window.csv")
    assert "core_exposure" in selections.columns


def test_run_config_core_overlay_rebound_writes_ablation(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _experiment_config(tmp_path, "core_overlay_rebound", "core_overlay_rebound")
    config["strategy_name"] = "core_overlay"
    config["strategy_params"] = {
        "core_exposure": 0.50,
        "overlay_max": 0.35,
        "max_total_exposure": 1.25,
        "trend_window": 20,
        "fast_trend_window": 5,
        "momentum_window": 5,
        "vol_window": 5,
        "vol_cap": 0.03,
        "crash_cut_exposure": 0.25,
        "rebound_boost": False,
        "use_rebound_module": True,
        "rolling_high_window": 20,
        "drawdown_trigger": -0.15,
        "rebound_momentum_window": 5,
        "rebound_momentum_threshold": 0.03,
        "reclaim_ma_window": 10,
        "rebound_position": 1.0,
        "rebound_hold_days": 5,
    }
    config_path = tmp_path / "core_overlay_rebound.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "core_overlay_rebound"
    assert (output_dir / "rebound_ablation_summary.csv").exists()
    ablation = pd.read_csv(output_dir / "rebound_ablation_summary.csv")
    assert ablation["variant"].tolist() == ["without_rebound", "with_rebound", "difference"]


def test_run_config_vol_target_minimal_dataset(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _experiment_config(tmp_path, "vol_target", "vol_target")
    config["strategy_name"] = "vol_target"
    config["strategy_params"] = {
        "target_ann_vol": 0.60,
        "realized_vol_window": 5,
        "min_exposure": 0.25,
        "max_exposure": 1.25,
        "trend_window": 20,
        "momentum_window": 5,
        "trend_multiplier_below_ma": 0.50,
        "momentum_boost": 1.15,
        "crash_vol_cutoff": 0.05,
        "crash_exposure": 0.25,
    }
    config_path = tmp_path / "vol_target.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "vol_target"
    assert (output_dir / "drawdown_chart_data.csv").exists()
    assert (output_dir / "vol_target_acceptance_summary.csv").exists()
    stitched = pd.read_csv(output_dir / "stitched_equity.csv")
    assert "target_exposure" in stitched.columns
    assert "realized_ann_vol" in stitched.columns
    assert "drawdown" in stitched.columns

    selections = pd.read_csv(output_dir / "parameter_selection_by_window.csv")
    assert "target_ann_vol" in selections.columns


def test_run_config_core_overlay_governor_writes_ablation(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _experiment_config(tmp_path, "core_overlay_governor", "core_overlay_governor")
    config["strategy_name"] = "core_overlay"
    config["strategy_params"] = {
        "core_exposure": 0.50,
        "overlay_max": 0.35,
        "max_total_exposure": 1.25,
        "trend_window": 20,
        "fast_trend_window": 5,
        "momentum_window": 5,
        "vol_window": 5,
        "vol_cap": 0.03,
        "crash_cut_exposure": 0.25,
        "rebound_boost": False,
        "use_drawdown_governor": True,
        "portfolio_dd_trigger": -0.20,
        "qqq_dd_trigger": -0.10,
        "reduced_exposure": 0.50,
        "recovery_ma_window": 10,
        "recovery_momentum_window": 5,
        "recovery_momentum_threshold": 0.03,
        "max_days_reduced": 20,
    }
    config_path = tmp_path / "core_overlay_governor.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "core_overlay_governor"
    assert (output_dir / "drawdown_governor_ablation_summary.csv").exists()
    assert (output_dir / "drawdown_governor_ablation_yearly_returns.csv").exists()
    ablation = pd.read_csv(output_dir / "drawdown_governor_ablation_summary.csv")
    assert ablation["variant"].tolist() == [
        "base_strategy",
        "base_governor",
        "base_governor_rebound",
    ]

    selections = pd.read_csv(output_dir / "parameter_selection_by_window.csv")
    assert "use_drawdown_governor" in selections.columns


def test_run_config_market_internals_writes_daily_and_skipped_symbols(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _experiment_config(tmp_path, "market_internals", "market_internals")
    config["strategy_name"] = "core_overlay"
    config["strategy_params"] = {
        "core_exposure": 0.50,
        "overlay_max": 0.35,
        "max_total_exposure": 1.00,
        "trend_window": 20,
        "fast_trend_window": 5,
        "momentum_window": 5,
        "vol_window": 5,
        "vol_cap": 0.03,
        "crash_cut_exposure": 0.25,
        "rebound_boost": False,
        "use_market_internals": True,
        "market_internals_signal_window": 20,
        "market_internals_risk_on_threshold": 0.60,
        "market_internals_risk_off_threshold": 0.40,
    }
    config_path = tmp_path / "market_internals.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "market_internals"
    assert (output_dir / "market_internals_daily.csv").exists()
    daily = pd.read_csv(output_dir / "market_internals_daily.csv")
    assert "market_internals_risk_score" in daily.columns

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    skipped = {
        item["symbol"]: item["reason"]
        for item in run_config["market_internals_data_status"]["skipped_symbols"]
    }
    assert skipped["RSP"] == "missing"
    assert skipped["HYG"] == "missing"


def test_run_config_rotation_writes_rotation_outputs_and_skips_missing_optional_asset(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _experiment_config(tmp_path, "rotation", "rotation")
    config["strategy_name"] = "rotation"
    config["strategy_params"] = {
        "risk_assets": ["TQQQ", "SOXL", "UPRO", "TECL"],
        "rebalance_frequency": "monthly",
        "top_n": 1,
        "max_asset_weight": 1.0,
        "max_total_leveraged_exposure": 1.0,
        "trend_filter_symbol": "QQQ",
        "trend_filter_window": 100,
        "defensive_asset": "CASH",
        "require_positive_momentum": True,
    }
    config_path = tmp_path / "rotation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "rotation"
    for filename in [
        "selected_assets.csv",
        "weights_by_date.csv",
        "turnover_summary.csv",
        "same_period_benchmark_summary.csv",
    ]:
        assert (output_dir / filename).exists()

    weights = pd.read_csv(output_dir / "weights_by_date.csv")
    assert "weight_TQQQ" in weights.columns

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    skipped = {
        item["symbol"]: item["reason"]
        for item in run_config["rotation_asset_status"]["skipped_symbols"]
    }
    assert skipped["TECL"] == "missing"


def test_run_config_soxl_core_overlay_writes_soxl_benchmark_outputs(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    config = _experiment_config(tmp_path, "soxl_core", "soxl_core")
    config["symbols"] = ["SOXL", "SOXX", "SMH", "QQQ", "TQQQ", "SGOV", "BIL"]
    config["strategy_name"] = "core_overlay"
    config["benchmark_symbol"] = "SOXL"
    config["asset_config"] = {
        "trade_asset": "SOXL",
        "primary_signal_asset": "SOXX",
        "secondary_filter_asset": "SMH",
        "benchmark_symbol": "SOXL",
        "risk_off_symbol": "BIL",
    }
    config["benchmark_comparison_symbols"] = ["TQQQ", "SOXX", "SMH"]
    config["soxl_experiment"] = True
    config["strategy_params"] = {
        "core_exposure": 0.50,
        "overlay_max": 0.35,
        "max_total_exposure": 1.00,
        "trend_window": 20,
        "fast_trend_window": 5,
        "momentum_window": 5,
        "vol_window": 5,
        "vol_cap": 0.03,
        "crash_cut_exposure": 0.25,
        "rebound_boost": True,
        "risk_off_symbol": "BIL",
        "risk_off_weight": 1.0,
    }
    config_path = tmp_path / "soxl_core.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run-config", str(config_path)]) == 0

    output_dir = tmp_path / "soxl_core"
    for filename in [
        "soxl_vs_soxl_summary.csv",
        "soxl_vs_tqqq_summary.csv",
        "soxl_vs_soxx_summary.csv",
        "soxl_yearly_returns.csv",
        "soxl_selected_params.csv",
    ]:
        assert (output_dir / filename).exists()

    stitched = pd.read_csv(output_dir / "stitched_equity.csv")
    assert "soxl_weight" in stitched.columns
    selections = pd.read_csv(output_dir / "soxl_selected_params.csv")
    assert selections.loc[0, "trade_asset"] == "SOXL"
    assert selections.loc[0, "primary_signal_asset"] == "SOXX"

    soxl_summary = pd.read_csv(output_dir / "soxl_vs_soxl_summary.csv")
    tqqq_summary = pd.read_csv(output_dir / "soxl_vs_tqqq_summary.csv")
    assert soxl_summary.loc[0, "benchmark_symbol"] == "SOXL"
    assert tqqq_summary.loc[0, "benchmark_symbol"] == "TQQQ"
