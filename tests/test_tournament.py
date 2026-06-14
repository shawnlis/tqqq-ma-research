from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.cli import main
from research.tournament import _write_ranked_aliases, _write_voltarget_deep_report


def _write_prices(path: Path) -> None:
    dates = pd.bdate_range("2011-01-03", "2019-12-31")
    steps = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame(
        {
            "Date": dates,
            "TQQQ": 100.0 + steps * 0.10,
            "QQQ": 100.0 + steps * 0.05,
        }
    )
    prices.to_csv(path, index=False)


def _ma_config(tmp_path: Path, name: str) -> dict:
    return {
        "experiment_name": name,
        "symbols": ["TQQQ", "QQQ"],
        "start_date": "2011-01-03",
        "end_date": "2019-12-31",
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
        "transaction_cost_bps": 10.0,
        "train_years": 5,
        "test_years": 1,
        "objective": "final_equity_ratio",
        "use_ensemble_wf": True,
        "ensemble_top_k": 1,
        "ensemble_weight_power": 2.0,
        "output_dir": str(tmp_path / name),
        "data_csv": str(tmp_path / "prices.csv"),
    }


def test_run_tournament_records_rankings_and_failures(tmp_path: Path) -> None:
    _write_prices(tmp_path / "prices.csv")
    good_config_path = tmp_path / "good.yaml"
    good_config_path.write_text(yaml.safe_dump(_ma_config(tmp_path, "good")), encoding="utf-8")

    bad = _ma_config(tmp_path, "bad")
    bad["strategy_name"] = "unsupported_strategy"
    bad_config_path = tmp_path / "bad.yaml"
    bad_config_path.write_text(yaml.safe_dump(bad), encoding="utf-8")

    tournament_path = tmp_path / "tournament.yaml"
    tournament_path.write_text(
        yaml.safe_dump(
            {
                "tournament_name": "tiny_tournament",
                "benchmark_symbol": "TQQQ",
                "output_dir": str(tmp_path / "tournament_out"),
                "strategies": [
                    {"family": "Good MA", "category": "tqqq", "config": str(good_config_path)},
                    {"family": "Bad Strategy", "category": "tqqq", "config": str(bad_config_path)},
                ],
            }
        ),
        encoding="utf-8",
    )

    stale_success_dir = tmp_path / "good" / "standard_5y_1y__10bps"
    stale_success_dir.mkdir(parents=True)
    (stale_success_dir / "error_summary.csv").write_text("stale", encoding="utf-8")
    (stale_success_dir / "error_report.md").write_text("stale", encoding="utf-8")
    stale_error_dir = tmp_path / "bad" / "standard_5y_1y__10bps"
    stale_error_dir.mkdir(parents=True)
    (stale_error_dir / "stitched_equity.csv").write_text("stale", encoding="utf-8")
    (stale_error_dir / "same_period_benchmark_summary.csv").write_text("stale", encoding="utf-8")

    assert main(["run-tournament", str(tournament_path)]) == 0

    output_dir = tmp_path / "tournament_out"
    for filename in [
        "tournament_summary.csv",
        "tournament_ranked_by_final_equity_ratio.csv",
        "tournament_ranked_by_robustness.csv",
        "tournament_failures.csv",
        "tournament_variant_results.csv",
        "report.md",
        "report_summary.csv",
        "run_config.json",
    ]:
        assert (output_dir / filename).exists()

    for filename in [
        "equity_vs_benchmark_log.png",
        "drawdown.png",
        "exposure_over_time.png",
        "cost_sensitivity.png",
        "validation_final_equity_ratio.png",
        "robustness_score.png",
    ]:
        assert (output_dir / "charts" / filename).exists()

    variants = pd.read_csv(output_dir / "tournament_variant_results.csv")
    assert len(variants[variants["family"] == "Good MA"]) == 9
    assert len(variants[variants["family"] == "Bad Strategy"]) == 9
    ok_variant = variants[(variants["family"] == "Good MA") & (variants["status"] == "ok")].iloc[0]
    ok_output_dir = Path(ok_variant["output_dir"])
    for filename in [
        "stitched_equity.csv",
        "walk_forward_windows.csv",
        "same_period_benchmark_summary.csv",
        "yearly_returns.csv",
        "run_config.json",
        "report.md",
    ]:
        assert (ok_output_dir / filename).exists(), filename
    assert not (ok_output_dir / "error_summary.csv").exists()
    assert not (ok_output_dir / "error_report.md").exists()
    failed_variant = variants[(variants["family"] == "Bad Strategy") & (variants["status"] != "ok")].iloc[0]
    failed_output_dir = Path(failed_variant["output_dir"])
    assert (failed_output_dir / "error_summary.csv").exists()
    assert (failed_output_dir / "error_report.md").exists()
    assert not (failed_output_dir / "stitched_equity.csv").exists()
    assert not (failed_output_dir / "same_period_benchmark_summary.csv").exists()

    summary = pd.read_csv(output_dir / "tournament_summary.csv")
    assert set(summary["family"]) == {"Good MA", "Bad Strategy"}
    assert "robustness_score" in summary.columns
    assert "cost_25_final_equity_ratio" in summary.columns

    failures = pd.read_csv(output_dir / "tournament_failures.csv")
    assert "Bad Strategy" in set(failures["family"])
    assert "failed_run" in set(failures["record_type"])
    assert "rejected_strategy" in set(failures["record_type"])


def test_run_tournament_dry_run_respects_max_configs(tmp_path: Path, capsys) -> None:
    _write_prices(tmp_path / "prices.csv")
    config_paths = []
    for name in ["first", "second", "third"]:
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(_ma_config(tmp_path, name)), encoding="utf-8")
        config_paths.append(config_path)

    tournament_path = tmp_path / "tournament_dry.yaml"
    tournament_path.write_text(
        yaml.safe_dump(
            {
                "tournament_name": "tiny_tournament_dry",
                "benchmark_symbol": "TQQQ",
                "output_dir": str(tmp_path / "tournament_dry_out"),
                "strategies": [
                    {"family": f"Family {idx}", "category": "tqqq", "config": str(path)}
                    for idx, path in enumerate(config_paths)
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(["run-tournament", str(tournament_path), "--dry-run", "--max-configs", "2"]) == 0

    captured = capsys.readouterr()
    assert "estimated_evaluations" in captured.out
    summary = pd.read_csv(tmp_path / "tournament_dry_out" / "tournament_dry_run_summary.csv")
    assert len(summary) == 2
    assert summary["tournament_total_configs"].iloc[0] == 3
    assert summary["tournament_selected_configs"].iloc[0] == 2
    assert not (tmp_path / "first" / "stitched_equity.csv").exists()


def test_run_tournament_runtime_guard_requires_explicit_allow_long_run(tmp_path: Path, capsys) -> None:
    _write_prices(tmp_path / "prices.csv")
    config_path = tmp_path / "single.yaml"
    config_path.write_text(yaml.safe_dump(_ma_config(tmp_path, "single")), encoding="utf-8")

    tournament_path = tmp_path / "guarded_tournament.yaml"
    tournament_path.write_text(
        yaml.safe_dump(
            {
                "tournament_name": "guarded_tournament",
                "benchmark_symbol": "TQQQ",
                "output_dir": str(tmp_path / "guarded_out"),
                "max_estimated_runtime_seconds": 0.0,
                "strategies": [
                    {"family": "Guarded MA", "category": "tqqq", "config": str(config_path)},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(["run-tournament", str(tournament_path)]) == 1
    captured = capsys.readouterr()
    assert "--allow-long-run" in captured.err

    assert (tmp_path / "guarded_out" / "tournament_runtime_guard_summary.csv").exists()
    assert main(["run-tournament", str(tournament_path), "--allow-long-run"]) == 0
    assert (tmp_path / "guarded_out" / "tournament_summary.csv").exists()


def test_tournament_voltarget_deep_dry_run_succeeds(tmp_path: Path) -> None:
    source = Path("configs") / "tournament_voltarget_deep.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["output_dir"] = str(tmp_path / "voltarget_deep")
    config_path = tmp_path / "tournament_voltarget_deep.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run-tournament", str(config_path), "--dry-run"]) == 0

    dry_run = pd.read_csv(tmp_path / "voltarget_deep" / "tournament_dry_run_summary.csv")
    assert set(dry_run["family"]) == {
        "VolTargetTQQQStrategy",
        "VolTarget + DrawdownGovernor",
        "MarketInternals VolTarget",
        "CoreOverlay + Rebound comparator",
        "SOXL VolTarget sector bet",
    }
    assert set(dry_run["execution_models"]) == {"close_to_close_shifted;next_open_to_next_open"}
    assert (dry_run["variant_count"] == 24).all()
    assert dry_run["estimated_runtime_seconds"].astype(float).sum() > 0.0


def test_voltarget_deep_rankings_exclude_soxl_sector_bets(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "family": "TQQQ VolTarget",
                "category": "tqqq_voltarget",
                "baseline_final_equity_ratio": 1.2,
                "robustness_score": 0.8,
            },
            {
                "family": "SOXL VolTarget sector bet",
                "category": "soxl_sector_bet",
                "baseline_final_equity_ratio": 3.0,
                "robustness_score": 2.0,
            },
        ]
    )

    _write_ranked_aliases(
        tmp_path,
        summary,
        {"separate_sector_bets": True, "sector_bet_categories": ["soxl_sector_bet"]},
    )

    tqqq_ranked = pd.read_csv(tmp_path / "ranked_by_final_equity_ratio.csv")
    sector_ranked = pd.read_csv(tmp_path / "sector_bet_ranked_by_final_equity_ratio.csv")
    assert set(tqqq_ranked["family"]) == {"TQQQ VolTarget"}
    assert set(sector_ranked["family"]) == {"SOXL VolTarget sector bet"}


def test_voltarget_deep_report_includes_sensitivity_and_alternate_variants(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "family": "VolTargetTQQQStrategy",
                "category": "tqqq_voltarget",
                "baseline_final_equity_ratio": 1.4,
                "median_final_equity_ratio": 1.2,
                "worst_final_equity_ratio": 0.9,
                "baseline_strategy_max_dd": -0.55,
                "cost_10_final_equity_ratio": 1.4,
                "cost_25_final_equity_ratio": 1.2,
                "cost_50_final_equity_ratio": 1.0,
                "cost_sensitivity_25_vs_10": 0.86,
                "cost_sensitivity_50_vs_10": 0.71,
                "robustness_score": 1.0,
                "accepted_candidate": True,
                "rejection_reason": "",
            },
            {
                "family": "SOXL VolTarget sector bet",
                "category": "soxl_sector_bet",
                "baseline_final_equity_ratio": 2.0,
                "median_final_equity_ratio": 1.7,
                "worst_final_equity_ratio": 0.8,
                "baseline_strategy_max_dd": -0.75,
                "cost_10_final_equity_ratio": 2.0,
                "cost_25_final_equity_ratio": 1.7,
                "cost_50_final_equity_ratio": 1.3,
                "cost_sensitivity_25_vs_10": 0.85,
                "cost_sensitivity_50_vs_10": 0.65,
                "robustness_score": 1.4,
                "native_benchmark_symbol": "SOXL",
                "baseline_native_final_equity_ratio": 0.5,
                "accepted_candidate": False,
                "rejection_reason": "sector bet",
            },
        ]
    )
    variants = pd.DataFrame(
        [
            {
                "family": "VolTargetTQQQStrategy",
                "category": "tqqq_voltarget",
                "variant": "alternate_3y_1y__25bps",
                "walk_forward_variant": "alternate_3y_1y",
                "walk_forward_mode": "rolling",
                "execution_model": "close_to_close_shifted",
                "transaction_cost_bps": 25.0,
                "final_equity_ratio": 1.1,
                "strategy_max_dd": -0.50,
                "status": "ok",
                "error": "",
            }
        ]
    )

    report_path = _write_voltarget_deep_report(
        output_dir=tmp_path,
        config={"report_type": "voltarget_deep", "tournament_name": "test"},
        summary=summary,
        variants=variants,
        failures=pd.DataFrame(),
    )

    assert report_path is not None
    report = report_path.read_text(encoding="utf-8")
    assert "## Cost sensitivity" in report
    assert "## Alternate walk-forward variants" in report
    assert "## TQQQ VolTarget candidates" in report
    assert "## SOXL sector-bet candidates" in report
    assert "SOXL rows are sector-bet diagnostics" in report
