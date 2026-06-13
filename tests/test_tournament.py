from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.cli import main


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
