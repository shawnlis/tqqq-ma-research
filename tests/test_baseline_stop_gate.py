from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.cli import main


def _set_failed_gate(monkeypatch, tmp_path: Path) -> Path:
    gate = tmp_path / "baseline_regression" / "BASELINE_GATE_FAILED.txt"
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text("failed\n", encoding="utf-8")
    monkeypatch.setenv("RESEARCH_BASELINE_GATE_FILE", str(gate))
    return gate


def _write_prices(path: Path) -> None:
    dates = pd.bdate_range("2011-01-03", "2018-12-31")
    steps = np.arange(len(dates), dtype=float)
    pd.DataFrame(
        {
            "Date": dates,
            "TQQQ": 100.0 + steps * 0.10,
            "QQQ": 100.0 + steps * 0.05,
        }
    ).to_csv(path, index=False)


def _ma_config(tmp_path: Path, name: str) -> dict:
    return {
        "experiment_name": name,
        "symbols": ["TQQQ", "QQQ"],
        "start_date": "2011-01-03",
        "end_date": "2018-12-31",
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


def _write_tournament(tmp_path: Path) -> Path:
    _write_prices(tmp_path / "prices.csv")
    config_path = tmp_path / "ma.yaml"
    config_path.write_text(yaml.safe_dump(_ma_config(tmp_path, "ma_gate")), encoding="utf-8")
    tournament_path = tmp_path / "tournament.yaml"
    tournament_path.write_text(
        yaml.safe_dump(
            {
                "tournament_name": "gate_tournament",
                "benchmark_symbol": "TQQQ",
                "output_dir": str(tmp_path / "tournament_out"),
                "strategies": [{"family": "MA Gate", "category": "tqqq", "config": str(config_path)}],
            }
        ),
        encoding="utf-8",
    )
    return tournament_path


def test_baseline_gate_failure_blocks_tournament(tmp_path: Path, monkeypatch, capsys) -> None:
    _set_failed_gate(monkeypatch, tmp_path)
    tournament_path = _write_tournament(tmp_path)

    assert main(["run-tournament", str(tournament_path)]) == 1

    captured = capsys.readouterr()
    assert "Baseline regression gate is failed" in captured.err
    assert "--accept-baseline-regression" in captured.err
    assert not (tmp_path / "tournament_out" / "tournament_summary.csv").exists()


def test_baseline_gate_failure_blocks_candidate_extraction(tmp_path: Path, monkeypatch, capsys) -> None:
    _set_failed_gate(monkeypatch, tmp_path)
    tournament_dir = tmp_path / "tournament"
    tournament_dir.mkdir()

    assert main(["extract-candidates", str(tournament_dir)]) == 1

    captured = capsys.readouterr()
    assert "Baseline regression gate is failed" in captured.err
    assert "--accept-baseline-regression" in captured.err


def test_accept_baseline_regression_allows_run_but_logs_warning(tmp_path: Path, monkeypatch, capsys) -> None:
    _set_failed_gate(monkeypatch, tmp_path)
    tournament_path = _write_tournament(tmp_path)

    assert main(["run-tournament", str(tournament_path), "--accept-baseline-regression"]) == 0

    captured = capsys.readouterr()
    assert "WARNING: baseline regression gate is failed" in captured.err
    assert (tmp_path / "tournament_out" / "tournament_summary.csv").exists()


def test_check_blockers_fails_when_baseline_gate_exists(tmp_path: Path, monkeypatch) -> None:
    _set_failed_gate(monkeypatch, tmp_path)

    assert main(["check-blockers"]) == 1
