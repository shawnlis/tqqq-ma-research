from pathlib import Path

import pandas as pd

from research.cli import main


def _write_prices(path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", "2020-02-14")
    prices = pd.DataFrame(
        {
            "Date": dates,
            "TQQQ": [100.0 + i * 0.50 for i in range(len(dates))],
            "QQQ": [100.0 + i * 0.20 for i in range(len(dates))],
        }
    )
    prices.to_csv(path, index=False)


def _window_rows() -> list[dict]:
    return [
        {
            "train_start": "2020-01-01",
            "train_end": "2020-01-31",
            "test_start": "2020-02-03",
            "test_end": "2020-02-14",
            "trend_window": 2,
            "momentum_window": 1,
            "vol_window": 2,
            "trend_on": 0.0,
            "trend_off": 0.0,
            "mom_on": 0.0,
            "mom_off": -0.01,
            "vol_cap": 1.0,
            "risk_on_leverage": 1.0,
            "risk_off_qqq_position": 1.0,
            "transition_position": 0.25,
            "cooldown_days": 0,
            "test_final_equity": 1.23,
            "test_avg_exposure": 0.75,
            "test_trades": 3,
        }
    ]


def _write_legacy_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    prices_path = tmp_path / "prices.csv"
    _write_prices(prices_path)
    windows_path = tmp_path / "windows.csv"
    pd.DataFrame(_window_rows()).to_csv(windows_path, index=False)
    dates = pd.bdate_range("2020-02-03", "2020-02-14")
    old = pd.DataFrame(
        {
            "Date": dates,
            "ret": [0.123] + [0.0] * (len(dates) - 1),
            "equity": [1.123] * len(dates),
            "position": [9.0] + [1.0] * (len(dates) - 1),
            "tqqq_weight": [9.0] + [1.0] * (len(dates) - 1),
            "qqq_weight": [9.0] + [0.0] * (len(dates) - 1),
            "turnover": [9.0] + [0.0] * (len(dates) - 1),
        }
    )
    old_path = tmp_path / "old_stitched.csv"
    old.to_csv(old_path, index=False)
    return prices_path, windows_path, old_path


def _run_replay(tmp_path: Path) -> Path:
    prices_path, windows_path, old_path = _write_legacy_inputs(tmp_path)
    output_dir = tmp_path / "replay"
    assert (
        main(
            [
                "replay-regime-windows",
                "--windows",
                str(windows_path),
                "--old-stitched",
                str(old_path),
                "--output-dir",
                str(output_dir),
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-02-14",
                "--data-csv",
                str(prices_path),
                "--tolerance",
                "1e-8",
            ]
        )
        == 1
    )
    return output_dir


def test_replay_regime_windows_does_not_run_grid_search(tmp_path: Path, monkeypatch) -> None:
    def fail_grid_search(*args, **kwargs):
        raise AssertionError("grid search should not run during replay")

    monkeypatch.setattr("research.validation.regime_search_on_window", fail_grid_search)
    output_dir = _run_replay(tmp_path)
    assert (output_dir / "replayed_stitched_equity.csv").exists()


def test_replay_regime_windows_preserves_old_selected_parameters(tmp_path: Path) -> None:
    output_dir = _run_replay(tmp_path)
    replay = pd.read_csv(output_dir / "window_param_replay.csv")
    old = _window_rows()[0]

    for column in [
        "trend_window",
        "momentum_window",
        "vol_window",
        "trend_on",
        "trend_off",
        "mom_on",
        "mom_off",
        "vol_cap",
        "risk_on_leverage",
        "risk_off_qqq_position",
        "transition_position",
        "cooldown_days",
    ]:
        assert replay[column].iloc[0] == old[column]
    assert replay["min_hold_days"].iloc[0] == 0


def test_replay_regime_windows_identifies_first_return_and_weight_divergence(tmp_path: Path) -> None:
    output_dir = _run_replay(tmp_path)
    summary = pd.read_csv(output_dir / "replay_compare_summary.csv")

    assert summary["first_ret_divergence_date"].iloc[0] == "2020-02-03"
    assert summary["first_weight_divergence_date"].iloc[0] == "2020-02-03"
    assert not bool(summary["passed"].iloc[0])


def test_replay_regime_windows_output_contract_is_complete(tmp_path: Path) -> None:
    output_dir = _run_replay(tmp_path)
    for filename in [
        "replayed_stitched_equity.csv",
        "replay_compare_summary.csv",
        "replay_compare_report.md",
        "window_param_replay.csv",
        "first_divergence.csv",
    ]:
        assert (output_dir / filename).exists()

    first_divergence = pd.read_csv(output_dir / "first_divergence.csv")
    expected = {
        "date",
        "old_ret",
        "replay_ret",
        "ret_diff",
        "old_position",
        "replay_position",
        "position_diff",
        "old_tqqq_weight",
        "replay_tqqq_weight",
        "old_qqq_weight",
        "replay_qqq_weight",
        "old_turnover",
        "replay_turnover",
        "old_equity",
        "replay_equity",
    }
    assert expected.issubset(first_divergence.columns)
