from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.cli import main


def _price_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.cumprod(1.0 + returns)


def _write_open_question_prices(path: Path) -> None:
    dates = pd.bdate_range("1999-03-10", "2019-12-31")
    t = np.arange(len(dates), dtype=float)
    base_ret = 0.00028 + 0.0025 * np.sin(t / 17.0) + 0.0015 * np.cos(t / 53.0)
    spy_ret = 0.00022 + 0.0018 * np.sin(t / 23.0)
    semi_ret = 0.00035 + 0.0030 * np.sin(t / 13.0)
    bond_ret = 0.00005 + 0.0006 * np.cos(t / 31.0)

    prices = pd.DataFrame(
        {
            "Date": dates,
            "QQQ": _price_from_returns(base_ret),
            "TQQQ": _price_from_returns(3.0 * base_ret - 0.00004),
            "SPY": _price_from_returns(spy_ret),
            "UPRO": _price_from_returns(3.0 * spy_ret - 0.00004),
            "RSP": _price_from_returns(spy_ret * 0.95 + 0.00002),
            "IWM": _price_from_returns(spy_ret * 1.10 - 0.00001),
            "HYG": _price_from_returns(0.35 * spy_ret + 0.00005),
            "LQD": _price_from_returns(bond_ret + 0.00003),
            "IEF": _price_from_returns(bond_ret),
            "TLT": _price_from_returns(1.6 * bond_ret),
            "XLK": _price_from_returns(base_ret * 1.05),
            "SMH": _price_from_returns(semi_ret),
            "SOXX": _price_from_returns(semi_ret * 0.98),
            "SOXL": _price_from_returns(3.0 * semi_ret - 0.00005),
            "SGOV": _price_from_returns(np.full(len(dates), 0.00003)),
            "BIL": _price_from_returns(np.full(len(dates), 0.000025)),
            "SHY": _price_from_returns(np.full(len(dates), 0.00004) + 0.00015 * np.sin(t / 67.0)),
            "GLD": _price_from_returns(0.00008 + 0.0012 * np.cos(t / 29.0)),
            "^VIX": 20.0 + 6.0 * np.sin(t / 41.0),
        }
    )
    prices.to_csv(path, index=False)


def test_run_open_questions_pack_writes_question_outputs(tmp_path: Path) -> None:
    data_csv = tmp_path / "prices.csv"
    _write_open_question_prices(data_csv)
    config_path = tmp_path / "open_questions.yaml"
    output_dir = tmp_path / "open_questions_out"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pack_name": "tiny_open_questions",
                "benchmark_symbol": "TQQQ",
                "output_dir": str(output_dir),
                "data_csv": str(data_csv),
                "start_date": "1999-03-10",
                "end_date": "2019-12-31",
                "transaction_cost_bps": 0.0,
                "train_years": 5,
                "test_years": 1,
                "objective": "objective_final_ratio",
                "core_exposure_values": [0.25, 1.0],
                "risk_off_symbols": ["CASH", "QQQ"],
                "adaptive_max_exposures": [1.0, 1.5],
                "soxl_variants": ["soxl_core_overlay", "rotation_tqqq_soxl_upro"],
                "synthetic_periods": [
                    {
                        "label": "real_tqqq_period",
                        "start_date": "2011-01-01",
                        "end_date": "2019-12-31",
                        "use_synthetic": False,
                    },
                    {
                        "label": "synthetic_full_qqq_history",
                        "start_date": "1999-03-10",
                        "end_date": "2019-12-31",
                        "use_synthetic": True,
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
                "q8_core_exposures": [0.50, 0.80],
                "q8_overlay_maxes": [0.35],
                "bootstrap_iterations": 5,
            }
        ),
        encoding="utf-8",
    )

    assert main(["run-open-questions", str(config_path)]) == 0

    expected_files = [
        "open_questions_summary.csv",
        "question_1_exposure_floor.csv",
        "question_2_rebound.csv",
        "question_3_risk_off.csv",
        "question_4_adaptive_leverage.csv",
        "question_5_market_internals.csv",
        "question_6_soxl.csv",
        "question_7_synthetic_history.csv",
        "question_8_overfit.csv",
        "open_questions_report.md",
        "run_config.json",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()

    summary = pd.read_csv(output_dir / "open_questions_summary.csv")
    assert set(summary["question_id"]) == set(range(1, 9))
    assert set(summary["answer"]).issubset({"yes", "no", "inconclusive"})
    assert "final_equity_ratio_versus_tqqq" in summary.columns
    assert "best_candidate_config" in summary.columns

    q3 = pd.read_csv(output_dir / "question_3_risk_off.csv")
    assert set(q3["risk_off_symbol"]) == {"CASH", "QQQ"}

    q8 = pd.read_csv(output_dir / "question_8_overfit.csv")
    assert "validation_summary" in set(q8["record_type"].dropna())
    assert (output_dir / "question_8_validation_artifacts" / "validation_summary.csv").exists()


def test_run_open_questions_dry_run_writes_estimates_without_case_outputs(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "open_questions_dry.yaml"
    output_dir = tmp_path / "open_questions_dry_out"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pack_name": "dry_open_questions",
                "benchmark_symbol": "TQQQ",
                "output_dir": str(output_dir),
                "start_date": "2011-01-01",
                "end_date": "2019-12-31",
                "run_questions": [1],
                "core_exposure_values": [0.25, 0.50, 1.0],
            }
        ),
        encoding="utf-8",
    )

    assert main(["run-open-questions", str(config_path), "--dry-run"]) == 0

    captured = capsys.readouterr()
    assert "dry_run" in captured.out
    summary = pd.read_csv(output_dir / "open_questions_summary.csv")
    assert summary.loc[summary["question_id"] == 1, "dry_run_experiment_count"].iloc[0] == 3
    q1 = pd.read_csv(output_dir / "question_1_exposure_floor.csv")
    assert set(q1["status"]) == {"dry_run"}
    assert "estimated_runtime_seconds" in q1.columns
    assert not (output_dir / "case_outputs").exists()


def test_run_open_questions_max_configs_skips_after_limit(tmp_path: Path) -> None:
    data_csv = tmp_path / "prices.csv"
    _write_open_question_prices(data_csv)
    config_path = tmp_path / "open_questions_max.yaml"
    output_dir = tmp_path / "open_questions_max_out"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pack_name": "max_open_questions",
                "benchmark_symbol": "TQQQ",
                "output_dir": str(output_dir),
                "data_csv": str(data_csv),
                "start_date": "2011-01-01",
                "end_date": "2019-12-31",
                "transaction_cost_bps": 0.0,
                "train_years": 5,
                "test_years": 1,
                "run_questions": [1],
                "core_exposure_values": [0.25, 0.50, 0.75, 1.0],
            }
        ),
        encoding="utf-8",
    )

    assert main(["run-open-questions", str(config_path), "--max-configs", "2"]) == 0

    q1 = pd.read_csv(output_dir / "question_1_exposure_floor.csv")
    assert int((q1["status"] == "ok").sum()) == 2
    assert int((q1["status"] == "skipped_max_configs").sum()) == 2
    summary = pd.read_csv(output_dir / "open_questions_summary.csv")
    row = summary.loc[summary["question_id"] == 1].iloc[0]
    assert row["successful_experiment_count"] == 2
    assert row["skipped_experiment_count"] == 2
