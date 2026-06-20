from pathlib import Path

import pandas as pd

from research.cli import main


def _write_price_cache(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    cache_dir = tmp_path / "price_cache"
    cache_dir.mkdir()
    dates = pd.bdate_range("2020-01-01", "2020-01-10")
    prices = pd.DataFrame(
        {
            "Date": dates,
            "TQQQ": [100.0, 103.0, 101.97, 104.0094, 106.089588, 103.96779624, 105.0074742024, 108.157698428472],
            "QQQ": [100.0, 101.0, 100.495, 101.49995, 102.5149495, 101.489800005, 101.997249005025, 103.01722149507525],
        }
    )
    prices.to_csv(cache_dir / "tqqq_qqq_prices.csv", index=False)
    return cache_dir, prices.set_index("Date")


def _write_old_stitched(
    path: Path,
    prices: pd.DataFrame,
    *,
    change_tqqq_date: str | None = None,
    change_qqq_date: str | None = None,
    include_return_columns: bool = True,
) -> pd.DataFrame:
    old_dates = pd.bdate_range("2020-01-06", "2020-01-10")
    tqqq_ret = prices["TQQQ"].pct_change().loc[old_dates]
    qqq_ret = prices["QQQ"].pct_change().loc[old_dates]
    tqqq_weight = pd.Series([1.0, 0.0, 0.5, 1.0, 0.0], index=old_dates)
    qqq_weight = pd.Series([0.0, 1.0, 0.5, 0.0, 1.0], index=old_dates)
    daily_ret = tqqq_weight * tqqq_ret + qqq_weight * qqq_ret
    old = pd.DataFrame(
        {
            "Date": old_dates,
            "ret": daily_ret.to_numpy(),
            "equity": (1.0 + daily_ret).cumprod().to_numpy(),
            "tqqq_weight": tqqq_weight.to_numpy(),
            "qqq_weight": qqq_weight.to_numpy(),
        }
    )
    if include_return_columns:
        old["daily_ret_tqqq"] = tqqq_ret.to_numpy()
        old["daily_ret_qqq"] = qqq_ret.to_numpy()
        old["daily_ret"] = daily_ret.to_numpy()
    if change_tqqq_date:
        old.loc[old["Date"] == pd.Timestamp(change_tqqq_date), "daily_ret_tqqq"] += 0.001
    if change_qqq_date:
        old.loc[old["Date"] == pd.Timestamp(change_qqq_date), "daily_ret_qqq"] -= 0.001
    old.to_csv(path, index=False)
    return old


def _run_audit(tmp_path: Path, old_path: Path, cache_dir: Path) -> Path:
    output_dir = tmp_path / "audit"
    assert main(
        [
            "audit-baseline-data",
            "--old-stitched",
            str(old_path),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
            "--end-date",
            "2020-01-10",
        ]
    ) == 0
    return output_dir


def test_audit_baseline_data_detects_matching_daily_returns(tmp_path: Path) -> None:
    cache_dir, prices = _write_price_cache(tmp_path)
    old_path = tmp_path / "old_stitched.csv"
    _write_old_stitched(old_path, prices)

    output_dir = _run_audit(tmp_path, old_path, cache_dir)

    summary = pd.read_csv(output_dir / "baseline_data_audit_summary.csv")
    assert summary.loc[0, "current_data_matches_old_baseline_returns"] == "yes"
    assert summary.loc[0, "max_abs_tqqq_return_diff"] <= 1e-8
    assert summary.loc[0, "max_abs_qqq_return_diff"] <= 1e-8
    differences = pd.read_csv(output_dir / "first_return_differences.csv")
    assert differences.empty


def test_audit_baseline_data_detects_changed_tqqq_return(tmp_path: Path) -> None:
    cache_dir, prices = _write_price_cache(tmp_path)
    old_path = tmp_path / "old_stitched.csv"
    _write_old_stitched(old_path, prices, change_tqqq_date="2020-01-07")

    output_dir = _run_audit(tmp_path, old_path, cache_dir)

    summary = pd.read_csv(output_dir / "baseline_data_audit_summary.csv")
    assert summary.loc[0, "current_data_matches_old_baseline_returns"] == "no"
    assert summary.loc[0, "first_tqqq_return_diff_date"] == "2020-01-07"
    assert summary.loc[0, "max_abs_tqqq_return_diff"] > 1e-8
    differences = pd.read_csv(output_dir / "first_return_differences.csv")
    assert differences.loc[0, "date"] == "2020-01-07"
    assert abs(differences.loc[0, "tqqq_return_diff"]) > 1e-8


def test_audit_baseline_data_detects_changed_qqq_return(tmp_path: Path) -> None:
    cache_dir, prices = _write_price_cache(tmp_path)
    old_path = tmp_path / "old_stitched.csv"
    _write_old_stitched(old_path, prices, change_qqq_date="2020-01-08")

    output_dir = _run_audit(tmp_path, old_path, cache_dir)

    summary = pd.read_csv(output_dir / "baseline_data_audit_summary.csv")
    assert summary.loc[0, "current_data_matches_old_baseline_returns"] == "no"
    assert summary.loc[0, "first_qqq_return_diff_date"] == "2020-01-08"
    assert summary.loc[0, "max_abs_qqq_return_diff"] > 1e-8
    differences = pd.read_csv(output_dir / "first_return_differences.csv")
    assert differences.loc[0, "date"] == "2020-01-08"
    assert abs(differences.loc[0, "qqq_return_diff"]) > 1e-8


def test_audit_baseline_data_reports_limited_audit_when_old_returns_absent(tmp_path: Path) -> None:
    cache_dir, prices = _write_price_cache(tmp_path)
    old_path = tmp_path / "old_stitched.csv"
    _write_old_stitched(old_path, prices, include_return_columns=False)

    output_dir = _run_audit(tmp_path, old_path, cache_dir)

    summary = pd.read_csv(output_dir / "baseline_data_audit_summary.csv")
    assert summary.loc[0, "current_data_matches_old_baseline_returns"] == "limited"
    assert bool(summary.loc[0, "return_level_audit_limited"])
    assert "return-level data audit is limited" in summary.loc[0, "warnings"]
