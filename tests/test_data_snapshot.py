from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.cli import main


def _write_snapshot_prices(path: Path, *, changed: bool = False) -> None:
    dates = pd.bdate_range("2020-01-01", "2020-01-08")
    aaa = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    if changed:
        aaa[2] = 109.0
    df = pd.DataFrame(
        {
            "Date": dates,
            "AAA": aaa,
            "AAA_OPEN": [value - 0.25 for value in aaa],
            "AAA_HIGH": [value + 0.50 for value in aaa],
            "AAA_LOW": [value - 0.75 for value in aaa],
            "AAA_CLOSE": aaa,
            "AAA_VOLUME": [1000, 1001, 1002, 1003, 1004, 1005],
            "BBB": [50.0, None, 51.0, 52.0, 53.0, 54.0],
        }
    )
    df.to_csv(path, index=False)


def _freeze(tmp_path: Path) -> tuple[Path, Path]:
    data_csv = tmp_path / "prices.csv"
    _write_snapshot_prices(data_csv)
    output_dir = tmp_path / "snapshot"
    assert (
        main(
            [
                "freeze-data-snapshot",
                "--symbols",
                "AAA",
                "BBB",
                "--output-dir",
                str(output_dir),
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-01-08",
                "--data-csv",
                str(data_csv),
            ]
        )
        == 0
    )
    return output_dir, data_csv


def test_freeze_manifest_writes_expected_fields(tmp_path: Path) -> None:
    output_dir, _ = _freeze(tmp_path)

    manifest = json.loads((output_dir / "data_snapshot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 1
    assert manifest["data_loader_version"]
    assert "yfinance_version" in manifest
    assert manifest["auto_adjust"] is True
    assert manifest["data_source"] == "data_csv"
    assert manifest["start_date"] == "2020-01-01"
    assert manifest["end_date"] == "2020-01-08"
    assert set(manifest["symbols"]) == {"AAA", "BBB"}
    assert manifest["per_symbol"]["AAA"]["first_date"] == "2020-01-01"
    assert manifest["per_symbol"]["AAA"]["last_date"] == "2020-01-08"
    assert manifest["per_symbol"]["AAA"]["row_count"] == 6
    assert manifest["per_symbol"]["AAA"]["close_hash_sha256"]
    assert manifest["per_symbol"]["AAA"]["ohlcv_hash_sha256"]
    assert manifest["per_symbol"]["BBB"]["missing_rows_vs_union_calendar"] == 1
    assert (output_dir / "data_snapshot_summary.csv").exists()
    assert (output_dir / "data_snapshot_report.md").exists()


def test_verify_snapshot_passes_when_data_unchanged(tmp_path: Path) -> None:
    output_dir, _ = _freeze(tmp_path)

    assert main(["verify-data-snapshot", str(output_dir / "data_snapshot_manifest.json")]) == 0

    verify = pd.read_csv(output_dir / "data_snapshot_verify_summary.csv")
    assert verify["passed"].all()


def test_verify_snapshot_fails_when_fixture_data_changes(tmp_path: Path) -> None:
    output_dir, data_csv = _freeze(tmp_path)
    _write_snapshot_prices(data_csv, changed=True)

    assert main(["verify-data-snapshot", str(output_dir / "data_snapshot_manifest.json")]) == 1

    verify = pd.read_csv(output_dir / "data_snapshot_verify_summary.csv")
    aaa = verify[verify["symbol"] == "AAA"].iloc[0]
    assert not bool(aaa["passed"])
    assert "close_hash_sha256" in aaa["mismatch_fields"]


def test_verify_report_includes_symbol_level_hash_mismatches(tmp_path: Path) -> None:
    output_dir, data_csv = _freeze(tmp_path)
    _write_snapshot_prices(data_csv, changed=True)

    assert main(["verify-data-snapshot", str(output_dir / "data_snapshot_manifest.json")]) == 1

    report = (output_dir / "data_snapshot_verify_report.md").read_text(encoding="utf-8")
    assert "AAA" in report
    assert "close_hash_sha256" in report
    assert "ohlcv_hash_sha256" in report
