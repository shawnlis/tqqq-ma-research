from pathlib import Path

import pandas as pd

from research.cli import main


def _write_run_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _rows() -> list[dict]:
    return [
        {"Date": "2020-01-01", "ret": 0.0, "equity": 1.0, "position": 0.0, "turnover": 0.0},
        {"Date": "2020-01-02", "ret": 0.01, "equity": 1.01, "position": 1.0, "turnover": 1.0},
        {"Date": "2020-01-03", "ret": -0.02, "equity": 0.9898, "position": 1.0, "turnover": 0.0},
    ]


def test_compare_run_identical_files_pass(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    _write_run_csv(old, _rows())
    _write_run_csv(new, _rows())
    output_dir = tmp_path / "comparison"

    assert main(
        [
            "compare-run",
            "--old",
            str(old),
            "--new",
            str(new),
            "--tolerance",
            "1e-8",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = pd.read_csv(output_dir / "compare_run_summary.csv")
    assert summary.loc[summary["check"] == "overall", "passed"].iloc[0]
    assert (output_dir / "compare_run_report.md").exists()


def test_compare_run_changed_return_fails(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    _write_run_csv(old, _rows())
    changed = _rows()
    changed[1]["ret"] = 0.02
    _write_run_csv(new, changed)
    output_dir = tmp_path / "comparison"

    assert main(
        [
            "compare-run",
            "--old",
            str(old),
            "--new",
            str(new),
            "--tolerance",
            "1e-8",
            "--output-dir",
            str(output_dir),
        ]
    ) == 1

    summary = pd.read_csv(output_dir / "compare_run_summary.csv")
    ret_row = summary[summary["check"] == "ret"].iloc[0]
    assert not bool(ret_row["passed"])
    assert ret_row["max_abs_diff"] > 1e-8
    assert ret_row["failure_type"] == "numeric_mismatch"


def test_compare_run_missing_dates_fail(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    _write_run_csv(old, _rows())
    _write_run_csv(new, _rows()[:-1])
    output_dir = tmp_path / "comparison"

    assert main(
        [
            "compare-run",
            "--old",
            str(old),
            "--new",
            str(new),
            "--tolerance",
            "1e-8",
            "--output-dir",
            str(output_dir),
        ]
    ) == 1

    summary = pd.read_csv(output_dir / "compare_run_summary.csv")
    date_row = summary[summary["check"] == "date_index"].iloc[0]
    row_count = summary[summary["check"] == "row_count"].iloc[0]
    assert not bool(date_row["passed"])
    assert not bool(row_count["passed"])
    assert date_row["failure_type"] == "date_range_mismatch"
    assert row_count["failure_type"] == "date_range_mismatch"
