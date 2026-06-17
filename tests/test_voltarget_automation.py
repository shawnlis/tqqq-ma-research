from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from research.cli import main
from research.voltarget_monitor_health import check_voltarget_monitor_health


def _write_health_fixture(tmp_path: Path, *, stale_days: int = 0, ledger_exists: bool = True) -> tuple[Path, Path, Path, Path]:
    live = tmp_path / "outputs" / "live_signal"
    drift = tmp_path / "outputs" / "execution_drift"
    reconcile = tmp_path / "outputs" / "paper_reconciliation"
    repo_root = tmp_path / "repo"
    for path in (live, drift, reconcile, repo_root / "research", repo_root / "scripts", repo_root / "automation"):
        path.mkdir(parents=True, exist_ok=True)
    (repo_root / "research" / "placeholder.py").write_text("print('paper monitor')\n", encoding="utf-8")
    (repo_root / "scripts" / "placeholder.template").write_text("PAPER MONITORING ONLY\n", encoding="utf-8")
    (repo_root / "automation" / "placeholder.template").write_text("PAPER MONITORING ONLY\n", encoding="utf-8")

    (live / "signal_today.json").write_text(
        json.dumps({"latest_price_date": "2026-06-12", "classification": "crash_control_candidate"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "latest_price_date": "2026-06-12",
                "as_of_date": "2026-06-17",
                "stale_days": stale_days,
                "stale_warning_days": 5,
                "status": "ok" if stale_days == 0 else "stale_data_warning",
                "warning": "",
            }
        ]
    ).to_csv(live / "data_quality_report.csv", index=False)
    (live / "latest_run_status.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (live / "financing_cost_report.md").write_text("# financing\n", encoding="utf-8")
    pd.DataFrame([{"date": "2026-06-12", "daily_financing_cost": 0.0}]).to_csv(
        live / "financing_cost_history.csv",
        index=False,
    )
    (drift / "execution_drift_report.md").write_text("# drift\n", encoding="utf-8")
    pd.DataFrame([{"signal_date": "2026-06-12"}]).to_csv(drift / "execution_drift_summary.csv", index=False)
    ledger = tmp_path / "data" / "paper_trading" / "voltarget_paper_trades.csv"
    if ledger_exists:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text("date,signal_date\n", encoding="utf-8")
        (reconcile / "paper_trade_reconciliation_report.md").write_text("# reconcile\n", encoding="utf-8")
        pd.DataFrame([{"signal_date": "2026-06-12"}]).to_csv(
            reconcile / "paper_trade_reconciliation.csv",
            index=False,
        )
    return live, drift, reconcile, repo_root


def test_daily_script_template_contains_no_broker_or_order_command() -> None:
    text = Path("scripts/run_voltarget_daily_monitor.ps1.template").read_text(encoding="utf-8").lower()
    forbidden_commands = ["submit_order", "place_order", "create_order", "ib_insync", "broker_api", "auto_trade"]
    assert all(command not in text for command in forbidden_commands)
    assert "paper monitoring only" in text
    assert "no auto-trading" in text


def test_scheduler_templates_reference_correct_powershell_scripts() -> None:
    daily = Path("automation/windows_task_voltarget_daily.xml.template").read_text(encoding="utf-8")
    weekly = Path("automation/windows_task_voltarget_weekly.xml.template").read_text(encoding="utf-8")
    assert "powershell.exe" in daily
    assert "-ExecutionPolicy Bypass -File" in daily
    assert "scripts\\run_voltarget_daily_monitor.ps1" in daily
    assert "<Monday />" in daily and "<Friday />" in daily
    assert "T07:00:00" in daily
    assert "scripts\\run_voltarget_weekly_review.ps1" in weekly
    assert "<Saturday />" in weekly
    assert "T09:00:00" in weekly


def test_health_check_passes_with_fixture_outputs(tmp_path: Path) -> None:
    live, drift, reconcile, repo_root = _write_health_fixture(tmp_path)
    result = check_voltarget_monitor_health(
        output_dir=live,
        ledger_path=tmp_path / "data" / "paper_trading" / "voltarget_paper_trades.csv",
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )
    assert result.overall_status == "ok"
    assert result.summary_path.exists()
    assert result.report_path.exists()


def test_health_check_warns_on_stale_data(tmp_path: Path) -> None:
    live, drift, reconcile, repo_root = _write_health_fixture(tmp_path, stale_days=6)
    result = check_voltarget_monitor_health(
        output_dir=live,
        ledger_path=tmp_path / "data" / "paper_trading" / "voltarget_paper_trades.csv",
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )
    assert result.overall_status == "warning"
    row = result.summary.set_index("check").loc["stale_data_days"]
    assert row["status"] == "warning"


def test_health_check_warns_if_ledger_missing_without_hard_fail(tmp_path: Path) -> None:
    live, drift, reconcile, repo_root = _write_health_fixture(tmp_path, ledger_exists=False)
    result = check_voltarget_monitor_health(
        output_dir=live,
        ledger_path=tmp_path / "data" / "paper_trading" / "voltarget_paper_trades.csv",
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )
    row = result.summary.set_index("check").loc["ledger_reconciliation_status"]
    assert result.overall_status == "warning"
    assert row["status"] == "warning"
    assert not bool(row["hard_fail"])


def test_health_check_cli_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    live, _, _, repo_root = _write_health_fixture(tmp_path, ledger_exists=False)
    monkeypatch.chdir(repo_root)
    assert main(["check-voltarget-monitor-health", "--output-dir", str(live)]) == 0
    assert (live / "monitor_health_summary.csv").exists()
    assert (live / "monitor_health_report.md").exists()


def test_daily_template_simulated_run_writes_latest_automation_status(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path("scripts/run_voltarget_daily_monitor.ps1.template")
    target = scripts / "run_voltarget_daily_monitor.ps1"
    shutil.copyfile(source, target)
    env = os.environ.copy()
    env["VOLTARGET_MONITOR_SIMULATE"] = "1"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(target)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    status_path = tmp_path / "outputs" / "live_signal" / "latest_automation_status.json"
    assert status_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    assert status["paper_trading_only"] is True
    assert status["no_auto_trading"] is True
    assert status["status"] == "ok"


def test_no_generated_automation_files_are_committed() -> None:
    assert not Path("scripts/run_voltarget_daily_monitor.ps1").exists()
    assert not Path("scripts/run_voltarget_weekly_review.ps1").exists()
    assert not Path("automation/windows_task_voltarget_daily.xml").exists()
    assert not Path("automation/windows_task_voltarget_weekly.xml").exists()
