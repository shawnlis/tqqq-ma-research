from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.cli import main
from research import voltarget_daily_monitor
from research.voltarget_daily_monitor import run_automated_voltarget_paper_monitor
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
    (live / "latest_automation_status.json").write_text(json.dumps({"status": "ok", "message": "ok"}), encoding="utf-8")
    (live / "latest_scheduler_status.json").write_text(json.dumps({"status": "ok", "message": "ok"}), encoding="utf-8")
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


def _write_auto_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "input_dir": "unused",
                "cache_dir": "./price_cache",
                "execution_assumption": "next_open_to_next_open",
                "transaction_cost_bps": 10.0,
                "slippage_bps": 0.0,
                "paper_ledger_mode": "auto",
                "paper_starting_equity": 100000,
                "paper_execution_model": "next_open_to_next_open",
                "paper_fill_price_source": "next_open",
                "fallback_fill_price_source": "latest_close",
                "assumed_slippage_bps": 5.0,
                "assumed_transaction_cost_bps": 10.0,
                "financing_annual_cost": 0.06,
                "annual_financing_rate_assumption": 0.06,
                "financing_applies_above_exposure": 1.0,
                "financing_day_count_basis": 252,
                "stale_data_warning_days": 5,
                "max_allowed_stale_days": 1,
                "target_symbol": "TQQQ",
                "benchmark_symbol": "TQQQ",
                "classification": "crash_control_candidate",
                "production_ready": False,
                "paper_trading_only": True,
                "auto_paper_ledger_enabled": True,
                "manual_ledger_required": False,
                "paper_ledger_start_mode": "historical_backfill",
                "paper_ledger_allow_historical_backfill": True,
                "paper_ledger_reset_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_auto_ledger(output_dir: Path, *, status: str = "ok", ledger_mode: str = "historical_backfill") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "signal_date": "2026-06-12",
                "execution_date": "2026-06-15",
                "status": status,
                "target_exposure": 1.1,
                "paper_equity": 100500.0,
                "relative_equity_vs_tqqq": 1.01,
                "ledger_mode": ledger_mode,
                "paper_ledger_start_date": "2026-06-12" if ledger_mode == "live_monitor" else "",
                "historical_backfill": ledger_mode == "historical_backfill",
                "live_ledger_initialized": ledger_mode == "live_monitor",
                "paper_trading_only": True,
                "no_broker_integration": True,
                "no_auto_trading": True,
            }
        ]
    ).to_csv(output_dir / "auto_paper_ledger.csv", index=False)


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
        config_path=tmp_path / "missing_config.yaml",
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
        config_path=tmp_path / "missing_config.yaml",
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
        config_path=tmp_path / "missing_config.yaml",
        ledger_path=tmp_path / "data" / "paper_trading" / "voltarget_paper_trades.csv",
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )
    row = result.summary.set_index("check").loc["ledger_reconciliation_status"]
    assert result.overall_status == "warning"
    assert row["status"] == "warning"
    assert not bool(row["hard_fail"])


def test_health_check_does_not_warn_on_missing_manual_ledger_in_auto_mode(tmp_path: Path) -> None:
    live, drift, reconcile, repo_root = _write_health_fixture(tmp_path, ledger_exists=False)
    config = _write_auto_config(tmp_path / "config.yaml")
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["paper_ledger_start_mode"] = "live_from_config_date"
    config_data["paper_ledger_allow_historical_backfill"] = False
    config.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    auto_dir = tmp_path / "outputs" / "auto_paper_ledger"
    _write_auto_ledger(auto_dir, ledger_mode="live_monitor")
    (auto_dir / "auto_paper_ledger_state.json").write_text(
        json.dumps({"ledger_mode": "live_monitor", "paper_ledger_start_date": "2026-06-12", "starting_equity": 100000}),
        encoding="utf-8",
    )

    result = check_voltarget_monitor_health(
        output_dir=live,
        config_path=config,
        ledger_path=tmp_path / "data" / "paper_trading" / "voltarget_paper_trades.csv",
        auto_paper_ledger_dir=auto_dir,
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )

    row = result.summary.set_index("check").loc["ledger_reconciliation_status"]
    assert result.overall_status == "ok"
    assert row["status"] == "ok"
    assert "manual ledger not required" in row["detail"]


def test_health_check_warns_if_auto_paper_ledger_is_stale(tmp_path: Path) -> None:
    live, drift, reconcile, repo_root = _write_health_fixture(tmp_path)
    config = _write_auto_config(tmp_path / "config.yaml")
    auto_dir = tmp_path / "outputs" / "auto_paper_ledger"
    _write_auto_ledger(auto_dir, status="stale_data")

    result = check_voltarget_monitor_health(
        output_dir=live,
        config_path=config,
        auto_paper_ledger_dir=auto_dir,
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )

    row = result.summary.set_index("check").loc["auto_paper_latest_status"]
    assert result.overall_status == "warning"
    assert row["status"] == "warning"


def test_health_check_fails_if_latest_automation_status_failed(tmp_path: Path) -> None:
    live, drift, reconcile, repo_root = _write_health_fixture(tmp_path)
    (live / "latest_automation_status.json").write_text(
        json.dumps({"status": "failed", "message": "wrapper failed"}),
        encoding="utf-8",
    )

    result = check_voltarget_monitor_health(
        output_dir=live,
        config_path=tmp_path / "missing_config.yaml",
        ledger_path=tmp_path / "data" / "paper_trading" / "voltarget_paper_trades.csv",
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )

    row = result.summary.set_index("check").loc["latest_automation_status"]
    assert result.overall_status == "failed"
    assert row["status"] == "failed"
    assert bool(row["hard_fail"])


def test_health_check_fails_if_latest_scheduler_status_failed(tmp_path: Path) -> None:
    live, drift, reconcile, repo_root = _write_health_fixture(tmp_path)
    (live / "latest_scheduler_status.json").write_text(
        json.dumps({"status": "failed", "message": "wrapper failed"}),
        encoding="utf-8",
    )

    result = check_voltarget_monitor_health(
        output_dir=live,
        config_path=tmp_path / "missing_config.yaml",
        ledger_path=tmp_path / "data" / "paper_trading" / "voltarget_paper_trades.csv",
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )

    row = result.summary.set_index("check").loc["latest_scheduler_status"]
    assert result.overall_status == "failed"
    assert row["status"] == "failed"
    assert bool(row["hard_fail"])


def test_health_check_distinguishes_live_monitor_from_historical_backfill(tmp_path: Path) -> None:
    live, drift, reconcile, repo_root = _write_health_fixture(tmp_path)
    config = _write_auto_config(tmp_path / "config.yaml")
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["paper_ledger_start_mode"] = "live_from_config_date"
    config_data["paper_ledger_allow_historical_backfill"] = False
    config.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    auto_dir = tmp_path / "outputs" / "auto_paper_ledger"
    _write_auto_ledger(auto_dir, ledger_mode="live_monitor")
    (auto_dir / "auto_paper_ledger_state.json").write_text(
        json.dumps({"ledger_mode": "live_monitor", "paper_ledger_start_date": "2026-06-12", "starting_equity": 100000}),
        encoding="utf-8",
    )

    result = check_voltarget_monitor_health(
        output_dir=live,
        config_path=config,
        auto_paper_ledger_dir=auto_dir,
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )
    by_check = result.summary.set_index("check")
    assert by_check.loc["current_ledger_kind", "detail"] == "live_monitor"
    assert by_check.loc["live_ledger_initialized", "status"] == "ok"

    _write_auto_ledger(auto_dir, ledger_mode="historical_backfill")
    result = check_voltarget_monitor_health(
        output_dir=live,
        config_path=config,
        auto_paper_ledger_dir=auto_dir,
        execution_drift_dir=drift,
        paper_reconciliation_dir=reconcile,
        repo_root=repo_root,
    )
    row = result.summary.set_index("check").loc["current_ledger_kind"]
    assert row["detail"] == "historical_backfill"
    assert row["status"] == "warning"


def test_health_check_cli_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    live, _, _, repo_root = _write_health_fixture(tmp_path, ledger_exists=False)
    monkeypatch.chdir(repo_root)
    assert main(["check-voltarget-monitor-health", "--output-dir", str(live)]) == 0
    assert (live / "monitor_health_summary.csv").exists()
    assert (live / "monitor_health_report.md").exists()


def test_daily_template_simulated_run_writes_latest_scheduler_status(tmp_path: Path) -> None:
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
    scheduler_status_path = tmp_path / "outputs" / "live_signal" / "latest_scheduler_status.json"
    assert not status_path.exists()
    assert scheduler_status_path.exists()
    status = json.loads(scheduler_status_path.read_text(encoding="utf-8-sig"))
    assert status["paper_trading_only"] is True
    assert status["no_auto_trading"] is True
    assert status["status"] == "ok"


def test_daily_template_allows_native_stderr_when_exit_code_is_zero(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path("scripts/run_voltarget_daily_monitor.ps1.template")
    target = scripts / "run_voltarget_daily_monitor.ps1"
    shutil.copyfile(source, target)
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    (fake_bin / "python.cmd").write_text(
        "@echo off\n"
        "echo native info log from stderr 1>&2\n"
        "exit /b 0\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(target)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    status_path = tmp_path / "outputs" / "live_signal" / "latest_scheduler_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    assert status["status"] == "ok"
    log_text = next((tmp_path / "logs" / "voltarget_monitor").glob("voltarget_daily_*.log")).read_text(
        encoding="utf-8",
    )
    assert "native info log from stderr" in log_text


def test_one_command_workflow_calls_expected_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    config = _write_auto_config(tmp_path / "config.yaml")

    def fake_daily(**_: object) -> SimpleNamespace:
        calls.append("daily")
        return SimpleNamespace(
            status_path=tmp_path / "live" / "latest_run_status.json",
            log_path=tmp_path / "live" / "daily_run_log.csv",
            status={"status": "ok", "steps": {"risk_policy_dashboard": {"status": "ok"}}},
        )

    def fake_ledger(**_: object) -> SimpleNamespace:
        calls.append("ledger")
        latest_row = {
            "status": "ok",
            "target_exposure": 1.2,
            "paper_equity": 101000.0,
            "relative_equity_vs_tqqq": 1.01,
        }
        return SimpleNamespace(
            latest_status="ok",
            latest_row=latest_row,
            ledger_path=tmp_path / "auto" / "auto_paper_ledger.csv",
            summary_path=tmp_path / "auto" / "auto_paper_summary.csv",
            report_path=tmp_path / "auto" / "auto_paper_report.md",
        )

    def fake_drift(**_: object) -> SimpleNamespace:
        calls.append("drift")
        return SimpleNamespace(
            summary_path=tmp_path / "drift" / "execution_drift_summary.csv",
            report_path=tmp_path / "drift" / "execution_drift_report.md",
            metrics={"drift_exceeds_policy_threshold": False},
        )

    def fake_health(**_: object) -> SimpleNamespace:
        calls.append("health")
        return SimpleNamespace(
            overall_status="ok",
            summary_path=tmp_path / "live" / "monitor_health_summary.csv",
            report_path=tmp_path / "live" / "monitor_health_report.md",
        )

    monkeypatch.setattr(voltarget_daily_monitor, "run_daily_voltarget_monitor", fake_daily)
    monkeypatch.setattr(voltarget_daily_monitor, "update_auto_paper_ledger", fake_ledger)
    monkeypatch.setattr(voltarget_daily_monitor, "run_execution_drift_tracking", fake_drift)
    monkeypatch.setattr(voltarget_daily_monitor, "check_voltarget_monitor_health", fake_health)

    result = run_automated_voltarget_paper_monitor(config_path=config, signal_dir=tmp_path / "live")

    assert result.status["status"] == "ok"
    assert calls == ["daily", "ledger", "drift", "health"]
    assert result.status_path.exists()
    assert result.log_path.exists()


def test_no_generated_automation_files_are_committed() -> None:
    assert not Path("scripts/run_voltarget_daily_monitor.ps1").exists()
    assert not Path("scripts/run_voltarget_weekly_review.ps1").exists()
    assert not Path("automation/windows_task_voltarget_daily.xml").exists()
    assert not Path("automation/windows_task_voltarget_weekly.xml").exists()
