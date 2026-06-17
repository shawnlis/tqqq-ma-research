from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .voltarget_live_monitor import load_monitor_config


FORBIDDEN_AUTOMATION_TOKENS = (
    "broker_api",
    "submit_order",
    "place_order",
    "create_order",
    "auto_trade",
    "ib_insync",
    "rebalance_account",
)


@dataclass(frozen=True)
class MonitorHealthResult:
    output_dir: Path
    summary_path: Path
    report_path: Path
    summary: pd.DataFrame
    overall_status: str


def _add(rows: List[Dict[str, object]], check: str, status: str, detail: str = "", hard_fail: bool = False) -> None:
    rows.append(
        {
            "check": check,
            "status": status,
            "detail": detail,
            "hard_fail": bool(hard_fail),
        }
    )


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_forbidden_tokens(paths: Iterable[Path]) -> List[str]:
    hits: List[str] = []
    for root in paths:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in files:
            if path.name == "voltarget_monitor_health.py":
                continue
            if path.suffix.lower() in {".pyc", ".png", ".csv", ".json"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            for token in FORBIDDEN_AUTOMATION_TOKENS:
                if token.lower() in text:
                    hits.append(f"{path}:{token}")
    return hits


def _overall_status(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "failed"
    if summary["hard_fail"].astype(bool).any():
        return "failed"
    if summary["status"].astype(str).eq("warning").any():
        return "warning"
    return "ok"


def _write_report(path: Path, summary: pd.DataFrame, overall_status: str) -> None:
    lines = [
        "# VolTarget Monitor Health Check",
        "",
        "**paper trading only**",
        "",
        f"- Overall status: `{overall_status}`",
        "",
        "## Checks",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Boundary",
        "The health check is read-only. It does not place trades, call broker APIs, or enable auto-trading.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def check_voltarget_monitor_health(
    *,
    output_dir: Path,
    config_path: Path = Path("configs/voltarget_live_monitor.yaml"),
    ledger_path: Path = Path("data/paper_trading/voltarget_paper_trades.csv"),
    auto_paper_ledger_dir: Path = Path("outputs/auto_paper_ledger"),
    execution_drift_dir: Path = Path("outputs/execution_drift"),
    paper_reconciliation_dir: Path = Path("outputs/paper_reconciliation"),
    repo_root: Path = Path("."),
) -> MonitorHealthResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    if Path(config_path).exists():
        config = load_monitor_config(Path(config_path))
    else:
        config = {"auto_paper_ledger_enabled": False, "manual_ledger_required": True}
    auto_ledger_enabled = bool(config.get("auto_paper_ledger_enabled", False))
    manual_ledger_required = bool(config.get("manual_ledger_required", True))

    signal_path = output_dir / "signal_today.json"
    if signal_path.exists():
        signal = _read_json(signal_path)
        _add(rows, "latest_signal_exists", "ok", str(signal_path))
        _add(rows, "latest_signal_date", "ok", str(signal.get("latest_price_date", "")))
    else:
        _add(rows, "latest_signal_exists", "failed", f"missing {signal_path}", hard_fail=True)
        signal = {}

    quality_path = output_dir / "data_quality_report.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        if quality.empty:
            _add(rows, "stale_data_days", "warning", "data_quality_report.csv is empty")
        else:
            stale_days = int(pd.to_numeric(quality.iloc[0].get("stale_days", 0), errors="coerce"))
            quality_status = str(quality.iloc[0].get("status", ""))
            status = "warning" if quality_status != "ok" else "ok"
            _add(rows, "stale_data_days", status, f"stale_days={stale_days}; status={quality_status}")
    else:
        _add(rows, "stale_data_days", "warning", f"missing {quality_path}")

    run_status_path = output_dir / "latest_run_status.json"
    if run_status_path.exists():
        run_status = _read_json(run_status_path)
        status_value = str(run_status.get("status", ""))
        status = "ok" if status_value == "ok" else "warning"
        _add(rows, "latest_run_status", status, status_value)
    else:
        _add(rows, "latest_run_status", "warning", f"missing {run_status_path}")

    drift_report = Path(execution_drift_dir) / "execution_drift_report.md"
    drift_summary = Path(execution_drift_dir) / "execution_drift_summary.csv"
    if drift_report.exists() and drift_summary.exists():
        _add(rows, "execution_drift_status", "ok", str(drift_report))
    else:
        _add(rows, "execution_drift_status", "warning", f"missing {drift_report} or {drift_summary}")

    financing_report = output_dir / "financing_cost_report.md"
    financing_history = output_dir / "financing_cost_history.csv"
    if financing_report.exists() and financing_history.exists():
        _add(rows, "financing_report_status", "ok", str(financing_report))
    else:
        _add(rows, "financing_report_status", "warning", f"missing {financing_report} or {financing_history}")

    if auto_ledger_enabled and not manual_ledger_required:
        auto_ledger_path = Path(auto_paper_ledger_dir) / "auto_paper_ledger.csv"
        if auto_ledger_path.exists():
            auto_ledger = pd.read_csv(auto_ledger_path)
            if auto_ledger.empty:
                _add(rows, "auto_paper_ledger_exists", "warning", f"empty {auto_ledger_path}")
            else:
                latest = auto_ledger.iloc[-1]
                latest_status = str(latest.get("status", ""))
                status = "ok" if latest_status == "ok" else "warning"
                _add(rows, "auto_paper_ledger_exists", "ok", str(auto_ledger_path))
                _add(rows, "auto_paper_latest_row_date", "ok", str(latest.get("signal_date", "")))
                _add(rows, "auto_paper_latest_status", status, latest_status)
                pending_count = int(auto_ledger["status"].astype(str).eq("pending_fill").sum())
                stale_count = int(auto_ledger["status"].astype(str).eq("stale_data").sum())
                failed_count = int(auto_ledger["status"].astype(str).eq("failed").sum())
                _add(rows, "auto_paper_pending_fills", "warning" if pending_count else "ok", f"pending_fill_count={pending_count}")
                _add(rows, "auto_paper_stale_data", "warning" if stale_count else "ok", f"stale_data_count={stale_count}")
                _add(rows, "auto_paper_failed_rows", "failed" if failed_count else "ok", f"failed_count={failed_count}", hard_fail=bool(failed_count))
                paper_equity = pd.to_numeric(auto_ledger.get("paper_equity"), errors="coerce")
                relative = pd.to_numeric(auto_ledger.get("relative_equity_vs_tqqq"), errors="coerce")
                _add(
                    rows,
                    "auto_paper_equity_updated",
                    "ok" if paper_equity.notna().any() else "warning",
                    f"latest_paper_equity={latest.get('paper_equity', '')}",
                )
                _add(
                    rows,
                    "auto_paper_relative_equity_updated",
                    "ok" if relative.notna().any() else "warning",
                    f"latest_relative_equity_vs_tqqq={latest.get('relative_equity_vs_tqqq', '')}",
                )
        else:
            _add(rows, "auto_paper_ledger_exists", "warning", f"missing {auto_ledger_path}")
        _add(rows, "ledger_reconciliation_status", "ok", "manual ledger not required in auto paper ledger mode")
    else:
        ledger_path = Path(ledger_path)
        reconciliation_report = Path(paper_reconciliation_dir) / "paper_trade_reconciliation_report.md"
        reconciliation_csv = Path(paper_reconciliation_dir) / "paper_trade_reconciliation.csv"
        if ledger_path.exists():
            if reconciliation_report.exists() and reconciliation_csv.exists():
                _add(rows, "ledger_reconciliation_status", "ok", str(reconciliation_report))
            else:
                _add(rows, "ledger_reconciliation_status", "warning", "ledger exists but reconciliation output is missing")
        else:
            _add(rows, "ledger_reconciliation_status", "warning", f"ledger missing: {ledger_path}")

    repo_root = Path(repo_root)
    hits = _scan_forbidden_tokens([repo_root / "research", repo_root / "scripts", repo_root / "automation"])
    if hits:
        _add(rows, "no_broker_or_autotrading_code", "failed", ";".join(hits), hard_fail=True)
    else:
        _add(rows, "no_broker_or_autotrading_code", "ok", "no forbidden automation tokens detected")

    summary = pd.DataFrame(rows)
    overall = _overall_status(summary)
    summary_path = output_dir / "monitor_health_summary.csv"
    report_path = output_dir / "monitor_health_report.md"
    summary.to_csv(summary_path, index=False)
    _write_report(report_path, summary, overall)
    return MonitorHealthResult(
        output_dir=output_dir,
        summary_path=summary_path,
        report_path=report_path,
        summary=summary,
        overall_status=overall,
    )
