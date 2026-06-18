from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .data_snapshot import verify_data_snapshot
from .auto_paper_ledger import update_auto_paper_ledger
from .execution_drift import run_execution_drift_tracking
from .market_data_refresh import refresh_recent_market_data
from .visualization import generate_equity_visualizations
from .voltarget_live_monitor import build_monitor_frame, generate_voltarget_signal, load_monitor_config
from .voltarget_monitor_health import check_voltarget_monitor_health
from .voltarget_risk_dashboard import build_voltarget_risk_dashboard


DEFAULT_SIGNAL_OUTPUT_DIR = Path("outputs/live_signal")
DEFAULT_VISUALIZATION_OUTPUT_DIR = Path("outputs/visualizations/voltarget_stage3")
DEFAULT_AUTO_LEDGER_OUTPUT_DIR = Path("outputs/auto_paper_ledger")
DEFAULT_EXECUTION_DRIFT_OUTPUT_DIR = Path("outputs/execution_drift")


@dataclass(frozen=True)
class DailyMonitorResult:
    output_dir: Path
    status_path: Path
    log_path: Path
    status: Dict[str, Any]


@dataclass(frozen=True)
class AutomatedPaperMonitorResult:
    output_dir: Path
    status_path: Path
    log_path: Path
    status: Dict[str, Any]


def _utc_now(generated_at: Optional[str]) -> str:
    return generated_at or datetime.now(timezone.utc).isoformat()


def _run_date(as_of_date: Optional[pd.Timestamp]) -> str:
    if as_of_date is not None:
        return pd.Timestamp(as_of_date).date().isoformat()
    return datetime.now(timezone.utc).date().isoformat()


def _append_daily_log(path: Path, row: Dict[str, Any]) -> pd.DataFrame:
    new_row = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new_row], ignore_index=True)
    else:
        combined = new_row
    combined = combined.sort_values("run_date")
    combined = combined.drop_duplicates(subset=["run_date"], keep="last").reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def _write_status(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_stale_signal_report(path: Path, refresh: Dict[str, Any], freshness: Dict[str, Any]) -> None:
    lines = [
        "# VolTarget Paper Signal Report",
        "",
        "**paper trading only**",
        "",
        "No new accepted signal was generated because market data remained stale after the refresh step.",
        "",
        "## Market Data Refresh",
        f"- Refresh attempted: `{refresh.get('attempted', False)}`",
        f"- Refresh success: `{refresh.get('refresh_success', False)}`",
        f"- Data source: `{refresh.get('source', '')}`",
        f"- Latest price date: `{refresh.get('latest_price_date', freshness.get('latest_price_date', ''))}`",
        f"- Stale days: `{refresh.get('stale_days', freshness.get('stale_days', ''))}`",
        f"- Max allowed stale days: `{refresh.get('max_allowed_stale_days', freshness.get('stale_warning_days', ''))}`",
        f"- Refresh error: `{refresh.get('error', '')}`",
        "",
        "## Status",
        "- Signal status: `stale_data`",
        "- Accepted signal updated: `false`",
        "- Previous `signal_today.json` and `signal_history.csv` were left unchanged.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_data_quality(path: Path, refresh: Dict[str, Any], freshness: Dict[str, Any]) -> None:
    pd.DataFrame(
        [
            {
                "latest_price_date": refresh.get("latest_price_date", freshness.get("latest_price_date", "")),
                "as_of_date": freshness.get("as_of_date", ""),
                "stale_days": refresh.get("stale_days", freshness.get("stale_days", "")),
                "stale_warning_days": refresh.get("max_allowed_stale_days", freshness.get("stale_warning_days", "")),
                "status": "stale_data_warning",
                "warning": freshness.get("warning", "latest data is stale"),
                "refresh_attempted": refresh.get("attempted", False),
                "refresh_success": refresh.get("refresh_success", False),
                "refresh_source": refresh.get("source", ""),
                "refresh_error": refresh.get("error", ""),
            }
        ]
    ).to_csv(path, index=False)


def _append_refresh_to_report(path: Path, refresh: Dict[str, Any]) -> None:
    if not path.exists():
        return
    section = [
        "",
        "## Market Data Refresh",
        f"- Refresh attempted: `{refresh.get('attempted', False)}`",
        f"- Refresh success: `{refresh.get('refresh_success', False)}`",
        f"- Data source: `{refresh.get('source', '')}`",
        f"- Latest price date after refresh: `{refresh.get('latest_price_date', '')}`",
        f"- Stale days after refresh: `{refresh.get('stale_days', '')}`",
        f"- Max allowed stale days: `{refresh.get('max_allowed_stale_days', '')}`",
        f"- Refresh error: `{refresh.get('error', '')}`",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(section))


def check_monitor_data_freshness(
    config: Dict[str, Any],
    *,
    data_csv: Optional[Path] = None,
    as_of_date: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    frame = build_monitor_frame(config, data_csv=data_csv, signal_as_of_date=as_of_date)
    latest_date = pd.Timestamp(frame.index[-1]).normalize()
    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()
    stale_days = int(max((as_of - latest_date).days, 0))
    stale_limit = int(config.get("max_allowed_stale_days", config["stale_data_warning_days"]))
    status = "ok" if stale_days <= stale_limit else "warning"
    return {
        "status": status,
        "latest_price_date": latest_date.date().isoformat(),
        "as_of_date": as_of.date().isoformat(),
        "stale_days": stale_days,
        "stale_warning_days": stale_limit,
        "warning": "" if status == "ok" else f"latest data is {stale_days} calendar days old",
    }


def _verify_snapshot_if_configured(
    config: Dict[str, Any],
    *,
    data_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest = str(config.get("data_snapshot_manifest", "")).strip()
    if not manifest:
        return {"status": "skipped", "configured": False, "manifest_path": ""}
    rows, passed = verify_data_snapshot(
        Path(manifest),
        allow_drift=bool(config.get("data_snapshot_allow_drift", False)),
        cache_dir=str(config.get("cache_dir", "./price_cache")),
        data_csv=data_csv,
    )
    return {
        "status": "ok" if passed else "failed",
        "configured": True,
        "manifest_path": manifest,
        "checked_rows": int(len(rows)),
    }


def _log_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    steps = payload.get("steps", {})
    signal = steps.get("generate_voltarget_signal", {})
    freshness = steps.get("data_freshness_check", {})
    snapshot = steps.get("data_snapshot_verification", {})
    risk = steps.get("risk_policy_dashboard", {})
    visualization = steps.get("visualization_update", {})
    return {
        "run_date": payload["run_date"],
        "generated_at": payload["generated_at"],
        "status": payload["status"],
        "latest_price_date": signal.get("latest_price_date", freshness.get("latest_price_date", "")),
        "data_freshness_status": freshness.get("status", ""),
        "stale_days": freshness.get("stale_days", ""),
        "snapshot_status": snapshot.get("status", ""),
        "signal_status": signal.get("status", ""),
        "risk_dashboard_status": risk.get("status", ""),
        "visualization_status": visualization.get("status", ""),
        "error": payload.get("error", ""),
    }


def _automation_log_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    steps = payload.get("steps", {})
    daily = steps.get("daily_monitor", {})
    ledger = steps.get("auto_paper_ledger", {})
    drift = steps.get("execution_drift", {})
    health = steps.get("health_check", {})
    latest_ledger = ledger.get("latest_row", {}) if isinstance(ledger.get("latest_row", {}), dict) else {}
    return {
        "run_date": payload["run_date"],
        "generated_at": payload["generated_at"],
        "status": payload["status"],
        "daily_monitor_status": daily.get("status", ""),
        "auto_paper_ledger_status": ledger.get("status", ""),
        "execution_drift_status": drift.get("status", ""),
        "health_status": health.get("overall_status", ""),
        "latest_target_exposure": latest_ledger.get("target_exposure", ""),
        "latest_paper_equity": latest_ledger.get("paper_equity", ""),
        "latest_relative_equity_vs_tqqq": latest_ledger.get("relative_equity_vs_tqqq", ""),
        "error": payload.get("error", ""),
    }


def run_daily_voltarget_monitor(
    *,
    config_path: Path,
    output_dir: Optional[Path] = None,
    data_csv: Optional[Path] = None,
    as_of_date: Optional[pd.Timestamp] = None,
    generated_at: Optional[str] = None,
) -> DailyMonitorResult:
    generated = _utc_now(generated_at)
    run_date = _run_date(as_of_date)
    effective_output_dir = Path(output_dir) if output_dir is not None else DEFAULT_SIGNAL_OUTPUT_DIR
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    status_path = effective_output_dir / "latest_run_status.json"
    log_path = effective_output_dir / "daily_run_log.csv"

    payload: Dict[str, Any] = {
        "generated_at": generated,
        "run_date": run_date,
        "status": "failed",
        "paper_trading_only": True,
        "production_ready": False,
        "steps": {},
        "outputs": {
            "latest_run_status": str(status_path),
            "daily_run_log": str(log_path),
        },
        "error": "",
    }

    try:
        config = load_monitor_config(Path(config_path))
        input_dir = Path(str(config["input_dir"]))
        risk_policy_path = str(config.get("risk_policy_config", "")).strip()
        risk_output_dir = Path(str(config.get("risk_dashboard_output_dir", effective_output_dir)))
        visualization_output_dir = Path(str(config.get("visualization_output_dir", DEFAULT_VISUALIZATION_OUTPUT_DIR)))

        refresh = refresh_recent_market_data(config, data_csv=data_csv, as_of_date=as_of_date)
        payload["steps"]["market_data_refresh"] = refresh.as_dict()

        freshness = check_monitor_data_freshness(config, data_csv=data_csv, as_of_date=as_of_date)
        if refresh.status == "stale_data":
            freshness["status"] = "warning"
            freshness["latest_price_date"] = refresh.latest_price_date
            freshness["stale_days"] = refresh.stale_days
            freshness["stale_warning_days"] = refresh.max_allowed_stale_days
            freshness["warning"] = f"latest data is {refresh.stale_days} calendar days old after refresh"
        payload["steps"]["data_freshness_check"] = freshness

        snapshot = _verify_snapshot_if_configured(config, data_csv=data_csv)
        payload["steps"]["data_snapshot_verification"] = snapshot
        if snapshot["status"] == "failed":
            raise ValueError("data snapshot verification failed")

        block_stale = bool(config.get("stale_data_blocks_new_signal", False)) and freshness["status"] != "ok"
        if block_stale:
            data_quality_path = effective_output_dir / "data_quality_report.csv"
            signal_report_path = effective_output_dir / "signal_report.md"
            _write_data_quality(data_quality_path, refresh.as_dict(), freshness)
            _write_stale_signal_report(signal_report_path, refresh.as_dict(), freshness)
            payload["steps"]["generate_voltarget_signal"] = {
                "status": "stale_data",
                "accepted_signal": False,
                "latest_price_date": freshness.get("latest_price_date", ""),
                "signal_today": str(effective_output_dir / "signal_today.json"),
                "signal_history": str(effective_output_dir / "signal_history.csv"),
                "signal_report": str(signal_report_path),
                "data_quality_report": str(data_quality_path),
            }
            payload["outputs"].update(
                {
                    "signal_report": str(signal_report_path),
                    "data_quality_report": str(data_quality_path),
                }
            )
            if bool(config.get("fail_on_stale_data", False)):
                raise ValueError("market data remained stale after refresh")
        else:
            signal_result = generate_voltarget_signal(
                config_path=Path(config_path),
                output_dir=effective_output_dir,
                data_csv=data_csv,
                signal_as_of_date=as_of_date,
                generated_at=generated,
            )
            _append_refresh_to_report(signal_result.signal_report_path, refresh.as_dict())
            payload["steps"]["generate_voltarget_signal"] = {
                "status": signal_result.signal_today.get("data_quality_status", "ok"),
                "accepted_signal": True,
                "latest_price_date": signal_result.signal_today.get("latest_price_date", ""),
                "signal_today": str(signal_result.signal_today_path),
                "signal_history": str(signal_result.signal_history_path),
                "signal_report": str(signal_result.signal_report_path),
                "data_quality_report": str(signal_result.data_quality_path),
            }
            payload["outputs"].update(
                {
                    "signal_today": str(signal_result.signal_today_path),
                    "signal_history": str(signal_result.signal_history_path),
                    "signal_report": str(signal_result.signal_report_path),
                    "data_quality_report": str(signal_result.data_quality_path),
                }
            )

        if risk_policy_path:
            risk_result = build_voltarget_risk_dashboard(
                config_path=Path(risk_policy_path),
                input_dir=input_dir,
                output_dir=risk_output_dir,
                as_of_date=as_of_date,
            )
            payload["steps"]["risk_policy_dashboard"] = {
                "status": "ok",
                "dashboard": str(risk_result.dashboard_path),
                "summary": str(risk_result.summary_path),
                "breaches": str(risk_result.breaches_path),
                "breach_count": int(risk_result.breaches["breached"].astype(bool).sum()),
            }
            payload["outputs"].update(
                {
                    "risk_dashboard": str(risk_result.dashboard_path),
                    "risk_dashboard_summary": str(risk_result.summary_path),
                    "risk_policy_breaches": str(risk_result.breaches_path),
                }
            )
        else:
            payload["steps"]["risk_policy_dashboard"] = {"status": "skipped", "reason": "risk_policy_config unavailable"}

        visualization_result = generate_equity_visualizations(input_dir=input_dir, output_dir=visualization_output_dir)
        payload["steps"]["visualization_update"] = {
            "status": "ok",
            "summary": str(visualization_result.summary_path),
            "report": str(visualization_result.report_path),
            "charts": {name: str(path) for name, path in visualization_result.chart_paths.items()},
        }
        payload["outputs"].update(
            {
                "visualization_summary": str(visualization_result.summary_path),
                "visualization_report": str(visualization_result.report_path),
            }
        )

        warning = freshness["status"] != "ok" or payload["steps"]["generate_voltarget_signal"]["status"] != "ok"
        payload["status"] = "warning" if warning else "ok"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)

    _write_status(status_path, payload)
    _append_daily_log(log_path, _log_row(payload))
    return DailyMonitorResult(
        output_dir=effective_output_dir,
        status_path=status_path,
        log_path=log_path,
        status=payload,
    )


def run_automated_voltarget_paper_monitor(
    *,
    config_path: Path,
    signal_dir: Path = DEFAULT_SIGNAL_OUTPUT_DIR,
    auto_ledger_output_dir: Path = DEFAULT_AUTO_LEDGER_OUTPUT_DIR,
    execution_drift_dir: Path = DEFAULT_EXECUTION_DRIFT_OUTPUT_DIR,
    data_csv: Optional[Path] = None,
    as_of_date: Optional[pd.Timestamp] = None,
    generated_at: Optional[str] = None,
    drift_threshold: float = 0.01,
) -> AutomatedPaperMonitorResult:
    generated = _utc_now(generated_at)
    run_date = _run_date(as_of_date)
    signal_dir = Path(signal_dir)
    signal_dir.mkdir(parents=True, exist_ok=True)
    status_path = signal_dir / "latest_automation_status.json"
    log_path = signal_dir / "daily_run_log.csv"
    payload: Dict[str, Any] = {
        "generated_at": generated,
        "run_date": run_date,
        "status": "failed",
        "paper_trading_only": True,
        "production_ready": False,
        "no_broker_integration": True,
        "no_auto_trading": True,
        "steps": {},
        "outputs": {
            "latest_automation_status": str(status_path),
            "daily_run_log": str(log_path),
        },
        "error": "",
    }

    try:
        daily = run_daily_voltarget_monitor(
            config_path=Path(config_path),
            output_dir=signal_dir,
            data_csv=data_csv,
            as_of_date=as_of_date,
            generated_at=generated,
        )
        payload["steps"]["daily_monitor"] = {
            "status": daily.status.get("status", "failed"),
            "status_path": str(daily.status_path),
            "log_path": str(daily.log_path),
            "market_data_refresh": daily.status.get("steps", {}).get("market_data_refresh", {}),
            "data_freshness_check": daily.status.get("steps", {}).get("data_freshness_check", {}),
            "generate_voltarget_signal": daily.status.get("steps", {}).get("generate_voltarget_signal", {}),
        }
        if daily.status.get("status") == "failed":
            raise ValueError(str(daily.status.get("error", "daily monitor failed")))

        ledger = update_auto_paper_ledger(
            config_path=Path(config_path),
            signal_dir=signal_dir,
            output_dir=Path(auto_ledger_output_dir),
            data_csv=data_csv,
        )
        payload["steps"]["auto_paper_ledger"] = {
            "status": ledger.latest_status,
            "ledger": str(ledger.ledger_path),
            "summary": str(ledger.summary_path),
            "report": str(ledger.report_path),
            "latest_row": ledger.latest_row,
        }
        payload["outputs"].update(
            {
                "auto_paper_ledger": str(ledger.ledger_path),
                "auto_paper_summary": str(ledger.summary_path),
                "auto_paper_report": str(ledger.report_path),
            }
        )

        drift = run_execution_drift_tracking(
            config_path=Path(config_path),
            signal_dir=signal_dir,
            output_dir=Path(execution_drift_dir),
            data_csv=data_csv,
            drift_threshold=float(drift_threshold),
        )
        payload["steps"]["execution_drift"] = {
            "status": "warning" if bool(drift.metrics.get("drift_exceeds_policy_threshold", False)) else "ok",
            "summary": str(drift.summary_path),
            "report": str(drift.report_path),
            "metrics": drift.metrics,
        }
        payload["outputs"].update(
            {
                "execution_drift_summary": str(drift.summary_path),
                "execution_drift_report": str(drift.report_path),
            }
        )

        payload["steps"]["financing_tracking"] = {
            "status": "included",
            "detail": "financing_cost_history.csv and financing_cost_report.md are written by the daily monitor",
        }
        payload["steps"]["risk_dashboard"] = {
            "status": daily.status.get("steps", {}).get("risk_policy_dashboard", {}).get("status", "unknown"),
        }

        health = check_voltarget_monitor_health(
            output_dir=signal_dir,
            config_path=Path(config_path),
            auto_paper_ledger_dir=Path(auto_ledger_output_dir),
            execution_drift_dir=Path(execution_drift_dir),
        )
        payload["steps"]["health_check"] = {
            "overall_status": health.overall_status,
            "summary": str(health.summary_path),
            "report": str(health.report_path),
        }
        payload["outputs"].update(
            {
                "monitor_health_summary": str(health.summary_path),
                "monitor_health_report": str(health.report_path),
            }
        )

        warning = (
            daily.status.get("status") == "warning"
            or ledger.latest_status in {"pending_fill", "stale_data", "missing_price_data"}
            or payload["steps"]["execution_drift"]["status"] == "warning"
            or health.overall_status == "warning"
        )
        payload["status"] = "warning" if warning else "ok"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)

    _write_status(status_path, payload)
    _append_daily_log(log_path, _automation_log_row(payload))
    return AutomatedPaperMonitorResult(
        output_dir=signal_dir,
        status_path=status_path,
        log_path=log_path,
        status=payload,
    )
