from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .data_snapshot import verify_data_snapshot
from .visualization import generate_equity_visualizations
from .voltarget_live_monitor import build_monitor_frame, generate_voltarget_signal, load_monitor_config
from .voltarget_risk_dashboard import build_voltarget_risk_dashboard


DEFAULT_SIGNAL_OUTPUT_DIR = Path("outputs/live_signal")
DEFAULT_VISUALIZATION_OUTPUT_DIR = Path("outputs/visualizations/voltarget_stage3")


@dataclass(frozen=True)
class DailyMonitorResult:
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
    stale_limit = int(config["stale_data_warning_days"])
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

        freshness = check_monitor_data_freshness(config, data_csv=data_csv, as_of_date=as_of_date)
        payload["steps"]["data_freshness_check"] = freshness

        snapshot = _verify_snapshot_if_configured(config, data_csv=data_csv)
        payload["steps"]["data_snapshot_verification"] = snapshot
        if snapshot["status"] == "failed":
            raise ValueError("data snapshot verification failed")

        signal_result = generate_voltarget_signal(
            config_path=Path(config_path),
            output_dir=effective_output_dir,
            data_csv=data_csv,
            signal_as_of_date=as_of_date,
            generated_at=generated,
        )
        payload["steps"]["generate_voltarget_signal"] = {
            "status": signal_result.signal_today.get("data_quality_status", "ok"),
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
