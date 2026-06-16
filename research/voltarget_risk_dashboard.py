from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .final_voltarget_audit import _resolve_output_dir, locate_plain_voltarget_candidate


REQUIRED_POLICY_FIELDS = (
    "max_account_allocation",
    "max_allowed_exposure",
    "max_daily_trade_delta",
    "max_drawdown_warning",
    "max_drawdown_stop_review",
    "stale_data_stop",
    "financing_cost_assumption",
    "paper_trade_start_date",
)


@dataclass(frozen=True)
class RiskDashboardResult:
    output_dir: Path
    summary: pd.DataFrame
    breaches: pd.DataFrame
    dashboard_path: Path
    summary_path: Path
    breaches_path: Path
    chart_paths: Dict[str, Path]


def load_risk_policy(config_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Risk policy config must be a mapping.")
    missing = [field for field in REQUIRED_POLICY_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Risk policy config missing required fields: {', '.join(missing)}")
    policy = dict(data)
    for field in REQUIRED_POLICY_FIELDS:
        if field == "paper_trade_start_date":
            policy[field] = str(policy[field])
        elif field == "stale_data_stop":
            policy[field] = int(policy[field])
        else:
            policy[field] = float(policy[field])
    return policy


def _read_stitched(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing stitched_equity.csv: {path}")
    data = pd.read_csv(path)
    if "Date" not in data.columns:
        first = data.columns[0]
        data = data.rename(columns={first: "Date"})
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").set_index("Date")
    if "equity" not in data.columns and "ret" in data.columns:
        data["equity"] = (1.0 + data["ret"].astype(float).fillna(0.0)).cumprod()
    if "drawdown" not in data.columns and "equity" in data.columns:
        equity = data["equity"].astype(float)
        data["drawdown"] = equity / equity.cummax() - 1.0
    return data


def _latest_float(row: pd.Series, column: str, default: float = np.nan) -> float:
    if column not in row:
        return default
    try:
        value = row[column]
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _state_from_stitched(stitched: pd.DataFrame, as_of_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    if stitched.empty:
        raise ValueError("stitched_equity.csv is empty.")
    latest = stitched.iloc[-1]
    latest_date = pd.Timestamp(stitched.index[-1]).normalize()
    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()
    current_exposure = _latest_float(latest, "position")
    target_exposure = _latest_float(latest, "target_exposure", current_exposure)
    required_trade_delta = target_exposure - current_exposure
    current_drawdown = _latest_float(latest, "drawdown")
    realized_ann_vol = _latest_float(latest, "realized_ann_vol")
    stale_days = int(max((as_of - latest_date).days, 0))
    tqqq_equity = (1.0 + stitched["daily_ret_tqqq"].astype(float).fillna(0.0)).cumprod()
    paper_equity = _latest_float(latest, "equity")
    relative_equity = paper_equity / float(tqqq_equity.iloc[-1]) if float(tqqq_equity.iloc[-1]) != 0.0 else np.nan
    return {
        "latest_data_date": str(latest_date.date()),
        "as_of_date": str(as_of.date()),
        "stale_days": stale_days,
        "current_exposure": current_exposure,
        "target_exposure_next_session": target_exposure,
        "required_trade_delta": required_trade_delta,
        "estimated_turnover": abs(required_trade_delta),
        "realized_ann_vol": realized_ann_vol,
        "trend_state": "below_trend" if int(_latest_float(latest, "below_trend", 0.0)) else "above_or_at_trend",
        "momentum_state": "strong_momentum" if int(_latest_float(latest, "strong_momentum", 0.0)) else "normal_momentum",
        "crash_control_state": "crash_regime" if int(_latest_float(latest, "crash_regime", 0.0)) else "normal",
        "current_drawdown": current_drawdown,
        "max_drawdown_to_date": float(stitched["drawdown"].astype(float).min()),
        "same_period_paper_equity": paper_equity,
        "relative_equity_vs_tqqq": relative_equity,
    }


def build_policy_breaches(policy: Dict[str, Any], state: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    def add(name: str, observed: float, limit: float, breached: bool, status: str) -> None:
        rows.append(
            {
                "policy": name,
                "observed_value": observed,
                "policy_limit": limit,
                "breached": bool(breached),
                "status": status,
            }
        )

    add(
        "max_account_allocation",
        np.nan,
        float(policy["max_account_allocation"]),
        False,
        "not_evaluated_no_account_balance",
    )
    max_exposure_observed = max(
        float(state["current_exposure"]),
        float(state["target_exposure_next_session"]),
    )
    add(
        "max_allowed_exposure",
        max_exposure_observed,
        float(policy["max_allowed_exposure"]),
        max_exposure_observed > float(policy["max_allowed_exposure"]),
        "breach" if max_exposure_observed > float(policy["max_allowed_exposure"]) else "ok",
    )
    trade_delta = abs(float(state["required_trade_delta"]))
    add(
        "max_daily_trade_delta",
        trade_delta,
        float(policy["max_daily_trade_delta"]),
        trade_delta > float(policy["max_daily_trade_delta"]),
        "breach" if trade_delta > float(policy["max_daily_trade_delta"]) else "ok",
    )
    drawdown = float(state["current_drawdown"])
    add(
        "max_drawdown_warning",
        drawdown,
        float(policy["max_drawdown_warning"]),
        drawdown <= float(policy["max_drawdown_warning"]),
        "warning" if drawdown <= float(policy["max_drawdown_warning"]) else "ok",
    )
    add(
        "max_drawdown_stop_review",
        drawdown,
        float(policy["max_drawdown_stop_review"]),
        drawdown <= float(policy["max_drawdown_stop_review"]),
        "stop_review" if drawdown <= float(policy["max_drawdown_stop_review"]) else "ok",
    )
    stale_days = int(state["stale_days"])
    add(
        "stale_data_stop",
        float(stale_days),
        float(policy["stale_data_stop"]),
        stale_days > int(policy["stale_data_stop"]),
        "stop_stale_data" if stale_days > int(policy["stale_data_stop"]) else "ok",
    )
    add(
        "financing_cost_assumption",
        float(policy["financing_cost_assumption"]),
        float(policy["financing_cost_assumption"]),
        False,
        "assumption_recorded",
    )
    return pd.DataFrame(rows)


def _write_signal_box(state: Dict[str, Any], breaches: pd.DataFrame, output_dir: Path) -> Path:
    breached = int(breaches["breached"].astype(bool).sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    text = "\n".join(
        [
            "VolTarget Paper Dashboard",
            "paper trading only",
            f"Latest data: {state['latest_data_date']}",
            f"Current exposure: {state['current_exposure']:.3f}x",
            f"Target next session: {state['target_exposure_next_session']:.3f}x",
            f"Trade delta: {state['required_trade_delta']:.3f}x",
            f"Breaches: {breached}",
        ]
    )
    ax.text(0.05, 0.95, text, va="top", ha="left", fontsize=14, family="monospace")
    path = output_dir / "current_signal_box.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_drawdown_chart(stitched: pd.DataFrame, policy: Dict[str, Any], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    stitched["drawdown"].astype(float).plot(ax=ax, label="VolTarget drawdown")
    ax.axhline(float(policy["max_drawdown_warning"]), color="orange", linestyle="--", label="warning")
    ax.axhline(float(policy["max_drawdown_stop_review"]), color="red", linestyle="--", label="stop review")
    ax.set_title("Drawdown Status")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "drawdown_status.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_dashboard(
    path: Path,
    *,
    policy: Dict[str, Any],
    state: Dict[str, Any],
    breaches: pd.DataFrame,
    chart_paths: Dict[str, Path],
) -> None:
    breach_count = int(breaches["breached"].astype(bool).sum())
    lines = [
        "# VolTarget Risk Policy Dashboard",
        "",
        "**paper trading only**",
        "",
        "**not production-ready**",
        "",
        "**Final classification: crash-control candidate, not persistent alpha**",
        "",
        "## Current State",
        f"- Latest data date: `{state['latest_data_date']}`",
        f"- Current exposure: `{state['current_exposure']}`",
        f"- Target exposure for next session: `{state['target_exposure_next_session']}`",
        f"- Required trade delta: `{state['required_trade_delta']}`",
        f"- Realized annual volatility: `{state['realized_ann_vol']}`",
        f"- Trend state: `{state['trend_state']}`",
        f"- Momentum state: `{state['momentum_state']}`",
        f"- Crash-control state: `{state['crash_control_state']}`",
        f"- Current drawdown: `{state['current_drawdown']}`",
        f"- Same-period paper equity: `{state['same_period_paper_equity']}`",
        f"- Relative equity vs TQQQ: `{state['relative_equity_vs_tqqq']}`",
        f"- Stale data days: `{state['stale_days']}`",
        "",
        "## Policy",
    ]
    lines.extend(f"- {field}: `{policy[field]}`" for field in REQUIRED_POLICY_FIELDS)
    lines.extend(
        [
            "",
            "## Breaches",
            f"- Breach count: `{breach_count}`",
            "```text",
            breaches.to_string(index=False),
            "```",
            "",
            "## Charts",
            f"- Current signal box: `{chart_paths['current_signal_box']}`",
            f"- Drawdown status: `{chart_paths['drawdown_status']}`",
            "",
            "## Boundary",
            "This dashboard is a paper-trading monitor only. It does not place trades, generate orders, or make investment recommendations.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_voltarget_risk_dashboard(
    *,
    config_path: Path,
    input_dir: Path,
    output_dir: Path,
    as_of_date: Optional[pd.Timestamp] = None,
) -> RiskDashboardResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_risk_policy(Path(config_path))
    candidate = locate_plain_voltarget_candidate(Path(input_dir))
    candidate_dir = _resolve_output_dir(Path(input_dir), candidate.get("baseline_output_dir"))
    stitched = _read_stitched(candidate_dir / "stitched_equity.csv")
    state = _state_from_stitched(stitched, as_of_date=as_of_date)
    breaches = build_policy_breaches(policy, state)

    summary = pd.DataFrame([{**state, **{f"policy_{k}": v for k, v in policy.items()}, "breach_count": int(breaches["breached"].astype(bool).sum())}])
    dashboard_path = output_dir / "dashboard.md"
    summary_path = output_dir / "dashboard_summary.csv"
    breaches_path = output_dir / "risk_policy_breaches.csv"
    signal_chart = _write_signal_box(state, breaches, output_dir)
    drawdown_chart = _write_drawdown_chart(stitched, policy, output_dir)
    chart_paths = {
        "current_signal_box": signal_chart,
        "drawdown_status": drawdown_chart,
    }
    summary.to_csv(summary_path, index=False)
    breaches.to_csv(breaches_path, index=False)
    _write_dashboard(dashboard_path, policy=policy, state=state, breaches=breaches, chart_paths=chart_paths)
    return RiskDashboardResult(
        output_dir=output_dir,
        summary=summary,
        breaches=breaches,
        dashboard_path=dashboard_path,
        summary_path=summary_path,
        breaches_path=breaches_path,
        chart_paths=chart_paths,
    )
