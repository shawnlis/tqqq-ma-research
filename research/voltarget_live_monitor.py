from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .execution import NEXT_OPEN_TO_NEXT_OPEN, normalize_execution_model
from .execution_financing import _load_price_data, _read_json, _resolve_output_dir, apply_slippage_and_financing, replay_locked_voltarget_candidate
from .final_voltarget_audit import locate_plain_voltarget_candidate
from .voltarget_risk_dashboard import build_policy_breaches, load_risk_policy


REQUIRED_CONFIG_FIELDS = (
    "input_dir",
    "execution_assumption",
    "transaction_cost_bps",
    "financing_annual_cost",
    "stale_data_warning_days",
    "target_symbol",
    "benchmark_symbol",
    "classification",
    "production_ready",
)


@dataclass(frozen=True)
class VolTargetSignalResult:
    output_dir: Path
    signal_today_path: Path
    signal_history_path: Path
    signal_report_path: Path
    data_quality_path: Path
    financing_history_path: Path
    financing_report_path: Path
    chart_paths: Dict[str, Path]
    signal_today: Dict[str, Any]


def load_monitor_config(config_path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("VolTarget live monitor config must be a mapping.")
    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in data]
    if missing:
        raise ValueError(f"VolTarget live monitor config missing required fields: {', '.join(missing)}")
    config = dict(data)
    config["input_dir"] = str(config["input_dir"])
    config["cache_dir"] = str(config.get("cache_dir", "./price_cache"))
    config["execution_assumption"] = normalize_execution_model(config.get("execution_assumption", NEXT_OPEN_TO_NEXT_OPEN))
    config["transaction_cost_bps"] = float(config["transaction_cost_bps"])
    config["slippage_bps"] = float(config.get("slippage_bps", 0.0))
    config["annual_financing_rate_assumption"] = float(
        config.get("annual_financing_rate_assumption", config["financing_annual_cost"])
    )
    config["financing_annual_cost"] = float(config["annual_financing_rate_assumption"])
    config["financing_applies_above_exposure"] = float(config.get("financing_applies_above_exposure", 1.0))
    config["financing_day_count_basis"] = int(config.get("financing_day_count_basis", 252))
    config["stale_data_warning_days"] = int(config["stale_data_warning_days"])
    config["max_allowed_stale_days"] = int(config.get("max_allowed_stale_days", config["stale_data_warning_days"]))
    config["target_symbol"] = str(config["target_symbol"]).upper()
    config["benchmark_symbol"] = str(config["benchmark_symbol"]).upper()
    config["classification"] = str(config["classification"])
    config["production_ready"] = bool(config["production_ready"])
    config["paper_trading_only"] = bool(config.get("paper_trading_only", True))
    config["paper_ledger_mode"] = str(config.get("paper_ledger_mode", "manual")).lower()
    config["paper_starting_equity"] = float(config.get("paper_starting_equity", 100000.0))
    config["paper_execution_model"] = normalize_execution_model(
        config.get("paper_execution_model", config["execution_assumption"])
    )
    config["paper_fill_price_source"] = str(config.get("paper_fill_price_source", "next_open")).lower()
    config["fallback_fill_price_source"] = str(config.get("fallback_fill_price_source", "latest_close")).lower()
    config["assumed_slippage_bps"] = float(config.get("assumed_slippage_bps", config.get("slippage_bps", 0.0)))
    config["assumed_transaction_cost_bps"] = float(
        config.get("assumed_transaction_cost_bps", config.get("transaction_cost_bps", 0.0))
    )
    config["auto_paper_ledger_enabled"] = bool(config.get("auto_paper_ledger_enabled", False))
    config["manual_ledger_required"] = bool(config.get("manual_ledger_required", True))
    return config


def _load_stage3_context(config: Dict[str, Any], data_csv: Optional[Path], signal_as_of_date: Optional[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    input_dir = Path(config["input_dir"])
    candidate = locate_plain_voltarget_candidate(input_dir)
    candidate_dir = _resolve_output_dir(input_dir, candidate.get("baseline_output_dir"))
    windows_path = candidate_dir / "walk_forward_windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing walk_forward_windows.csv: {windows_path}")
    windows = pd.read_csv(windows_path)
    if windows.empty:
        raise ValueError(f"Window file is empty: {windows_path}")
    run_config = _read_json(candidate_dir / "run_config.json")
    data = _load_price_data(
        run_config=run_config,
        windows=windows,
        data_csv=data_csv,
        cache_dir=str(config.get("cache_dir", "./price_cache")),
    )
    if signal_as_of_date is not None:
        data = data.loc[data.index <= pd.Timestamp(signal_as_of_date)].copy()
    if data.empty:
        raise ValueError("No price data available for live monitor after applying signal_as_of_date.")
    return windows, data, {"candidate_dir": candidate_dir, "run_config": run_config}


def build_monitor_frame(config: Dict[str, Any], *, data_csv: Optional[Path] = None, signal_as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    windows, data, _ = _load_stage3_context(config, data_csv, signal_as_of_date)
    stitched = replay_locked_voltarget_candidate(
        data=data,
        windows=windows,
        transaction_cost_bps=float(config["transaction_cost_bps"]),
        execution_model=str(config["execution_assumption"]),
    )
    adjusted = apply_slippage_and_financing(
        stitched,
        slippage_bps=float(config.get("slippage_bps", 0.0)),
        financing_annual_cost=float(config["financing_annual_cost"]),
    )
    tqqq_equity = (1.0 + adjusted["daily_ret_tqqq"].astype(float).fillna(0.0)).cumprod()
    adjusted["tqqq_paper_equity"] = tqqq_equity
    adjusted["relative_equity_vs_tqqq"] = adjusted["equity"].astype(float) / tqqq_equity
    adjusted["trade_delta"] = adjusted["target_exposure"].astype(float) - adjusted["position"].astype(float)
    adjusted["estimated_transaction_cost"] = adjusted["trade_delta"].abs() * (float(config["transaction_cost_bps"]) / 10000.0)
    adjusted["estimated_financing_cost"] = (
        adjusted["target_exposure"].astype(float) - float(config["financing_applies_above_exposure"])
    ).clip(lower=0.0) * (
        float(config["annual_financing_rate_assumption"]) / float(config["financing_day_count_basis"])
    )
    return adjusted


def build_financing_cost_history(config: Dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    exposure = frame["position"].astype(float)
    threshold = float(config["financing_applies_above_exposure"])
    annual_rate = float(config["annual_financing_rate_assumption"])
    day_count = float(config["financing_day_count_basis"])
    excess_exposure = (exposure - threshold).clip(lower=0.0)
    daily_financing_cost = excess_exposure * (annual_rate / day_count)
    ret_after_financing = frame["ret"].astype(float).fillna(0.0)
    ret_before_financing = ret_after_financing + daily_financing_cost
    equity_before = (1.0 + ret_before_financing).cumprod()
    equity_after = (1.0 + ret_after_financing).cumprod()
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(idx).date().isoformat() for idx in frame.index],
            "exposure": exposure.to_numpy(),
            "excess_exposure": excess_exposure.to_numpy(),
            "annual_financing_rate_assumption": annual_rate,
            "financing_applies_above_exposure": threshold,
            "financing_day_count_basis": int(day_count),
            "daily_financing_cost": daily_financing_cost.to_numpy(),
            "cumulative_financing_cost": daily_financing_cost.cumsum().to_numpy(),
            "strategy_equity_before_financing": equity_before.to_numpy(),
            "strategy_equity_after_financing": equity_after.to_numpy(),
            "equity_impact": (equity_after - equity_before).to_numpy(),
            "equity_impact_pct": (equity_after / equity_before - 1.0).replace([np.inf, -np.inf], np.nan).to_numpy(),
        }
    )


def _financing_sensitivity(config: Dict[str, Any], frame: pd.DataFrame, rates: tuple[float, ...] = (0.03, 0.06, 0.09, 0.12)) -> pd.DataFrame:
    rows = []
    exposure = frame["position"].astype(float)
    threshold = float(config["financing_applies_above_exposure"])
    day_count = float(config["financing_day_count_basis"])
    excess_exposure = (exposure - threshold).clip(lower=0.0)
    baseline_history = build_financing_cost_history(config, frame)
    baseline_daily_cost = pd.Series(
        baseline_history["daily_financing_cost"].astype(float).to_numpy(),
        index=frame.index,
    )
    ret_before_financing = frame["ret"].astype(float).fillna(0.0) + baseline_daily_cost
    equity_before = (1.0 + ret_before_financing).cumprod()
    for rate in rates:
        scenario_daily_cost = excess_exposure * (float(rate) / day_count)
        equity_after = (1.0 + ret_before_financing - scenario_daily_cost).cumprod()
        rows.append(
            {
                "annual_financing_rate_assumption": float(rate),
                "cumulative_financing_cost": float(scenario_daily_cost.cumsum().iloc[-1]) if len(scenario_daily_cost) else 0.0,
                "final_equity_before_financing": float(equity_before.iloc[-1]) if len(equity_before) else np.nan,
                "final_equity_after_financing": float(equity_after.iloc[-1]) if len(equity_after) else np.nan,
                "final_equity_impact": float(equity_after.iloc[-1] - equity_before.iloc[-1]) if len(equity_after) else np.nan,
                "final_equity_impact_pct": float(equity_after.iloc[-1] / equity_before.iloc[-1] - 1.0) if len(equity_after) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_financing_report(path: Path, config: Dict[str, Any], history: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    exposure_days = int((history["exposure"].astype(float) > 1.0).sum()) if not history.empty else 0
    average_excess = float(history["excess_exposure"].astype(float).mean()) if not history.empty else 0.0
    cumulative_drag = float(history["cumulative_financing_cost"].iloc[-1]) if not history.empty else 0.0
    final_before = float(history["strategy_equity_before_financing"].iloc[-1]) if not history.empty else np.nan
    final_after = float(history["strategy_equity_after_financing"].iloc[-1]) if not history.empty else np.nan
    final_impact = final_after - final_before if not np.isnan(final_before) and not np.isnan(final_after) else np.nan
    final_impact_pct = final_after / final_before - 1.0 if final_before and not np.isnan(final_before) else np.nan
    sensitivity_table = sensitivity.to_string(index=False)
    lines = [
        "# VolTarget Paper Financing Cost Report",
        "",
        "**paper trading only**",
        "",
        "This report tracks financing-cost assumptions for the paper monitor. It does not change strategy logic, place trades, or call broker APIs.",
        "",
        "## Assumptions",
        f"- Annual financing rate assumption: `{config['annual_financing_rate_assumption']}`",
        f"- Financing applies above exposure: `{config['financing_applies_above_exposure']}`",
        f"- Financing day-count basis: `{config['financing_day_count_basis']}`",
        "",
        "## Summary",
        f"- Days exposure > 1.0: `{exposure_days}`",
        f"- Average excess exposure: `{average_excess}`",
        f"- Cumulative financing drag: `{cumulative_drag}`",
        f"- Final equity before financing: `{final_before}`",
        f"- Final equity after financing: `{final_after}`",
        f"- Final equity impact: `{final_impact}`",
        f"- Final equity impact pct: `{final_impact_pct}`",
        "",
        "## Sensitivity",
        "```text",
        sensitivity_table,
        "```",
        "",
        "## Boundary",
        "Financing-cost tracking is an audit layer for paper monitoring only. It is not production-ready and is not an investment recommendation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _data_quality(config: Dict[str, Any], frame: pd.DataFrame, as_of_date: Optional[pd.Timestamp]) -> pd.DataFrame:
    latest_date = pd.Timestamp(frame.index[-1]).normalize()
    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()
    stale_days = int(max((as_of - latest_date).days, 0))
    stale_limit = int(config.get("max_allowed_stale_days", config["stale_data_warning_days"]))
    required_columns = [
        "target_exposure",
        "position",
        "realized_ann_vol",
        "below_trend",
        "strong_momentum",
        "crash_regime",
        "equity",
        "tqqq_paper_equity",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    status = "ok"
    warning = ""
    if missing:
        status = "missing_required_columns"
        warning = ";".join(missing)
    elif stale_days > stale_limit:
        status = "stale_data_warning"
        warning = f"latest data is {stale_days} calendar days old"
    return pd.DataFrame(
        [
            {
                "latest_price_date": str(latest_date.date()),
                "as_of_date": str(as_of.date()),
                "stale_days": stale_days,
                "stale_warning_days": stale_limit,
                "status": status,
                "warning": warning,
            }
        ]
    )


def _risk_breaches(config: Dict[str, Any], latest: pd.Series, quality: pd.DataFrame) -> pd.DataFrame:
    policy_path = config.get("risk_policy_config")
    if not policy_path:
        return pd.DataFrame(
            [{"policy": "risk_policy_config", "observed_value": np.nan, "policy_limit": np.nan, "breached": False, "status": "not_configured"}]
        )
    policy = load_risk_policy(Path(str(policy_path)))
    state = {
        "current_exposure": float(latest["position"]),
        "target_exposure_next_session": float(latest["target_exposure"]),
        "required_trade_delta": float(latest["trade_delta"]),
        "current_drawdown": float(latest["drawdown"]),
        "stale_days": int(quality.iloc[0]["stale_days"]),
    }
    return build_policy_breaches(policy, state)


def _signal_dict(config: Dict[str, Any], frame: pd.DataFrame, quality: pd.DataFrame, breaches: pd.DataFrame, generated_at: Optional[str]) -> Dict[str, Any]:
    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else latest
    latest_date = pd.Timestamp(frame.index[-1]).date().isoformat()
    breach_count = int(breaches["breached"].astype(bool).sum()) if "breached" in breaches.columns else 0
    return {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "latest_price_date": latest_date,
        "target_symbol": str(config["target_symbol"]),
        "benchmark_symbol": str(config["benchmark_symbol"]),
        "target_exposure_next_session": float(latest["target_exposure"]),
        "previous_target_exposure": float(previous["target_exposure"]),
        "current_exposure": float(latest["position"]),
        "trade_delta": float(latest["trade_delta"]),
        "execution_assumption": str(config["execution_assumption"]),
        "financing_assumption": float(config["financing_annual_cost"]),
        "classification": str(config["classification"]),
        "production_ready": False,
        "paper_trading_only": True,
        "data_quality_status": str(quality.iloc[0]["status"]),
        "risk_policy_breach_count": breach_count,
        "no_auto_trading": True,
    }


def _history_rows(config: Dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    previous_target = frame["target_exposure"].astype(float).shift(1).fillna(frame["target_exposure"].astype(float))
    rows = pd.DataFrame(
        {
            "latest_price_date": [pd.Timestamp(idx).date().isoformat() for idx in frame.index],
            "target_symbol": str(config["target_symbol"]),
            "benchmark_symbol": str(config["benchmark_symbol"]),
            "target_exposure_next_session": frame["target_exposure"].astype(float).to_numpy(),
            "previous_target_exposure": previous_target.to_numpy(),
            "current_exposure": frame["position"].astype(float).to_numpy(),
            "trade_delta": frame["trade_delta"].astype(float).to_numpy(),
            "realized_ann_vol": frame["realized_ann_vol"].astype(float).to_numpy(),
            "trend_state": np.where(frame["below_trend"].astype(int).to_numpy() == 1, "below_trend", "above_or_at_trend"),
            "momentum_state": np.where(frame["strong_momentum"].astype(int).to_numpy() == 1, "strong_momentum", "normal_momentum"),
            "crash_control_state": np.where(frame["crash_regime"].astype(int).to_numpy() == 1, "crash_regime", "normal"),
            "estimated_transaction_cost": frame["estimated_transaction_cost"].astype(float).to_numpy(),
            "estimated_financing_cost": frame["estimated_financing_cost"].astype(float).to_numpy(),
            "current_paper_equity": frame["equity"].astype(float).to_numpy(),
            "tqqq_paper_equity": frame["tqqq_paper_equity"].astype(float).to_numpy(),
            "relative_equity_vs_tqqq": frame["relative_equity_vs_tqqq"].astype(float).to_numpy(),
            "execution_assumption": str(config["execution_assumption"]),
        }
    )
    return rows


def _merge_history(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()
    combined = combined.sort_values("latest_price_date")
    combined = combined.drop_duplicates(subset=["latest_price_date"], keep="last")
    return combined.reset_index(drop=True)


def _write_chart(frame: pd.DataFrame, output_dir: Path, filename: str, columns: list[str], title: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    frame[columns].astype(float).plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_report(path: Path, config: Dict[str, Any], signal: Dict[str, Any], latest: pd.Series, quality: pd.DataFrame, breaches: pd.DataFrame, chart_paths: Dict[str, Path]) -> None:
    breach_count = int(breaches["breached"].astype(bool).sum()) if "breached" in breaches.columns else 0
    quality_row = quality.iloc[0]
    trend_state = "below_trend" if int(latest["below_trend"]) else "above_or_at_trend"
    momentum_state = "strong_momentum" if int(latest["strong_momentum"]) else "normal_momentum"
    crash_state = "crash_regime" if int(latest["crash_regime"]) else "normal"
    lines = [
        "# VolTarget Paper Signal Report",
        "",
        "**paper trading only**",
        "",
        "**not production-ready**",
        "",
        "This report does not place trades, create orders, call broker APIs, or provide investment advice.",
        "",
        "## Daily Signal",
        f"- Latest data date: `{signal['latest_price_date']}`",
        f"- Data stale days: `{int(quality_row['stale_days'])}`",
        f"- Current target exposure: `{signal['target_exposure_next_session']}`",
        f"- Previous target exposure: `{signal['previous_target_exposure']}`",
        f"- Current exposure: `{signal['current_exposure']}`",
        f"- Required trade delta: `{signal['trade_delta']}`",
        f"- Realized volatility: `{float(latest['realized_ann_vol'])}`",
        f"- Trend state: `{trend_state}`",
        f"- Momentum state: `{momentum_state}`",
        f"- Crash-control state: `{crash_state}`",
        f"- Estimated transaction cost: `{float(latest['estimated_transaction_cost'])}`",
        f"- Estimated financing cost: `{float(latest['estimated_financing_cost'])}`",
        f"- Current paper equity: `{float(latest['equity'])}`",
        f"- TQQQ paper equity: `{float(latest['tqqq_paper_equity'])}`",
        f"- Relative equity vs TQQQ: `{float(latest['relative_equity_vs_tqqq'])}`",
        f"- Risk policy breaches: `{breach_count}`",
        f"- Data quality status: `{quality_row['status']}`",
        "",
        "## Assumptions",
        f"- Execution assumption: `{config['execution_assumption']}`",
        f"- Financing assumption: `{config['financing_annual_cost']}`",
        f"- Classification: `{config['classification']}`",
        "- Production ready: `false`",
        "",
        "## Risk Policy Breaches",
        "```text",
        breaches.to_string(index=False),
        "```",
        "",
        "## Charts",
        f"- Paper equity curve: `{chart_paths['paper_equity_curve']}`",
        f"- Relative equity vs TQQQ: `{chart_paths['relative_equity_vs_tqqq']}`",
        f"- Exposure history: `{chart_paths['exposure_history']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_voltarget_signal(
    *,
    config_path: Path,
    output_dir: Path,
    data_csv: Optional[Path] = None,
    signal_as_of_date: Optional[pd.Timestamp] = None,
    generated_at: Optional[str] = None,
) -> VolTargetSignalResult:
    config = load_monitor_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_monitor_frame(config, data_csv=data_csv, signal_as_of_date=signal_as_of_date)
    quality = _data_quality(config, frame, signal_as_of_date)
    latest = frame.iloc[-1]
    breaches = _risk_breaches(config, latest, quality)
    signal = _signal_dict(config, frame, quality, breaches, generated_at)

    signal_today_path = output_dir / "signal_today.json"
    signal_history_path = output_dir / "signal_history.csv"
    signal_report_path = output_dir / "signal_report.md"
    data_quality_path = output_dir / "data_quality_report.csv"
    financing_history_path = output_dir / "financing_cost_history.csv"
    financing_report_path = output_dir / "financing_cost_report.md"
    signal_today_path.write_text(json.dumps(signal, indent=2, sort_keys=True), encoding="utf-8")
    history = _merge_history(signal_history_path, _history_rows(config, frame))
    history.to_csv(signal_history_path, index=False)
    quality.to_csv(data_quality_path, index=False)
    financing_history = build_financing_cost_history(config, frame)
    financing_history.to_csv(financing_history_path, index=False)
    financing_sensitivity = _financing_sensitivity(config, frame)
    _write_financing_report(financing_report_path, config, financing_history, financing_sensitivity)
    chart_paths = {
        "paper_equity_curve": _write_chart(frame, output_dir, "paper_equity_curve.png", ["equity", "tqqq_paper_equity"], "Paper Equity Curve", "Equity"),
        "relative_equity_vs_tqqq": _write_chart(frame, output_dir, "relative_equity_vs_tqqq.png", ["relative_equity_vs_tqqq"], "Relative Equity vs TQQQ", "Relative equity"),
        "exposure_history": _write_chart(frame, output_dir, "exposure_history.png", ["position", "target_exposure"], "Exposure History", "Exposure"),
    }
    _write_report(signal_report_path, config, signal, latest, quality, breaches, chart_paths)
    return VolTargetSignalResult(
        output_dir=output_dir,
        signal_today_path=signal_today_path,
        signal_history_path=signal_history_path,
        signal_report_path=signal_report_path,
        data_quality_path=data_quality_path,
        financing_history_path=financing_history_path,
        financing_report_path=financing_report_path,
        chart_paths=chart_paths,
        signal_today=signal,
    )
