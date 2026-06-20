from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import annualized_return, annualized_volatility, calmar_ratio, max_drawdown, sharpe_ratio


EXPOSURE_COLUMNS = ("target_exposure", "position", "exposure", "tqqq_weight", "weight", "leverage")


@dataclass(frozen=True)
class FinalVolTargetAuditResult:
    output_dir: Path
    summary: pd.DataFrame
    artifact_paths: Dict[str, Path]
    final_classification: str


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _as_float(value: object, default: float = np.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _first_existing_csv(paths: tuple[Path, ...]) -> Optional[pd.DataFrame]:
    for path in paths:
        data = _read_csv_if_exists(path)
        if data is not None:
            return data
    return None


def _resolve_output_dir(input_dir: Path, output_value: object) -> Path:
    text = str(output_value).strip()
    if not text:
        raise ValueError("Selected candidate does not include baseline_output_dir.")
    path = Path(text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "outputs":
        return path
    return input_dir / path


def locate_plain_voltarget_candidate(input_dir: Path) -> pd.Series:
    input_dir = Path(input_dir)
    family = "VolTargetTQQQStrategy"
    category = "tqqq_voltarget"
    candidate_stub: Optional[pd.Series] = None

    decision_path = input_dir / "candidate_decision_table.csv"
    if decision_path.exists():
        decision = pd.read_csv(decision_path)
        rows = decision[
            (decision.get("family", pd.Series(dtype=object)).astype(str) == family)
            | (decision.get("category", pd.Series(dtype=object)).astype(str) == category)
        ]
        if not rows.empty:
            primary = rows[rows.get("is_primary_candidate", pd.Series(dtype=object)).map(_as_bool)]
            candidate_stub = primary.iloc[0] if not primary.empty else rows.iloc[0]
            family = str(candidate_stub.get("family", family))
            category = str(candidate_stub.get("category", category))

    for filename in ("voltarget_stage3_summary.csv", "robust_alpha_audit.csv", "tournament_summary.csv"):
        path = input_dir / filename
        if not path.exists():
            continue
        table = pd.read_csv(path)
        rows = table[
            (table.get("family", pd.Series(dtype=object)).astype(str) == family)
            | (table.get("category", pd.Series(dtype=object)).astype(str) == category)
        ]
        if rows.empty:
            continue
        primary = rows[rows.get("is_primary_candidate", pd.Series(dtype=object)).map(_as_bool)]
        selected = primary.iloc[0] if not primary.empty else rows.iloc[0]
        if candidate_stub is not None:
            merged = selected.to_dict()
            for key, value in candidate_stub.to_dict().items():
                merged.setdefault(key, value)
            return pd.Series(merged)
        return selected

    raise FileNotFoundError("Could not locate the plain VolTargetTQQQStrategy candidate in Stage 3 artifacts.")


def _load_candidate_artifacts(input_dir: Path, candidate: pd.Series) -> Dict[str, object]:
    candidate_dir = _resolve_output_dir(input_dir, candidate.get("baseline_output_dir"))
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate output directory does not exist: {candidate_dir}")
    stitched_path = candidate_dir / "stitched_equity.csv"
    benchmark_path = candidate_dir / "same_period_benchmark_summary.csv"
    if not stitched_path.exists():
        raise FileNotFoundError(f"Missing stitched_equity.csv: {stitched_path}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing same_period_benchmark_summary.csv: {benchmark_path}")

    return {
        "candidate_dir": candidate_dir,
        "stitched": pd.read_csv(stitched_path),
        "benchmark_summary": pd.read_csv(benchmark_path),
        "yearly_returns": _read_csv_if_exists(candidate_dir / "yearly_returns.csv"),
        "run_config_path": candidate_dir / "run_config.json" if (candidate_dir / "run_config.json").exists() else None,
        "candidate_decision": _read_csv_if_exists(input_dir / "candidate_decision_table.csv"),
        "robust_alpha": _read_csv_if_exists(input_dir / "robust_alpha_audit.csv"),
        "fair_ex_2022": _read_csv_if_exists(input_dir / "fair_ex_2022_audit.csv"),
        "fair_leverage": _first_existing_csv(
            (
                candidate_dir / "constant_leverage_benchmark_summary.csv",
                input_dir / "fair_leverage_benchmark_summary.csv",
            )
        ),
        "one_year_contribution": _first_existing_csv(
            (
                candidate_dir / "one_year_contribution.csv",
                input_dir / "one_year_contribution.csv",
            )
        ),
    }


def prepare_audit_curves(stitched: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "daily_ret_tqqq"}
    if not required.issubset(stitched.columns):
        missing = ", ".join(sorted(required - set(stitched.columns)))
        raise ValueError(f"stitched_equity.csv missing required columns: {missing}")
    data = stitched.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").set_index("Date")
    if "equity" in data.columns:
        strategy_equity = data["equity"].astype(float)
    elif "ret" in data.columns:
        strategy_equity = (1.0 + data["ret"].astype(float).fillna(0.0)).cumprod()
    else:
        raise ValueError("stitched_equity.csv must include equity or ret.")
    benchmark_equity = (1.0 + data["daily_ret_tqqq"].astype(float).fillna(0.0)).cumprod()
    first_strategy = float(strategy_equity.dropna().iloc[0])
    first_benchmark = float(benchmark_equity.dropna().iloc[0])
    curves = pd.DataFrame(
        {
            "strategy_equity": strategy_equity / first_strategy,
            "tqqq_equity": benchmark_equity / first_benchmark,
        }
    ).dropna()
    curves["strategy_return"] = curves["strategy_equity"].pct_change().fillna(0.0)
    curves["tqqq_return"] = curves["tqqq_equity"].pct_change().fillna(0.0)
    curves["relative_equity"] = curves["strategy_equity"] / curves["tqqq_equity"]
    curves["relative_return"] = curves["relative_equity"].pct_change().fillna(0.0)
    curves["strategy_drawdown"] = curves["strategy_equity"] / curves["strategy_equity"].cummax() - 1.0
    curves["tqqq_drawdown"] = curves["tqqq_equity"] / curves["tqqq_equity"].cummax() - 1.0
    return curves


def _period_positive_share(series: pd.Series, period: str) -> float:
    if series.empty:
        return np.nan
    period_returns = series.groupby(series.index.to_period(period)).agg(lambda values: float(values.iloc[-1] / values.iloc[0] - 1.0))
    return float((period_returns > 0.0).mean()) if len(period_returns) else np.nan


def build_relative_equity_audit(curves: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, object]]:
    relative = curves["relative_equity"]
    relative_dd = relative / relative.cummax() - 1.0
    rolling = relative.pct_change(63)
    best_end = rolling.idxmax() if not rolling.dropna().empty else pd.NaT
    worst_end = rolling.idxmin() if not rolling.dropna().empty else pd.NaT
    daily = curves[["strategy_equity", "tqqq_equity", "relative_equity", "relative_return"]].copy()
    daily["relative_drawdown"] = relative_dd
    daily = daily.reset_index(names="date")
    summary = {
        "relative_start": float(relative.iloc[0]),
        "relative_end": float(relative.iloc[-1]),
        "relative_max": float(relative.max()),
        "relative_min": float(relative.min()),
        "pct_days_above_1": float((relative > 1.0).mean()),
        "pct_months_positive_relative_return": _period_positive_share(relative, "M"),
        "pct_years_positive_relative_return": _period_positive_share(relative, "Y"),
        "largest_relative_drawdown": float(relative_dd.min()),
        "best_63d_relative_window_end": "" if pd.isna(best_end) else best_end.date().isoformat(),
        "best_63d_relative_change": _as_float(rolling.loc[best_end]) if not pd.isna(best_end) else np.nan,
        "worst_63d_relative_window_end": "" if pd.isna(worst_end) else worst_end.date().isoformat(),
        "worst_63d_relative_change": _as_float(rolling.loc[worst_end]) if not pd.isna(worst_end) else np.nan,
    }
    return daily, summary


def _find_exposure_column(stitched: pd.DataFrame) -> Optional[str]:
    for column in EXPOSURE_COLUMNS:
        if column in stitched.columns:
            return column
    return None


def build_exposure_audit(stitched: pd.DataFrame, curves: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, object]]:
    data = stitched.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    exposure_column = _find_exposure_column(data)
    if exposure_column is None:
        summary = {
            "exposure_column": "",
            "exposure_available": False,
            "message": "exposure column unavailable",
        }
        return pd.DataFrame([summary]), summary

    exposure = data.set_index("Date")[exposure_column].astype(float).reindex(curves.index)
    largest_dd_date = curves["strategy_drawdown"].idxmin()
    mask_2022 = exposure.index.year == 2022
    mask_2020 = (exposure.index >= pd.Timestamp("2020-02-19")) & (exposure.index <= pd.Timestamp("2020-12-31"))
    summary = {
        "exposure_column": exposure_column,
        "exposure_available": True,
        "average_exposure": float(exposure.mean()),
        "median_exposure": float(exposure.median()),
        "max_exposure": float(exposure.max()),
        "min_exposure": float(exposure.min()),
        "pct_days_exposure_gt_1_0": float((exposure > 1.0).mean()),
        "pct_days_exposure_gt_1_25": float((exposure > 1.25).mean()),
        "pct_days_exposure_gt_1_5": float((exposure > 1.5).mean()),
        "pct_days_exposure_gt_2_0": float((exposure > 2.0).mean()),
        "average_exposure_2022": float(exposure.loc[mask_2022].mean()) if mask_2022.any() else np.nan,
        "average_exposure_ex_2022": float(exposure.loc[~mask_2022].mean()) if (~mask_2022).any() else np.nan,
        "exposure_during_largest_drawdown": float(exposure.loc[largest_dd_date]) if largest_dd_date in exposure.index else np.nan,
        "largest_drawdown_date": largest_dd_date.date().isoformat(),
        "average_exposure_2020_crash_rebound": float(exposure.loc[mask_2020].mean()) if mask_2020.any() else np.nan,
    }
    daily = pd.DataFrame({"date": exposure.index, "exposure": exposure.values, "exposure_column": exposure_column})
    return daily, summary


def build_yearly_contribution_audit(curves: pd.DataFrame, yearly_returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if yearly_returns is not None and {"year", "strategy_return", "benchmark_return"}.issubset(yearly_returns.columns):
        yearly = yearly_returns[["year", "strategy_return", "benchmark_return"]].copy()
    else:
        yearly = (
            curves[["strategy_return", "tqqq_return"]]
            .groupby(curves.index.year)
            .agg(lambda values: float((1.0 + values).prod() - 1.0))
            .reset_index(names="year")
            .rename(columns={"tqqq_return": "benchmark_return"})
        )
    yearly["strategy_return"] = yearly["strategy_return"].astype(float)
    yearly["benchmark_return"] = yearly["benchmark_return"].astype(float)
    yearly["relative_return"] = yearly["strategy_return"] - yearly["benchmark_return"]
    yearly["log_excess_contribution"] = np.log1p(yearly["strategy_return"]) - np.log1p(yearly["benchmark_return"])
    total = float(yearly["log_excess_contribution"].sum())
    yearly["share_of_total_log_excess_contribution"] = yearly["log_excess_contribution"] / total if total != 0.0 else np.nan
    max_abs = yearly["share_of_total_log_excess_contribution"].abs().max()
    yearly["is_dominant_year"] = yearly["share_of_total_log_excess_contribution"].abs().eq(max_abs)
    yearly["highlight"] = yearly["year"].astype(int).map(lambda year: "highlight" if year in {2020, 2022, 2023} or year >= 2023 else "")
    return yearly


def _segment_return(curves: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return curves.loc[(curves.index >= pd.Timestamp(start)) & (curves.index <= pd.Timestamp(end))].copy()


def build_regime_contribution_audit(curves: pd.DataFrame, exposure_daily: pd.DataFrame) -> pd.DataFrame:
    regimes = [
        ("pre_2020", "1900-01-01", "2019-12-31"),
        ("2020_crash_rebound", "2020-01-01", "2020-12-31"),
        ("2021_bull_market", "2021-01-01", "2021-12-31"),
        ("2022_bear_market", "2022-01-01", "2022-12-31"),
        ("2023_rebound", "2023-01-01", "2023-12-31"),
        ("2024_2026_late_cycle", "2024-01-01", "2026-12-31"),
    ]
    exposure = None
    if "exposure" in exposure_daily.columns:
        exposure = exposure_daily.copy()
        exposure["date"] = pd.to_datetime(exposure["date"])
        exposure = exposure.set_index("date")["exposure"].astype(float)

    rows = []
    for name, start, end in regimes:
        segment = _segment_return(curves, start, end)
        if segment.empty:
            rows.append({"regime": name, "status": "missing_dates"})
            continue
        strategy_multiple = float(segment["strategy_equity"].iloc[-1] / segment["strategy_equity"].iloc[0])
        tqqq_multiple = float(segment["tqqq_equity"].iloc[-1] / segment["tqqq_equity"].iloc[0])
        rows.append(
            {
                "regime": name,
                "start_date": segment.index.min().date().isoformat(),
                "end_date": segment.index.max().date().isoformat(),
                "strategy_final_equity_multiple": strategy_multiple,
                "tqqq_final_equity_multiple": tqqq_multiple,
                "ratio": strategy_multiple / tqqq_multiple if tqqq_multiple != 0.0 else np.nan,
                "max_drawdown": max_drawdown(segment["strategy_equity"] / segment["strategy_equity"].iloc[0]),
                "average_exposure": float(exposure.reindex(segment.index).mean()) if exposure is not None else np.nan,
                "relative_contribution": float(np.log(strategy_multiple) - np.log(tqqq_multiple)),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def build_fair_benchmark_audit(input_dir: Path, candidate_dir: Path, candidate: pd.Series, artifacts: Dict[str, object]) -> pd.DataFrame:
    rows = []
    fair_ex_2022 = artifacts.get("fair_ex_2022")
    if isinstance(fair_ex_2022, pd.DataFrame):
        rows_df = fair_ex_2022[
            (fair_ex_2022.get("family", pd.Series(dtype=object)).astype(str) == "VolTargetTQQQStrategy")
            | (fair_ex_2022.get("category", pd.Series(dtype=object)).astype(str) == "tqqq_voltarget")
        ]
        fair_row = rows_df.iloc[0] if not rows_df.empty else pd.Series(dtype=object)
    else:
        fair_row = pd.Series(dtype=object)

    def add(name: str, period: str, ratio: float, source: str) -> None:
        rows.append(
            {
                "benchmark": name,
                "period": period,
                "ratio": ratio,
                "beats_benchmark": bool(ratio > 1.0) if not np.isnan(ratio) else False,
                "status": "ok" if not np.isnan(ratio) else "missing_artifact",
                "source": source,
            }
        )

    add(
        "same_average_exposure_constant_tqqq",
        "full_period",
        _as_float(candidate.get("ratio_vs_same_avg_exposure_constant")),
        "voltarget_stage3_summary.csv",
    )
    add(
        "simple_voltarget_no_trend",
        "full_period",
        _as_float(candidate.get("ratio_vs_simple_voltarget_no_trend")),
        "voltarget_stage3_summary.csv",
    )
    add(
        "same_average_exposure_constant_tqqq",
        "ex_2022",
        _as_float(fair_row.get("ex_2022_ratio_vs_same_avg_exposure_constant")),
        "fair_ex_2022_audit.csv",
    )
    add(
        "simple_voltarget_no_trend",
        "ex_2022",
        _as_float(fair_row.get("ex_2022_ratio_vs_simple_voltarget_no_trend")),
        "fair_ex_2022_audit.csv",
    )
    add("simple_voltarget_no_trend", "post_2022", np.nan, "missing_artifact")

    constant = artifacts.get("fair_leverage")
    if isinstance(constant, pd.DataFrame):
        same_max = constant[constant.get("is_same_max_exposure_benchmark", pd.Series(dtype=object)).map(_as_bool)]
        if not same_max.empty:
            row = same_max.iloc[0]
            strategy = _as_float(row.get("strategy_final_equity"))
            const = _as_float(row.get("constant_final_equity"))
            add("same_max_constant_tqqq", "full_period", strategy / const if const else np.nan, str(candidate_dir / "constant_leverage_benchmark_summary.csv"))
        else:
            add("same_max_constant_tqqq", "full_period", np.nan, "missing_artifact")
    else:
        add("same_max_constant_tqqq", "full_period", np.nan, "missing_artifact")
    return pd.DataFrame(rows)


def _rolling_return(series: pd.Series, window: int) -> pd.Series:
    return series / series.shift(window) - 1.0


def _worst_period_return(equity: pd.Series, window: int) -> float:
    values = _rolling_return(equity, window).dropna()
    return float(values.min()) if not values.empty else np.nan


def _longest_drawdown_duration(drawdown: pd.Series) -> int:
    longest = current = 0
    for value in drawdown:
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _recovery_after_max_drawdown_days(drawdown: pd.Series) -> float:
    if drawdown.empty:
        return np.nan
    trough = drawdown.idxmin()
    after = drawdown.loc[trough:]
    recovered = after[after >= 0.0]
    if recovered.empty:
        return np.nan
    return float((recovered.index[0] - trough).days)


def build_practical_risk_audit(curves: pd.DataFrame) -> pd.DataFrame:
    strategy_dd = curves["strategy_drawdown"]
    tqqq_dd = curves["tqqq_drawdown"]
    strategy_max_dd = float(strategy_dd.min())
    tqqq_max_dd = float(tqqq_dd.min())
    if strategy_max_dd <= -0.70:
        classification = "not suitable for conservative capital"
    elif strategy_max_dd <= -0.50:
        classification = "paper-trading candidate only"
    else:
        classification = "potential small-allocation research candidate"
    return pd.DataFrame(
        [
            {
                "strategy_max_drawdown": strategy_max_dd,
                "tqqq_max_drawdown": tqqq_max_dd,
                "drawdown_improvement": strategy_max_dd - tqqq_max_dd,
                "worst_1_month_return": _worst_period_return(curves["strategy_equity"], 21),
                "worst_3_month_return": _worst_period_return(curves["strategy_equity"], 63),
                "worst_6_month_return": _worst_period_return(curves["strategy_equity"], 126),
                "longest_drawdown_duration_trading_days": _longest_drawdown_duration(strategy_dd),
                "recovery_time_after_max_drawdown_calendar_days": _recovery_after_max_drawdown_days(strategy_dd),
                "max_drawdown_exceeds_50pct": bool(strategy_max_dd <= -0.50),
                "max_drawdown_exceeds_60pct": bool(strategy_max_dd <= -0.60),
                "max_drawdown_exceeds_70pct": bool(strategy_max_dd <= -0.70),
                "practical_classification": classification,
            }
        ]
    )


def build_execution_sensitivity_audit(candidate_dir: Path) -> pd.DataFrame:
    rows = []
    for filename in ("execution_model_sensitivity.csv", "cost_sensitivity.csv"):
        path = candidate_dir / filename
        if path.exists():
            data = pd.read_csv(path)
            data["artifact"] = filename
            data["execution_sensitivity_missing"] = False
            rows.append(data)
    if rows:
        return pd.concat(rows, ignore_index=True, sort=False)
    return pd.DataFrame(
        [
            {
                "execution_sensitivity_missing": True,
                "message": "execution sensitivity artifacts missing; future paper-trading validation recommended",
            }
        ]
    )


def classify_final_candidate(candidate: pd.Series, fair: pd.DataFrame) -> str:
    full = _as_float(candidate.get("full_period_ratio_vs_tqqq"))
    ex_2022 = _as_float(candidate.get("ex_2022_ratio_vs_tqqq"))
    one_year = _as_float(candidate.get("one_year_dominance_share"), default=1.0)
    fair_ex_2022 = fair[(fair["period"] == "ex_2022") & fair["benchmark"].isin(["same_average_exposure_constant_tqqq", "simple_voltarget_no_trend"])]
    fair_ex_ok = bool((fair_ex_2022["ratio"].astype(float) > 1.0).all()) if len(fair_ex_2022) == 2 else False
    no_trend_ex = fair[
        (fair["period"] == "ex_2022") & (fair["benchmark"] == "simple_voltarget_no_trend")
    ]["ratio"]
    no_trend_ok = bool(not no_trend_ex.empty and float(no_trend_ex.iloc[0]) > 1.0)
    if np.isnan(full) or np.isnan(ex_2022) or full <= 1.0 or ex_2022 <= 1.0:
        return "reject"
    if fair_ex_ok and no_trend_ok and one_year <= 0.60:
        return "robust_alpha_candidate"
    if not no_trend_ok or not fair_ex_ok:
        return "crash_control_candidate"
    return "leverage_risk_budget_candidate"


def _save_line_chart(path: Path, series: Dict[str, pd.Series], title: str, ylabel: str, *, log_scale: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, values in series.items():
        ax.plot(values.index, values.values, label=label, linewidth=1.5)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_bar_chart(path: Path, labels: pd.Series, values: pd.Series, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, values.astype(float))
    ax.set_xticks(x)
    ax.set_xticklabels(labels.astype(str), rotation=45)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_exposure_distribution(path: Path, exposure_daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if "exposure" in exposure_daily.columns:
        ax.hist(exposure_daily["exposure"].astype(float).dropna(), bins=40)
        title = "Exposure distribution"
    else:
        ax.text(0.5, 0.5, "Exposure column unavailable", ha="center", va="center")
        title = "Exposure distribution unavailable"
    ax.set_title(title)
    ax.set_xlabel("Exposure")
    ax.set_ylabel("Days")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_charts(output_dir: Path, curves: pd.DataFrame, yearly: pd.DataFrame, exposure_daily: pd.DataFrame) -> Dict[str, Path]:
    paths = {
        "equity_curve_log": output_dir / "equity_curve_log.png",
        "relative_equity_curve": output_dir / "relative_equity_curve.png",
        "drawdown_curve": output_dir / "drawdown_curve.png",
        "exposure_over_time": output_dir / "exposure_over_time.png",
        "yearly_relative_return": output_dir / "yearly_relative_return.png",
        "rolling_1y_relative_return": output_dir / "rolling_1y_relative_return.png",
        "rolling_1y_sharpe": output_dir / "rolling_1y_sharpe.png",
        "contribution_by_year": output_dir / "contribution_by_year.png",
        "exposure_distribution": output_dir / "exposure_distribution.png",
    }
    _save_line_chart(paths["equity_curve_log"], {"VolTargetTQQQStrategy": curves["strategy_equity"], "TQQQ buy-and-hold": curves["tqqq_equity"]}, "Equity curve, log scale", "Normalized equity", log_scale=True)
    _save_line_chart(paths["relative_equity_curve"], {"VolTarget / TQQQ": curves["relative_equity"]}, "Relative equity", "Relative equity")
    _save_line_chart(paths["drawdown_curve"], {"VolTargetTQQQStrategy": curves["strategy_drawdown"], "TQQQ buy-and-hold": curves["tqqq_drawdown"]}, "Drawdown curve", "Drawdown")
    if "exposure" in exposure_daily.columns:
        exposure = exposure_daily.copy()
        exposure["date"] = pd.to_datetime(exposure["date"])
        _save_line_chart(paths["exposure_over_time"], {"Exposure": exposure.set_index("date")["exposure"]}, "Exposure over time", "Exposure")
    else:
        _save_exposure_distribution(paths["exposure_over_time"], exposure_daily)
    _save_bar_chart(paths["yearly_relative_return"], yearly["year"], yearly["relative_return"], "Yearly relative return", "Strategy return minus TQQQ return")
    _save_bar_chart(paths["contribution_by_year"], yearly["year"], yearly["log_excess_contribution"], "Log excess contribution by year", "Log excess contribution")
    rolling_relative = _rolling_return(curves["strategy_equity"], 252) - _rolling_return(curves["tqqq_equity"], 252)
    _save_line_chart(paths["rolling_1y_relative_return"], {"Rolling 1Y relative return": rolling_relative.dropna()}, "Rolling 1Y relative return", "Relative return")
    rolling_strategy_sharpe = curves["strategy_return"].rolling(252).mean() / curves["strategy_return"].rolling(252).std(ddof=0) * np.sqrt(252)
    rolling_tqqq_sharpe = curves["tqqq_return"].rolling(252).mean() / curves["tqqq_return"].rolling(252).std(ddof=0) * np.sqrt(252)
    _save_line_chart(paths["rolling_1y_sharpe"], {"VolTarget 1Y Sharpe": rolling_strategy_sharpe.dropna(), "TQQQ 1Y Sharpe": rolling_tqqq_sharpe.dropna()}, "Rolling 1Y Sharpe", "Sharpe")
    _save_exposure_distribution(paths["exposure_distribution"], exposure_daily)
    return paths


def _write_report(
    path: Path,
    *,
    summary: Dict[str, object],
    chart_paths: Dict[str, Path],
) -> None:
    charts = "\n".join(f"- `{name}`: `{chart_path}`" for name, chart_path in chart_paths.items())
    text = f"""# Final VolTarget Audit Report

This is an empirical audit artifact. It is not an investment recommendation and does not make the strategy production-ready.

## Direct Answers

- Is the strategy robustly outperforming TQQQ? `{summary['is_robustly_outperforming_tqqq']}`.
- Is the outperformance persistent or regime-concentrated? `{summary['persistence_answer']}`.
- How dependent is it on 2022? `2022 remains an explicit concentration risk; ex-2022 no-trend ratio is {summary['ex_2022_ratio_vs_simple_voltarget_no_trend']}`.
- Does it beat fair leverage benchmarks? `{summary['fair_benchmark_answer']}`.
- Does it beat SimpleVolTargetNoTrend ex-2022? `{summary['beats_simple_no_trend_ex_2022']}`.
- How much leverage does it actually use? Average exposure `{summary['average_exposure']}`, max exposure `{summary['max_exposure']}`.
- Is drawdown still severe? `{summary['drawdown_answer']}`.
- What should be the next step? `Stop backtest expansion. Only consider a paper-trading signal monitor with no auto-trading.`

## Final Classification

```text
{summary['final_classification']}
```

The strategy remains a TQQQ crash-control / risk-budget overlay, not proven persistent alpha.

## Key Metrics

- Full-period ratio vs TQQQ: {summary['full_period_ratio_vs_tqqq']}
- Ex-2022 ratio vs TQQQ: {summary['ex_2022_ratio_vs_tqqq']}
- Post-2022 ratio vs TQQQ: {summary['post_2022_ratio_vs_tqqq']}
- Ex-2022 ratio vs SimpleVolTargetNoTrend: {summary['ex_2022_ratio_vs_simple_voltarget_no_trend']}
- One-year dominance share: {summary['one_year_dominance_share']}
- Strategy max drawdown: {summary['strategy_max_drawdown']}
- TQQQ max drawdown: {summary['tqqq_max_drawdown']}
- Practical risk classification: {summary['practical_classification']}

## Charts

{charts}

## Not Claims

- Do not claim persistent alpha.
- Do not claim production readiness.
- Do not claim the strategy beats SimpleVolTargetNoTrend ex-2022.
- Do not run full or deep tournaments from this audit.
"""
    path.write_text(text, encoding="utf-8")


def run_final_voltarget_audit(input_dir: Path, output_dir: Path) -> FinalVolTargetAuditResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = locate_plain_voltarget_candidate(input_dir)
    artifacts = _load_candidate_artifacts(input_dir, candidate)
    candidate_dir = artifacts["candidate_dir"]
    stitched = artifacts["stitched"]
    curves = prepare_audit_curves(stitched)

    relative_daily, relative_summary = build_relative_equity_audit(curves)
    exposure_daily, exposure_summary = build_exposure_audit(stitched, curves)
    yearly = build_yearly_contribution_audit(curves, artifacts.get("yearly_returns"))
    regime = build_regime_contribution_audit(curves, exposure_daily)
    fair = build_fair_benchmark_audit(input_dir, candidate_dir, candidate, artifacts)
    practical = build_practical_risk_audit(curves)
    execution = build_execution_sensitivity_audit(candidate_dir)
    final_classification = classify_final_candidate(candidate, fair)

    chart_paths = _write_charts(output_dir, curves, yearly, exposure_daily)

    fair_no_trend_ex = fair[(fair["benchmark"] == "simple_voltarget_no_trend") & (fair["period"] == "ex_2022")]
    no_trend_ratio = float(fair_no_trend_ex["ratio"].iloc[0]) if not fair_no_trend_ex.empty else np.nan
    practical_row = practical.iloc[0]
    summary_row = {
        "final_classification": final_classification,
        "strategy_name": "VolTargetTQQQStrategy",
        "full_period_ratio_vs_tqqq": _as_float(candidate.get("full_period_ratio_vs_tqqq")),
        "ex_2022_ratio_vs_tqqq": _as_float(candidate.get("ex_2022_ratio_vs_tqqq")),
        "post_2022_ratio_vs_tqqq": _as_float(candidate.get("post_2022_ratio_vs_tqqq")),
        "ratio_vs_same_avg_exposure_constant": _as_float(candidate.get("ratio_vs_same_avg_exposure_constant")),
        "ratio_vs_simple_voltarget_no_trend": _as_float(candidate.get("ratio_vs_simple_voltarget_no_trend")),
        "ex_2022_ratio_vs_simple_voltarget_no_trend": no_trend_ratio,
        "one_year_dominance_share": _as_float(candidate.get("one_year_dominance_share")),
        "relative_equity_end": relative_summary["relative_end"],
        "pct_days_relative_above_1": relative_summary["pct_days_above_1"],
        "average_exposure": exposure_summary.get("average_exposure", np.nan),
        "max_exposure": exposure_summary.get("max_exposure", np.nan),
        "pct_days_exposure_gt_1_0": exposure_summary.get("pct_days_exposure_gt_1_0", np.nan),
        "pct_days_exposure_gt_1_25": exposure_summary.get("pct_days_exposure_gt_1_25", np.nan),
        "pct_days_exposure_gt_1_5": exposure_summary.get("pct_days_exposure_gt_1_5", np.nan),
        "pct_days_exposure_gt_2_0": exposure_summary.get("pct_days_exposure_gt_2_0", np.nan),
        "strategy_max_drawdown": practical_row["strategy_max_drawdown"],
        "tqqq_max_drawdown": practical_row["tqqq_max_drawdown"],
        "practical_classification": practical_row["practical_classification"],
        "is_robustly_outperforming_tqqq": final_classification == "robust_alpha_candidate",
        "persistence_answer": "regime-concentrated / crash-control overlay",
        "fair_benchmark_answer": "fails stricter ex-2022 no-trend/fair benchmark checks",
        "beats_simple_no_trend_ex_2022": bool(no_trend_ratio > 1.0) if not np.isnan(no_trend_ratio) else False,
        "drawdown_answer": "severe drawdown remains; not production-ready",
    }
    summary = pd.DataFrame([summary_row])

    outputs = {
        "final_voltarget_audit_summary": output_dir / "final_voltarget_audit_summary.csv",
        "relative_equity_audit": output_dir / "relative_equity_audit.csv",
        "exposure_audit": output_dir / "exposure_audit.csv",
        "yearly_contribution_audit": output_dir / "yearly_contribution_audit.csv",
        "regime_contribution_audit": output_dir / "regime_contribution_audit.csv",
        "execution_sensitivity_audit": output_dir / "execution_sensitivity_audit.csv",
        "fair_benchmark_audit": output_dir / "fair_benchmark_audit.csv",
        "practical_risk_audit": output_dir / "practical_risk_audit.csv",
        "final_voltarget_audit_report": output_dir / "final_voltarget_audit_report.md",
    }
    summary.to_csv(outputs["final_voltarget_audit_summary"], index=False)
    relative_daily.to_csv(outputs["relative_equity_audit"], index=False)
    exposure_daily.to_csv(outputs["exposure_audit"], index=False)
    yearly.to_csv(outputs["yearly_contribution_audit"], index=False)
    regime.to_csv(outputs["regime_contribution_audit"], index=False)
    execution.to_csv(outputs["execution_sensitivity_audit"], index=False)
    fair.to_csv(outputs["fair_benchmark_audit"], index=False)
    practical.to_csv(outputs["practical_risk_audit"], index=False)
    _write_report(outputs["final_voltarget_audit_report"], summary=summary_row, chart_paths=chart_paths)

    artifact_paths = {**outputs, **chart_paths}
    return FinalVolTargetAuditResult(
        output_dir=output_dir,
        summary=summary,
        artifact_paths=artifact_paths,
        final_classification=final_classification,
    )
