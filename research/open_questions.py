from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .anti_overfit import run_anti_overfit_validation
from .experiments import _load_price_data, _run_walk_forward, load_yaml_file
from .reports import _markdown_table, compare_to_benchmark, save_run_config


QUESTION_OUTPUTS = {
    1: "question_1_exposure_floor.csv",
    2: "question_2_rebound.csv",
    3: "question_3_risk_off.csv",
    4: "question_4_adaptive_leverage.csv",
    5: "question_5_market_internals.csv",
    6: "question_6_soxl.csv",
    7: "question_7_synthetic_history.csv",
    8: "question_8_overfit.csv",
}

QUESTION_TEXT = {
    1: "Does the strategy fail because average exposure is too low?",
    2: "Does fast re-entry after crashes improve performance?",
    3: "Is cash better than QQQ during risk-off?",
    4: "Does adaptive leverage beat fixed 1x TQQQ exposure?",
    5: "Can market internals improve timing?",
    6: "Does SOXL help beat TQQQ, or is it just a sector bet?",
    7: "Do results survive longer synthetic history?",
    8: "Are results just parameter overfitting?",
}

DEFAULT_RISK_OFF_SYMBOLS = ("CASH", "SGOV", "BIL", "SHY", "IEF", "TLT", "GLD", "QQQ")
MARKET_INTERNALS_SYMBOLS = ("SPY", "QQQ", "RSP", "IWM", "HYG", "LQD", "IEF", "TLT", "^VIX", "XLK", "SMH", "SOXX")


@dataclass
class CaseResult:
    row: Dict[str, Any]
    config: Dict[str, Any]
    data: pd.DataFrame
    wf_table: pd.DataFrame
    stitched: pd.DataFrame
    summary: pd.DataFrame
    yearly: pd.DataFrame


def _as_list(value: Any, default: Optional[Sequence[Any]] = None) -> List[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _unique_symbols(symbols: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in symbols:
        if value is None:
            continue
        symbol = str(value).upper()
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(value)


def _best_ok_row(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and pd.notna(_safe_float(row.get("final_equity_ratio")))
    ]
    if not ok_rows:
        return None
    return max(ok_rows, key=lambda row: _safe_float(row.get("final_equity_ratio")))


def _answer_summary(
    question_id: int,
    rows: Sequence[Dict[str, Any]],
    answer: str,
    caveats: str,
    best: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    best = best or _best_ok_row(rows)
    return {
        "question_id": question_id,
        "question": QUESTION_TEXT[question_id],
        "answer": answer,
        "best_candidate_config": best.get("candidate_config", "") if best else "",
        "final_equity_ratio_versus_tqqq": _safe_float(best.get("final_equity_ratio")) if best else np.nan,
        "max_drawdown": _safe_float(best.get("strategy_max_dd")) if best else np.nan,
        "caveats": caveats,
    }


def _year_value(yearly: pd.DataFrame, year: int, column: str) -> float:
    if yearly.empty or "year" not in yearly.columns or column not in yearly.columns:
        return np.nan
    match = yearly.loc[pd.to_numeric(yearly["year"], errors="coerce") == int(year), column]
    return _safe_float(match.iloc[0]) if not match.empty else np.nan


def _compound_years(yearly: pd.DataFrame, start_year: int, column: str) -> float:
    if yearly.empty or "year" not in yearly.columns or column not in yearly.columns:
        return np.nan
    years = pd.to_numeric(yearly["year"], errors="coerce")
    values = pd.to_numeric(yearly.loc[years >= int(start_year), column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def _question_rows_to_frame(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame([{"status": "no_result"}])


class OpenQuestionsExperimentPack:
    def __init__(self, pack: Dict[str, Any], config_path: Optional[Path] = None):
        self.pack = copy.deepcopy(pack)
        self.config_path = Path(config_path) if config_path is not None else None
        self.pack_name = str(self.pack.get("pack_name", self.pack.get("experiment_pack_name", "open_questions")))
        self.output_dir = Path(self.pack.get("output_dir", Path("outputs") / self.pack_name))
        self.benchmark_symbol = str(self.pack.get("benchmark_symbol", "TQQQ")).upper()
        self.question_frames: Dict[int, pd.DataFrame] = {}
        self.summary_rows: List[Dict[str, Any]] = []

    def run(self) -> pd.DataFrame:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_questions = {
            int(value)
            for value in _as_list(self.pack.get("run_questions"), range(1, 9))
        }

        runners = {
            1: self._question_1_exposure_floor,
            2: self._question_2_rebound,
            3: self._question_3_risk_off,
            4: self._question_4_adaptive_leverage,
            5: self._question_5_market_internals,
            6: self._question_6_soxl,
            7: self._question_7_synthetic_history,
            8: self._question_8_overfit,
        }
        for question_id, runner in runners.items():
            if question_id not in run_questions:
                frame = pd.DataFrame([{"status": "skipped", "question_id": question_id}])
                summary = _answer_summary(question_id, frame.to_dict("records"), "inconclusive", "Question skipped by config.")
            else:
                frame, summary = runner()
            self.question_frames[question_id] = frame
            self.summary_rows.append(summary)
            frame.to_csv(self.output_dir / QUESTION_OUTPUTS[question_id], index=False)

        summary_df = pd.DataFrame(self.summary_rows)
        summary_df.to_csv(self.output_dir / "open_questions_summary.csv", index=False)
        self._write_report(summary_df)
        save_run_config(
            self.output_dir,
            {
                "command": "run-open-questions",
                "config_path": str(self.config_path) if self.config_path is not None else None,
                "pack_name": self.pack_name,
                **self.pack,
            },
        )
        self._print_terminal_summary(summary_df)
        return summary_df

    def _common_config(
        self,
        *,
        experiment_name: str,
        symbols: Sequence[str],
        strategy_name: str,
        strategy_params: Optional[Dict[str, Any]] = None,
        parameter_grid: Optional[Dict[str, Any]] = None,
        benchmark_symbol: str = "TQQQ",
        asset_config: Optional[Dict[str, Any]] = None,
        trade_symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Any = None,
        output_name: Optional[str] = None,
        use_synthetic: bool = False,
    ) -> Dict[str, Any]:
        output_name = output_name or experiment_name
        config: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "symbols": _unique_symbols(symbols),
            "start_date": str(start_date or self.pack.get("start_date", "2011-01-01")),
            "end_date": self.pack.get("end_date") if end_date is None else end_date,
            "strategy_name": strategy_name,
            "benchmark_symbol": str(benchmark_symbol).upper(),
            "transaction_cost_bps": float(self.pack.get("transaction_cost_bps", 25.0)),
            "train_years": int(self.pack.get("train_years", 5)),
            "test_years": int(self.pack.get("test_years", 1)),
            "objective": str(self.pack.get("objective", "objective_final_ratio")),
            "output_dir": str(self.output_dir / "case_outputs" / output_name),
            "n_jobs": int(self.pack.get("n_jobs", 1)),
        }
        if parameter_grid is not None:
            config["parameter_grid"] = parameter_grid
        else:
            config["strategy_params"] = strategy_params or {}
        if asset_config is not None:
            config["asset_config"] = asset_config
        if trade_symbol is not None:
            config["trade_symbol"] = str(trade_symbol).upper()
        if use_synthetic:
            config.update(
                {
                    "use_synthetic_leverage": True,
                    "synthetic_base_symbol": str(self.pack.get("synthetic_base_symbol", "QQQ")).upper(),
                    "synthetic_leverage": float(self.pack.get("synthetic_leverage", 3.0)),
                    "synthetic_expense_ratio": float(self.pack.get("synthetic_expense_ratio", 0.0095)),
                    "synthetic_financing_spread": float(self.pack.get("synthetic_financing_spread", 0.0)),
                }
            )

        for key in ("data_csv", "cache_dir", "use_csv_if_exists", "download_ohlc", "execution_model"):
            if key in self.pack:
                config[key] = self.pack[key]

        global_overrides = self.pack.get("global_overrides", {})
        if isinstance(global_overrides, dict):
            for key, value in global_overrides.items():
                if key in {"experiment_name", "strategy_name", "strategy_params", "parameter_grid", "output_dir"}:
                    continue
                config[key] = copy.deepcopy(value)

        return config

    def _run_case(
        self,
        *,
        question_id: int,
        test_name: str,
        candidate_config: str,
        config: Dict[str, Any],
        compare_symbol: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CaseResult:
        compare_symbol = str(compare_symbol or self.benchmark_symbol).upper()
        extra = extra or {}
        base_row = {
            "question_id": question_id,
            "question": QUESTION_TEXT[question_id],
            "test_name": test_name,
            "candidate_config": candidate_config,
            "strategy_name": config.get("strategy_name", ""),
            "native_benchmark_symbol": str(config.get("benchmark_symbol", "")).upper(),
            "comparison_benchmark_symbol": compare_symbol,
            "transaction_cost_bps": config.get("transaction_cost_bps", np.nan),
            "train_years": config.get("train_years", np.nan),
            "test_years": config.get("test_years", np.nan),
            **extra,
        }
        try:
            data = _load_price_data(config)
            wf_table, stitched, grid_size = _run_walk_forward(config, data)
            if stitched.empty:
                raise ValueError("walk-forward produced no stitched equity")
            summary, yearly = compare_to_benchmark(stitched, data, benchmark_symbol=compare_symbol)
            row = {**base_row, **summary.iloc[0].to_dict()}
            native_symbol = str(config.get("benchmark_symbol", compare_symbol)).upper()
            if native_symbol != compare_symbol and native_symbol in data.columns:
                native_summary, _ = compare_to_benchmark(stitched, data, benchmark_symbol=native_symbol)
                native = native_summary.iloc[0]
                row["native_final_equity_ratio"] = native.get("final_equity_ratio", np.nan)
                row["native_strategy_cagr"] = native.get("strategy_cagr", np.nan)
                row["native_benchmark_cagr"] = native.get("benchmark_cagr", np.nan)
            row.update(
                {
                    "status": "ok",
                    "error": "",
                    "grid_size": int(grid_size),
                    "walk_forward_windows": int(len(wf_table)),
                    "avg_exposure": _safe_float(stitched["position"].mean()) if "position" in stitched.columns else np.nan,
                    "avg_leveraged_exposure": _safe_float(stitched["leveraged_exposure"].mean()) if "leveraged_exposure" in stitched.columns else np.nan,
                    "avg_risk_off_weight": _safe_float(stitched["risk_off_weight"].mean()) if "risk_off_weight" in stitched.columns else np.nan,
                    "total_turnover": _safe_float(stitched["turnover"].sum()) if "turnover" in stitched.columns else np.nan,
                    "total_cost": _safe_float(stitched["cost"].sum()) if "cost" in stitched.columns else np.nan,
                    "rebound_year_capture": summary.iloc[0].get("average_relative_return_in_rebound_years", np.nan),
                }
            )
            return CaseResult(row, config, data, wf_table, stitched, summary, yearly)
        except Exception as exc:
            row = {
                **base_row,
                "status": "error",
                "error": str(exc),
                "final_equity_ratio": np.nan,
                "strategy_max_dd": np.nan,
            }
            return CaseResult(row, config, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    def _core_overlay_params(
        self,
        *,
        core_exposure: float = 0.65,
        rebound: bool = False,
        governor: bool = False,
        risk_off_symbol: str = "CASH",
        risk_off_weight: float = 0.0,
        market_components: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "core_exposure": float(core_exposure),
            "overlay_max": float(self.pack.get("core_overlay_overlay_max", 0.35)),
            "max_total_exposure": float(self.pack.get("core_overlay_max_total_exposure", 1.25)),
            "trend_window": int(self.pack.get("core_overlay_trend_window", 150)),
            "fast_trend_window": int(self.pack.get("core_overlay_fast_trend_window", 50)),
            "momentum_window": int(self.pack.get("core_overlay_momentum_window", 63)),
            "vol_window": int(self.pack.get("core_overlay_vol_window", 20)),
            "vol_cap": float(self.pack.get("core_overlay_vol_cap", 0.026)),
            "crash_cut_exposure": float(self.pack.get("core_overlay_crash_cut_exposure", 0.25)),
            "rebound_boost": bool(self.pack.get("core_overlay_rebound_boost", False)),
            "risk_off_symbol": str(risk_off_symbol).upper(),
            "risk_off_weight": float(risk_off_weight),
        }
        if rebound:
            params.update(self._rebound_params())
        if governor:
            params.update(self._governor_params())
        if market_components is not None:
            params.update(self._market_internals_params(market_components))
        return params

    def _regime_params(
        self,
        *,
        rebound: bool = False,
        governor: bool = False,
        risk_off_symbol: str = "QQQ",
        risk_off_weight: Optional[float] = None,
        market_components: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "trend_window": int(self.pack.get("regime_trend_window", 150)),
            "momentum_window": int(self.pack.get("regime_momentum_window", 20)),
            "vol_window": int(self.pack.get("regime_vol_window", 20)),
            "trend_on": float(self.pack.get("regime_trend_on", 0.0)),
            "trend_off": float(self.pack.get("regime_trend_off", -0.01)),
            "mom_on": float(self.pack.get("regime_mom_on", 0.0)),
            "mom_off": float(self.pack.get("regime_mom_off", -0.02)),
            "vol_cap": float(self.pack.get("regime_vol_cap", 0.026)),
            "risk_on_leverage": float(self.pack.get("regime_risk_on_leverage", 1.0)),
            "risk_off_qqq_position": float(self.pack.get("regime_risk_off_qqq_position", 1.0)),
            "transition_position": float(self.pack.get("regime_transition_position", 0.35)),
            "min_hold_days": int(self.pack.get("regime_min_hold_days", 3)),
            "cooldown_days": int(self.pack.get("regime_cooldown_days", 2)),
            "risk_off_symbol": str(risk_off_symbol).upper(),
        }
        if risk_off_weight is not None:
            params["risk_off_weight"] = float(risk_off_weight)
        if rebound:
            params.update(self._rebound_params())
        if governor:
            params.update(self._governor_params())
        if market_components is not None:
            params.update(self._market_internals_params(market_components))
        return params

    def _vol_target_params(
        self,
        *,
        max_exposure: float = 1.25,
        governor: bool = False,
        market_components: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "target_ann_vol": float(self.pack.get("vol_target_ann_vol", 0.60)),
            "realized_vol_window": int(self.pack.get("vol_realized_vol_window", 20)),
            "min_exposure": float(self.pack.get("vol_min_exposure", 0.25)),
            "max_exposure": float(max_exposure),
            "trend_window": int(self.pack.get("vol_trend_window", 150)),
            "momentum_window": int(self.pack.get("vol_momentum_window", 63)),
            "trend_multiplier_below_ma": float(self.pack.get("vol_trend_multiplier_below_ma", 0.50)),
            "momentum_boost": float(self.pack.get("vol_momentum_boost", 1.15)),
            "crash_vol_cutoff": float(self.pack.get("vol_crash_vol_cutoff", 0.050)),
            "crash_exposure": float(self.pack.get("vol_crash_exposure", 0.25)),
            "risk_off_symbol": str(self.pack.get("vol_risk_off_symbol", "CASH")).upper(),
            "risk_off_weight": float(self.pack.get("vol_risk_off_weight", 0.0)),
        }
        if governor:
            params.update(self._governor_params())
        if market_components is not None:
            params.update(self._market_internals_params(market_components))
        return params

    def _rebound_params(self) -> Dict[str, Any]:
        return {
            "use_rebound_module": True,
            "rolling_high_window": int(self.pack.get("rebound_rolling_high_window", 126)),
            "drawdown_trigger": float(self.pack.get("rebound_drawdown_trigger", -0.20)),
            "rebound_momentum_window": int(self.pack.get("rebound_momentum_window", 10)),
            "rebound_momentum_threshold": float(self.pack.get("rebound_momentum_threshold", 0.05)),
            "reclaim_ma_window": int(self.pack.get("rebound_reclaim_ma_window", 20)),
            "rebound_position": float(self.pack.get("rebound_position", 1.25)),
            "rebound_hold_days": int(self.pack.get("rebound_hold_days", 10)),
        }

    def _governor_params(self) -> Dict[str, Any]:
        return {
            "use_drawdown_governor": True,
            "portfolio_dd_trigger": float(self.pack.get("governor_portfolio_dd_trigger", -0.30)),
            "qqq_dd_trigger": float(self.pack.get("governor_qqq_dd_trigger", -0.15)),
            "reduced_exposure": float(self.pack.get("governor_reduced_exposure", 0.50)),
            "recovery_ma_window": int(self.pack.get("governor_recovery_ma_window", 20)),
            "recovery_momentum_window": int(self.pack.get("governor_recovery_momentum_window", 10)),
            "recovery_momentum_threshold": float(self.pack.get("governor_recovery_momentum_threshold", 0.05)),
            "max_days_reduced": int(self.pack.get("governor_max_days_reduced", 40)),
        }

    def _market_internals_params(self, components: Sequence[str]) -> Dict[str, Any]:
        return {
            "use_market_internals": True,
            "market_internals_signal_window": int(self.pack.get("market_internals_signal_window", 100)),
            "market_internals_risk_on_threshold": float(self.pack.get("market_internals_risk_on_threshold", 0.60)),
            "market_internals_risk_off_threshold": float(self.pack.get("market_internals_risk_off_threshold", 0.40)),
            "market_internals_components": list(components),
        }

    def _question_1_exposure_floor(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        values = [float(v) for v in _as_list(self.pack.get("core_exposure_values"), (0.25, 0.50, 0.65, 0.80, 1.00))]
        for core_exposure in values:
            config = self._common_config(
                experiment_name=f"q1_core_exposure_{core_exposure:g}",
                symbols=["TQQQ", "QQQ", "CASH"],
                strategy_name="core_overlay",
                strategy_params=self._core_overlay_params(core_exposure=core_exposure),
            )
            result = self._run_case(
                question_id=1,
                test_name=f"core_exposure_{core_exposure:g}",
                candidate_config=f"core_overlay_core_{core_exposure:g}",
                config=config,
                extra={"core_exposure": core_exposure},
            )
            rows.append(result.row)

        frame = _question_rows_to_frame(rows)
        best = _best_ok_row(rows)
        low_row = next((row for row in rows if row.get("status") == "ok" and row.get("core_exposure") == min(values)), None)
        if best is None or low_row is None:
            answer = "inconclusive"
            caveats = "No successful exposure-floor comparison was produced."
        else:
            improvement = _safe_float(best.get("final_equity_ratio")) - _safe_float(low_row.get("final_equity_ratio"))
            if _safe_float(best.get("core_exposure")) >= 0.80 and improvement > 0.02:
                answer = "yes"
                caveats = "Higher core exposure materially improved same-period final_equity_ratio; drawdown tradeoff still needs review."
            else:
                answer = "no"
                caveats = "Higher core exposure did not clearly dominate the low-exposure baseline in this targeted pass."
        return frame, _answer_summary(1, rows, answer, caveats, best)

    def _question_2_rebound(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        cases = [
            ("regime_without_rebound", "regime", self._regime_params(rebound=False), ["TQQQ", "QQQ"]),
            ("regime_with_rebound", "regime", self._regime_params(rebound=True), ["TQQQ", "QQQ"]),
            ("core_overlay_without_rebound", "core_overlay", self._core_overlay_params(rebound=False), ["TQQQ", "QQQ", "CASH"]),
            ("core_overlay_with_rebound", "core_overlay", self._core_overlay_params(rebound=True), ["TQQQ", "QQQ", "CASH"]),
        ]
        for test_name, strategy_name, params, symbols in cases:
            config = self._common_config(
                experiment_name=f"q2_{test_name}",
                symbols=symbols,
                strategy_name=strategy_name,
                strategy_params=params,
            )
            result = self._run_case(
                question_id=2,
                test_name=test_name,
                candidate_config=test_name,
                config=config,
                extra={"uses_rebound_module": "with_rebound" in test_name},
            )
            row = result.row
            for year in (2019, 2020, 2023):
                strategy_return = _year_value(result.yearly, year, "strategy_return")
                benchmark_return = _year_value(result.yearly, year, "benchmark_return")
                row[f"strategy_return_{year}"] = strategy_return
                row[f"benchmark_return_{year}"] = benchmark_return
                row[f"relative_return_{year}"] = (
                    strategy_return - benchmark_return
                    if pd.notna(strategy_return) and pd.notna(benchmark_return)
                    else np.nan
                )
            strategy_post_2022 = _compound_years(result.yearly, 2023, "strategy_return")
            benchmark_post_2022 = _compound_years(result.yearly, 2023, "benchmark_return")
            row["strategy_post_2022_recovery_return"] = strategy_post_2022
            row["benchmark_post_2022_recovery_return"] = benchmark_post_2022
            row["relative_post_2022_recovery_return"] = (
                strategy_post_2022 - benchmark_post_2022
                if pd.notna(strategy_post_2022) and pd.notna(benchmark_post_2022)
                else np.nan
            )
            rows.append(row)

        frame = _question_rows_to_frame(rows)
        deltas = []
        for prefix in ("regime", "core_overlay"):
            without = next((row for row in rows if row.get("test_name") == f"{prefix}_without_rebound"), None)
            with_rebound = next((row for row in rows if row.get("test_name") == f"{prefix}_with_rebound"), None)
            if without and with_rebound and without.get("status") == with_rebound.get("status") == "ok":
                deltas.append(
                    (
                        _safe_float(with_rebound.get("final_equity_ratio")) - _safe_float(without.get("final_equity_ratio")),
                        _safe_float(with_rebound.get("relative_post_2022_recovery_return"))
                        - _safe_float(without.get("relative_post_2022_recovery_return")),
                    )
                )
        if not deltas:
            answer = "inconclusive"
            caveats = "No complete with/without rebound pair succeeded."
        elif any(final_delta > 0.0 and recovery_delta > 0.0 for final_delta, recovery_delta in deltas):
            answer = "yes"
            caveats = "At least one rebound-enabled pair improved final_equity_ratio and post-2022 relative recovery."
        else:
            answer = "no"
            caveats = "Rebound did not improve both final_equity_ratio and post-2022 recovery in the targeted pair tests."
        return frame, _answer_summary(2, rows, answer, caveats)

    def _question_3_risk_off(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        risk_off_symbols = [str(v).upper() for v in _as_list(self.pack.get("risk_off_symbols"), DEFAULT_RISK_OFF_SYMBOLS)]
        for symbol in risk_off_symbols:
            config = self._common_config(
                experiment_name=f"q3_risk_off_{symbol.lower().replace('^', '')}",
                symbols=_unique_symbols(["TQQQ", "QQQ", symbol]),
                strategy_name="core_overlay",
                strategy_params=self._core_overlay_params(risk_off_symbol=symbol, risk_off_weight=1.0),
            )
            result = self._run_case(
                question_id=3,
                test_name=f"risk_off_{symbol}",
                candidate_config=f"core_overlay_risk_off_{symbol}",
                config=config,
                extra={"risk_off_symbol": symbol, "risk_off_weight": 1.0},
            )
            rows.append(result.row)

        cash = next((row for row in rows if row.get("risk_off_symbol") == "CASH" and row.get("status") == "ok"), None)
        qqq = next((row for row in rows if row.get("risk_off_symbol") == "QQQ" and row.get("status") == "ok"), None)
        if cash is None or qqq is None:
            answer = "inconclusive"
            caveats = "CASH and QQQ risk-off rows were not both successful."
        elif _safe_float(cash.get("final_equity_ratio")) > _safe_float(qqq.get("final_equity_ratio")):
            answer = "yes"
            caveats = "CASH beat QQQ as the risk-off sleeve under identical strategy parameters."
        else:
            answer = "no"
            caveats = "QQQ matched or beat CASH as the risk-off sleeve under identical strategy parameters."
        return _question_rows_to_frame(rows), _answer_summary(3, rows, answer, caveats)

    def _question_4_adaptive_leverage(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        max_exposures = [float(v) for v in _as_list(self.pack.get("adaptive_max_exposures"), (1.00, 1.25, 1.50, 2.00))]
        for max_exposure in max_exposures:
            config = self._common_config(
                experiment_name=f"q4_vol_target_max_{max_exposure:g}",
                symbols=["TQQQ", "QQQ", "CASH"],
                strategy_name="vol_target",
                strategy_params=self._vol_target_params(max_exposure=max_exposure),
            )
            result = self._run_case(
                question_id=4,
                test_name=f"vol_target_max_{max_exposure:g}",
                candidate_config=f"vol_target_max_exposure_{max_exposure:g}",
                config=config,
                extra={"max_exposure": max_exposure},
            )
            rows.append(result.row)

        best = _best_ok_row(rows)
        one_x = next((row for row in rows if row.get("status") == "ok" and _safe_float(row.get("max_exposure")) == 1.0), None)
        if best is None or one_x is None:
            answer = "inconclusive"
            caveats = "No complete 1.0x versus higher-max-exposure comparison was produced."
        elif _safe_float(best.get("max_exposure")) > 1.0 and _safe_float(best.get("final_equity_ratio")) > _safe_float(one_x.get("final_equity_ratio")):
            answer = "yes"
            caveats = "A higher max_exposure variant improved final_equity_ratio after configured transaction costs."
        else:
            answer = "no"
            caveats = "Extra max exposure did not improve final_equity_ratio over the 1.0 cap in this targeted pass."
        return _question_rows_to_frame(rows), _answer_summary(4, rows, answer, caveats, best)

    def _question_5_market_internals(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        variants = [
            ("qqq_only_trend_momentum_vol", None),
            ("qqq_spy_credit", ("qqq_trend", "spy_trend", "credit_risk")),
            ("qqq_breadth_credit_vix", ("qqq_trend", "spy_trend", "equal_weight_strength", "small_cap_strength", "credit_risk", "vix_regime")),
        ]
        for test_name, components in variants:
            symbols = ["TQQQ", "QQQ", "CASH"]
            if components is not None:
                symbols.extend(MARKET_INTERNALS_SYMBOLS)
            config = self._common_config(
                experiment_name=f"q5_{test_name}",
                symbols=_unique_symbols(symbols),
                strategy_name="core_overlay",
                strategy_params=self._core_overlay_params(market_components=components),
            )
            result = self._run_case(
                question_id=5,
                test_name=test_name,
                candidate_config=f"core_overlay_{test_name}",
                config=config,
                extra={
                    "uses_market_internals": components is not None,
                    "market_internals_components": ";".join(components or ()),
                },
            )
            rows.append(result.row)

        base = next((row for row in rows if row.get("test_name") == "qqq_only_trend_momentum_vol" and row.get("status") == "ok"), None)
        best = _best_ok_row(rows)
        if base is None or best is None:
            answer = "inconclusive"
            caveats = "The QQQ-only baseline or internals variants failed."
        elif best.get("test_name") != "qqq_only_trend_momentum_vol" and _safe_float(best.get("final_equity_ratio")) > _safe_float(base.get("final_equity_ratio")):
            answer = "yes"
            caveats = "A market-internals variant beat the QQQ-only baseline out of sample."
        else:
            answer = "no"
            caveats = "Market internals did not improve final_equity_ratio over the QQQ-only baseline in this pass."
        return _question_rows_to_frame(rows), _answer_summary(5, rows, answer, caveats, best)

    def _question_6_soxl(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        variants = [str(v) for v in _as_list(self.pack.get("soxl_variants"), ("soxl_core_overlay", "soxl_vol_target", "soxl_regime", "soxl_rotation", "rotation_tqqq_soxl_upro"))]
        for variant in variants:
            config = self._soxl_or_rotation_config(variant)
            result = self._run_case(
                question_id=6,
                test_name=variant,
                candidate_config=variant,
                config=config,
                compare_symbol="TQQQ",
                extra={"soxl_variant": variant},
            )
            row = result.row
            row["soxl_vs_tqqq_final_equity_ratio"] = row.get("final_equity_ratio", np.nan)
            row["soxl_vs_soxl_final_equity_ratio"] = row.get("native_final_equity_ratio", np.nan)
            rows.append(row)

        ok_rows = [row for row in rows if row.get("status") == "ok"]
        if not ok_rows:
            answer = "inconclusive"
            caveats = "No SOXL or rotation row completed successfully."
        elif any(_safe_float(row.get("final_equity_ratio")) > 1.0 for row in ok_rows):
            answer = "yes"
            caveats = "At least one SOXL or rotation candidate beat same-period TQQQ raw wealth; sector concentration remains a key caveat."
        else:
            answer = "no"
            caveats = "No SOXL or rotation candidate beat same-period TQQQ raw wealth in this targeted pass."
        return _question_rows_to_frame(rows), _answer_summary(6, rows, answer, caveats)

    def _question_7_synthetic_history(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        periods = self.pack.get("synthetic_periods")
        if periods is None:
            periods = [
                {"label": "real_tqqq_period", "start_date": "2011-01-01", "end_date": self.pack.get("end_date"), "use_synthetic": False},
                {"label": "synthetic_full_qqq_history", "start_date": "1999-03-10", "end_date": self.pack.get("end_date"), "use_synthetic": True},
                {"label": "synthetic_2000_2002", "start_date": "1999-03-10", "end_date": "2002-12-31", "use_synthetic": True, "train_years": 1, "test_years": 1},
                {"label": "synthetic_2008", "start_date": "2003-01-01", "end_date": "2008-12-31", "use_synthetic": True, "train_years": 5, "test_years": 1},
            ]
        for item in periods:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "synthetic_period"))
            use_synthetic = bool(item.get("use_synthetic", True))
            config = self._common_config(
                experiment_name=f"q7_{label}",
                symbols=["QQQ", "TQQQ", "CASH"],
                strategy_name="core_overlay",
                strategy_params=self._core_overlay_params(),
                start_date=str(item.get("start_date", "1999-03-10")),
                end_date=item.get("end_date"),
                use_synthetic=use_synthetic,
            )
            if "train_years" in item:
                config["train_years"] = int(item["train_years"])
            if "test_years" in item:
                config["test_years"] = int(item["test_years"])
            result = self._run_case(
                question_id=7,
                test_name=label,
                candidate_config=f"core_overlay_{label}",
                config=config,
                compare_symbol="TQQQ",
                extra={
                    "use_synthetic_leverage": use_synthetic,
                    "synthetic_period": label,
                    "walk_forward_note": "shortened train/test for early crash windows" if int(config["train_years"]) < 5 else "",
                },
            )
            rows.append(result.row)

        synthetic_rows = [
            row
            for row in rows
            if row.get("status") == "ok" and _bool_from_value(row.get("use_synthetic_leverage"))
        ]
        if not synthetic_rows:
            answer = "inconclusive"
            caveats = "No synthetic-history rows completed successfully."
        elif all(_safe_float(row.get("final_equity_ratio")) > 1.0 for row in synthetic_rows):
            answer = "yes"
            caveats = "All completed synthetic-history rows beat synthetic 3x QQQ buy-and-hold; early subperiod rows may use shortened training windows."
        else:
            answer = "no"
            caveats = "At least one completed synthetic-history row failed to beat synthetic 3x QQQ buy-and-hold."
        return _question_rows_to_frame(rows), _answer_summary(7, rows, answer, caveats)

    def _question_8_overfit(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        grid = {
            "core_exposure": [float(v) for v in _as_list(self.pack.get("q8_core_exposures"), (0.50, 0.65, 0.80))],
            "overlay_max": [float(v) for v in _as_list(self.pack.get("q8_overlay_maxes"), (0.35, 0.50))],
            "max_total_exposure": [float(self.pack.get("core_overlay_max_total_exposure", 1.25))],
            "trend_window": [int(self.pack.get("core_overlay_trend_window", 150))],
            "fast_trend_window": [int(self.pack.get("core_overlay_fast_trend_window", 50))],
            "momentum_window": [int(self.pack.get("core_overlay_momentum_window", 63))],
            "vol_window": [int(self.pack.get("core_overlay_vol_window", 20))],
            "vol_cap": [float(self.pack.get("core_overlay_vol_cap", 0.026))],
            "crash_cut_exposure": [float(self.pack.get("core_overlay_crash_cut_exposure", 0.25))],
            "rebound_boost": [False],
            "risk_off_symbol": ["CASH"],
            "risk_off_weight": [0.0],
        }
        config = self._common_config(
            experiment_name="q8_core_overlay_neighborhood",
            symbols=["TQQQ", "QQQ", "CASH"],
            strategy_name="core_overlay",
            parameter_grid=grid,
        )
        config["anti_overfit_validation"] = {
            "bootstrap_iterations": int(self.pack.get("bootstrap_iterations", self.pack.get("q8_bootstrap_iterations", 200))),
            "bootstrap_block_sizes": [20, 60],
        }
        result = self._run_case(
            question_id=8,
            test_name="core_overlay_neighborhood_validation",
            candidate_config="core_overlay_neighborhood_validation",
            config=config,
        )
        rows: List[Dict[str, Any]] = [result.row]
        validation_outputs: Dict[str, pd.DataFrame] = {}
        if result.row.get("status") == "ok":
            try:
                validation_outputs = run_anti_overfit_validation(
                    config=config,
                    data=result.data,
                    output_dir=self.output_dir / "question_8_validation_artifacts",
                    standard_wf_table=result.wf_table,
                    standard_stitched=result.stitched,
                    standard_summary=result.summary,
                    grid_size=int(result.row.get("grid_size", 0)),
                    walk_forward_runner=_run_walk_forward,
                )
            except Exception as exc:
                rows.append(
                    {
                        "question_id": 8,
                        "question": QUESTION_TEXT[8],
                        "test_name": "anti_overfit_validation",
                        "candidate_config": "core_overlay_neighborhood_validation",
                        "record_type": "validation_error",
                        "status": "error",
                        "error": str(exc),
                    }
                )
            for table_name, table in validation_outputs.items():
                if table_name == "standard_summary" or table.empty:
                    continue
                for _, table_row in table.iterrows():
                    row = {
                        "question_id": 8,
                        "question": QUESTION_TEXT[8],
                        "test_name": "anti_overfit_validation",
                        "candidate_config": "core_overlay_neighborhood_validation",
                        "record_type": table_name,
                        "status": table_row.get("status", "ok"),
                    }
                    row.update(table_row.to_dict())
                    rows.append(row)

        validation = validation_outputs.get("validation_summary", pd.DataFrame())
        if validation.empty:
            answer = "inconclusive"
            caveats = "Anti-overfit validation did not produce a validation summary."
        else:
            vrow = validation.iloc[0]
            if _bool_from_value(vrow.get("candidate_viable", False)):
                answer = "no"
                caveats = "The candidate passed the configured anti-overfit viability gates."
            elif _bool_from_value(vrow.get("standard_walk_forward_pass", False)):
                answer = "yes"
                caveats = "The standard walk-forward passed, but one or more alternate walk-forward, cost, or parameter-neighborhood gates failed."
            else:
                answer = "inconclusive"
                caveats = "The candidate did not pass the standard walk-forward gate, so overfitting is not the only failure mode."
        best = _best_ok_row([result.row])
        return _question_rows_to_frame(rows), _answer_summary(8, [result.row], answer, caveats, best)

    def _soxl_or_rotation_config(self, variant: str) -> Dict[str, Any]:
        variant = str(variant)
        soxl_asset_config = {
            "trade_asset": "SOXL",
            "primary_signal_asset": "SOXX",
            "secondary_filter_asset": "SMH",
            "benchmark_symbol": "SOXL",
            "risk_off_symbol": "BIL",
        }
        if variant == "soxl_core_overlay":
            return self._common_config(
                experiment_name="q6_soxl_core_overlay",
                symbols=["SOXL", "SOXX", "SMH", "QQQ", "TQQQ", "BIL"],
                strategy_name="core_overlay",
                strategy_params=self._core_overlay_params(core_exposure=0.50, risk_off_symbol="BIL", risk_off_weight=1.0),
                benchmark_symbol="SOXL",
                asset_config=soxl_asset_config,
            )
        if variant == "soxl_vol_target":
            params = self._vol_target_params(max_exposure=1.25)
            params["risk_off_symbol"] = "BIL"
            params["risk_off_weight"] = 1.0
            return self._common_config(
                experiment_name="q6_soxl_vol_target",
                symbols=["SOXL", "SOXX", "SMH", "QQQ", "TQQQ", "BIL"],
                strategy_name="vol_target",
                strategy_params=params,
                benchmark_symbol="SOXL",
                asset_config=soxl_asset_config,
            )
        if variant == "soxl_regime":
            return self._common_config(
                experiment_name="q6_soxl_regime",
                symbols=["SOXL", "SOXX", "SMH", "QQQ", "TQQQ", "BIL"],
                strategy_name="regime",
                strategy_params=self._regime_params(risk_off_symbol="BIL", risk_off_weight=1.0),
                benchmark_symbol="SOXL",
                asset_config=soxl_asset_config,
            )
        if variant == "soxl_rotation":
            return self._common_config(
                experiment_name="q6_soxl_rotation",
                symbols=["SOXL", "SOXX", "SMH", "QQQ", "TQQQ", "UPRO", "BIL", "CASH"],
                strategy_name="rotation",
                strategy_params={
                    "risk_assets": ["SOXL", "TQQQ"],
                    "rebalance_frequency": "monthly",
                    "top_n": 1,
                    "max_asset_weight": 1.0,
                    "max_total_leveraged_exposure": 1.0,
                    "trend_filter_symbol": "SOXX",
                    "trend_filter_window": 100,
                    "defensive_asset": "BIL",
                    "require_positive_momentum": True,
                },
                benchmark_symbol="SOXL",
                asset_config=soxl_asset_config,
            )
        return self._common_config(
            experiment_name="q6_rotation_tqqq_soxl_upro",
            symbols=["TQQQ", "SOXL", "UPRO", "QQQ", "SPY", "CASH", "BIL"],
            strategy_name="rotation",
            strategy_params={
                "risk_assets": ["TQQQ", "SOXL", "UPRO"],
                "rebalance_frequency": "monthly",
                "top_n": 1,
                "max_asset_weight": 1.0,
                "max_total_leveraged_exposure": 1.0,
                "trend_filter_symbol": "QQQ",
                "trend_filter_window": 100,
                "defensive_asset": "CASH",
                "require_positive_momentum": True,
            },
            benchmark_symbol="TQQQ",
        )

    def _write_report(self, summary_df: pd.DataFrame) -> Path:
        report_path = self.output_dir / "open_questions_report.md"
        lines = [
            f"# Open Questions Experiment Pack: {self.pack_name}",
            "",
            "This report summarizes empirical backtest outputs only. It is not an investment recommendation.",
            "",
            "## Summary",
            _markdown_table(summary_df, max_rows=20),
            "",
        ]
        for question_id in range(1, 9):
            frame = self.question_frames.get(question_id, pd.DataFrame())
            summary_row = summary_df.loc[summary_df["question_id"] == question_id]
            row = summary_row.iloc[0].to_dict() if not summary_row.empty else {}
            lines.extend(
                [
                    f"## Question {question_id}",
                    QUESTION_TEXT[question_id],
                    "",
                    f"- answer: {row.get('answer', 'inconclusive')}",
                    f"- best candidate config: {row.get('best_candidate_config', '')}",
                    f"- final_equity_ratio versus TQQQ: {row.get('final_equity_ratio_versus_tqqq', np.nan)}",
                    f"- max drawdown: {row.get('max_drawdown', np.nan)}",
                    f"- caveats: {row.get('caveats', '')}",
                    "",
                    _markdown_table(frame, max_rows=12),
                    "",
                ]
            )
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def _print_terminal_summary(self, summary_df: pd.DataFrame) -> None:
        print("\n=== Open questions summary ===")
        for _, row in summary_df.iterrows():
            ratio = _safe_float(row.get("final_equity_ratio_versus_tqqq"))
            ratio_text = "n/a" if pd.isna(ratio) else f"{ratio:.6f}"
            print(f"Q{int(row['question_id'])}: {row['answer']} | best={row.get('best_candidate_config', '')} | ratio={ratio_text}")


def run_open_questions_config(config_path: Path) -> pd.DataFrame:
    config_path = Path(config_path)
    pack = load_yaml_file(config_path)
    return OpenQuestionsExperimentPack(pack, config_path=config_path).run()
