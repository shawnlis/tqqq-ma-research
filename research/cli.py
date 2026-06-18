from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .data import load_prices
from .auto_paper_ledger import initialize_auto_paper_ledger, update_auto_paper_ledger
from .baseline_gate import accept_baseline_regression, check_blockers, create_baseline_regression_report
from .experiments import ExperimentInfrastructureError, dry_run_experiment_config, run_batch_config, run_experiment_config
from .execution_drift import run_execution_drift_tracking
from .execution_financing import run_execution_financing_audit
from .final_voltarget_audit import run_final_voltarget_audit
from .metrics import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio
from .reports import (
    compare_to_benchmark,
    make_heatmap_table,
    print_benchmark_summary,
    save_benchmark_comparison,
    save_run_config,
    summarize_param_stability,
    summarize_stitched,
)
from .strategies.ma_strategy import make_param_grid
from .strategies.regime_strategy import make_regime_param_grid
from .validation import (
    grid_search_on_window,
    log_multiple_testing_warning,
    log_workload_hint,
    regime_search_on_window,
    walk_forward_regime_search,
    walk_forward_search,
    walk_forward_search_ensemble,
    walk_forward_windows,
)
from .audit import audit_baseline_data, audit_config, audit_data
from .candidates import extract_final_candidates
from .comparison import compare_run_files
from .controlled_runs import GATE_CONTROLLED_RUNS, summarize_controlled_runs
from .data_snapshot import freeze_data_snapshot, verify_data_snapshot
from .open_questions import run_open_questions_config
from .paper_trade_reconciliation import reconcile_paper_trades
from .replay import replay_regime_windows
from .regime_allocation_audit import audit_regime_allocation
from .tournament import run_tournament_config
from .visualization import generate_equity_visualizations
from .voltarget_simplification_battle import run_simplification_battle
from .voltarget_execution_semantics import run_execution_semantics_audit
from .voltarget_daily_monitor import run_automated_voltarget_paper_monitor, run_daily_voltarget_monitor
from .voltarget_live_monitor import generate_voltarget_signal
from .voltarget_monitor_health import check_voltarget_monitor_health
from .voltarget_risk_dashboard import build_voltarget_risk_dashboard
from .voltarget_timing_audit import audit_voltarget_timing
from .voltarget_stage import summarize_voltarget_stage
from .voltarget_fair_leverage import write_stage2_fair_leverage_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_search(output_dir: str = "./ma_search_output_v7", objective: Optional[str] = None) -> None:
    data = load_prices(start="2011-01-01", use_csv_if_exists=True)

    # Search space kept moderate by default
    short_range = range(3, 21)                   # 3..20
    long_range = range(20, 201, 10)             # 20..200 step 10
    ma_types = ("sma", "ema")
    signal_assets = ("TQQQ", "QQQ")
    thresholds = (0.0, 0.005, 0.01)             # 0%, 0.5%, 1.0%
    cooldown_days_options = (0, 3)
    qqq_filter_options = (False, True)
    qqq_filter_windows = (200,)

    train_objective = str(objective or "excess_score")
    transaction_cost_bps = 10.0
    n_jobs = 1                                  # set >1 for parallel search
    use_ensemble_wf = True
    ensemble_top_k = 5
    ensemble_weight_power = 2.0

    grid = make_param_grid(
        short_range=short_range,
        long_range=long_range,
        ma_types=ma_types,
        signal_assets=signal_assets,
        thresholds=thresholds,
        cooldown_days_options=cooldown_days_options,
        use_qqq_filter_options=qqq_filter_options,
        qqq_filter_windows=qqq_filter_windows,
    )

    logger.info("Loaded %d trading days", len(data))
    logger.info("Parameter grid size: %d", len(grid))
    log_multiple_testing_warning(len(grid), train_objective)

    wf_windows = walk_forward_windows(data.index, train_years=5, test_years=1)
    log_workload_hint(len(grid), len(wf_windows), n_jobs)

    # Full-sample research only
    full_results = grid_search_on_window(
        full_data=data,
        grid=grid,
        eval_start=data.index.min(),
        eval_end=data.index.max(),
        objective=train_objective,
        transaction_cost_bps=transaction_cost_bps,
        n_jobs=n_jobs,
    )

    print("\n=== Top 20 full-sample results (research only) ===")
    print(
        full_results[
            [
                "short",
                "long",
                "ma_type",
                "signal_asset",
                "threshold",
                "cooldown_days",
                "qqq_filter",
                "qqq_filter_window",
                "cagr",
                "sharpe",
                "max_dd",
                "calmar",
                "excess_cagr",
                "excess_sharpe",
                "dd_penalty",
                "excess_score",
                "trades",
                "avg_exposure",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # Walk-forward
    if use_ensemble_wf:
        wf_table, stitched = walk_forward_search_ensemble(
            data=data,
            grid=grid,
            train_objective=train_objective,
            transaction_cost_bps=transaction_cost_bps,
            train_years=5,
            test_years=1,
            n_jobs=n_jobs,
            top_k=ensemble_top_k,
            weight_power=ensemble_weight_power,
        )
    else:
        wf_table, stitched = walk_forward_search(
            data=data,
            grid=grid,
            train_objective=train_objective,
            transaction_cost_bps=transaction_cost_bps,
            train_years=5,
            test_years=1,
            n_jobs=n_jobs,
        )

    print("\n=== Walk-forward windows ===")
    if wf_table.empty:
        print("No walk-forward windows produced.")
    else:
        print(wf_table.to_string(index=False))

    print("\n=== Walk-forward stitched summary ===")
    benchmark_summary = None
    benchmark_yearly = None
    if stitched.empty:
        print("No stitched walk-forward equity available.")
    else:
        summary = summarize_stitched(stitched)
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")
        benchmark_summary, benchmark_yearly = compare_to_benchmark(stitched, data)
        print_benchmark_summary(benchmark_summary)

    # Buy & hold benchmark
    bh_ret = data["TQQQ"].pct_change().fillna(0.0)
    bh_equity = (1.0 + bh_ret).cumprod()

    print("\n=== Buy & Hold TQQQ full-sample ===")
    print(f"bh_final_equity: {bh_equity.iloc[-1]:.6f}")
    print(f"bh_cagr: {annualized_return(bh_equity):.6f}")
    print(f"bh_vol: {annualized_volatility(bh_ret):.6f}")
    print(f"bh_sharpe: {sharpe_ratio(bh_ret):.6f}")
    print(f"bh_max_dd: {max_drawdown(bh_equity):.6f}")

    # Stability
    stability_metric = train_objective
    stability = summarize_param_stability(
        full_results,
        metric=stability_metric,
        short_radius=2,
        long_radius=20,
        score_mode="difference",
    )

    heatmap = make_heatmap_table(
        results=full_results,
        metric=stability_metric,
        ma_type="ema",
        signal_asset="QQQ",
        threshold=0.0,
        cooldown_days=0,
        qqq_filter=True,
        qqq_filter_window=200,
    )

    print("\n=== Top 20 neighborhood-stable parameter sets ===")
    if stability.empty:
        print("No stability table available.")
    else:
        print(
            stability[
                [
                    "short",
                    "long",
                    "ma_type",
                    "signal_asset",
                    "threshold",
                    "cooldown_days",
                    "qqq_filter",
                    "qqq_filter_window",
                    f"{stability_metric}_point",
                    "neigh_mean",
                    "neigh_std",
                    "neigh_count",
                    "robustness_score",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    # Save outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_results.to_csv(out_dir / "grid_search_results.csv", index=False)
    wf_table.to_csv(out_dir / "walk_forward_windows.csv", index=False)
    if not stitched.empty:
        stitched.to_csv(out_dir / "walk_forward_stitched_equity.csv")
        save_benchmark_comparison(out_dir, benchmark_summary, benchmark_yearly)
    if not stability.empty:
        stability.to_csv(out_dir / "param_stability_summary.csv", index=False)
    if not heatmap.empty:
        heatmap.to_csv(out_dir / f"heatmap_ema_QQQsignal_filter_{stability_metric}.csv")

    save_run_config(
        out_dir,
        {
            "command": "run-ma-search",
            "data_start": "2011-01-01",
            "transaction_cost_bps": transaction_cost_bps,
            "train_objective": train_objective,
            "n_jobs": n_jobs,
            "short_range": list(short_range),
            "long_range": list(long_range),
            "ma_types": ma_types,
            "signal_assets": signal_assets,
            "thresholds": thresholds,
            "cooldown_days_options": cooldown_days_options,
            "qqq_filter_options": qqq_filter_options,
            "qqq_filter_windows": qqq_filter_windows,
            "train_years": 5,
            "test_years": 1,
            "use_ensemble_wf": use_ensemble_wf,
            "ensemble_top_k": ensemble_top_k,
            "ensemble_weight_power": ensemble_weight_power,
            "output_dir": str(out_dir),
        },
    )

    logger.info("Saved outputs to %s", out_dir.resolve())


def run_regime_search(
    output_dir: str = "./ma_search_output_v11",
    objective: Optional[str] = None,
    start_date: str = "2011-01-01",
    end_date: Optional[str] = None,
) -> None:
    data = load_prices(start=start_date, end=end_date, use_csv_if_exists=True)

    trend_windows = (150, 200, 250)
    momentum_windows = (20, 40)
    vol_windows = (20, 40)
    trend_on_levels = (0.0, 0.01)
    trend_off_levels = (-0.01, 0.0)
    mom_on_levels = (0.0, 0.02)
    mom_off_levels = (-0.02, 0.0)
    vol_caps = (0.018, 0.022, 0.026)
    risk_on_leverages = (1.00, 1.10, 1.25)
    risk_off_qqq_positions = (1.0,)
    transition_positions = (0.25, 0.40)
    min_hold_days_options = (3, 6)
    cooldown_days_options = (0, 2)

    train_objective = str(objective or "state_score")
    transaction_cost_bps = 10.0
    n_jobs = 1
    recent_days = 252

    grid = make_regime_param_grid(
        trend_windows=trend_windows,
        momentum_windows=momentum_windows,
        vol_windows=vol_windows,
        trend_on_levels=trend_on_levels,
        trend_off_levels=trend_off_levels,
        mom_on_levels=mom_on_levels,
        mom_off_levels=mom_off_levels,
        vol_caps=vol_caps,
        risk_on_leverages=risk_on_leverages,
        risk_off_qqq_positions=risk_off_qqq_positions,
        transition_positions=transition_positions,
        min_hold_days_options=min_hold_days_options,
        cooldown_days_options=cooldown_days_options,
    )

    logger.info("Loaded %d trading days", len(data))
    logger.info("Regime grid size: %d", len(grid))

    wf_windows = walk_forward_windows(data.index, train_years=5, test_years=1)
    logger.warning(
        "Regime workload estimate: %d evaluations before retries/errors. n_jobs=%d.",
        len(grid) * (1 + len(wf_windows)),
        n_jobs,
    )

    full_results = regime_search_on_window(
        full_data=data,
        grid=grid,
        eval_start=data.index.min(),
        eval_end=data.index.max(),
        objective=train_objective,
        transaction_cost_bps=transaction_cost_bps,
        n_jobs=n_jobs,
        recent_days=recent_days,
    )

    print("\n=== Top 20 full-sample regime results ===")
    print(
        full_results[
            [
                "trend_window",
                "momentum_window",
                "vol_window",
                "trend_on",
                "trend_off",
                "mom_on",
                "mom_off",
                "vol_cap",
                "risk_on_leverage",
                "risk_off_qqq_position",
                "transition_position",
                "min_hold_days",
                "cooldown_days",
                "cagr",
                "sharpe",
                "max_dd",
                "calmar",
                "excess_cagr",
                "excess_sharpe",
                "recent_excess_cagr",
                "recent_excess_sharpe",
                "state_score",
                "avg_exposure",
                "trades",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    wf_table, stitched = walk_forward_regime_search(
        data=data,
        grid=grid,
        train_objective=train_objective,
        transaction_cost_bps=transaction_cost_bps,
        train_years=5,
        test_years=1,
        n_jobs=n_jobs,
        recent_days=recent_days,
    )

    print("\n=== Regime walk-forward windows ===")
    if wf_table.empty:
        print("No walk-forward windows produced.")
    else:
        print(wf_table.to_string(index=False))

    print("\n=== Regime walk-forward stitched summary ===")
    benchmark_summary = None
    benchmark_yearly = None
    if stitched.empty:
        print("No stitched walk-forward equity available.")
    else:
        summary = summarize_stitched(stitched)
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")
        benchmark_summary, benchmark_yearly = compare_to_benchmark(stitched, data)
        print_benchmark_summary(benchmark_summary)

    bh_ret = data["TQQQ"].pct_change().fillna(0.0)
    bh_equity = (1.0 + bh_ret).cumprod()
    print("\n=== Buy & Hold TQQQ full-sample ===")
    print(f"bh_final_equity: {bh_equity.iloc[-1]:.6f}")
    print(f"bh_cagr: {annualized_return(bh_equity):.6f}")
    print(f"bh_vol: {annualized_volatility(bh_ret):.6f}")
    print(f"bh_sharpe: {sharpe_ratio(bh_ret):.6f}")
    print(f"bh_max_dd: {max_drawdown(bh_equity):.6f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_results.to_csv(out_dir / "regime_grid_search_results.csv", index=False)
    wf_table.to_csv(out_dir / "regime_walk_forward_windows.csv", index=False)
    if not stitched.empty:
        stitched.to_csv(out_dir / "regime_walk_forward_stitched_equity.csv")
        save_benchmark_comparison(out_dir, benchmark_summary, benchmark_yearly)

    save_run_config(
        out_dir,
        {
            "command": "run-existing-regime",
            "data_start": start_date,
            "data_end": end_date,
            "transaction_cost_bps": transaction_cost_bps,
            "train_objective": train_objective,
            "n_jobs": n_jobs,
            "recent_days": recent_days,
            "trend_windows": trend_windows,
            "momentum_windows": momentum_windows,
            "vol_windows": vol_windows,
            "trend_on_levels": trend_on_levels,
            "trend_off_levels": trend_off_levels,
            "mom_on_levels": mom_on_levels,
            "mom_off_levels": mom_off_levels,
            "vol_caps": vol_caps,
            "risk_on_leverages": risk_on_leverages,
            "risk_off_qqq_positions": risk_off_qqq_positions,
            "transition_positions": transition_positions,
            "min_hold_days_options": min_hold_days_options,
            "cooldown_days_options": cooldown_days_options,
            "train_years": 5,
            "test_years": 1,
            "output_dir": str(out_dir),
        },
    )

    logger.info("Saved regime outputs to %s", out_dir.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TQQQ strategy research runner")
    subparsers = parser.add_subparsers(dest="command")

    regime = subparsers.add_parser(
        "run-existing-regime",
        help="Reproduce the current regime search and walk-forward run.",
    )
    regime.add_argument(
        "--output-dir",
        default="./ma_search_output_v11",
        help="Directory for regime CSV outputs and run_config.json.",
    )
    regime.add_argument("--objective", help="Override the training objective name.")
    regime.add_argument("--start-date", default="2011-01-01", help="Start date for the regime data load.")
    regime.add_argument("--end-date", help="Inclusive end date for the regime data load.")

    ma = subparsers.add_parser(
        "run-ma-search",
        help="Run the existing moving-average search workflow.",
    )
    ma.add_argument(
        "--output-dir",
        default="./ma_search_output_v7",
        help="Directory for MA CSV outputs and run_config.json.",
    )
    ma.add_argument("--objective", help="Override the training objective name.")

    run_config = subparsers.add_parser(
        "run-config",
        help="Run one YAML experiment config.",
    )
    run_config.add_argument("config_path", help="Path to a YAML experiment config.")
    run_config.add_argument("--objective", help="Override the config objective name.")
    run_config.add_argument("--dry-run", action="store_true", help="Print workload estimate without running backtests.")

    run_batch = subparsers.add_parser(
        "run-batch",
        help="Run a YAML batch config containing multiple experiments.",
    )
    run_batch.add_argument("batch_path", help="Path to a YAML batch config.")
    run_batch.add_argument("--objective", help="Override each experiment objective name.")
    run_batch.add_argument("--max-configs", type=int, help="Run only the first N configs in the batch.")
    run_batch.add_argument("--dry-run", action="store_true", help="Print workload estimates without running backtests.")

    run_tournament = subparsers.add_parser(
        "run-tournament",
        help="Run a tournament YAML config across candidate strategy families.",
    )
    run_tournament.add_argument("tournament_path", help="Path to a YAML tournament config.")
    run_tournament.add_argument("--max-configs", type=int, help="Run only the first N tournament configs.")
    run_tournament.add_argument("--dry-run", action="store_true", help="Print workload estimates without running backtests.")
    run_tournament.add_argument(
        "--allow-long-run",
        action="store_true",
        help="Allow a full tournament whose estimated runtime exceeds the configured safety threshold.",
    )
    run_tournament.add_argument(
        "--accept-baseline-regression",
        action="store_true",
        help="Proceed despite BASELINE_GATE_FAILED.txt after explicitly accepting the baseline regression.",
    )

    summarize_voltarget_stage_parser = subparsers.add_parser(
        "summarize-voltarget-stage",
        help="Summarize staged VolTarget tournament outputs.",
    )
    summarize_voltarget_stage_parser.add_argument(
        "output_dir",
        help="Tournament stage output directory containing tournament_summary.csv.",
    )
    summarize_voltarget_stage_parser.add_argument(
        "--fair-leverage",
        action="store_true",
        help="Also write Stage 2 fair leverage benchmark summary/report from completed VolTarget outputs.",
    )

    plot_equity_curve = subparsers.add_parser(
        "plot-equity-curve",
        help="Generate equity-curve visualizations for the selected Stage 3 VolTarget candidate.",
    )
    plot_equity_curve.add_argument(
        "--input-dir",
        required=True,
        help="Stage 3 tournament output directory containing candidate decision artifacts.",
    )
    plot_equity_curve.add_argument(
        "--output-dir",
        required=True,
        help="Directory for visualization PNGs, summary CSV, and report.",
    )

    audit_final_voltarget = subparsers.add_parser(
        "audit-final-voltarget",
        help="Build the final audit pack for the selected Stage 3 VolTarget candidate.",
    )
    audit_final_voltarget.add_argument(
        "--input-dir",
        required=True,
        help="Stage 3 tournament output directory containing final VolTarget artifacts.",
    )
    audit_final_voltarget.add_argument(
        "--output-dir",
        required=True,
        help="Directory for final VolTarget audit CSV, Markdown, and chart outputs.",
    )


    execution_financing = subparsers.add_parser(
        "audit-voltarget-execution-financing",
        help="Replay the locked Stage 3 VolTarget candidate under execution, slippage, and financing assumptions.",
    )
    execution_financing.add_argument(
        "--input-dir",
        required=True,
        help="Stage 3 tournament output directory containing the selected VolTarget candidate.",
    )
    execution_financing.add_argument(
        "--output-dir",
        required=True,
        help="Directory for execution/financing summary, report, and heatmap outputs.",
    )
    execution_financing.add_argument(
        "--data-csv",
        help="Optional fixture CSV with Date, close, and OHLC columns for offline replay.",
    )
    execution_financing.add_argument(
        "--cache-dir",
        default="./price_cache",
        help="Price cache directory used when --data-csv is not supplied.",
    )
    execution_semantics = subparsers.add_parser(
        "audit-voltarget-execution-semantics",
        help="Audit VolTarget trade lifecycle, return decomposition, and valid execution semantics.",
    )
    execution_semantics.add_argument(
        "--input-dir",
        default="outputs/tournament_voltarget_stage3",
        help="Stage 3 tournament output directory containing the selected VolTarget candidate.",
    )
    execution_semantics.add_argument(
        "--output-dir",
        required=True,
        help="Directory for execution semantics CSV, report, and chart outputs.",
    )
    execution_semantics.add_argument(
        "--data-csv",
        help="Optional fixture CSV with Date, close, and OHLC columns for offline replay.",
    )
    execution_semantics.add_argument(
        "--cache-dir",
        default="./price_cache",
        help="Price cache directory used when --data-csv is not supplied.",
    )
    simplification_battle = subparsers.add_parser(
        "compare-voltarget-simplification",
        help="Compare the locked Stage 3 VolTarget candidate against simpler alternatives without re-optimization.",
    )
    simplification_battle.add_argument(
        "--input-dir",
        required=True,
        help="Stage 3 tournament output directory containing the selected VolTarget candidate.",
    )
    simplification_battle.add_argument(
        "--output-dir",
        required=True,
        help="Directory for simplification comparison CSV, report, and charts.",
    )
    simplification_battle.add_argument(
        "--data-csv",
        help="Optional fixture CSV with Date, close, and OHLC columns for offline replay.",
    )
    simplification_battle.add_argument(
        "--cache-dir",
        default="./price_cache",
        help="Price cache directory used when --data-csv is not supplied.",
    )
    risk_dashboard = subparsers.add_parser(
        "build-voltarget-risk-dashboard",
        help="Build a paper-trading-only risk policy dashboard for the locked Stage 3 VolTarget candidate.",
    )
    risk_dashboard.add_argument(
        "--config",
        required=True,
        help="Path to voltarget_risk_policy.yaml.",
    )
    risk_dashboard.add_argument(
        "--input-dir",
        default="outputs/tournament_voltarget_stage3",
        help="Stage 3 tournament output directory containing the selected VolTarget candidate.",
    )
    risk_dashboard.add_argument(
        "--output-dir",
        required=True,
        help="Directory for risk dashboard Markdown, CSV, and chart outputs.",
    )
    voltarget_signal = subparsers.add_parser(
        "generate-voltarget-signal",
        help="Generate a paper-trading-only signal monitor report for the locked Stage 3 VolTarget candidate.",
    )
    voltarget_signal.add_argument(
        "--config",
        required=True,
        help="Path to voltarget_live_monitor.yaml.",
    )
    voltarget_signal.add_argument(
        "--output-dir",
        required=True,
        help="Directory for signal JSON, history CSV, report, charts, and data-quality output.",
    )
    voltarget_signal.add_argument(
        "--data-csv",
        help="Optional fixture CSV with Date, close, and OHLC columns for offline replay.",
    )
    daily_voltarget_monitor = subparsers.add_parser(
        "run-daily-voltarget-monitor",
        help="Run the paper-trading-only daily VolTarget monitor wrapper.",
    )
    daily_voltarget_monitor.add_argument(
        "--config",
        required=True,
        help="Path to voltarget_live_monitor.yaml.",
    )
    auto_paper_ledger = subparsers.add_parser(
        "update-auto-paper-ledger",
        help="Update the automated VolTarget paper ledger from generated signals and observable OHLC prices.",
    )
    auto_paper_ledger.add_argument(
        "--config",
        required=True,
        help="Path to voltarget_live_monitor.yaml.",
    )
    auto_paper_ledger.add_argument(
        "--signal-dir",
        default="outputs/live_signal",
        help="Directory containing signal_history.csv and signal_today.json.",
    )
    auto_paper_ledger.add_argument(
        "--output-dir",
        default="outputs/auto_paper_ledger",
        help="Directory for auto paper ledger outputs.",
    )
    auto_paper_ledger.add_argument(
        "--data-csv",
        help="Optional fixture CSV with Date and adjusted OHLC columns.",
    )
    init_auto_paper_ledger = subparsers.add_parser(
        "initialize-auto-paper-ledger",
        help="Initialize the live automated VolTarget paper ledger state without broker or order actions.",
    )
    init_auto_paper_ledger.add_argument("--config", required=True, help="Path to voltarget_live_monitor.yaml.")
    init_auto_paper_ledger.add_argument("--start-date", required=True, help="Live paper ledger start date, YYYY-MM-DD.")
    init_auto_paper_ledger.add_argument("--starting-equity", type=float, required=True, help="Starting paper equity.")
    init_auto_paper_ledger.add_argument("--output-dir", default="outputs/auto_paper_ledger", help="Auto paper ledger output directory.")
    init_auto_paper_ledger.add_argument(
        "--confirm-reset",
        action="store_true",
        help="Create/reset auto_paper_ledger.csv. Without this, only the state file is written unless no ledger exists.",
    )
    automated_voltarget_monitor = subparsers.add_parser(
        "run-automated-voltarget-paper-monitor",
        help="Run the full paper-monitor workflow with automated ledger, drift tracking, financing, dashboard, and health checks.",
    )
    automated_voltarget_monitor.add_argument(
        "--config",
        required=True,
        help="Path to voltarget_live_monitor.yaml.",
    )
    automated_voltarget_monitor.add_argument(
        "--data-csv",
        help="Optional fixture CSV with Date and adjusted OHLC columns.",
    )
    reconcile_paper_trades_parser = subparsers.add_parser(
        "reconcile-paper-trades",
        help="Reconcile manual VolTarget paper ledger rows against generated paper signals.",
    )
    reconcile_paper_trades_parser.add_argument(
        "--signal-dir",
        required=True,
        help="Directory containing signal_history.csv from the VolTarget paper monitor.",
    )
    reconcile_paper_trades_parser.add_argument(
        "--ledger",
        required=True,
        help="Manual CSV ledger path. Missing ledgers are treated as all signal rows missing manual entries.",
    )
    reconcile_paper_trades_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for paper trade reconciliation CSV and Markdown report.",
    )
    execution_drift = subparsers.add_parser(
        "track-execution-drift",
        help="Track VolTarget paper-monitor execution drift against observable TQQQ OHLC prices.",
    )
    execution_drift.add_argument(
        "--config",
        default="configs/voltarget_live_monitor.yaml",
        help="Path to voltarget_live_monitor.yaml.",
    )
    execution_drift.add_argument(
        "--signal-dir",
        default="outputs/live_signal",
        help="Directory containing signal_history.csv from the paper monitor.",
    )
    execution_drift.add_argument(
        "--output-dir",
        default="outputs/execution_drift",
        help="Directory for execution drift CSV, report, and charts.",
    )
    execution_drift.add_argument(
        "--data-csv",
        help="Optional fixture CSV with Date and adjusted OHLC columns.",
    )
    execution_drift.add_argument(
        "--drift-threshold",
        type=float,
        default=0.01,
        help="Absolute drift threshold used for policy-threshold warnings.",
    )
    health_check = subparsers.add_parser(
        "check-voltarget-monitor-health",
        help="Run read-only health checks for the VolTarget paper monitor outputs.",
    )
    health_check.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing live signal outputs and receiving health-check artifacts.",
    )
    health_check.add_argument(
        "--config",
        default="configs/voltarget_live_monitor.yaml",
        help="Path to voltarget_live_monitor.yaml.",
    )
    run_open_questions = subparsers.add_parser(
        "run-open-questions",
        help="Run the OpenQuestionsExperimentPack diagnostic experiments.",
    )
    run_open_questions.add_argument("pack_path", help="Path to an OpenQuestionsExperimentPack YAML config.")
    run_open_questions.add_argument("--max-configs", type=int, help="Run only the first N experiment cases in the pack.")
    run_open_questions.add_argument("--dry-run", action="store_true", help="Print workload estimates without running backtests.")
    run_open_questions.add_argument(
        "--only",
        action="append",
        help="Run only one open question by id/name, e.g. 7, q7, or question_7. Repeat for multiple questions.",
    )

    extract_candidates = subparsers.add_parser(
        "extract-candidates",
        help="Extract final candidates from an existing tournament output directory.",
    )
    extract_candidates.add_argument("tournament_dir", help="Path to a tournament output directory.")
    extract_candidates.add_argument(
        "--accept-baseline-regression",
        action="store_true",
        help="Proceed despite BASELINE_GATE_FAILED.txt after explicitly accepting the baseline regression.",
    )

    check_blockers_parser = subparsers.add_parser(
        "check-blockers",
        help="Check hard research blockers before launching downstream runs.",
    )
    check_blockers_parser.set_defaults(command="check-blockers")

    audit_config_parser = subparsers.add_parser(
        "audit-config",
        help="Audit experiment or tournament YAML before launching long-running runs.",
    )
    audit_config_parser.add_argument("config_path", help="Path to an experiment, batch, or tournament YAML config.")

    audit_data_parser = subparsers.add_parser(
        "audit-data",
        help="Audit required data availability for an experiment or tournament YAML config.",
    )
    audit_data_parser.add_argument("config_path", help="Path to an experiment, batch, or tournament YAML config.")

    audit_baseline_data_parser = subparsers.add_parser(
        "audit-baseline-data",
        help="Audit current TQQQ/QQQ price returns against a legacy stitched baseline.",
    )
    audit_baseline_data_parser.add_argument("--old-stitched", required=True, help="Path to legacy stitched equity CSV.")
    audit_baseline_data_parser.add_argument("--output-dir", required=True, help="Directory for baseline data audit outputs.")
    audit_baseline_data_parser.add_argument("--start-date", help="Optional old stitched/data start date.")
    audit_baseline_data_parser.add_argument("--end-date", help="Optional old stitched/data end date.")
    audit_baseline_data_parser.add_argument("--cache-dir", default="./price_cache", help="Price cache directory.")
    audit_baseline_data_parser.add_argument("--tolerance", type=float, default=1e-8, help="Return comparison tolerance.")

    compare_run = subparsers.add_parser(
        "compare-run",
        help="Compare two stitched run CSVs for baseline numerical equivalence.",
    )
    compare_run.add_argument("--old", required=True, help="Path to the old baseline CSV.")
    compare_run.add_argument("--new", required=True, help="Path to the new run CSV.")
    compare_run.add_argument("--tolerance", type=float, default=1e-8, help="Absolute tolerance for numeric comparisons.")
    compare_run.add_argument(
        "--output-dir",
        default="outputs/comparison",
        help="Directory for compare_run_summary.csv and compare_run_report.md.",
    )

    controlled_runs = subparsers.add_parser(
        "summarize-controlled-runs",
        help="Summarize controlled real-data runs and classify raw TQQQ outperformance.",
    )
    controlled_runs.add_argument(
        "--mode",
        choices=("full", "gate"),
        default="full",
        help="Summary mode: full controlled configs or bounded gate configs.",
    )
    controlled_runs.add_argument(
        "--output-dir",
        default=None,
        help="Directory for controlled_run_summary.csv and controlled_run_report.md.",
    )

    replay_regime = subparsers.add_parser(
        "replay-regime-windows",
        help="Replay selected legacy regime walk-forward windows without running grid search.",
    )
    replay_regime.add_argument("--windows", required=True, help="Path to legacy regime_walk_forward_windows.csv.")
    replay_regime.add_argument("--old-stitched", required=True, help="Path to legacy stitched equity CSV.")
    replay_regime.add_argument("--output-dir", required=True, help="Directory for replay diagnostic outputs.")
    replay_regime.add_argument("--start-date", help="Data start date. Defaults to the earliest train_start in windows.")
    replay_regime.add_argument("--end-date", help="Inclusive data/replay end date.")
    replay_regime.add_argument("--tolerance", type=float, default=1e-8, help="Numeric comparison tolerance.")
    replay_regime.add_argument("--transaction-cost-bps", type=float, help="Override transaction cost bps.")
    replay_regime.add_argument("--execution-model", help="Override execution model.")
    replay_regime.add_argument("--data-csv", help="Optional fixture CSV with Date, TQQQ, and QQQ columns.")

    audit_regime_allocation_parser = subparsers.add_parser(
        "audit-regime-allocation",
        help="Audit legacy regime allocation signals, weights, turnover, returns, and compounding.",
    )
    audit_regime_allocation_parser.add_argument("--windows", required=True, help="Path to legacy regime walk-forward windows CSV.")
    audit_regime_allocation_parser.add_argument("--old-stitched", required=True, help="Path to legacy stitched equity CSV.")
    audit_regime_allocation_parser.add_argument("--output-dir", required=True, help="Directory for regime allocation audit outputs.")
    audit_regime_allocation_parser.add_argument("--start-date", help="Optional old stitched/data start date.")
    audit_regime_allocation_parser.add_argument("--end-date", help="Optional old stitched/data end date.")
    audit_regime_allocation_parser.add_argument("--tolerance", type=float, default=1e-8, help="Numeric comparison tolerance.")
    audit_regime_allocation_parser.add_argument("--transaction-cost-bps", type=float, help="Override transaction cost bps.")
    audit_regime_allocation_parser.add_argument("--execution-model", help="Override execution model.")
    audit_regime_allocation_parser.add_argument("--data-csv", help="Optional fixture CSV with Date, TQQQ, and QQQ columns.")
    audit_regime_allocation_parser.add_argument("--cache-dir", default="./price_cache", help="Price cache directory.")

    audit_voltarget_timing_parser = subparsers.add_parser(
        "audit-voltarget-timing",
        help="Audit VolTarget generated daily output for look-ahead and one-bar signal timing.",
    )
    audit_voltarget_timing_parser.add_argument("config_path", help="Path to a VolTarget YAML config.")
    audit_voltarget_timing_parser.add_argument(
        "--output-dir",
        default="outputs/audits",
        help="Directory for voltarget_timing_audit_summary.csv and voltarget_timing_audit_report.md.",
    )
    audit_voltarget_timing_parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Absolute numeric tolerance for timing comparisons.",
    )

    baseline_regression = subparsers.add_parser(
        "baseline-regression-report",
        help="Create baseline regression summary/report and write BASELINE_GATE_FAILED.txt unless equivalence passes.",
    )
    baseline_regression.add_argument("--output-dir", default="outputs/baseline_regression")
    baseline_regression.add_argument(
        "--compare-summary",
        default="outputs/comparison/baseline_clipped/compare_run_summary.csv",
        help="Matched-date compare-run summary CSV.",
    )
    baseline_regression.add_argument(
        "--replay-summary",
        default="outputs/replay/v10_regime_windows/replay_compare_summary.csv",
        help="Legacy replay summary CSV.",
    )
    baseline_regression.add_argument(
        "--replay-params",
        default="outputs/replay/v10_regime_windows/window_param_replay.csv",
        help="Legacy replay selected-parameter CSV.",
    )
    baseline_regression.add_argument(
        "--old-windows",
        default="ma_search_output_v10/regime_walk_forward_windows.csv",
        help="Old v10 selected window CSV.",
    )
    baseline_regression.add_argument(
        "--baseline-data-summary",
        default="outputs/comparison/baseline_data_audit/baseline_data_audit_summary.csv",
        help="Baseline price/return audit summary CSV.",
    )
    baseline_regression.add_argument(
        "--allocation-summary",
        default="outputs/comparison/regime_allocation_audit/regime_allocation_audit_summary.csv",
        help="Regime allocation audit summary CSV.",
    )
    baseline_regression.add_argument("--tolerance", type=float, default=1e-8)

    accept_baseline = subparsers.add_parser(
        "accept-baseline-regression",
        help="Formally accept v10 baseline regression as a data-vintage mismatch after diagnostic checks pass.",
    )
    accept_baseline.add_argument("--reason", required=True, help="Acceptance rationale.")
    accept_baseline.add_argument(
        "--max-final-equity-rel-diff",
        type=float,
        required=True,
        help="Maximum allowed abs(replay_final - old_final) / abs(old_final).",
    )
    accept_baseline.add_argument(
        "--require-weight-match",
        action="store_true",
        default=True,
        help="Require TQQQ/QQQ/position/turnover diffs to be floating-noise small.",
    )
    accept_baseline.add_argument("--weight-tolerance", type=float, default=1e-12)
    accept_baseline.add_argument("--output-dir", help="Baseline regression output directory. Defaults to gate directory.")
    accept_baseline.add_argument(
        "--replay-summary",
        default="outputs/replay/v10_regime_windows/replay_compare_summary.csv",
    )
    accept_baseline.add_argument(
        "--baseline-data-summary",
        default="outputs/comparison/baseline_data_audit/baseline_data_audit_summary.csv",
    )
    accept_baseline.add_argument(
        "--allocation-summary",
        default="outputs/comparison/regime_allocation_audit/regime_allocation_audit_summary.csv",
    )
    accept_baseline.add_argument(
        "--baseline-regression-summary",
        default="outputs/baseline_regression/baseline_regression_summary.csv",
    )

    freeze_snapshot = subparsers.add_parser(
        "freeze-data-snapshot",
        help="Write a manifest of current cached/downloaded price-data hashes.",
    )
    freeze_snapshot.add_argument("--symbols", nargs="+", required=True, help="Symbols to include in the snapshot.")
    freeze_snapshot.add_argument("--output-dir", required=True, help="Directory for data snapshot outputs.")
    freeze_snapshot.add_argument("--start-date", default="2011-01-01", help="Snapshot start date.")
    freeze_snapshot.add_argument("--end-date", help="Optional snapshot end date.")
    freeze_snapshot.add_argument("--cache-dir", default="./price_cache", help="Price cache directory.")
    freeze_snapshot.add_argument("--data-csv", help="Optional fixture CSV with Date and symbol columns.")

    verify_snapshot = subparsers.add_parser(
        "verify-data-snapshot",
        help="Reload current data and verify it against a data snapshot manifest.",
    )
    verify_snapshot.add_argument("manifest_path", help="Path to data_snapshot_manifest.json.")
    verify_snapshot.add_argument("--allow-drift", action="store_true", help="Exit 0 even if hashes differ.")
    verify_snapshot.add_argument("--cache-dir", help="Override price cache directory.")
    verify_snapshot.add_argument("--data-csv", help="Override data CSV for verification.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run-existing-regime"
    objective = getattr(args, "objective", None)

    if command == "run-existing-regime":
        run_regime_search(
            output_dir=args.output_dir if args.command else "./ma_search_output_v11",
            objective=objective,
            start_date=getattr(args, "start_date", "2011-01-01"),
            end_date=getattr(args, "end_date", None),
        )
        return 0
    if command == "run-ma-search":
        run_search(output_dir=args.output_dir, objective=objective)
        return 0
    if command == "run-config":
        if args.dry_run:
            dry_run_experiment_config(Path(args.config_path), objective_override=objective)
        else:
            try:
                run_experiment_config(Path(args.config_path), objective_override=objective)
            except ExperimentInfrastructureError as exc:
                print(f"ERROR: {exc.classification}: {exc}", file=sys.stderr)
                return 1
        return 0
    if command == "run-batch":
        run_batch_config(
            Path(args.batch_path),
            objective_override=objective,
            max_configs=args.max_configs,
            dry_run=args.dry_run,
        )
        return 0
    if command == "run-tournament":
        try:
            run_tournament_config(
                Path(args.tournament_path),
                max_configs=args.max_configs,
                dry_run=args.dry_run,
                allow_long_run=args.allow_long_run,
                accept_baseline_regression=args.accept_baseline_regression,
            )
            return 0
        except (RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "summarize-voltarget-stage":
        try:
            summarize_voltarget_stage(Path(args.output_dir))
            if bool(getattr(args, "fair_leverage", False)):
                write_stage2_fair_leverage_outputs(Path(args.output_dir))
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "plot-equity-curve":
        try:
            result = generate_equity_visualizations(
                input_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir),
            )
            print(f"Wrote visualization summary: {result.summary_path}")
            print(f"Wrote visualization report: {result.report_path}")
            for name, path in result.chart_paths.items():
                print(f"Wrote {name}: {path}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "audit-final-voltarget":
        try:
            result = run_final_voltarget_audit(
                input_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir),
            )
            print(f"Wrote final VolTarget audit to: {result.output_dir}")
            print(f"Final classification: {result.final_classification}")
            for name, path in result.artifact_paths.items():
                print(f"Wrote {name}: {path}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "audit-voltarget-execution-financing":
        try:
            result = run_execution_financing_audit(
                input_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir),
                data_csv=Path(args.data_csv) if args.data_csv else None,
                cache_dir=args.cache_dir,
            )
            print(f"Wrote execution/financing summary: {result.summary_path}")
            print(f"Wrote execution/financing report: {result.report_path}")
            print(f"Wrote sensitivity heatmap: {result.heatmap_path}")
            print(
                "Still beats TQQQ across all tested scenarios: "
                f"{result.still_beats_tqqq_after_realistic_financing}"
            )
            print(f"Breaking assumption: {result.breaking_assumption}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "audit-voltarget-execution-semantics":
        try:
            result = run_execution_semantics_audit(
                input_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir),
                data_csv=Path(args.data_csv) if args.data_csv else None,
                cache_dir=args.cache_dir,
            )
            print(f"Wrote execution semantics summary: {result.summary_path}")
            print(f"Wrote return decomposition: {result.decomposition_path}")
            print(f"Wrote execution semantics report: {result.report_path}")
            for name, path in result.chart_paths.items():
                print(f"Wrote {name}: {path}")
            print(f"Classification: {result.classification}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "compare-voltarget-simplification":
        try:
            result = run_simplification_battle(
                input_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir),
                data_csv=Path(args.data_csv) if args.data_csv else None,
                cache_dir=args.cache_dir,
            )
            print(f"Wrote simplification battle summary: {result.summary_path}")
            print(f"Wrote simplification battle report: {result.report_path}")
            for name, path in result.chart_paths.items():
                print(f"Wrote {name}: {path}")
            print(f"Full-period winner: {result.full_period_winner}")
            print(f"Ex-2022 winner: {result.ex_2022_winner}")
            print(f"Post-2022 winner: {result.post_2022_winner}")
            print(f"Best drawdown model: {result.best_drawdown_model}")
            print(f"Trend/momentum answer: {result.trend_momentum_answer}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "build-voltarget-risk-dashboard":
        try:
            result = build_voltarget_risk_dashboard(
                config_path=Path(args.config),
                input_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir),
            )
            print(f"Wrote dashboard: {result.dashboard_path}")
            print(f"Wrote dashboard summary: {result.summary_path}")
            print(f"Wrote risk policy breaches: {result.breaches_path}")
            for name, path in result.chart_paths.items():
                print(f"Wrote {name}: {path}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "generate-voltarget-signal":
        try:
            result = generate_voltarget_signal(
                config_path=Path(args.config),
                output_dir=Path(args.output_dir),
                data_csv=Path(args.data_csv) if args.data_csv else None,
            )
            print(f"Wrote signal JSON: {result.signal_today_path}")
            print(f"Wrote signal history: {result.signal_history_path}")
            print(f"Wrote signal report: {result.signal_report_path}")
            print(f"Wrote data quality report: {result.data_quality_path}")
            print(f"Wrote financing cost history: {result.financing_history_path}")
            print(f"Wrote financing cost report: {result.financing_report_path}")
            for name, path in result.chart_paths.items():
                print(f"Wrote {name}: {path}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "run-daily-voltarget-monitor":
        result = run_daily_voltarget_monitor(config_path=Path(args.config))
        print(f"Wrote daily run status: {result.status_path}")
        print(f"Wrote daily run log: {result.log_path}")
        print(f"Daily monitor status: {result.status['status']}")
        if result.status.get("error"):
            print(f"ERROR: {result.status['error']}", file=sys.stderr)
        return 0 if result.status["status"] in {"ok", "warning"} else 1
    if command == "update-auto-paper-ledger":
        try:
            result = update_auto_paper_ledger(
                config_path=Path(args.config),
                signal_dir=Path(args.signal_dir),
                output_dir=Path(args.output_dir),
                data_csv=Path(args.data_csv) if args.data_csv else None,
            )
            print(f"Wrote auto paper ledger: {result.ledger_path}")
            print(f"Wrote auto paper summary: {result.summary_path}")
            print(f"Wrote auto paper report: {result.report_path}")
            for name, path in result.chart_paths.items():
                print(f"Wrote {name}: {path}")
            print(f"Latest auto paper status: {result.latest_status}")
            print(f"Latest target exposure: {result.latest_row.get('target_exposure', '')}")
            print(f"Latest paper equity: {result.latest_row.get('paper_equity', '')}")
            print(f"Latest relative equity vs TQQQ: {result.latest_row.get('relative_equity_vs_tqqq', '')}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "initialize-auto-paper-ledger":
        try:
            result = initialize_auto_paper_ledger(
                config_path=Path(args.config),
                start_date=args.start_date,
                starting_equity=float(args.starting_equity),
                output_dir=Path(args.output_dir),
                confirm_reset=bool(args.confirm_reset),
            )
            print(f"Wrote auto paper ledger state: {result.state_path}")
            print(f"Auto paper ledger path: {result.ledger_path}")
            print(f"Reset performed: {result.reset_performed}")
            print(f"Paper ledger start date: {result.state['paper_ledger_start_date']}")
            print(f"Starting equity: {result.state['starting_equity']}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "run-automated-voltarget-paper-monitor":
        result = run_automated_voltarget_paper_monitor(
            config_path=Path(args.config),
            data_csv=Path(args.data_csv) if args.data_csv else None,
        )
        print(f"Wrote automation status: {result.status_path}")
        print(f"Wrote daily run log: {result.log_path}")
        print(f"Automated paper monitor status: {result.status['status']}")
        ledger_step = result.status.get("steps", {}).get("auto_paper_ledger", {})
        latest = ledger_step.get("latest_row", {}) if isinstance(ledger_step, dict) else {}
        if latest:
            print(f"Latest auto paper ledger status: {latest.get('status', '')}")
            print(f"Latest target exposure: {latest.get('target_exposure', '')}")
            print(f"Latest paper equity: {latest.get('paper_equity', '')}")
            print(f"Latest relative equity vs TQQQ: {latest.get('relative_equity_vs_tqqq', '')}")
        if result.status.get("error"):
            print(f"ERROR: {result.status['error']}", file=sys.stderr)
        return 0 if result.status["status"] in {"ok", "warning"} else 1
    if command == "reconcile-paper-trades":
        try:
            result = reconcile_paper_trades(
                signal_dir=Path(args.signal_dir),
                ledger_path=Path(args.ledger),
                output_dir=Path(args.output_dir),
            )
            print(f"Wrote paper trade reconciliation: {result.reconciliation_path}")
            print(f"Wrote paper trade reconciliation report: {result.report_path}")
            warnings = int(result.reconciliation["warnings"].fillna("").astype(str).ne("").sum()) if not result.reconciliation.empty else 0
            print(f"Rows with warnings: {warnings}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "track-execution-drift":
        try:
            result = run_execution_drift_tracking(
                config_path=Path(args.config),
                signal_dir=Path(args.signal_dir),
                output_dir=Path(args.output_dir),
                data_csv=Path(args.data_csv) if args.data_csv else None,
                drift_threshold=float(args.drift_threshold),
            )
            print(f"Wrote execution drift summary: {result.summary_path}")
            print(f"Wrote execution drift report: {result.report_path}")
            for name, path in result.chart_paths.items():
                print(f"Wrote {name}: {path}")
            print(f"Drift exceeds policy threshold: {result.metrics['drift_exceeds_policy_threshold']}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "check-voltarget-monitor-health":
        result = check_voltarget_monitor_health(output_dir=Path(args.output_dir), config_path=Path(args.config))
        print(f"Wrote monitor health summary: {result.summary_path}")
        print(f"Wrote monitor health report: {result.report_path}")
        print(f"Monitor health status: {result.overall_status}")
        return 1 if result.overall_status == "failed" else 0
    if command == "run-open-questions":
        run_open_questions_config(
            Path(args.pack_path),
            max_configs=args.max_configs,
            dry_run=args.dry_run,
            only_questions=args.only,
        )
        return 0
    if command == "extract-candidates":
        try:
            extract_final_candidates(
                Path(args.tournament_dir),
                accept_baseline_regression=args.accept_baseline_regression,
            )
            return 0
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "check-blockers":
        summary = check_blockers()
        failed = summary["status"].astype(str).str.lower().eq("failed").any()
        return 1 if failed else 0
    if command == "audit-config":
        audit_config(Path(args.config_path))
        return 0
    if command == "audit-data":
        audit_data(Path(args.config_path))
        return 0
    if command == "audit-baseline-data":
        audit_baseline_data(
            old_stitched_path=Path(args.old_stitched),
            output_dir=Path(args.output_dir),
            start_date=args.start_date,
            end_date=args.end_date,
            cache_dir=args.cache_dir,
            tolerance=float(args.tolerance),
        )
        return 0
    if command == "compare-run":
        _, passed = compare_run_files(
            Path(args.old),
            Path(args.new),
            tolerance=float(args.tolerance),
            output_dir=Path(args.output_dir),
        )
        return 0 if passed else 1
    if command == "summarize-controlled-runs":
        output_dir = Path(
            args.output_dir
            if args.output_dir
            else ("outputs/controlled_gate" if args.mode == "gate" else "outputs/controlled_runs")
        )
        if args.mode == "gate":
            summarize_controlled_runs(output_dir=output_dir, run_specs=GATE_CONTROLLED_RUNS, mode=args.mode)
        else:
            summarize_controlled_runs(output_dir=output_dir, mode=args.mode)
        return 0
    if command == "replay-regime-windows":
        summary = replay_regime_windows(
            windows_path=Path(args.windows),
            old_stitched_path=Path(args.old_stitched),
            output_dir=Path(args.output_dir),
            start_date=args.start_date,
            end_date=args.end_date,
            tolerance=float(args.tolerance),
            transaction_cost_bps=args.transaction_cost_bps,
            execution_model=args.execution_model,
            data_csv=Path(args.data_csv) if args.data_csv else None,
        )
        passed = bool(summary["passed"].iloc[0]) if not summary.empty else False
        return 0 if passed else 1
    if command == "audit-regime-allocation":
        audit_regime_allocation(
            windows_path=Path(args.windows),
            old_stitched_path=Path(args.old_stitched),
            output_dir=Path(args.output_dir),
            start_date=args.start_date,
            end_date=args.end_date,
            tolerance=float(args.tolerance),
            transaction_cost_bps=args.transaction_cost_bps,
            execution_model=args.execution_model,
            data_csv=Path(args.data_csv) if args.data_csv else None,
            cache_dir=args.cache_dir,
        )
        return 0
    if command == "audit-voltarget-timing":
        try:
            _, passed = audit_voltarget_timing(
                Path(args.config_path),
                output_dir=Path(args.output_dir),
                tolerance=float(args.tolerance),
            )
            return 0 if passed else 1
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "baseline-regression-report":
        create_baseline_regression_report(
            output_dir=Path(args.output_dir),
            compare_summary_path=Path(args.compare_summary),
            replay_summary_path=Path(args.replay_summary),
            replay_params_path=Path(args.replay_params),
            old_windows_path=Path(args.old_windows),
            baseline_data_summary_path=Path(args.baseline_data_summary),
            allocation_summary_path=Path(args.allocation_summary),
            tolerance=float(args.tolerance),
        )
        return 0
    if command == "accept-baseline-regression":
        try:
            accept_baseline_regression(
                reason=args.reason,
                max_final_equity_rel_diff=float(args.max_final_equity_rel_diff),
                require_weight_match=bool(args.require_weight_match),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                replay_summary_path=Path(args.replay_summary),
                baseline_data_summary_path=Path(args.baseline_data_summary),
                allocation_summary_path=Path(args.allocation_summary),
                baseline_regression_summary_path=Path(args.baseline_regression_summary),
                weight_tolerance=float(args.weight_tolerance),
            )
            return 0
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if command == "freeze-data-snapshot":
        freeze_data_snapshot(
            symbols=args.symbols,
            output_dir=Path(args.output_dir),
            start_date=args.start_date,
            end_date=args.end_date,
            cache_dir=args.cache_dir,
            data_csv=Path(args.data_csv) if args.data_csv else None,
        )
        return 0
    if command == "verify-data-snapshot":
        try:
            _, passed = verify_data_snapshot(
                Path(args.manifest_path),
                allow_drift=bool(args.allow_drift),
                cache_dir=args.cache_dir,
                data_csv=Path(args.data_csv) if args.data_csv else None,
            )
            return 0 if passed else 1
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
