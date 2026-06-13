from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from .data import load_prices
from .experiments import run_batch_config, run_experiment_config
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
from .candidates import extract_final_candidates
from .open_questions import run_open_questions_config
from .tournament import run_tournament_config

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


def run_regime_search(output_dir: str = "./ma_search_output_v11", objective: Optional[str] = None) -> None:
    data = load_prices(start="2011-01-01", use_csv_if_exists=True)

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
            "data_start": "2011-01-01",
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

    run_batch = subparsers.add_parser(
        "run-batch",
        help="Run a YAML batch config containing multiple experiments.",
    )
    run_batch.add_argument("batch_path", help="Path to a YAML batch config.")
    run_batch.add_argument("--objective", help="Override each experiment objective name.")

    run_tournament = subparsers.add_parser(
        "run-tournament",
        help="Run a tournament YAML config across candidate strategy families.",
    )
    run_tournament.add_argument("tournament_path", help="Path to a YAML tournament config.")

    run_open_questions = subparsers.add_parser(
        "run-open-questions",
        help="Run the OpenQuestionsExperimentPack diagnostic experiments.",
    )
    run_open_questions.add_argument("pack_path", help="Path to an OpenQuestionsExperimentPack YAML config.")

    extract_candidates = subparsers.add_parser(
        "extract-candidates",
        help="Extract final candidates from an existing tournament output directory.",
    )
    extract_candidates.add_argument("tournament_dir", help="Path to a tournament output directory.")

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
        )
        return 0
    if command == "run-ma-search":
        run_search(output_dir=args.output_dir, objective=objective)
        return 0
    if command == "run-config":
        run_experiment_config(Path(args.config_path), objective_override=objective)
        return 0
    if command == "run-batch":
        run_batch_config(Path(args.batch_path), objective_override=objective)
        return 0
    if command == "run-tournament":
        run_tournament_config(Path(args.tournament_path))
        return 0
    if command == "run-open-questions":
        run_open_questions_config(Path(args.pack_path))
        return 0
    if command == "extract-candidates":
        extract_final_candidates(Path(args.tournament_dir))
        return 0

    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
