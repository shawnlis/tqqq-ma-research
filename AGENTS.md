# AGENTS.md

## Project Purpose

This repository is a quantitative research framework for testing TQQQ timing and allocation strategies against same-period TQQQ buy-and-hold. The main research question is raw outperformance versus TQQQ final wealth after realistic validation, costs, and robustness checks. Risk metrics matter, but they are secondary to same-period final_equity_ratio versus TQQQ for the primary project goal.

Do not make investment recommendations from these results. Treat all outputs as empirical backtest artifacts with assumptions and limitations.

## Package Layout

- `TQQQ_MA.py` is a backward-compatible wrapper that delegates to the package CLI.
- `research/data.py` handles price loading, CASH synthetic series, and synthetic leveraged ETF series.
- `research/backtest.py` applies strategy positions, execution models, and transaction costs.
- `research/metrics.py` contains performance metrics and objective functions.
- `research/validation.py` contains walk-forward search and out-of-sample stitching.
- `research/experiments.py` runs YAML-driven experiments and writes experiment outputs.
- `research/tournament.py` runs multi-strategy tournament comparisons.
- `research/open_questions.py` runs targeted diagnostic experiment packs.
- `research/candidates.py` extracts final candidates from tournament artifacts.
- `research/reports.py` writes benchmark summaries, reports, and charts.
- `research/strategies/` contains strategy modules: MA, regime, core overlay, vol target, rotation, rebound, drawdown governor, and market internals.
- `configs/` contains YAML experiment, batch, tournament, and diagnostic pack configs.
- `tests/` contains pytest coverage for metrics, data, backtesting, validation, strategies, experiments, tournament, reports, open questions, and candidate extraction.

## Required Validation

Before claiming a code change is complete, run:

```powershell
python -m pytest tests -q
```

If this fails, stop and report the failure. Do not commit a requested change after a failing required validation run unless the user explicitly asks for a failing work-in-progress commit.

## Preferred Smoke Tests

Use these commands for narrower checks when touching specific areas:

```powershell
python -m pytest tests/test_metrics.py tests/test_backtest.py tests/test_reports.py -q
python -m pytest tests/test_experiments.py -q
python -m pytest tests/test_tournament.py tests/test_candidates.py -q
python -m pytest tests/test_open_questions.py -q
python -m research.cli --help
python -m research.cli extract-candidates --help
```

For real-data workflow checks, prefer small config runs before broad tournaments:

```powershell
python -m research.cli run-config configs/core_overlay_tqqq.yaml
python -m research.cli run-open-questions configs/open_questions.yaml
python -m research.cli run-tournament configs/tournament.yaml
python -m research.cli extract-candidates outputs/tournament
```

## Generated Files And Commit Hygiene

Do not commit generated outputs, caches, or transient Python/test artifacts:

- `outputs/`
- `price_cache/`
- `ma_search_output_*`
- `__pycache__/`
- `.pytest_cache/`
- `.pytest-tmp/`

The `.gitignore` should keep these out of Git. If any of them appear in `git status`, fix the ignore/staging state before committing.

## Research Rules

- Do not claim strategy success unless `same_period_benchmark_summary.csv` shows `final_equity_ratio > 1.0` versus `TQQQ` over the exact stitched strategy dates.
- Do not optimize for Sharpe or Calmar as the primary objective for the main TQQQ-outperformance goal. Use `objective_final_ratio` first unless a task explicitly asks for a different diagnostic objective.
- Preserve look-ahead prevention. Strategy signals formed at close `t` must be shifted one bar before returns are applied. Do not remove or bypass the position shift in backtests.
- Every strategy output must include same-period benchmark comparison artifacts, especially `same_period_benchmark_summary.csv` and yearly return comparison output.
- Keep transaction costs applied to allocation changes, including risk-off asset allocation changes.
- Same-period benchmark comparisons must use the stitched strategy dates, not full available price history.
- Record failed experiments in tournament, open-question, and candidate extraction outputs. Do not hide failed runs.

## Safe Editing Guidance

- Prefer small, scoped changes that preserve existing strategy behavior unless the user explicitly asks to change strategy logic.
- When adding a new strategy, config, objective, execution mode, or validation rule, add focused pytest coverage.
- When touching backtest timing, costs, benchmark comparison, or walk-forward stitching, add or run tests that prove no look-ahead and no duplicate stitched dates.
- Use YAML configs in `configs/` for reproducible experiments.
- Save `run_config.json` beside experiment outputs when adding new output-producing workflows.
