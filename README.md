# TQQQ Strategy Research

This repository is structured for serious strategy research against same-period TQQQ buy-and-hold. The current code is a modularized version of the existing `TQQQ_MA.py` workflow; strategy logic has not been optimized or replaced.

## Running The Existing Regime Workflow

```powershell
python -m research.cli run-existing-regime
```

## Running Configured Experiments

Single experiment:

```powershell
python -m research.cli run-config configs/example_ma.yaml
python -m research.cli run-config configs/example_ma.yaml --objective objective_excess_cagr_with_dd_guard
```

Batch:

```powershell
python -m research.cli run-batch configs/example_batch.yaml
python -m research.cli run-batch configs/example_batch.yaml --objective objective_final_ratio
```

Synthetic long-history baseline:

```powershell
python -m research.cli run-batch configs/synthetic_tqqq_long_history_baseline.yaml
```

Core-overlay TQQQ experiment:

```powershell
python -m research.cli run-config configs/core_overlay_tqqq.yaml
```

Rebound module experiments:

```powershell
python -m research.cli run-config configs/core_overlay_with_rebound.yaml
python -m research.cli run-config configs/regime_with_rebound.yaml
```

Vol-target TQQQ experiments:

```powershell
python -m research.cli run-config configs/vol_target_tqqq.yaml
python -m research.cli run-config configs/vol_target_synthetic_long_history.yaml
```

Drawdown governor experiments:

```powershell
python -m research.cli run-config configs/core_overlay_governor.yaml
python -m research.cli run-config configs/vol_target_governor.yaml
```

Risk-off asset comparison:

```powershell
python -m research.cli run-batch configs/batch_risk_off_assets.yaml
```

Market-internals experiments:

```powershell
python -m research.cli run-config configs/core_overlay_market_internals.yaml
python -m research.cli run-config configs/vol_target_market_internals.yaml
```

Cross-asset leveraged ETF rotation:

```powershell
python -m research.cli run-config configs/rotation_tqqq_soxl_upro.yaml
python -m research.cli run-config configs/rotation_monthly.yaml
python -m research.cli run-config configs/rotation_weekly.yaml
```

SOXL-specific experiments:

```powershell
python -m research.cli run-config configs/soxl_core_overlay.yaml
python -m research.cli run-config configs/soxl_vol_target.yaml
python -m research.cli run-config configs/soxl_regime.yaml
python -m research.cli run-config configs/soxl_rotation.yaml
```

Candidate strategy tournament:

```powershell
python -m research.cli run-tournament configs/tournament.yaml
```

Experiment YAML files live in `configs/` and include:

```yaml
experiment_name: example_ma_final_equity_ratio
symbols: [TQQQ, QQQ]
start_date: "2011-01-01"
end_date:
strategy_name: ma
strategy_params:
  short: 3
  long: 20
  ma_type: sma
benchmark_symbol: TQQQ
transaction_cost_bps: 10.0
execution_model: close_to_close_shifted
train_years: 5
test_years: 1
objective: final_equity_ratio
output_dir: outputs/example
```

Use `parameter_grid` instead of `strategy_params` for grid searches. Batch YAML files contain a `configs` list and optional `output_dir`.

The primary ranking metric for this project is raw final wealth versus the same-period benchmark. Risk metrics such as Calmar and drawdown are secondary diagnostics unless explicitly selected as part of an objective.

Supported objective names include:

- `objective_final_ratio`: `final_equity / benchmark_final_equity`.
- `objective_excess_cagr_with_dd_guard`: excess CAGR minus a penalty for worse drawdown than the benchmark and annual turnover.
- `objective_raw_outperformance_with_survival`: penalizes any strategy that does not beat final benchmark wealth; otherwise adds a Calmar bonus.
- `objective_yearly_consistency`: final wealth ratio plus a yearly win-count bonus, less the worst relative yearly loss.
- `objective_rebound_capture`: final wealth ratio plus relative return in rebound years after benchmark drawdowns worse than -30%.

Configs include an `objectives` list documenting at least `objective_final_ratio` and `objective_excess_cagr_with_dd_guard`; the active ranking objective is the scalar `objective` field or the CLI `--objective` override.

Asset selection is configurable through `asset_config`:

```yaml
asset_config:
  trade_asset: SOXL
  primary_signal_asset: SOXX
  secondary_filter_asset: SMH
  benchmark_symbol: SOXL
  risk_off_symbol: BIL
benchmark_symbol: SOXL
benchmark_comparison_symbols: [TQQQ, SOXX, SMH]
```

`trade_asset` is the ETF whose exposure is traded. `primary_signal_asset` drives momentum-style signals. `secondary_filter_asset` drives trend, volatility, rebound, and governor filter checks where those modules use a market filter. `benchmark_symbol` controls the same-period benchmark used for parameter selection and the standard summary. Existing TQQQ configs default to `TQQQ` traded against QQQ signals and a TQQQ benchmark.

`core_overlay` maintains a permanent core exposure to the configured `trade_asset` and adds tactical overlay exposure when the configured signal/filter assets show favorable regimes. Its walk-forward selection ranks by `final_equity_ratio`, then `excess_cagr`, then `max_dd`.

The optional rebound re-entry module can be attached to `regime` and `core_overlay`. It marks QQQ drawdowns as recovery candidates, then forces faster TQQQ re-entry when QQQ reclaims a short moving average with positive rebound momentum. Rebound-enabled runs write `rebound_ablation_summary.csv`, comparing without rebound, with rebound, and the difference for 2019, 2020, 2023 returns plus `final_equity_ratio`.

`vol_target` holds the configured `trade_asset` with exposure scaled by realized traded-asset volatility, then adjusts exposure using configured signal and filter assets. It writes daily `target_exposure`, `realized_daily_vol`, `realized_ann_vol`, strategy `equity`, and `drawdown` in `stitched_equity.csv`. Vol-target runs also write `drawdown_chart_data.csv` and `vol_target_acceptance_summary.csv`; a failure is defined as `final_equity_ratio <= 1.0` versus the configured same-period benchmark.

The optional drawdown governor can be attached to `regime`, `core_overlay`, and `vol_target`. It caps exposure only after severe portfolio or QQQ drawdowns, restores exposure on QQQ recovery signals, and gradually ramps back after `max_days_reduced` unless QQQ makes a new low. Governor-enabled runs write:

- `drawdown_governor_ablation_summary.csv`
- `drawdown_governor_ablation_yearly_returns.csv`

The ablation variants are `base_strategy`, `base_governor`, and `base_governor_rebound`.

Risk-off allocation is available through `risk_off_symbol` and `risk_off_weight`. When traded-asset exposure is below 1.0, unused capital can be allocated to `CASH`, `SGOV`, `BIL`, `SHY`, `IEF`, `TLT`, `GLD`, `QQQ`, or any loaded symbol. `CASH` is a synthetic zero-return price series. Transaction costs are charged on both traded-asset allocation changes and risk-off allocation changes. Risk-off batch runs write `risk_off_comparison.csv` with final-equity ratio, CAGR, max drawdown, Calmar, worst year, 2022 return, and 2023 return.

Market-internals scoring is available for `regime`, `core_overlay`, and `vol_target` through `use_market_internals: true`. The module builds a composite `market_internals_risk_score` from available proxies: SPY, QQQ, RSP, IWM, HYG, LQD, IEF, TLT, `^VIX`, XLK, SMH, and SOXX. Components include QQQ/SPY trend, equal-weight and small-cap relative strength, credit risk, duration stress, tech strength, semiconductor strength, and VIX regime. Auto-added proxy symbols are optional; missing or insufficient-history symbols are skipped and recorded in `run_config.json` under `market_internals_data_status`.

Market-internals runs write `market_internals_daily.csv`, and the same columns are also included in `stitched_equity.csv`:

- `market_internals_risk_score`
- `market_internals_component_count`
- `market_internals_risk_on`
- `market_internals_risk_off`
- `mi_*` component columns

`rotation` / `CrossAssetLeveragedMomentumStrategy` ranks leveraged ETF candidates (`TQQQ`, `SOXL`, `UPRO`, and `TECL` when available) on rebalance dates. The rank score averages cross-sectional ranks for 20-day momentum, 63-day momentum, 126-day momentum, inverse 20-day volatility, momentum divided by volatility, and drawdown from the 63-day high. The strategy holds the top `top_n` assets when the QQQ/SPY trend filter allows risk and, when configured, momentum is positive. Otherwise it holds the configured defensive asset (`CASH`, `SGOV`, `BIL`, or `QQQ`). Missing optional candidates are skipped and recorded in `run_config.json` under `rotation_asset_status`.

Rotation runs write:

- `selected_assets.csv`
- `weights_by_date.csv`
- `turnover_summary.csv`
- `stitched_equity.csv`
- `same_period_benchmark_summary.csv`
- `yearly_returns.csv`

The original rotation benchmark remains same-period buy-and-hold TQQQ. SOXL rotation configs use the configured same-period SOXL benchmark for parameter selection and also write explicit TQQQ comparison output. Do not judge rotation against a blended custom benchmark when answering whether it beats TQQQ or only adds complexity.

SOXL configs use the same strategy engines with `trade_asset: SOXL`, SOXX/SMH signal filters, and strict same-period benchmark comparisons. SOXL runs write:

- `soxl_vs_soxl_summary.csv`
- `soxl_vs_tqqq_summary.csv`
- `soxl_vs_soxx_summary.csv` and/or `soxl_vs_smh_summary.csv` when those symbols are loaded
- `soxl_yearly_returns.csv`
- `soxl_selected_params.csv`

Use these files to answer whether SOXL timing improves on SOXL buy-and-hold, whether it beats TQQQ buy-and-hold on the exact same dates, and whether the return profile only works during semiconductor bull markets.

Synthetic leveraged history can be enabled in any experiment:

```yaml
use_synthetic_leverage: true
synthetic_base_symbol: QQQ
synthetic_leverage: 3.0
synthetic_expense_ratio: 0.0095
synthetic_financing_spread: 0.0
```

When enabled, the framework builds a TQQQ-like synthetic series from the base symbol's daily returns and keeps real `TQQQ` separately as `REAL_TQQQ` where available. Synthetic experiments also write:

- `synthetic_tracking_summary.csv`
- `synthetic_tracking_equity.csv`

The legacy entry point remains available:

```powershell
python TQQQ_MA.py
```

Both commands default to the existing regime output directory, `ma_search_output_v11`, and write a `run_config.json` beside the generated CSV outputs. Use `--output-dir` to write to a different directory.

Every run with stitched walk-forward equity also writes:

- `same_period_benchmark_summary.csv`
- `same_period_yearly_returns.csv`

Config-driven experiments write normalized artifact names:

- `stitched_equity.csv`
- `walk_forward_windows.csv`
- `same_period_benchmark_summary.csv`
- `yearly_returns.csv`
- `validation_summary.csv`
- `walk_forward_variant_summary.csv`
- `bootstrap_summary.csv`
- `subperiod_summary.csv`
- `parameter_stability.csv`
- `cost_sensitivity.csv`
- `execution_model_sensitivity.csv`
- `report.md`
- `report_summary.csv`
- `charts/*.png`
- `run_config.json`

Batch runs write:

- `batch_summary.csv`
- `ranked_by_final_equity_ratio.csv`
- `ranked_by_excess_cagr.csv`
- `ranked_by_calmar.csv`

Tournament runs compare candidate strategy families across 5/1, 3/1, and 7/1 walk-forward variants at 10, 25, and 50 bps transaction costs. The tournament benchmark defaults to same-period TQQQ even when a strategy, such as a SOXL variant, selects parameters against its native configured benchmark.

Tournament runs write:

- `tournament_summary.csv`
- `tournament_ranked_by_final_equity_ratio.csv`
- `tournament_ranked_by_robustness.csv`
- `tournament_failures.csv`
- `tournament_variant_results.csv`
- `report.md`
- `report_summary.csv`
- `charts/*.png`
- `run_config.json`

`tournament_failures.csv` includes both failed variant runs and rejected strategy-family summaries. Failed experiments are retained instead of being dropped from rankings.

## Package Layout

- `research/assets.py`: shared `AssetConfig` defaults and serialization helpers.
- `research/data.py`: price loading and cache handling.
- `research/execution.py`: execution-model return alignment and OHLC fallback helpers.
- `research/metrics.py`: return, volatility, drawdown, Sharpe, Calmar, and scoring helpers.
- `research/backtest.py`: position-shifted backtest engines and window evaluation pieces.
- `research/strategies/ma_strategy.py`: moving-average signal logic and MA parameter grid.
- `research/strategies/regime_strategy.py`: current regime signal logic and regime parameter grid.
- `research/strategies/core_overlay_strategy.py`: core-plus-overlay TQQQ exposure logic.
- `research/strategies/vol_target_strategy.py`: volatility-targeted TQQQ exposure logic.
- `research/strategies/market_internals.py`: optional composite market-internals risk score.
- `research/strategies/rotation_strategy.py`: cross-asset leveraged ETF momentum rotation.
- `research/strategies/rebound.py`: optional rebound re-entry overlay for supported strategies.
- `research/strategies/drawdown_governor.py`: optional portfolio/QQQ drawdown exposure cap and recovery ramp.
- `research/validation.py`: grid search, walk-forward splitting, and stitched walk-forward construction.
- `research/anti_overfit.py`: anti-overfit validation reports for walk-forward variants, bootstrap checks, subperiods, and parameter stability.
- `research/sensitivity.py`: cost and execution-model sensitivity reports.
- `research/tournament.py`: candidate-family tournament runner, rankings, robustness score, and failure capture.
- `research/reports.py`: stitched summaries, stability tables, heatmaps, and run config writing.
- `research/cli.py`: command-line runners.

## Signal Timing Assumptions

Signals are formed using information available at the close of day `t`. Positions are shifted by one bar with `shift(1)`, so the position takes effect on day `t+1`. This is the repo's explicit guard against same-bar look-ahead.

Market-internals components are computed from the same close-aligned daily price table. They affect the unshifted target exposure for day `t`, and the existing backtest shift applies that exposure starting on day `t+1`.

Windowed evaluation uses full history up to each evaluation end date for indicator warm-up, then slices the evaluation window. Walk-forward windows carry the ending position from the prior stitched test window into the next window so first-bar turnover and costs are not understated.

## Transaction Cost Assumptions

Transaction costs are modeled as basis points of absolute turnover:

```text
cost = turnover * transaction_cost_bps / 10000
```

The current default is `10.0` bps. MA strategy turnover is based on absolute position changes. Regime strategy turnover sums absolute changes in traded-asset and risk-off weights.

## Walk-Forward Methodology

The existing walk-forward workflow uses rolling calendar-year windows. The current defaults train for 5 years, test for 1 year, rank parameters only on the training window, then apply the selected parameters directly to the following test window.

Stitched walk-forward equity is built only from test windows. Duplicate dates are dropped after concatenation, preserving the first occurrence. This prevents overlapping windows from double-counting returns.

Config-driven experiments also write anti-overfit validation reports. These include the standard 5-year train / 1-year test walk-forward, 3-year and 7-year train alternatives, an anchored expanding-window variant, 25 bps and 50 bps cost sensitivity, 20-day and 60-day block bootstrap estimates, fixed subperiod summaries, and selected-parameter neighborhood stability.

A strategy is marked `candidate_viable` only if it beats the same-period benchmark in the standard 5x1 walk-forward, beats it in at least one alternate walk-forward, survives both cost-sensitivity runs, and does not rely on isolated selected-parameter neighborhoods. This is a research gate, not an optimization target.

## Execution Models

Backtests support three execution models through `execution_model`:

- `close_to_close_shifted`: signal at close `t`, position applied to close-to-close return on `t+1`.
- `next_open_to_close`: signal at close `t`, trade at next open, hold next open to same-day close.
- `next_open_to_next_open`: signal at close `t`, trade at next open, hold to the following open.

Config-driven downloads request adjusted OHLC data by default and keep the close price under the original symbol column, so existing signal logic still reads close prices. OHLC columns are stored as `SYMBOL_OPEN`, `SYMBOL_HIGH`, `SYMBOL_LOW`, and `SYMBOL_CLOSE`.

If a requested open-based execution model lacks OHLC for a traded symbol, the backtest falls back to `close_to_close_shifted` and records the warning in `stitched_equity.csv`, `execution_model_sensitivity.csv`, and `run_config.json`.

Every config run writes sensitivity reports:

- `cost_sensitivity.csv`: runs the candidate strategy at 0, 10, 25, 50, and 100 bps with the configured execution model.
- `execution_model_sensitivity.csv`: runs the candidate strategy under all three execution models at the configured transaction cost.

Do not treat a strategy as robust if it only beats TQQQ at 0 bps. The 25 bps row in `cost_sensitivity.csv` is the first practical friction check.

## Same-Period Benchmark Requirement

Every strategy result must be compared with its configured buy-and-hold benchmark over the exact same dates as the strategy evaluation. TQQQ remains the default project benchmark unless a config explicitly sets another `benchmark_symbol`. Walk-forward test-window metrics use the configured benchmark sliced to the same test window.

Do not compare a strategy window to a benchmark computed over a different date range.

The stitched walk-forward benchmark comparison first restricts benchmark prices to `stitched_df.index`, then computes benchmark returns from that restricted series. The hard raw outperformance flag is:

```text
raw_outperformance_pass = final_equity_ratio > 1.0
```
