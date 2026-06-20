# TQQQ Strategy Research Handoff Summary

Last updated: 2026-06-15
Branch: `fix/research-gates`

This document is intended for loading into a fresh ChatGPT/Codex session. It summarizes the current research state, decisions, validation gates, important files, and next actions. It is not an investment recommendation.

## 1. Project Goal

The repository is a quantitative research framework for testing whether any TQQQ timing or risk-budget strategy can beat same-period TQQQ buy-and-hold without self-deception.

The primary research gate is:

```text
final_equity_ratio > 1.0 versus same-period TQQQ
```

Risk metrics such as Sharpe, Calmar, and drawdown are secondary. A strategy that reduces drawdown but fails raw same-period TQQQ final wealth is not a primary candidate.

## 2. Current High-Level Decision

The project has narrowed from broad strategy search to VolTarget validation.

Current classification:

```text
Plain VolTargetTQQQStrategy:
- Primary candidate.
- Beats TQQQ in Stage 2 full period.
- Survives ex-2022 and leave-one-year-out versus TQQQ.
- Beats same-average constant exposure and SimpleVolTargetNoTrend on full period.
- Stage 3 classified it as `crash_control_candidate`.
- Final classification: useful crash-control / risk-budget overlay, not proven persistent alpha.
- Full deep tournament remains deferred.

VolTarget + DrawdownGovernor:
- Secondary comparator only.
- Classified as governor_overfit_to_2022.
- Do not spend broad grid budget on it unless plain VolTarget fails.

CoreOverlay + Rebound:
- Comparator only.

Full tournament:
- Deferred by research judgment, not infrastructure.

Full VolTarget deep tournament:
- Deferred. Stage 3 did not justify running it.
```

## 3. Key Research Decisions Made

1. Same-period benchmark is mandatory.
   Every stitched walk-forward result must compare to TQQQ buy-and-hold over the exact stitched dates.

2. Baseline v10 numerical equivalence was not claimed.
   The legacy baseline mismatch was formally accepted as a data-vintage mismatch after replay/allocation diagnostics. The baseline gate is accepted with warning, not silently ignored.

3. Generated outputs are not committed.
   Source, configs, tests, and docs are committed. `outputs/`, `price_cache/`, logs, and generated artifacts stay local.

4. Full research grids are separated from gate configs.
   Gate configs are bounded smoke/validation runs. Full configs remain available but are not run blindly.

5. VolTarget is the only strategy family currently worth deeper TQQQ validation.
   Other families remain as comparators or diagnostic baselines.

6. Fair leverage attribution is required before declaring alpha.
   A strategy must be compared against:
   - 1.0x TQQQ
   - same-max constant TQQQ exposure
   - same-average-exposure constant TQQQ exposure
   - SimpleVolTargetNoTrend

7. 2022 is a known concentration risk.
   VolTarget survives some ex-2022 checks, but 2022 still contributes materially to the edge. Stage 3 was added specifically to distinguish robust alpha from crash-control.

## 4. Completed Architecture Work

The original single-file script was converted into a package-style framework.

Core package structure:

```text
research/
  data.py
  metrics.py
  backtest.py
  strategies/
    ma_strategy.py
    regime_strategy.py
    core_overlay_strategy.py
    vol_target_strategy.py
    rotation_strategy.py
    rebound.py
    drawdown_governor.py
    market_internals.py
    constant_exposure_strategy.py
  validation.py
  reports.py
  experiments.py
  tournament.py
  open_questions.py
  controlled_runs.py
  fairness.py
  voltarget_stage.py
  voltarget_stage2.py
  voltarget_stage3.py
  voltarget_fair_leverage.py
  voltarget_synthetic_history.py
  voltarget_timing_audit.py
  candidates.py
  baseline_gate.py
  replay.py
  regime_allocation_audit.py
  data_snapshot.py
  cli.py

configs/
outputs/
tests/
```

`TQQQ_MA.py` is intended to remain a compatibility wrapper around the CLI.

## 5. Important CLI Commands

General checks:

```powershell
cd "C:\Strategies\TQQQ MA"
python -m pytest tests -q
python -m research.cli check-blockers
```

Controlled/gate flow:

```powershell
python -m research.cli run-existing-regime
python -m research.cli run-config configs/regime_tqqq.yaml
python -m research.cli run-config configs/core_overlay_tqqq_gate.yaml
python -m research.cli run-config configs/core_overlay_with_rebound_gate.yaml
python -m research.cli run-config configs/vol_target_tqqq_gate.yaml
python -m research.cli summarize-controlled-runs --mode gate
```

Tournament gate:

```powershell
python -m research.cli run-tournament configs/tournament_gate.yaml --dry-run
python -m research.cli run-tournament configs/tournament_gate.yaml
```

VolTarget validation:

```powershell
python -m research.cli audit-voltarget-timing configs/vol_target_tqqq_gate.yaml
python -m research.cli run-config configs/voltarget_synthetic_history.yaml
python -m research.cli run-tournament configs/tournament_voltarget_stage1.yaml --dry-run
python -m research.cli run-tournament configs/tournament_voltarget_stage1.yaml
python -m research.cli summarize-voltarget-stage outputs/tournament_voltarget_stage1/
python -m research.cli run-tournament configs/tournament_voltarget_stage2_2022_sensitivity.yaml --dry-run
python -m research.cli run-tournament configs/tournament_voltarget_stage2_2022_sensitivity.yaml
python -m research.cli summarize-voltarget-stage outputs/tournament_voltarget_stage2/ --fair-leverage
python -m research.cli run-tournament configs/tournament_voltarget_stage3_robust_alpha.yaml --dry-run
```

Do not run yet unless Stage 3 actual output recommends it:

```powershell
python -m research.cli run-tournament configs/tournament_voltarget_deep.yaml --allow-long-run
```

## 6. Completed Validation Gates

### 6.1 Baseline Gate

Status: accepted with warning.

The legacy `ma_search_output_v10` comparison failed exact numerical equivalence after date/cache fixes. Diagnostic work added:

- `compare-run`
- `replay-regime-windows`
- `audit-baseline-data`
- `audit-regime-allocation`
- `baseline-regression-report`
- `accept-baseline-regression`
- `check-blockers`

Conclusion:

```text
Exact v10 numerical equivalence not claimed.
Baseline regression accepted as data-vintage mismatch.
Allocation replay was close enough to proceed under explicit warning.
```

### 6.2 Data Snapshot

Status: passed.

Current data snapshot can be frozen and verified with:

```powershell
python -m research.cli freeze-data-snapshot `
  --symbols TQQQ QQQ SOXL SOXX SMH UPRO SPY SGOV BIL SHY IEF TLT GLD RSP IWM HYG LQD XLK `
  --output-dir outputs/data_snapshot/current

python -m research.cli verify-data-snapshot outputs/data_snapshot/current/data_snapshot_manifest.json
```

### 6.3 Controlled Gate

Status: passed after bounded gate configs were added.

Full controlled configs are intentionally slow and remain separate from gate configs.

Gate configs:

- `configs/core_overlay_tqqq_gate.yaml`
- `configs/core_overlay_with_rebound_gate.yaml`
- `configs/vol_target_tqqq_gate.yaml`

Gate summary:

- `outputs/controlled_gate/controlled_run_summary.csv`
- `outputs/controlled_gate/controlled_run_report.md`

### 6.4 Tournament Gate

Status: passed.

The bounded tournament gate ran successfully. Existing MA ensemble wiring was fixed.

The gate is a pipeline/infrastructure validation, not a final research conclusion.

### 6.5 Open Questions

Status: diagnostic pack run and used to narrow the research direction.

Key outcome:

```text
VolTarget emerged as the strongest TQQQ family.
CoreOverlay/Rebound was not strong enough as primary.
SOXL remains a separate sector-bet question, not part of the TQQQ conclusion.
```

### 6.6 VolTarget Timing Audit

Status: passed.

Purpose:

- confirm shifted position timing
- prevent same-day signal/return leakage
- verify output timing assumptions

Relevant file:

- `research/voltarget_timing_audit.py`

### 6.7 VolTarget Synthetic History

Status: completed.

Config:

- `configs/voltarget_synthetic_history.yaml`

Outputs:

- `outputs/voltarget_synthetic_history/voltarget_synthetic_history.csv`
- `outputs/voltarget_synthetic_history/voltarget_synthetic_history_report.md`

Interpretation:

```text
Synthetic full QQQ history and major drawdown regimes were supportive.
2000-2002, 2008, and 2022 were strong.
2020 crash/rebound was weaker.
This supports the thesis that VolTarget works better in prolonged drawdowns than violent crash-and-rebound regimes.
```

### 6.8 VolTarget Stage 1

Status: completed.

Stage 1 established VolTarget as the strongest family.

Important observed results from prior run:

```text
VolTarget + DrawdownGovernor final_equity_ratio: about 2.3320
VolTargetTQQQStrategy final_equity_ratio: about 1.8744
CoreOverlay + Rebound comparator final_equity_ratio: about 0.9489
Top VolTarget+Governor survived 25 bps and 50 bps
One-year contribution audit showed 2022 dominance
```

### 6.9 VolTarget Stage 2

Status: completed.

Config:

- `configs/tournament_voltarget_stage2_2022_sensitivity.yaml`

Outputs:

- `outputs/tournament_voltarget_stage2/voltarget_stage2_summary.csv`
- `outputs/tournament_voltarget_stage2/voltarget_stage2_report.md`
- `outputs/tournament_voltarget_stage2/leave_one_year_out.csv`
- `outputs/tournament_voltarget_stage2/subperiod_summary.csv`
- `outputs/tournament_voltarget_stage2/year_dominance_audit.csv`

Key Stage 2 results:

```text
VolTargetTQQQStrategy:
- full ratio: 2.4072
- ex-2022 ratio: 1.0836
- minimum leave-one-year-out ratio: 1.0836
- passes full period
- passes ex-2022
- passes leave-one-year
- still one-year dominated by 2022

VolTarget + DrawdownGovernor:
- full ratio: 2.2115
- ex-2022 ratio: 0.9955
- minimum leave-one-year-out ratio: 0.9955
- fails hard ex-2022 gate
- governor does not add value outside 2022

CoreOverlay + Rebound comparator:
- full ratio: 1.0107
- ex-2022 ratio: 1.2868
- minimum leave-one-year-out ratio: 0.8317
- comparator only
```

Stage 2 interpretation:

```text
Plain VolTarget is not purely a 2022 artifact.
However, 2022 remains a dominant contributor.
Governor is likely over-defensive outside 2022.
```

### 6.10 Stage 2 Fair Leverage Benchmark

Status: completed.

Commit:

- `e50bae3 Add Stage 2 fair leverage benchmark report`

Outputs:

- `outputs/tournament_voltarget_stage2/fair_leverage_benchmark_summary.csv`
- `outputs/tournament_voltarget_stage2/fair_leverage_benchmark_report.md`

Plain VolTarget classification:

```text
persistent_candidate
```

Important ratios:

```text
VolTargetTQQQStrategy:
- ratio_vs_1x_tqqq: about 2.2624
- ratio_vs_same_max_constant_tqqq: about 8.3457
- ratio_vs_same_avg_exposure_constant_tqqq: about 1.9360
- ratio_vs_simple_voltarget_no_trend: about 1.1148
- ex_2022_ratio_vs_same_avg_exposure_constant_tqqq: about 0.4441
- ex_2022_ratio_vs_simple_voltarget_no_trend: about 0.5811

VolTarget + DrawdownGovernor:
- classification: governor_overfit_to_2022
```

Important caveat:

```text
Plain VolTarget remains interesting ex-2022 versus 1.0x TQQQ, but not versus fair leverage/no-trend baselines ex-2022.
```

### 6.11 VolTarget Stage 3

Status: completed.

Commit:

- `16ce20b Add VolTarget Stage 3 robust-alpha gate`

Config:

- `configs/tournament_voltarget_stage3_robust_alpha.yaml`

Dry-run result:

```text
Total estimated runtime: about 57.222 seconds
Plain VolTarget estimate: about 57.024 seconds
Governor comparator estimate: about 0.198 seconds
```

Actual Stage 3 result:

```text
VolTargetTQQQStrategy classification: crash_control_candidate
Recommendation: defer_deep_tournament
```

Plain VolTarget Stage 3 metrics:

```text
full_period_ratio_vs_tqqq: 3.5287695693589667
ex_2022_ratio_vs_tqqq: 2.5367728830614986
post_2022_ratio_vs_tqqq: 1.1420330794412785
ratio_vs_same_avg_exposure_constant: 3.3684138514488438
ratio_vs_simple_voltarget_no_trend: 1.5296526074462844
ex_2022_ratio_vs_simple_voltarget_no_trend: 0.7004733508234562
one_year_dominance_share: 0.5390646723490914
```

Outputs:

- `outputs/tournament_voltarget_stage3/voltarget_stage3_summary.csv`
- `outputs/tournament_voltarget_stage3/voltarget_stage3_report.md`
- `outputs/tournament_voltarget_stage3/robust_alpha_audit.csv`
- `outputs/tournament_voltarget_stage3/fair_ex_2022_audit.csv`
- `outputs/tournament_voltarget_stage3/candidate_decision_table.csv`

Stage 3 classification options:

```text
robust_alpha_candidate
crash_control_candidate
leverage_risk_budget_candidate
reject
```

Full VolTarget deep tournament should only run if Stage 3 says:

```text
robust_alpha_candidate
```

Stage 3 did not say this. Do not run the full VolTarget deep tournament.

## 7. Important Files and Roles

### Core Runtime

- `research/cli.py`
  Main command router. Important commands include `run-config`, `run-tournament`, `summarize-voltarget-stage`, `audit-voltarget-timing`, `check-blockers`, and baseline tools.

- `research/experiments.py`
  Config-driven experiment runner. Enforces output contract for successful runs.

- `research/tournament.py`
  Tournament runner. Handles validation variants, cost scenarios, runtime guard, summary/failure files, and stage report hooks.

- `research/backtest.py`
  Core backtest mechanics. Handles shifted positions, risk-off allocation, transaction costs, and execution models.

- `research/reports.py`
  Same-period benchmark comparison and general report generation.

- `research/data.py`
  Price loader. Handles cached data, date slicing, CASH synthetic series, and OHLC loading.

### Strategy Logic

- `research/strategies/vol_target_strategy.py`
  Main VolTarget strategy. Do not change alpha logic during validation tasks.

- `research/strategies/core_overlay_strategy.py`
  Core-overlay strategy used as comparator.

- `research/strategies/regime_strategy.py`
  Legacy regime strategy.

- `research/strategies/ma_strategy.py`
  Existing MA ensemble strategy.

- `research/strategies/drawdown_governor.py`
  Optional governor layer. Currently deprioritized for primary research.

- `research/strategies/rebound.py`
  Optional rebound module.

- `research/strategies/constant_exposure_strategy.py`
  Constant exposure benchmark implementation.

### VolTarget Validation

- `research/voltarget_timing_audit.py`
  Look-ahead/timing audit.

- `research/voltarget_synthetic_history.py`
  Synthetic long-history validation.

- `research/fairness.py`
  Constant leverage fairness calculations and per-experiment VolTarget fairness reports.

- `research/voltarget_stage.py`
  Stage 1/staged VolTarget summary.

- `research/voltarget_stage2.py`
  Year-sensitivity and 2022-dominance validation.

- `research/voltarget_fair_leverage.py`
  Stage 2 fair leverage/no-trend attribution.

- `research/voltarget_stage3.py`
  Stage 3 robust-alpha gate and candidate decision report.

### Baseline and Safety Gates

- `research/baseline_gate.py`
  Baseline regression gate and acceptance logic.

- `research/replay.py`
  Legacy regime window replay.

- `research/regime_allocation_audit.py`
  Allocation/signal audit for regime baseline mismatch.

- `research/data_snapshot.py`
  Freeze and verify local data snapshots.

- `research/candidates.py`
  Final candidate extraction; should not run against incomplete tournament outputs.

### Configs

Important VolTarget configs:

- `configs/vol_target_tqqq.yaml`
- `configs/vol_target_governor.yaml`
- `configs/voltarget_synthetic_history.yaml`
- `configs/tournament_voltarget_stage1.yaml`
- `configs/tournament_voltarget_stage2_2022_sensitivity.yaml`
- `configs/tournament_voltarget_stage3_robust_alpha.yaml`
- `configs/tournament_voltarget_deep.yaml`

Do not run:

- `configs/tournament.yaml`
- `configs/tournament_voltarget_deep.yaml`

Stage 3 completed and did not justify either run.

## 8. Important Commit History

Recent important commits:

```text
16ce20b Add VolTarget Stage 3 robust-alpha gate
e50bae3 Add Stage 2 fair leverage benchmark report
9eb71c9 Add VolTarget Stage 2 year-sensitivity validation
e1983ca Add VolTarget one-year contribution audit
c079d44 Clarify VolTarget constant-leverage benchmark
4c467b3 Add staged VolTarget validation configs
d68f62f Add VolTarget synthetic-history validation
0d73be9 Add VolTarget timing audit
a54bce1 Add constant-leverage fairness benchmarks
0f67909 Add VolTarget deep validation tournament config
f95340a Fix MA ensemble tournament wiring
5f9538c Add tournament gate variant artifacts
```

## 9. Current Test Status

Latest known full test status:

```text
python -m pytest tests -q
141 passed, 1 warning
```

The warning is from a third-party protobuf datetime deprecation, not a project failure.

## 10. Current Git Status

Latest known state:

```text
Branch: fix/research-gates
Worktree: clean
```

Verify in a new session:

```powershell
cd "C:\Strategies\TQQQ MA"
git status --short --branch
```

## 11. Immediate Next Step

Freeze the research phase.

Final classification:

```text
VolTargetTQQQStrategy = crash_control_candidate
Not persistent alpha
Not production-ready
Do not run deep tournament
Do not run full tournament
```

Final wording:

```text
The research found that VolTarget is the only serious candidate family. It historically outperformed TQQQ in the tested framework and survived ex-2022 versus TQQQ, but it did not pass the stricter robust-alpha attribution gate because ex-2022 performance versus SimpleVolTargetNoTrend remained weak. Therefore, it should be classified as a TQQQ crash-control / risk-budget overlay, not as proven persistent alpha.
```

The only worthwhile next project is a paper-trading signal monitor, not more backtest searching.

## 12. What Not To Do Next

Do not:

- run full `configs/tournament.yaml`
- run full `configs/tournament_voltarget_deep.yaml`
- widen all grids
- change VolTarget alpha logic
- mix SOXL sector-bet results into the TQQQ strategy conclusion
- declare final success based only on full-period or 2022-driven performance
- claim this is proven persistent alpha
- claim this is production-ready

## 13. Research Framing Going Forward

The clean final statement is:

```text
The research found that VolTarget is the only serious candidate family. It historically outperformed TQQQ in the tested framework and survived ex-2022 versus TQQQ, but it did not pass the stricter robust-alpha attribution gate because ex-2022 performance versus SimpleVolTargetNoTrend remained weak. Therefore, it should be classified as a TQQQ crash-control / risk-budget overlay, not as proven persistent alpha.
```
