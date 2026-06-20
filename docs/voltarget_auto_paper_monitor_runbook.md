# VolTarget Automated Paper Monitor Runbook

This workflow is paper monitoring only. It is not real trading, not broker integration, not production-ready, and not an investment recommendation.

The frozen strategy classification remains `crash_control_candidate`. Do not change strategy alpha logic, tune parameters, run deep tournaments, or add live execution paths from this workflow.

## One-Command Daily Workflow

Run:

```powershell
cd "C:\Strategies\TQQQ MA"
python -m research.cli run-automated-voltarget-paper-monitor --config configs\voltarget_live_monitor.yaml
```

The command runs:

1. Data freshness check.
2. Recent TQQQ/QQQ OHLC refresh and cache merge.
3. VolTarget signal generation, only when data is fresh enough.
4. Automated paper ledger update.
5. Execution drift tracking.
6. Financing tracking through the signal monitor.
7. Risk dashboard refresh.
8. Health check.
9. `outputs/live_signal/latest_automation_status.json` update.
10. `outputs/live_signal/daily_run_log.csv` update.

If data remains stale after refresh and `stale_data_blocks_new_signal: true`, no new accepted signal is generated. The previous `signal_today.json` and `signal_history.csv` are left unchanged, and the workflow exits 0 with warning unless `fail_on_stale_data: true`.

## Initialize Live Paper Ledger

The default config does not allow historical backfill:

```yaml
paper_ledger_start_mode: live_from_config_date
paper_ledger_allow_historical_backfill: false
paper_ledger_start_date:
```

Before unattended use, initialize the live ledger:

```powershell
python -m research.cli initialize-auto-paper-ledger --config configs\voltarget_live_monitor.yaml --start-date YYYY-MM-DD --starting-equity 100000 --output-dir outputs\auto_paper_ledger --confirm-reset
```

This writes local generated state under:

```text
outputs/auto_paper_ledger/auto_paper_ledger_state.json
outputs/auto_paper_ledger/auto_paper_ledger.csv
```

Without `--confirm-reset`, the command refuses to reset an existing ledger.

## Auto Paper Fill Assumption

The auto ledger reads:

- `outputs/live_signal/signal_history.csv`
- `outputs/live_signal/signal_today.json`
- TQQQ OHLC price data

For each signal date, the default fill is the next available TQQQ open. The ledger calculates target notional from configured paper equity and target exposure, converts that to paper shares, applies configured slippage and transaction costs, deducts financing cost when exposure is above the configured threshold, and updates paper cash, paper shares, paper equity, TQQQ buy-and-hold benchmark equity, and relative equity versus TQQQ.

No real order is created. No broker account is read or written.

Historical backfill remains available only if `paper_ledger_allow_historical_backfill: true`. Do not enable it for live unattended paper monitoring.

## Status Meanings

- `ok`: the signal has an observable paper fill.
- `pending_fill`: the next open required for the paper fill is not available yet.
- `stale_data`: the latest signal was flagged stale by the signal monitor.
- `missing_price_data`: required OHLC data was not available.
- `duplicate_skipped`: duplicate signal dates were ignored during ledger normalization.
- `initialized`: live ledger state exists, but no eligible live signal has been processed yet.
- `needs_start_date`: live ledger mode is enabled but no start date or initialized state exists.
- `failed`: the ledger update failed.

`pending_fill` is expected when the signal exists but the next trading session open is not yet available. Rerun the one-command workflow after the next open data appears.

`stale_data` means the monitor is using data older than the configured freshness threshold. Do not treat the signal as current paper evidence until the data issue is resolved.

## Outputs

Auto ledger outputs are saved under:

```text
outputs/auto_paper_ledger/
```

Important files:

- `auto_paper_ledger.csv`
- `auto_paper_summary.csv`
- `auto_paper_report.md`
- `paper_equity_vs_tqqq.png`
- `relative_equity_vs_tqqq.png`
- `paper_exposure_history.png`
- `paper_trade_delta_history.png`

Daily automation status is saved to:

```text
outputs/live_signal/latest_automation_status.json
outputs/live_signal/daily_run_log.csv
```

## Daily Inspection

Inspect:

```powershell
Get-Content outputs\live_signal\latest_automation_status.json
Import-Csv outputs\auto_paper_ledger\auto_paper_summary.csv
Import-Csv outputs\auto_paper_ledger\auto_paper_ledger.csv | Select-Object -Last 5
```

Check:

- latest automation status
- latest refresh attempt and refresh success
- latest price date and stale days
- latest auto paper ledger status
- whether the ledger is `live_monitor` or `historical_backfill`
- whether the live ledger is initialized
- latest target exposure
- latest paper equity
- latest relative equity versus TQQQ
- pending fills
- stale data rows
- execution drift warnings

## Disable Automation

If Windows Scheduled Tasks were created later, disable them with:

```powershell
schtasks /Change /TN "VolTarget Paper Monitor Daily" /DISABLE
schtasks /Change /TN "VolTarget Paper Monitor Weekly" /DISABLE
```

The committed files are templates only. Task creation still requires copying reviewed template files and typing `CREATE` in the task helper.

## Reset The Paper Ledger

To reset the simulated paper ledger, archive or delete only generated output files under:

```text
outputs/auto_paper_ledger/
```

Do not delete source code, configs, or tests. The next run will rebuild the auto ledger from `signal_history.csv` and available price data.

## Switch Back To Manual Ledger Mode

Edit `configs/voltarget_live_monitor.yaml`:

```yaml
paper_ledger_mode: manual
auto_paper_ledger_enabled: false
manual_ledger_required: true
```

Then maintain:

```text
data/paper_trading/voltarget_paper_trades.csv
```

from:

```text
data/paper_trading/voltarget_paper_trades_template.csv
```

Manual ledger reconciliation remains available through:

```powershell
python -m research.cli reconcile-paper-trades --signal-dir outputs\live_signal --ledger data\paper_trading\voltarget_paper_trades.csv --output-dir outputs\paper_reconciliation
```
