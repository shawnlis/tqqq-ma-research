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
2. VolTarget signal generation.
3. Automated paper ledger update.
4. Execution drift tracking.
5. Financing tracking through the signal monitor.
6. Risk dashboard refresh.
7. Health check.
8. `outputs/live_signal/latest_automation_status.json` update.
9. `outputs/live_signal/daily_run_log.csv` update.

## Auto Paper Fill Assumption

The auto ledger reads:

- `outputs/live_signal/signal_history.csv`
- `outputs/live_signal/signal_today.json`
- TQQQ OHLC price data

For each signal date, the default fill is the next available TQQQ open. The ledger calculates target notional from configured paper equity and target exposure, converts that to paper shares, applies configured slippage and transaction costs, deducts financing cost when exposure is above the configured threshold, and updates paper cash, paper shares, paper equity, TQQQ buy-and-hold benchmark equity, and relative equity versus TQQQ.

No real order is created. No broker account is read or written.

## Status Meanings

- `ok`: the signal has an observable paper fill.
- `pending_fill`: the next open required for the paper fill is not available yet.
- `stale_data`: the latest signal was flagged stale by the signal monitor.
- `missing_price_data`: required OHLC data was not available.
- `duplicate_skipped`: duplicate signal dates were ignored during ledger normalization.
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
- latest auto paper ledger status
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
