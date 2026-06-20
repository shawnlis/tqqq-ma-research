# VolTarget Paper-Monitor Automation Runbook

This workflow is for operational paper-monitoring only.

It is not production-ready, not persistent alpha, not auto-trading, and not broker integration. The frozen model classification remains `crash_control_candidate`.

## Manual Daily Run

Run after the market close data is available, or the next Singapore morning:

```powershell
cd "C:\Strategies\TQQQ MA"
python -m research.cli run-automated-voltarget-paper-monitor --config configs\voltarget_live_monitor.yaml
```

The one-command workflow runs signal generation, the automated paper ledger, execution drift tracking, financing tracking, the risk dashboard, and health checks.

Manual ledger reconciliation remains optional. If the manual ledger exists, you may also run:

```powershell
python -m research.cli reconcile-paper-trades --signal-dir outputs\live_signal --ledger data\paper_trading\voltarget_paper_trades.csv --output-dir outputs\paper_reconciliation
```

If `auto_paper_ledger_enabled: true` and `manual_ledger_required: false`, a missing manual ledger is not an automation warning.

## Manual Weekly Review

Weekly review is optional and only runs when the CLI exists:

```powershell
cd "C:\Strategies\TQQQ MA"
python -m research.cli weekly-voltarget-review --signal-dir outputs\live_signal --ledger data\paper_trading\voltarget_paper_trades.csv --output-dir outputs\weekly_review
```

If `weekly-voltarget-review` is not implemented, the weekly template logs a warning and exits successfully.

## Template Scripts

Templates are committed under:

- `scripts/run_voltarget_daily_monitor.ps1.template`
- `scripts/run_voltarget_weekly_review.ps1.template`
- `scripts/create_voltarget_monitor_tasks.ps1.template`
- `automation/windows_task_voltarget_daily.xml.template`
- `automation/windows_task_voltarget_weekly.xml.template`

Do not run `.template` files as production automation. Copy and review them first:

```powershell
Copy-Item scripts\run_voltarget_daily_monitor.ps1.template scripts\run_voltarget_daily_monitor.ps1
Copy-Item scripts\run_voltarget_weekly_review.ps1.template scripts\run_voltarget_weekly_review.ps1
Copy-Item scripts\create_voltarget_monitor_tasks.ps1.template scripts\create_voltarget_monitor_tasks.ps1
Copy-Item automation\windows_task_voltarget_daily.xml.template automation\windows_task_voltarget_daily.xml
Copy-Item automation\windows_task_voltarget_weekly.xml.template automation\windows_task_voltarget_weekly.xml
```

Then edit the copied XML files and replace:

```text
__REPO_ROOT__
```

with:

```text
C:\Strategies\TQQQ MA
```

Generated `.ps1` and `.xml` files are local operational files and should not be committed.

## Enable Windows Task Scheduler

Review the copied files first. Then run:

```powershell
cd "C:\Strategies\TQQQ MA"
powershell -ExecutionPolicy Bypass -File scripts\create_voltarget_monitor_tasks.ps1
```

The helper prints the exact `schtasks` commands and requires typing:

```text
CREATE
```

before creating tasks. It must not create tasks silently.

Daily schedule:

- Singapore local time
- Monday to Friday
- 7:00 AM
- Runs `scripts\run_voltarget_daily_monitor.ps1`

Weekly schedule:

- Singapore local time
- Saturday
- 9:00 AM
- Runs `scripts\run_voltarget_weekly_review.ps1`

## Disable Scheduled Tasks

```powershell
schtasks /Change /TN "VolTarget Paper Monitor Daily" /DISABLE
schtasks /Change /TN "VolTarget Paper Monitor Weekly" /DISABLE
```

To delete them:

```powershell
schtasks /Delete /TN "VolTarget Paper Monitor Daily" /F
schtasks /Delete /TN "VolTarget Paper Monitor Weekly" /F
```

## Logs

Automation logs are written under:

```text
logs/voltarget_monitor/
```

Daily automation status is written to:

```text
outputs/live_signal/latest_automation_status.json
```

Health-check outputs are written to:

```text
outputs/live_signal/monitor_health_summary.csv
outputs/live_signal/monitor_health_report.md
```

## Morning Inspection Checklist

Inspect these files each morning:

- `outputs/live_signal/latest_automation_status.json`
- `outputs/auto_paper_ledger/auto_paper_report.md`
- `outputs/auto_paper_ledger/auto_paper_summary.csv`
- `outputs/auto_paper_ledger/auto_paper_ledger.csv`
- `outputs/live_signal/latest_run_status.json`
- `outputs/live_signal/signal_today.json`
- `outputs/live_signal/data_quality_report.csv`
- `outputs/live_signal/financing_cost_report.md`
- `outputs/execution_drift/execution_drift_report.md`
- `outputs/paper_reconciliation/paper_trade_reconciliation_report.md`, if the optional manual ledger exists
- latest file under `logs/voltarget_monitor/`

## Warning Meanings

- Stale data: the latest price date is older than the configured freshness threshold.
- Pending fill: the next open needed for an automated paper fill is not available yet.
- Missing manual ledger: ignored when auto paper ledger mode is enabled and manual ledger is not required.
- Execution drift warning: observable open/close execution behavior is diverging from model assumptions.
- Financing warning: exposure above 1.0 creates financing drag; review `financing_cost_report.md`.
- Missing weekly CLI: weekly review automation is not installed; daily paper monitoring is unaffected.

## What Not To Do

Do not:

- Change strategy logic.
- Change parameters.
- Run the full tournament.
- Run the deep tournament.
- Add broker APIs.
- Add auto-trading.
- Treat the paper signal as a live order.
- Commit generated `outputs/`, `logs/`, copied `.ps1`, or copied `.xml` files.

This automation is only a disciplined way to run the frozen VolTarget Paper Monitor v1.
