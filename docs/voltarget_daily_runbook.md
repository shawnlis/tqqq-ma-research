# VolTarget Daily Paper-Monitor Runbook

This runbook is for the locked VolTarget paper monitor only.

It is not a production trading system, not a broker workflow, and not an investment recommendation. The monitor is classified as `crash_control_candidate`, not proven persistent alpha.

## Daily Command

Run after the market close and after the daily TQQQ/QQQ price cache has been refreshed:

```powershell
cd "C:\Strategies\TQQQ MA"
python -m research.cli run-daily-voltarget-monitor --config configs\voltarget_live_monitor.yaml
```

The wrapper runs these checks and reports in order:

1. Data freshness check.
2. Data snapshot verification, only when `data_snapshot_manifest` is configured.
3. VolTarget paper signal generation.
4. Risk policy dashboard refresh.
5. Equity-curve visualization refresh.
6. Daily run status logging.

## Files To Inspect

Inspect these files after each run:

- `outputs/live_signal/latest_run_status.json`
- `outputs/live_signal/daily_run_log.csv`
- `outputs/live_signal/signal_today.json`
- `outputs/live_signal/signal_report.md`
- `outputs/live_signal/data_quality_report.csv`
- `outputs/live_signal/risk_policy_breaches.csv`
- `outputs/live_signal/dashboard.md`
- `outputs/visualizations/voltarget_stage3/visualization_report.md`

If `latest_run_status.json` says `failed`, do not use the signal for paper tracking until the failure is understood and rerun successfully.

If it says `warning`, inspect `data_quality_report.csv`, stale days, and any risk policy breaches before recording a paper-trade observation.

## What Stale Data Means

Stale data means the latest price date used by the monitor is older than the configured freshness limit in `configs/voltarget_live_monitor.yaml`.

The default field is:

```yaml
stale_data_warning_days: 5
```

Stale data does not mean the strategy found a new trade. It means the monitor may be using old prices and the current signal may not reflect the most recent close.

## What Not To Do

Do not:

- Treat this as proven persistent alpha.
- Treat this as production-ready.
- Use the monitor to place trades automatically.
- Add broker integration to this monitor.
- Convert the output directly into live trading instructions.
- Tune parameters in response to one daily signal.
- Run the deep tournament or full tournament from this daily workflow.
- Ignore stale data, failed snapshot verification, or failed run status.

## Paper Trading Only

This workflow is for paper-trading observation and operational validation only.

Allowed uses:

- Record the daily target exposure for a paper log.
- Compare expected paper equity against monitor output.
- Track stale data warnings and risk policy breaches.
- Review execution and financing assumptions.

Not allowed in this workflow:

- Auto-trading.
- Broker calls.
- Account allocation decisions.
- Production signal distribution.
- Investment recommendations.

## Failure Handling

When the wrapper fails, it still writes:

- `outputs/live_signal/latest_run_status.json`
- `outputs/live_signal/daily_run_log.csv`

Use those files to identify which step failed. Fix the data/configuration issue, rerun the wrapper, and keep the paper monitor disabled until the latest run is `ok` or an explicitly accepted `warning`.
