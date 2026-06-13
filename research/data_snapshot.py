from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf

from .data import CASH_SYMBOL, load_prices
from .execution import OHLC_FIELDS, ohlc_column
from .reports import _markdown_table


MANIFEST_VERSION = 1
DATA_LOADER_VERSION = "research.data.load_prices:v1"
AUTO_ADJUST = True
VOLUME_FIELD = "VOLUME"


def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
    out: List[str] = []
    for symbol in symbols:
        text = str(symbol).strip().upper()
        if text and text not in out:
            out.append(text)
    if not out:
        raise ValueError("At least one symbol is required.")
    return out


def _cache_file_path(
    *,
    symbols: Sequence[str],
    cache_dir: str,
    include_ohlc: bool,
) -> Path:
    non_cash = tuple(symbol for symbol in _normalize_symbols(symbols) if symbol != CASH_SYMBOL)
    if not non_cash:
        return Path(cache_dir) / "cash_ohlc.csv" if include_ohlc else Path(cache_dir) / "cash_prices.csv"
    cache_name = (
        "tqqq_qqq_prices.csv"
        if non_cash == ("TQQQ", "QQQ") and not include_ohlc
        else f"{'_'.join(s.lower() for s in non_cash)}_{'ohlc' if include_ohlc else 'prices'}.csv"
    )
    return Path(cache_dir) / cache_name


def _read_data_csv(path: Path, *, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    data = data.loc[data.index >= pd.Timestamp(start_date)].copy()
    if end_date:
        data = data.loc[data.index <= pd.Timestamp(end_date)].copy()
    data.index.name = "Date"
    return data


def _load_snapshot_data(
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: Optional[str],
    cache_dir: str,
    data_csv: Optional[Path],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    normalized = _normalize_symbols(symbols)
    cache_path = _cache_file_path(symbols=normalized, cache_dir=cache_dir, include_ohlc=True)
    cache_exists_before = cache_path.exists()
    if data_csv is not None:
        data = _read_data_csv(data_csv, start_date=start_date, end_date=end_date)
        source = "data_csv"
    else:
        data = load_prices(
            start=start_date,
            end=end_date,
            symbols=normalized,
            use_csv_if_exists=True,
            cache_dir=cache_dir,
            dropna=False,
            allow_missing_symbols=True,
            include_ohlc=True,
        )
        source = "cache" if cache_exists_before else "download"
        if source == "download" and cache_path.exists():
            # Hash the persisted cache representation so immediate verification
            # compares like-for-like after a fresh download creates the cache.
            data = load_prices(
                start=start_date,
                end=end_date,
                symbols=normalized,
                use_csv_if_exists=True,
                cache_dir=cache_dir,
                dropna=False,
                allow_missing_symbols=True,
                include_ohlc=True,
            )
    metadata = {
        "cache_path_used": str(cache_path),
        "data_source": source,
        "came_from_cache": source == "cache",
        "download_occurred": source == "download",
        "data_csv": str(data_csv) if data_csv is not None else "",
    }
    return data, metadata


def _format_hash_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    try:
        return format(float(value), ".17g")
    except (TypeError, ValueError):
        return str(value)


def _hash_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    hasher = hashlib.sha256()
    columns = [str(column) for column in frame.columns]
    hasher.update(("columns|" + "|".join(columns) + "\n").encode("utf-8"))
    for idx, row in frame.iterrows():
        date_text = pd.Timestamp(idx).date().isoformat()
        values = [_format_hash_value(row[column]) for column in frame.columns]
        hasher.update((date_text + "|" + "|".join(values) + "\n").encode("utf-8"))
    return hasher.hexdigest()


def _hash_close_series(data: pd.DataFrame, symbol: str) -> str:
    if symbol not in data.columns:
        return ""
    frame = pd.DataFrame({"close": pd.to_numeric(data[symbol], errors="coerce")}, index=data.index).dropna()
    return _hash_frame(frame)


def _ohlcv_columns(data: pd.DataFrame, symbol: str) -> List[str]:
    candidates = [ohlc_column(symbol, field) for field in OHLC_FIELDS] + [ohlc_column(symbol, VOLUME_FIELD)]
    return [column for column in candidates if column in data.columns]


def _hash_ohlcv_series(data: pd.DataFrame, symbol: str) -> str:
    columns = _ohlcv_columns(data, symbol)
    if not columns:
        return ""
    frame = data[columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all")
    return _hash_frame(frame)


def _symbol_manifest(symbol: str, data: pd.DataFrame, union_index: pd.Index) -> Dict[str, Any]:
    if symbol not in data.columns:
        return {
            "symbol": symbol,
            "loaded": False,
            "first_date": "",
            "last_date": "",
            "row_count": 0,
            "missing_rows_vs_union_calendar": int(len(union_index)),
            "close_hash_sha256": "",
            "ohlcv_hash_sha256": "",
            "ohlcv_columns": [],
        }

    close = pd.to_numeric(data[symbol], errors="coerce").reindex(union_index)
    non_missing = close.dropna()
    return {
        "symbol": symbol,
        "loaded": not non_missing.empty,
        "first_date": non_missing.index.min().date().isoformat() if not non_missing.empty else "",
        "last_date": non_missing.index.max().date().isoformat() if not non_missing.empty else "",
        "row_count": int(len(non_missing)),
        "missing_rows_vs_union_calendar": int(close.isna().sum()),
        "close_hash_sha256": _hash_close_series(data.reindex(union_index), symbol),
        "ohlcv_hash_sha256": _hash_ohlcv_series(data.reindex(union_index), symbol),
        "ohlcv_columns": _ohlcv_columns(data, symbol),
    }


def build_data_snapshot_manifest(
    *,
    symbols: Sequence[str],
    output_dir: Path,
    start_date: str = "2011-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "./price_cache",
    data_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    normalized = _normalize_symbols(symbols)
    data, metadata = _load_snapshot_data(
        symbols=normalized,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        data_csv=data_csv,
    )
    union_index = data.index.sort_values().unique()
    per_symbol = {
        symbol: _symbol_manifest(symbol, data, union_index)
        for symbol in normalized
    }
    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_loader_version": DATA_LOADER_VERSION,
        "yfinance_version": getattr(yf, "__version__", ""),
        "auto_adjust": AUTO_ADJUST,
        "include_ohlc": True,
        "symbols": normalized,
        "start_date": start_date,
        "end_date": end_date or "",
        "cache_dir": cache_dir,
        "cache_path_used": metadata["cache_path_used"],
        "data_source": metadata["data_source"],
        "came_from_cache": bool(metadata["came_from_cache"]),
        "download_occurred": bool(metadata["download_occurred"]),
        "data_csv": metadata["data_csv"],
        "union_calendar_row_count": int(len(union_index)),
        "per_symbol": per_symbol,
    }


def _manifest_to_summary(manifest: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, info in manifest.get("per_symbol", {}).items():
        rows.append(
            {
                "symbol": symbol,
                "loaded": bool(info.get("loaded", False)),
                "first_date": info.get("first_date", ""),
                "last_date": info.get("last_date", ""),
                "row_count": int(info.get("row_count", 0)),
                "missing_rows_vs_union_calendar": int(info.get("missing_rows_vs_union_calendar", 0)),
                "close_hash_sha256": info.get("close_hash_sha256", ""),
                "ohlcv_hash_sha256": info.get("ohlcv_hash_sha256", ""),
                "ohlcv_columns": ";".join(info.get("ohlcv_columns", [])),
            }
        )
    return pd.DataFrame(rows)


def _write_snapshot_report(output_dir: Path, manifest: Dict[str, Any], summary: pd.DataFrame) -> Path:
    report_path = output_dir / "data_snapshot_report.md"
    lines = [
        "# Data Snapshot Manifest",
        "",
        "This manifest records hashes for loaded close and OHLCV data so future runs can detect data drift.",
        "",
        "## Global",
        f"- Generated UTC: `{manifest.get('generated_at_utc', '')}`",
        f"- Data loader version: `{manifest.get('data_loader_version', '')}`",
        f"- yfinance version: `{manifest.get('yfinance_version', '')}`",
        f"- Data source: `{manifest.get('data_source', '')}`",
        f"- Cache path used: `{manifest.get('cache_path_used', '')}`",
        f"- Start date: `{manifest.get('start_date', '')}`",
        f"- End date: `{manifest.get('end_date', '')}`",
        f"- Auto adjust: `{manifest.get('auto_adjust', '')}`",
        "",
        "## Symbols",
        _markdown_table(summary, max_rows=200),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def freeze_data_snapshot(
    *,
    symbols: Sequence[str],
    output_dir: Path,
    start_date: str = "2011-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "./price_cache",
    data_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_data_snapshot_manifest(
        symbols=symbols,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        data_csv=data_csv,
    )
    summary = _manifest_to_summary(manifest)
    (output_dir / "data_snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary.to_csv(output_dir / "data_snapshot_summary.csv", index=False)
    _write_snapshot_report(output_dir, manifest, summary)
    print(summary.to_string(index=False))
    print(f"\nData snapshot manifest written to: {output_dir / 'data_snapshot_manifest.json'}")
    return manifest


def _load_manifest(path: Path) -> Dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return loaded


def _verify_rows(old: Dict[str, Any], current: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    old_symbols = _normalize_symbols(old.get("symbols", []))
    current_per_symbol = current.get("per_symbol", {})
    old_per_symbol = old.get("per_symbol", {})
    for symbol in old_symbols:
        old_info = old_per_symbol.get(symbol, {})
        current_info = current_per_symbol.get(symbol, {})
        close_match = old_info.get("close_hash_sha256", "") == current_info.get("close_hash_sha256", "")
        ohlcv_match = old_info.get("ohlcv_hash_sha256", "") == current_info.get("ohlcv_hash_sha256", "")
        loaded_match = bool(old_info.get("loaded", False)) == bool(current_info.get("loaded", False))
        first_match = str(old_info.get("first_date", "")) == str(current_info.get("first_date", ""))
        last_match = str(old_info.get("last_date", "")) == str(current_info.get("last_date", ""))
        row_count_match = int(old_info.get("row_count", 0)) == int(current_info.get("row_count", 0))
        passed = bool(close_match and ohlcv_match and loaded_match and first_match and last_match and row_count_match)
        mismatch_fields = []
        for name, ok in (
            ("loaded", loaded_match),
            ("first_date", first_match),
            ("last_date", last_match),
            ("row_count", row_count_match),
            ("close_hash_sha256", close_match),
            ("ohlcv_hash_sha256", ohlcv_match),
        ):
            if not ok:
                mismatch_fields.append(name)
        rows.append(
            {
                "symbol": symbol,
                "passed": passed,
                "mismatch_fields": ";".join(mismatch_fields),
                "old_first_date": old_info.get("first_date", ""),
                "current_first_date": current_info.get("first_date", ""),
                "old_last_date": old_info.get("last_date", ""),
                "current_last_date": current_info.get("last_date", ""),
                "old_row_count": old_info.get("row_count", 0),
                "current_row_count": current_info.get("row_count", 0),
                "old_close_hash_sha256": old_info.get("close_hash_sha256", ""),
                "current_close_hash_sha256": current_info.get("close_hash_sha256", ""),
                "old_ohlcv_hash_sha256": old_info.get("ohlcv_hash_sha256", ""),
                "current_ohlcv_hash_sha256": current_info.get("ohlcv_hash_sha256", ""),
            }
        )
    return pd.DataFrame(rows)


def _write_verify_report(output_dir: Path, manifest_path: Path, rows: pd.DataFrame, passed: bool) -> Path:
    report_path = output_dir / "data_snapshot_verify_report.md"
    mismatches = rows[~rows["passed"].astype(bool)].copy() if not rows.empty else rows
    lines = [
        "# Data Snapshot Verification",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Passed: `{passed}`",
        "",
        "## Symbol Checks",
        _markdown_table(rows, max_rows=200),
        "",
        "## Mismatches",
        _markdown_table(mismatches, max_rows=200),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def verify_data_snapshot(
    manifest_path: Path,
    *,
    allow_drift: bool = False,
    cache_dir: Optional[str] = None,
    data_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, bool]:
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    effective_cache_dir = cache_dir or str(manifest.get("cache_dir", "./price_cache"))
    effective_data_csv = data_csv
    if effective_data_csv is None and manifest.get("data_source") == "data_csv":
        data_csv_text = str(manifest.get("data_csv", ""))
        effective_data_csv = Path(data_csv_text) if data_csv_text else None
    current = build_data_snapshot_manifest(
        symbols=manifest.get("symbols", []),
        output_dir=manifest_path.parent,
        start_date=str(manifest.get("start_date", "2011-01-01")),
        end_date=str(manifest.get("end_date", "")) or None,
        cache_dir=effective_cache_dir,
        data_csv=effective_data_csv,
    )
    rows = _verify_rows(manifest, current)
    passed = bool(rows["passed"].all()) if not rows.empty else False
    output_dir = manifest_path.parent
    rows.to_csv(output_dir / "data_snapshot_verify_summary.csv", index=False)
    _write_verify_report(output_dir, manifest_path, rows, passed)
    print(rows.to_string(index=False))
    if not passed:
        print(f"\nData snapshot drift detected. Verification report: {output_dir / 'data_snapshot_verify_report.md'}")
    else:
        print(f"\nData snapshot verification passed. Verification report: {output_dir / 'data_snapshot_verify_report.md'}")
    return rows, bool(passed or allow_drift)
