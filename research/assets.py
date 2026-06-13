from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AssetConfig:
    trade_asset: str = "TQQQ"
    primary_signal_asset: str = "QQQ"
    secondary_filter_asset: str = "QQQ"
    benchmark_symbol: str = "TQQQ"
    risk_off_symbol: str = "CASH"

    def normalized(self) -> "AssetConfig":
        return AssetConfig(
            trade_asset=str(self.trade_asset).upper(),
            primary_signal_asset=str(self.primary_signal_asset).upper(),
            secondary_filter_asset=str(self.secondary_filter_asset).upper(),
            benchmark_symbol=str(self.benchmark_symbol).upper(),
            risk_off_symbol=str(self.risk_off_symbol).upper(),
        )


DEFAULT_ASSET_CONFIG = AssetConfig()


def asset_config_from_mapping(
    values: Mapping[str, Any] | None = None,
    *,
    benchmark_symbol: str = "TQQQ",
    risk_off_symbol: str = "CASH",
) -> AssetConfig:
    values = values or {}
    return AssetConfig(
        trade_asset=str(values.get("trade_asset", DEFAULT_ASSET_CONFIG.trade_asset)).upper(),
        primary_signal_asset=str(
            values.get("primary_signal_asset", values.get("signal_asset", DEFAULT_ASSET_CONFIG.primary_signal_asset))
        ).upper(),
        secondary_filter_asset=str(
            values.get(
                "secondary_filter_asset",
                values.get("filter_asset", DEFAULT_ASSET_CONFIG.secondary_filter_asset),
            )
        ).upper(),
        benchmark_symbol=str(values.get("benchmark_symbol", benchmark_symbol)).upper(),
        risk_off_symbol=str(values.get("risk_off_symbol", risk_off_symbol)).upper(),
    )


def asset_config_row_fields(asset_config: AssetConfig) -> dict[str, str]:
    asset_config = asset_config.normalized()
    return {
        "trade_asset": asset_config.trade_asset,
        "primary_signal_asset": asset_config.primary_signal_asset,
        "secondary_filter_asset": asset_config.secondary_filter_asset,
        "benchmark_symbol": asset_config.benchmark_symbol,
        "asset_config_risk_off_symbol": asset_config.risk_off_symbol,
    }


def asset_config_from_row(row: Mapping[str, Any], fallback: AssetConfig | None = None) -> AssetConfig:
    fallback = (fallback or DEFAULT_ASSET_CONFIG).normalized()
    return AssetConfig(
        trade_asset=str(row.get("trade_asset", fallback.trade_asset)).upper(),
        primary_signal_asset=str(row.get("primary_signal_asset", fallback.primary_signal_asset)).upper(),
        secondary_filter_asset=str(row.get("secondary_filter_asset", fallback.secondary_filter_asset)).upper(),
        benchmark_symbol=str(row.get("benchmark_symbol", fallback.benchmark_symbol)).upper(),
        risk_off_symbol=str(row.get("asset_config_risk_off_symbol", fallback.risk_off_symbol)).upper(),
    )
