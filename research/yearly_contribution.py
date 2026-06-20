from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


ONE_YEAR_CONTRIBUTION_COLUMNS = [
    "family",
    "category",
    "experiment_name",
    "config_path",
    "variant",
    "walk_forward_variant",
    "transaction_cost_bps",
    "execution_model",
    "output_dir",
    "year",
    "yearly_strategy_return",
    "yearly_tqqq_return",
    "yearly_excess_return",
    "yearly_log_excess_return",
    "total_log_excess_return",
    "contribution_to_total_log_excess",
    "largest_contribution_year",
    "largest_single_year_contribution_share",
    "one_year_contributes_more_than_60pct",
    "audit_status",
    "audit_error",
]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata_value(metadata: Optional[Dict[str, Any]], key: str, default: Any = "") -> Any:
    if not metadata:
        return default
    value = metadata.get(key, default)
    if value is None:
        return default
    return value


def _blank_row(
    *,
    metadata: Optional[Dict[str, Any]] = None,
    status: str,
    error: str = "",
) -> Dict[str, Any]:
    row = {column: "" for column in ONE_YEAR_CONTRIBUTION_COLUMNS}
    for key in (
        "family",
        "category",
        "experiment_name",
        "config_path",
        "variant",
        "walk_forward_variant",
        "transaction_cost_bps",
        "execution_model",
        "output_dir",
    ):
        row[key] = _metadata_value(metadata, key, "")
    for key in (
        "year",
        "yearly_strategy_return",
        "yearly_tqqq_return",
        "yearly_excess_return",
        "yearly_log_excess_return",
        "total_log_excess_return",
        "contribution_to_total_log_excess",
        "largest_contribution_year",
        "largest_single_year_contribution_share",
    ):
        row[key] = np.nan
    row["one_year_contributes_more_than_60pct"] = False
    row["audit_status"] = status
    row["audit_error"] = error
    return row


def compute_one_year_contribution(
    yearly_returns: pd.DataFrame,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    threshold: float = 0.60,
) -> pd.DataFrame:
    required = {"year", "strategy_return", "benchmark_return"}
    if yearly_returns.empty:
        return pd.DataFrame(
            [_blank_row(metadata=metadata, status="missing_yearly_returns", error="yearly returns are empty")],
            columns=ONE_YEAR_CONTRIBUTION_COLUMNS,
        )
    missing = sorted(required.difference(yearly_returns.columns))
    if missing:
        return pd.DataFrame(
            [
                _blank_row(
                    metadata=metadata,
                    status="invalid_yearly_returns",
                    error=f"missing columns: {','.join(missing)}",
                )
            ],
            columns=ONE_YEAR_CONTRIBUTION_COLUMNS,
        )

    data = yearly_returns.copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["yearly_strategy_return"] = pd.to_numeric(data["strategy_return"], errors="coerce")
    data["yearly_tqqq_return"] = pd.to_numeric(data["benchmark_return"], errors="coerce")
    data = data.dropna(subset=["year", "yearly_strategy_return", "yearly_tqqq_return"]).copy()
    if data.empty:
        return pd.DataFrame(
            [_blank_row(metadata=metadata, status="invalid_yearly_returns", error="no numeric yearly rows")],
            columns=ONE_YEAR_CONTRIBUTION_COLUMNS,
        )

    valid_log = (data["yearly_strategy_return"] > -1.0) & (data["yearly_tqqq_return"] > -1.0)
    data = data[valid_log].copy()
    if data.empty:
        return pd.DataFrame(
            [_blank_row(metadata=metadata, status="invalid_yearly_returns", error="returns <= -100%")],
            columns=ONE_YEAR_CONTRIBUTION_COLUMNS,
        )

    data["yearly_excess_return"] = data["yearly_strategy_return"] - data["yearly_tqqq_return"]
    data["yearly_log_excess_return"] = np.log1p(data["yearly_strategy_return"]) - np.log1p(
        data["yearly_tqqq_return"]
    )
    total_log_excess = float(data["yearly_log_excess_return"].sum())
    if total_log_excess != 0.0 and np.isfinite(total_log_excess):
        data["contribution_to_total_log_excess"] = data["yearly_log_excess_return"] / total_log_excess
    else:
        data["contribution_to_total_log_excess"] = np.nan

    if total_log_excess > 0.0:
        contribution = pd.to_numeric(data["contribution_to_total_log_excess"], errors="coerce")
        largest_idx = contribution.idxmax()
        largest_share = float(contribution.loc[largest_idx])
        largest_year = int(data.loc[largest_idx, "year"])
        dominated = bool(largest_share > float(threshold))
        status = "ok"
        error = ""
    else:
        largest_share = 0.0
        largest_year = np.nan
        dominated = False
        status = "non_positive_total_log_excess"
        error = "total log excess return is not positive"

    rows = []
    for _, item in data.sort_values("year").iterrows():
        row = {column: "" for column in ONE_YEAR_CONTRIBUTION_COLUMNS}
        for key in (
            "family",
            "category",
            "experiment_name",
            "config_path",
            "variant",
            "walk_forward_variant",
            "transaction_cost_bps",
            "execution_model",
            "output_dir",
        ):
            row[key] = _metadata_value(metadata, key, "")
        row.update(
            {
                "year": int(item["year"]),
                "yearly_strategy_return": float(item["yearly_strategy_return"]),
                "yearly_tqqq_return": float(item["yearly_tqqq_return"]),
                "yearly_excess_return": float(item["yearly_excess_return"]),
                "yearly_log_excess_return": float(item["yearly_log_excess_return"]),
                "total_log_excess_return": total_log_excess,
                "contribution_to_total_log_excess": _safe_float(
                    item.get("contribution_to_total_log_excess")
                ),
                "largest_contribution_year": largest_year,
                "largest_single_year_contribution_share": largest_share,
                "one_year_contributes_more_than_60pct": dominated,
                "audit_status": status,
                "audit_error": error,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=ONE_YEAR_CONTRIBUTION_COLUMNS)


def read_yearly_contribution_from_output(
    output_dir: Path,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    threshold: float = 0.60,
) -> pd.DataFrame:
    yearly_path = Path(output_dir) / "yearly_returns.csv"
    if not yearly_path.exists():
        return pd.DataFrame(
            [
                _blank_row(
                    metadata=metadata,
                    status="missing_yearly_returns",
                    error=f"missing {yearly_path}",
                )
            ],
            columns=ONE_YEAR_CONTRIBUTION_COLUMNS,
        )
    return compute_one_year_contribution(
        pd.read_csv(yearly_path),
        metadata=metadata,
        threshold=threshold,
    )


def candidate_contribution_summary(contribution: pd.DataFrame) -> Dict[str, Any]:
    if contribution.empty:
        return {
            "performance_dominated_by_one_year": "",
            "dominant_year": "",
            "dominant_year_excess_share": np.nan,
        }
    valid = contribution[contribution["audit_status"].astype(str).isin(["ok", "non_positive_total_log_excess"])]
    row = valid.iloc[0] if not valid.empty else contribution.iloc[0]
    year = row.get("largest_contribution_year", np.nan)
    if pd.isna(year):
        year_value: Any = ""
    else:
        year_value = str(int(float(year)))
    return {
        "performance_dominated_by_one_year": bool(row.get("one_year_contributes_more_than_60pct", False)),
        "dominant_year": year_value,
        "dominant_year_excess_share": _safe_float(row.get("largest_single_year_contribution_share")),
    }


def build_one_year_contribution_from_variants(
    summary: pd.DataFrame,
    variants: pd.DataFrame,
    *,
    baseline_variant: str = "standard_5y_1y__10bps",
    threshold: float = 0.60,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=ONE_YEAR_CONTRIBUTION_COLUMNS)

    rows = []
    for _, candidate in summary.iterrows():
        family = str(candidate.get("family", ""))
        category = str(candidate.get("category", ""))
        experiment_name = str(candidate.get("experiment_name", ""))
        config_path = str(candidate.get("config_path", ""))
        variant_name = str(candidate.get("baseline_variant", baseline_variant) or baseline_variant)

        group = variants.copy() if not variants.empty else pd.DataFrame()
        if not group.empty:
            mask = (
                (group.get("family", pd.Series(dtype=str)).astype(str) == family)
                & (group.get("category", pd.Series(dtype=str)).astype(str) == category)
                & (group.get("variant", pd.Series(dtype=str)).astype(str) == variant_name)
                & (group.get("status", pd.Series(dtype=str)).astype(str) == "ok")
            )
            if "experiment_name" in group.columns and experiment_name:
                mask &= group["experiment_name"].astype(str) == experiment_name
            if "config_path" in group.columns and config_path:
                mask &= group["config_path"].astype(str) == config_path
            matches = group[mask]
        else:
            matches = pd.DataFrame()

        metadata = {
            "family": family,
            "category": category,
            "experiment_name": experiment_name,
            "config_path": config_path,
            "variant": variant_name,
        }
        if matches.empty:
            rows.append(
                _blank_row(
                    metadata=metadata,
                    status="missing_candidate_variant",
                    error=f"missing ok variant {variant_name}",
                )
            )
            continue

        variant = matches.iloc[0]
        output_dir_text = str(variant.get("output_dir", "")).strip()
        metadata.update(
            {
                "walk_forward_variant": variant.get("walk_forward_variant", ""),
                "transaction_cost_bps": variant.get("transaction_cost_bps", ""),
                "execution_model": variant.get("execution_model", ""),
                "output_dir": output_dir_text,
            }
        )
        if not output_dir_text or output_dir_text.lower() == "nan":
            rows.append(
                _blank_row(
                    metadata=metadata,
                    status="missing_candidate_output_dir",
                    error=f"missing output_dir for {variant_name}",
                )
            )
            continue
        output_dir = Path(output_dir_text)
        rows.extend(
            read_yearly_contribution_from_output(
                output_dir,
                metadata=metadata,
                threshold=threshold,
            ).to_dict("records")
        )

    return pd.DataFrame(rows, columns=ONE_YEAR_CONTRIBUTION_COLUMNS)


def markdown_one_year_contribution_table(contribution: pd.DataFrame, *, max_rows: int = 30) -> str:
    if contribution.empty:
        return "_No one-year contribution rows._"
    cols = [
        "family",
        "variant",
        "year",
        "yearly_strategy_return",
        "yearly_tqqq_return",
        "yearly_log_excess_return",
        "contribution_to_total_log_excess",
        "largest_single_year_contribution_share",
        "one_year_contributes_more_than_60pct",
        "audit_status",
    ]
    table = contribution[[column for column in cols if column in contribution.columns]].head(max_rows).fillna("")
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in table.columns) + " |")
    return "\n".join(lines)
