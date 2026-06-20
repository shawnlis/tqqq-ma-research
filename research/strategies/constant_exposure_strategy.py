from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ConstantExposureParams:
    exposure: float
    trade_asset: str = "TQQQ"


def build_constant_exposure_position(
    index: pd.Index,
    params: ConstantExposureParams,
) -> pd.Series:
    exposure = max(float(params.exposure), 0.0)
    return pd.Series(exposure, index=index, name="position")
