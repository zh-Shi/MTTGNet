from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnchorTimes:
    daily: list[pd.Timestamp]
    yearly: list[pd.Timestamp]


def safe_shift_year(
    timestamp: pd.Timestamp,
    years: int,
) -> pd.Timestamp:
    if years < 1:
        raise ValueError("years must be >= 1")

    target_year = timestamp.year - years

    try:
        return timestamp.replace(year=target_year)
    except ValueError:
        # Feb 29 → Feb 28 in non-leap years
        return timestamp.replace(
            year=target_year,
            month=2,
            day=28,
        )


def get_safe_daily_anchor_times(
    target_time: pd.Timestamp,
    forecast_origin: pd.Timestamp,
    num_anchors: int,
) -> list[pd.Timestamp]:
    if target_time <= forecast_origin:
        raise ValueError(
            "target_time must be after forecast_origin"
        )

    horizon_days = (
        target_time - forecast_origin
    ) / pd.Timedelta(days=1)

    # Step back at least horizon_days so anchors don't leak future info
    first_lag_days = max(
        1,
        math.ceil(float(horizon_days)),
    )

    anchors = [
        target_time - pd.Timedelta(days=k)
        for k in range(
            first_lag_days,
            first_lag_days + num_anchors,
        )
    ]

    if any(anchor > forecast_origin for anchor in anchors):
        raise RuntimeError(
            "Future leakage in daily anchors."
        )

    return anchors


def get_yearly_anchor_times(
    target_time: pd.Timestamp,
    num_anchors: int,
) -> list[pd.Timestamp]:
    return [
        safe_shift_year(target_time, years=k)
        for k in range(1, num_anchors + 1)
    ]


def build_anchor_times(
    target_time: pd.Timestamp,
    forecast_origin: pd.Timestamp,
    daily_anchors: int,
    yearly_anchors: int,
) -> AnchorTimes:
    return AnchorTimes(
        daily=get_safe_daily_anchor_times(
            target_time,
            forecast_origin,
            daily_anchors,
        ),
        yearly=get_yearly_anchor_times(
            target_time,
            yearly_anchors,
        ),
    )


def calendar_features(
    timestamp: pd.Timestamp,
    horizon_index: int,
    total_horizon: int,
) -> np.ndarray:
    hour_fraction = (
        timestamp.hour
        + timestamp.minute / 60.0
    ) / 24.0

    year_length = (
        366.0 if timestamp.is_leap_year else 365.0
    )

    day_fraction = (
        timestamp.dayofyear - 1 + hour_fraction
    ) / year_length

    horizon_fraction = (
        horizon_index / max(total_horizon, 1)
    )

    return np.asarray(
        [
            np.sin(2 * np.pi * hour_fraction),
            np.cos(2 * np.pi * hour_fraction),
            np.sin(2 * np.pi * day_fraction),
            np.cos(2 * np.pi * day_fraction),
            horizon_fraction,
        ],
        dtype=np.float32,
    )
