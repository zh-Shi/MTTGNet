import pandas as pd

from src.time_utils import (
    get_safe_daily_anchor_times,
    safe_shift_year,
)


def test_daily_anchor_not_future() -> None:
    origin = pd.Timestamp("2024-03-30 12:00:00")
    target = pd.Timestamp("2024-04-01 12:00:00")

    anchors = get_safe_daily_anchor_times(
        target_time=target,
        forecast_origin=origin,
        num_anchors=7,
    )

    assert all(
        anchor <= origin for anchor in anchors
    )


def test_leap_day() -> None:
    shifted = safe_shift_year(
        pd.Timestamp("2024-02-29 12:00:00"),
        years=1,
    )

    assert shifted == pd.Timestamp("2023-02-28 12:00:00")
