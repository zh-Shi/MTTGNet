"""Shared data-processing helpers for the MTTGNet data pipelines.

Centralizes two pieces of logic that were duplicated (and buggy) in every
``prepare_*.py`` script so all pipelines use one verified implementation:

1. ``delete_6hour`` — removes all Feb-29 (leap day) 6-hourly steps in a single
   date-mask pass.

   The old per-year offset arithmetic computed ``index + 365*4`` for each leap
   year, which is day **365** (Dec 31) — not Feb 29 (day 59, offset 236) — and
   it never accounted for array drift caused by earlier ``np.delete`` calls.
   It silently corrupted the temporal alignment of every dataset (each leap
   year shifted by 4 steps, 48 steps by 2024).

2. ``compute_t2m_anomaly`` — t2m anomaly against a **fixed 1981-2010**
   (train-period) daily/hourly climatology.

   Dataset A previously used a full-record (1980-2024) climatology, which is
   label leakage: the 2023-2024 test years were part of the anomaly baseline.
   The climatology is keyed by ``(month, day, hour)`` rather than
   ``(day_of_year, hour)`` so leap and non-leap years share the same daily
   mean (day-of-year is ambiguous for Mar 1 in a leap year).

All functions assume a chronological 6-hourly series starting at
1980-01-01 00:00 (the ERA5 / co2 / nino pipelines).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEAP_FREE = 65700           # 45 y * 365 d * 4 steps
LEAP_INCLUSIVE = 65748      # 45 y * (365 d + 12 leap days) * 4 steps
ANOMALY_BASELINE = ("1981-01-01", "2010-12-31")


def leap_free_mask(n_steps: int) -> np.ndarray:
    """Boolean mask selecting all-but-Feb-29 steps of a 6-hourly series.

    ``n_steps`` is the length of the *leap-inclusive* series.  The mask is
    built from an absolute 6-hourly calendar starting 1980-01-01, so it stays
    correct even if the caller previously truncated the series.
    """
    ts = pd.date_range("1980-01-01", periods=n_steps + 50, freq="6h")
    leap = (ts.month == 2) & (ts.day == 29)
    # ``leap`` is already a numpy boolean array — do NOT call ``.values``
    # on it (that's a pandas Series attribute; DatetimeIndex.month returns an
    # ndarray in modern pandas, so `.values` raises AttributeError).
    return ~leap[:n_steps]


def delete_6hour(data: np.ndarray) -> np.ndarray:
    """Remove Feb-29 steps from a chronological 1980+ 6-hourly series.

    Already-leap-free series of length ``LEAP_FREE`` are returned unchanged.

    NOTE (interface trap): a length of exactly 65700 is ambiguous — it could be
    a *genuinely leap-free* series, OR an old buggy dataset that was truncated
    from 65748 to 65700 *without* removing Feb 29 (it kept Feb 29 and dropped
    the last 48 steps of Dec 2024).  ``delete_6hour`` cannot tell the two apart
    by length alone, so the shortcut below must only be relied on for series
    produced by this pipeline.  If you feed an old 65700-length file here, it
    will be passed through UNCHANGED (Feb 29 intact).  Use
    ``verify_dataset_alignment`` on the assembled dataset to confirm.
    """
    data = np.asarray(data)
    if len(data) == LEAP_FREE:
        return data
    if len(data) != LEAP_INCLUSIVE:
        print(f"  WARNING delete_6hour: expected {LEAP_INCLUSIVE} "
              f"(leap-inclusive) or {LEAP_FREE} steps, got {len(data)} — a "
              f"source file may be missing; alignment cannot be guaranteed.")
    mask = leap_free_mask(len(data))
    return data[mask]


def leap_free_timestamps(n_steps: int = LEAP_FREE) -> pd.DatetimeIndex:
    """6-hourly timestamps for 1980-01-01 .. with Feb-29 steps removed."""
    ts = pd.date_range("1980-01-01", periods=n_steps + 50, freq="6h")
    return ts[~((ts.month == 2) & (ts.day == 29))][:n_steps]


def require_length(n: int, context: str = "") -> None:
    """Hard-guard against silently discarding data.

    After Feb-29 removal every variable in a prepare_* pipeline must be the
    natural leap-free length (``LEAP_FREE`` = 65700).  If the min length is
    SHORTER, that means a source file is missing (or the time range differs),
    and ``values[:N]`` would truncate ALL variables and throw away the tail.
    The previous code did exactly that (truncating a 65748 leap-inclusive
    array to 65700, dropping the last 48 steps of Dec 2024).  We now raise
    instead of producing a silently-truncated dataset.

    ``context`` names the dataset for the error message.
    """
    if n != LEAP_FREE:
        raise ValueError(
            f"{context}: expected {LEAP_FREE} leap-free steps after Feb-29 "
            f"removal, got {n}. A source file is likely missing (or the time "
            f"range differs). Refusing to write a dataset that silently "
            f"discards data. Fix the source, then re-run."
        )


def verify_dataset_alignment(values: np.ndarray, timestamps,
                             dataset_id: str = "") -> bool:
    """Sanity-check that a stored dataset is leap-free and length-consistent.

    Confirms:
      1. ``len(values) == len(timestamps)``
      2. timestamps contain NO Feb-29 steps (i.e. the leap-day removal ran)
      3. the grid covers the expected 1980-01-01 → 2024-12-31 18:00 span

    This does NOT prove the values themselves have no Feb-29 rows (length
    alone can't tell); it validates the timestamps/grid that the values are
    aligned to.  Returns True on success.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
    ok = True
    if len(values) != len(ts):
        print(f"  [verify:{dataset_id}] FAIL: {len(values)} values vs "
              f"{len(ts)} timestamps")
        ok = False
    n_feb29 = int(((ts.month == 2) & (ts.day == 29)).sum())
    if n_feb29:
        print(f"  [verify:{dataset_id}] FAIL: timestamps still contain "
              f"{n_feb29} Feb-29 steps — leap-day removal did NOT run")
        ok = False
    if len(ts):
        if ts[0] != pd.Timestamp("1980-01-01"):
            print(f"  [verify:{dataset_id}] WARN: grid starts at {ts[0]} "
                  f"(expected 1980-01-01)")
        if ts[-1] != pd.Timestamp("2024-12-31 18:00"):
            print(f"  [verify:{dataset_id}] WARN: grid ends at {ts[-1]} "
                  f"(expected 2024-12-31 18:00)")
    if ok:
        print(f"  [verify:{dataset_id}] OK: {len(values)} steps, leap-free, "
              f"{ts[0]} → {ts[-1]}")
    return ok


# NOTE: despite the "t2m" in the name, this function is generic — it subtracts
# a (month,day,hour) climatology from ANY 1-D series.  It is also used for
# TISR anomaly computation in prepare_datasets.py.
def compute_t2m_anomaly(
    raw: np.ndarray,
    ts: pd.DatetimeIndex,
    baseline_start: str = ANOMALY_BASELINE[0],
    baseline_end: str = ANOMALY_BASELINE[1],
) -> np.ndarray:
    """t2m anomaly vs a fixed (month, day, hour) climatology over the baseline.

    ``raw`` and ``ts`` must already be leap-aligned (same length).  The
    baseline window (default 1981-2010) lies entirely inside the training
    split, so no test-year information leaks into the target.
    """
    ref = (ts >= baseline_start) & (ts <= baseline_end)
    # DatetimeIndex.month/.day/.hour already return numpy arrays in modern
    # pandas — no `.values` (which would raise AttributeError).
    month = ts.month
    day = ts.day
    hour = ts.hour
    anomaly = np.zeros_like(raw)
    for h in (0, 6, 12, 18):
        for d in range(1, 32):
            for m in range(1, 13):
                slot = (month == m) & (day == d) & (hour == h)
                ref_slot = slot & ref
                if ref_slot.sum() > 0:
                    anomaly[slot] = raw[slot] - raw[ref_slot].mean()
    return anomaly
