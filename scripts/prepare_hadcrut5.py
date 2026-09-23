#!/usr/bin/env python3
"""Prepare monthly HadCRUT5 + CMIP6 historical CO2 dataset (1850–2014).

Data sources:
  - HadCRUT5 global monthly anomaly  (1850-01 → 2026-05)
  - CMIP6 historical monthly CO2     (1850-01 → 2014-12)

The dataset is truncated to 2014-12 (end of CMIP6 historical CO2).
SSP CO2 pathways (2015–2100) are used separately for projection experiments.

Output:
  - data/processed/hadcrut5_values.npy     [T, 2]  anomaly / CO2
  - data/processed/hadcrut5_timestamps.npy [T]

Columns: [0] anomaly (degC)   [1] CO2 (ppm)

Usage:
    python scripts/prepare_hadcrut5.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hadcrut5(path: str) -> pd.DataFrame:
    """Read HadCRUT5 CSV → mid-month index, column 'anomaly'."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["Time"].str.strip())
    df = df.rename(columns={"Anomaly (deg C)": "anomaly"})
    df = df[["timestamp", "anomaly"]].set_index("timestamp").sort_index()
    df.index = df.index + pd.offsets.Day(14)
    return df


def load_co2_cmip6(rel_path: str) -> pd.Series:
    """Load CMIP6 historical monthly CO2 (pre-processed npy)."""
    vals = np.load(str(PROJECT_ROOT / rel_path))
    idx = (pd.date_range("1850-01-01", periods=len(vals), freq="MS")
           + pd.offsets.Day(14))
    s = pd.Series(vals, index=idx, name="co2")
    print(f"  CMIP6 historical CO2 : {len(s)} months  "
          f"({s.index[0].date()} → {s.index[-1].date()})")
    return s


def main() -> None:
    print("=" * 56)
    print("  HadCRUT5 + CMIP6 historical CO2  (1850–2014)")
    print("=" * 56)

    # ---- load ----------------------------------------------------------
    print("\n[1] HadCRUT5")
    had = load_hadcrut5(
        str(PROJECT_ROOT
            / "data/HadCRUT5/"
            / "HadCRUT.5.1.0.0.analysis.summary_series.global.monthly.csv"))
    print(f"    {len(had)} months  ({had.index[0].date()} → {had.index[-1].date()})")

    print("\n[2] CMIP6 historical CO2")
    co2 = load_co2_cmip6("data/processed/co2_historical_monthly.npy")

    # ---- align on common period ----------------------------------------
    start = max(had.index[0], co2.index[0])
    end = min(had.index[-1], co2.index[-1])
    print(f"\n[3] Common period: {start.date()} → {end.date()}")

    common = pd.date_range(start, end, freq="MS") + pd.offsets.Day(14)
    aligned = pd.DataFrame(index=common)
    aligned["anomaly"] = had["anomaly"].reindex(common, method="nearest")
    aligned["co2"] = co2.reindex(common, method="nearest")
    aligned = aligned.dropna()

    n_dropped = len(had[had.index > end])
    print(f"    {len(aligned)} months aligned")
    print(f"    {n_dropped} HadCRUT5 months beyond 2014-12 dropped "
          f"(available for SSP validation)")

    print(f"    Anomaly  [{aligned['anomaly'].min():+.4f}, "
          f"{aligned['anomaly'].max():+.4f}]  "
          f"mean = {aligned['anomaly'].mean():+.4f}")
    print(f"    CO2      [{aligned['co2'].min():.1f}, "
          f"{aligned['co2'].max():.1f}]  "
          f"mean = {aligned['co2'].mean():.1f}")

    # ---- save ----------------------------------------------------------
    values = aligned[["anomaly", "co2"]].values.astype(np.float32)
    ts_str = aligned.index.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

    vp = OUT_DIR / "hadcrut5_values.npy"
    tp = OUT_DIR / "hadcrut5_timestamps.npy"
    np.save(vp, values)
    np.save(tp, ts_str)
    print(f"\n[4] Saved")
    print(f"    {vp}   {tuple(values.shape)}")
    print(f"    {tp}")

    # ---- verify & count origins ----------------------------------------
    v = np.load(vp)
    ts = pd.DatetimeIndex(pd.to_datetime(np.load(tp, allow_pickle=True)))
    assert ts.is_monotonic_increasing and not ts.has_duplicates
    assert len(v) == len(ts)

    L, H = 24, 12

    def count(idx, t0, t1):
        s, e = pd.Timestamp(t0), pd.Timestamp(t1)
        cand = np.flatnonzero((idx >= s) & (idx <= e))
        n = 0
        for i in cand:
            if i - L + 1 < 0 or i + H >= len(idx) or idx[i + H - 1] > e:
                continue
            n += 1
        return n

    print(f"\n[5] Origin counts  (L={L}, H={H})")
    for label, t0, t1 in [
        ("train  1850→1990", str(ts[0]), "1990-12-15"),
        ("val    1991→2004", "1991-01-15", "2004-12-15"),
        ("test   2005→2014", "2005-01-15", "2014-12-15"),
    ]:
        print(f"    {label}:  {count(ts, t0, t1):5d}")

    # Also report how many 2015-2026 months are available in HadCRUT5
    # (for SSP near-term validation)
    extra = had[had.index > end]
    if len(extra) > 0:
        print(f"\n    HadCRUT5 2015+ : {len(extra)} months available "
              f"({extra.index[0].date()} → {extra.index[-1].date()})")
        print(f"    → use for SSP near-term validation")

    print("\nDone.")


if __name__ == "__main__":
    main()
