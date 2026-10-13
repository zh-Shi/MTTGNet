#!/usr/bin/env python3
"""Convert Niño3.4 CSV to 6-hourly (piecewise constant), calendar-aligned.

Input:  data/nino34.anom.csv  — monthly Niño3.4 anomaly
Output: data/processed/nino34_6hourly.npy (65700 steps, leap-free grid)

The old implementation did ``np.repeat(monthly, 122)[:65700]`` — 122 steps
per month does not match real month lengths (28-31 days = 112-124 steps) nor
leap days, so the piecewise-constant series drifted off the ERA5 6-hourly
calendar (verified: 66% of steps held the wrong month's value).  We now map
each leap-free 6-hourly timestamp to its containing calendar month and take
that month's Niño3.4 value, which aligns perfectly with the A/B/C/D datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import leap_free_timestamps  # noqa: E402

# The on-disk file has historically been named "nina34.anom.csv" (sic);
# accept both spellings.
CSV_PATH = None
for _cand in ("data/nino34.anom.csv", "data/nina34.anom.csv",
              "data/processed/nino34.anom.csv", "data/processed/nina34.anom.csv"):
    _p = PROJECT_ROOT / _cand
    if _p.exists():
        CSV_PATH = _p
        break
OUT_DIR = PROJECT_ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if CSV_PATH is None:
        print("ERROR: Niño3.4 CSV not found.")
        print("Put your Niño3.4 CSV file at data/nino34.anom.csv")
        return

    df = pd.read_csv(CSV_PATH, skiprows=1)  # skip header line with units

    # Parse date column (format: YYYY/M/D)
    dates = pd.to_datetime(df.iloc[:, 0])
    nino34_raw = df.iloc[:, 1].values.astype(np.float32)

    # Filter missing values (-99.99 and -9999)
    valid = nino34_raw > -99
    dates = dates[valid]
    nino34_raw = nino34_raw[valid]

    # Filter 1980-2024
    mask = (dates.dt.year >= 1980) & (dates.dt.year <= 2024)
    monthly_dates = pd.DatetimeIndex(dates[mask])
    monthly = nino34_raw[mask]
    print(f"ERA5 period: {len(monthly)} months "
          f"({monthly.min():.2f} to {monthly.max():.2f})")

    # Build the leap-free 6-hourly grid (same as the datasets' timestamps)
    ts = leap_free_timestamps()
    # Map each 6h step to the index of its containing calendar month
    base_year, base_month = monthly_dates[0].year, monthly_dates[0].month
    month_pos = np.array(
        [(t.year - base_year) * 12 + (t.month - base_month) for t in ts],
        dtype=np.int64,
    )
    # Guard against any month out of range (data gaps at the boundaries)
    month_pos = np.clip(month_pos, 0, len(monthly) - 1)
    sixh = monthly[month_pos].astype(np.float32)

    print(f"6-hourly: {len(sixh)} steps "
          f"({ts[0]} → {ts[-1]})")
    np.save(OUT_DIR / "nino34_6hourly.npy", sixh)
    print(f"Saved to {OUT_DIR}/nino34_6hourly.npy")


if __name__ == "__main__":
    main()
