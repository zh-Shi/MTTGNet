#!/usr/bin/env python3
"""Interpolate a supplied NOAA Mauna Loa daily CSV without fetching new data.

This retrospective builder records the raw file hash and interpolation rule.
It does not recover the historical CSV version used by the manuscript.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_daily_series(source, start='1980-01-01', end='2024-12-31'):
    rows = pd.read_csv(source, comment='#', header=None)
    # NOAA downloads have no column header; legacy exports may have one.
    numeric = rows.iloc[:, :5].apply(pd.to_numeric, errors='coerce').dropna()
    numeric.columns = ['year', 'month', 'day', 'decimal_year', 'co2']
    dates = pd.to_datetime(numeric[['year', 'month', 'day']])
    series = pd.Series(numeric.co2.to_numpy(), index=dates).sort_index()
    series = series[~series.index.duplicated(keep='first')]
    series = series[series > 0].loc[start:end]
    grid = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=18), freq='6h')
    if series.empty or series.index[0] > grid[0]:
        raise ValueError('Daily CSV does not cover the requested start; no backward extrapolation')
    result = series.reindex(grid).interpolate().to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Interpolation left missing CO2 values')
    return result, grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, default=ROOT/'data/co2/co2_daily_mlo.csv')
    p.add_argument('--out', type=Path, default=ROOT/'data/ERA5_global/co2_new_6hourly.npy')
    p.add_argument('--start', default='1980-01-01')
    p.add_argument('--end', default='2024-12-31')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.out.exists() and not a.overwrite:
        p.error('Output already exists; choose another --out or explicitly use --overwrite')
    values, grid = build_daily_series(a.csv, a.start, a.end)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, values)
    record = dict(source_file=a.csv.name, source_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest(),
                  product='NOAA GML Mauna Loa daily means (includes documented Maunakea substitution)',
                  start=str(grid[0]), end=str(grid[-1]), steps=len(values), leap_days='retained',
                  interpolation='linear over 6-hour grid; final daily value carried through end of day',
                  interpretation='retrospective interpolation; historical publication vintages not reconstructed')
    a.out.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Saved {len(values)} values to {a.out}')


if __name__ == '__main__':
    main()
