#!/usr/bin/env python3
"""Process CMIP6 CO2 scenario data for MTTGNet climate conditioning.

Reads CNRM-ESM2-1 historical + SSP scenarios, interpolates from
monthly to 6-hourly, and saves per-scenario arrays.

Output:
  data/processed/co2_historical.npy  — 1850-2014, 6-hourly
  data/processed/co2_ssp126.npy     — 2015-2100, 6-hourly (SSP1-2.6)
  data/processed/co2_ssp245.npy     — 2015-2100, 6-hourly (SSP2-4.5)
  data/processed/co2_ssp370.npy     — 2015-2100, 6-hourly (SSP3-7.0)
  data/processed/co2_ssp585.npy     — 2015-2100, 6-hourly (SSP5-8.5)
  data/processed/co2_all.npy        — full timeseries per scenario [scenario, T]
"""

import numpy as np
from pathlib import Path
import argparse
import json

# ── Config ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/co2/CMIP6"
OUT_DIR = PROJECT_ROOT / "data/processed"

SCENARIOS = {
    "historical": "co2_Amon_CNRM-ESM2-1_historical_r3i1p1f2_gr_185001-201412.nc",
    "ssp126":     "co2_Amon_CNRM-ESM2-1_ssp126_r3i1p1f2_gr_201501-210012.nc",
    "ssp245":     "co2_Amon_CNRM-ESM2-1_ssp245_r3i1p1f2_gr_201501-210012.nc",
    "ssp370":     "co2_Amon_CNRM-ESM2-1_ssp370_r3i1p1f2_gr_201501-210012.nc",
    "ssp585":     "co2_Amon_CNRM-ESM2-1_ssp585_r3i1p1f2_gr_201501-210012.nc",
}

STEPS_PER_DAY = 4  # 6-hourly


def read_co2_monthly(path: Path, pressure_pa: float | None = None) -> np.ndarray:
    """Read monthly CO2 from CMIP6 netCDF4.

    Data is 4D [time, plev=19, lat=128, lon=256], unit = mol/mol.
    Select the highest-pressure level by default, then average valid cells.
    Explicit pressure_pa=100 reproduces the earlier 100-Pa layer choice.
    """
    import netCDF4 as nc

    with nc.Dataset(str(path)) as ds:
        levels = np.asarray(ds['plev'][:], dtype=float)
        unit = str(ds['plev'].units).lower()
        if unit in ('hpa', 'mbar'):
            levels *= 100.
        elif unit != 'pa':
            raise ValueError(f'Unsupported pressure unit: {unit}')
        idx = int(np.argmax(levels)) if pressure_pa is None else int(np.argmin(np.abs(levels-pressure_pa)))
        if pressure_pa is not None and not np.isclose(levels[idx],pressure_pa):
            raise ValueError(f'Pressure {pressure_pa} Pa is not available in {path.name}')
        print(f'      selected pressure: {levels[idx]:g} Pa (index {idx})')
        field = np.asarray(np.ma.filled(ds['co2'][:,idx,:,:],np.nan),dtype=float)
        weights = np.cos(np.deg2rad(np.asarray(ds['lat'][:],dtype=float)))[None,:,None]
        valid = np.isfinite(field)
        # Near-surface pressure levels have below-ground missing cells.
        denominator = np.sum(valid*weights,axis=(1,2))
        if np.any(denominator <= 0):
            raise ValueError('Selected pressure level has a month with no valid cells')
        mean = np.sum(np.where(valid,field,0.)*weights,axis=(1,2))/denominator
    return mean.astype(np.float32)*1e6


def monthly_to_6hourly(monthly: np.ndarray, start_year: int) -> np.ndarray:
    """Linear interpolation from monthly to 6-hourly.

    Accounts for actual days per month including leap years.
    """
    import calendar
    n_months = len(monthly)
    days_per_month = np.array([
        calendar.monthrange(start_year + i // 12, (i % 12) + 1)[1]
        for i in range(n_months)
    ])
    total_days = int(days_per_month.sum())
    n_steps = total_days * 4  # 6-hourly

    # Preserve the legacy month-start interpolation convention.
    monthly_x = np.arange(n_months, dtype=np.float64)
    # Cumulate days → fractional month index for each 6h step
    cum_days = np.concatenate([[0], np.cumsum(days_per_month)[:-1]])
    sixhourly_days = np.linspace(0, total_days - 0.25, n_steps)
    # Map each 6h day to fractional month index
    sixhourly_x = np.interp(sixhourly_days, cum_days, monthly_x)

    return np.interp(sixhourly_x, monthly_x, monthly).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--out-dir',type=Path,default=OUT_DIR)
    parser.add_argument('--pressure-pa',type=float,default=None,
                        help='Default: highest-pressure layer. Use 100 for the archived manuscript inputs.')
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    out_dir = args.out_dir
    existing = [p for name in SCENARIOS for p in (out_dir/f'co2_{name}.npy',out_dir/f'co2_{name}_monthly.npy') if p.exists()]
    if existing and not args.overwrite:
        parser.error('Existing CO2 arrays are protected; choose a new --out-dir or explicitly use --overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_co2 = {}  # scenario_name → [T]

    # Start years for each scenario (used to build the 6-hourly timeline)
    start_year = {"historical": 1850, "ssp126": 2015, "ssp245": 2015,
                  "ssp370": 2015, "ssp585": 2015}

    for scenario, filename in SCENARIOS.items():
        fpath = args.data_dir / filename
        if not fpath.exists():
            print(f"  SKIP {scenario}: file not found ({fpath})")
            continue

        print(f"  Reading {scenario} ...")
        monthly = read_co2_monthly(fpath,args.pressure_pa)
        v0 = float(monthly.flat[0]) if hasattr(monthly, 'flat') else float(monthly[0])
        v1 = float(monthly.flat[-1]) if hasattr(monthly, 'flat') else float(monthly[-1])
        print(f"    {len(monthly)} months, range [{v0:.1f}, {v1:.1f}]")

        np.save(out_dir / f"co2_{scenario}_monthly.npy", monthly)
        (out_dir/f'co2_{scenario}_processing.json').write_text(json.dumps(dict(
            source_file=filename,pressure_pa=args.pressure_pa,
            default_pressure_rule='highest available pressure',
            averaging='cosine-latitude weighted mean over valid cells',
            manuscript_compatibility='archived arrays used the last pressure layer, 100 Pa'),indent=2),encoding='utf-8')

        # Also write the 6-hourly arrays promised in the module docstring.
        # (Previously only *_monthly.npy was saved; the 6-hourly files were
        # listed as outputs but never produced by this script.)
        sixh = monthly_to_6hourly(monthly, start_year.get(scenario, 2015))
        np.save(out_dir / f"co2_{scenario}.npy", sixh)
        print(f"      6-hourly: {len(sixh)} steps")

        all_co2[scenario] = monthly

    # ── Summary (monthly is more meaningful than 6-hourly here) ──
    print(f"\nSaved to {out_dir}/:")
    for scenario in SCENARIOS:
        mpath = out_dir / f"co2_{scenario}_monthly.npy"
        if mpath.exists():
            arr = np.load(mpath)
            print(f"  co2_{scenario}_monthly.npy  [{len(arr)} months]  "
                  f"{float(arr[0]):.1f} → {float(arr[-1]):.1f} ppm")

    # ── Scenario comparison ──
    print(f"\n{'Scenario':<12s} {'Start':>8s} {'2030':>8s} {'2050':>8s} {'2100':>8s} {'Δ':>8s}")
    print("-" * 52)
    for scenario in SCENARIOS:
        mpath = out_dir / f"co2_{scenario}_monthly.npy"
        if not mpath.exists():
            continue
        arr = np.load(mpath)
        start = float(arr[0])
        end = float(arr[-1])
        if scenario == "historical":
            print(f"  {scenario:<10s} {start:>8.1f} {'—':>8s} {'—':>8s} {'—':>8s} {end-start:>+8.1f}")
        else:
            y30 = float(arr[min(15 * 12, len(arr)-1)])
            y50 = float(arr[min(35 * 12, len(arr)-1)])
            y100 = float(arr[-1])
            print(f"  {scenario:<10s} {start:>8.1f} {y30:>8.1f} {y50:>8.1f} {y100:>8.1f} {end-start:>+8.1f}")


if __name__ == "__main__":
    main()
