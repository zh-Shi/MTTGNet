#!/usr/bin/env python3
"""Reduce CMIP6 `tas` NetCDF to global-mean monthly series, paired with CO2.

Input : data/cmip6_tas/<model>/tas_Amon_<model>_<experiment>_*.nc   (download_cmip6_tas.py)
Output: data/processed/cmip6_tas/<model>/<experiment>_values.npy     [T,2] = [tas_K, co2_ppm]
        data/processed/cmip6_tas/<model>/<experiment>_timestamps.npy
        data/processed/cmip6_tas/<model>/<experiment>_meta.json

The global mean uses the same cosine-latitude area weighting as the ERA5
Dataset-D pipeline, so the CMIP6 target is the same physical quantity as the
Observed GMST the weather-scale model predicts.

CO2 pairing: the scenario concentration comes from this project's existing
pathways (`data/processed/co2_historical_monthly.npy` + `co2_ssp*_monthly.npy`,
which are CNRM-ESM2-1).  For the CNRM run that is exactly the model's own
forcing; for a different ESM the prescribed pathway is close but not identical —
`<experiment>_meta.json` records `co2_source` so this is never silently assumed.

Usage:
    python scripts/prepare_cmip6_tas.py                 # every model found
    python scripts/prepare_cmip6_tas.py --models CNRM-ESM2-1
    python scripts/prepare_cmip6_tas.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NC_ROOT = ROOT / "data/cmip6_tas"
OUT_ROOT = ROOT / "data/processed/cmip6_tas"
CO2_DIR = ROOT / "data/processed"

EXPERIMENTS = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]

# First month still counted as "historical" rather than scenario.  The 1850-1900
# window is the pre-industrial reference used throughout the paper.
PI_START = pd.Timestamp("1850-01-01")
PI_END = pd.Timestamp("1900-12-31")


def co2_series(experiment: str) -> tuple[np.ndarray, np.ndarray]:
    """Scenario CO2 [T] and its monthly timestamps, from the project's pathways."""
    if experiment == "historical":
        path = CO2_DIR / "co2_historical_monthly.npy"
    else:
        path = CO2_DIR / f"co2_{experiment}_monthly.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing CO2 pathway: {path}")
    co2 = np.load(path).astype(np.float64).reshape(-1)
    return co2


def _co2_timestamps(experiment: str) -> pd.DatetimeIndex:
    """Monthly timeline the CO2 pathway covers (1850-01..2014-12 / 2015-01..2100-12).

    Mid-month stamps, matching the convention used for the CMIP6 tas series and
    for `hadcrut5_timestamps.npy` (a bare ``freq="MS"`` range would land on the
    1st of each month and never intersect them).
    """
    start, end = ("1850-01-01", "2014-12-01") if experiment == "historical" \
        else ("2015-01-01", "2100-12-01")
    return pd.date_range(start, end, freq="MS") + pd.Timedelta(days=14)


def month_index(nc_path: Path, n_time: int) -> pd.DatetimeIndex:
    """Monthly timeline taken from the file NAME, not from the time coordinate.

    Two reasons not to decode the coordinate:

    * several models (CanESM5, GFDL-ESM4, ...) use a 365_day calendar, which
      xarray cannot decode without `cftime`;
    * the ``units`` attribute points at the calendar origin (e.g. 1850-01-01),
      NOT at the first sample — an MPI chunk named ``187001-188912`` still says
      "days since 1850-01-01", so the units alone would misdate it by 20 years.

    CMIP6 monthly files are named ``<...>_<YYYYMM>-<YYYYMM>.nc``, which is
    authoritative; and because monthly data has exactly 12 samples per year, the
    whole sequence follows from the start month and the length.
    """
    m = re.search(r"_(\d{6})-(\d{6})\.nc$", nc_path.name)
    if not m:
        raise ValueError(f"{nc_path.name}: no YYYYMM-YYYYMM range in the name")
    y, mo = int(m.group(1)[:4]), int(m.group(1)[4:6])
    # Build on month STARTS and shift afterwards.  Starting the range at the
    # 15th instead would keep 1850-01-15 as the first point but snap every later
    # point to a month start, so after the +14d shift the first stamp lands on
    # 1850-01-29 and drops out of the intersection with the CO2 pathway.
    idx = pd.date_range(f"{y}-{mo:02d}-01", periods=n_time, freq="MS")
    idx = idx + pd.Timedelta(days=14)
    expect = (int(m.group(2)[:4]) - y) * 12 + (int(m.group(2)[4:6]) - mo) + 1
    if expect != n_time:
        raise ValueError(f"{nc_path.name}: name implies {expect} months, "
                         f"file has {n_time}")
    return idx


def global_mean(nc_path: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Area-weighted global-mean tas [T] (K) and its monthly timestamps."""
    import xarray as xr

    # decode_times=False: the time coordinate may use a 365_day calendar, which
    # would need cftime.  month_index() derives the dates from the file name.
    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
    except (ImportError, ValueError):
        ds = xr.open_dataset(nc_path, engine="h5netcdf", decode_times=False)

    var = "tas" if "tas" in ds else next(
        v for v in ds.data_vars if v not in ("time_bounds", "lat_bnds", "lon_bnds"))
    da = ds[var]

    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"
    # cos(lat) weights, broadcastable against (lat, lon)
    w_lat = np.cos(np.deg2rad(da[lat_name].values)).reshape(-1, 1)
    w_lat = np.broadcast_to(w_lat, (da.sizes[lat_name], da.sizes[lon_name]))
    weights = np.broadcast_to(w_lat, da.shape[1:])

    with np.errstate(invalid="ignore"):
        ts_mean = da.weighted(xr.DataArray(weights, dims=da.dims[1:])).mean(
            dim=[lat_name, lon_name]).values.astype(np.float64)

    times = month_index(nc_path, da.sizes["time"])
    ds.close()
    return ts_mean, times


def process_experiment(nc_paths: list[Path], experiment: str, *, overwrite: bool) -> dict:
    """Global-mean series for one experiment, concatenating its chunks.

    Several models (MPI-ESM1-2-LR, MIROC6, GFDL-ESM4, ...) publish `historical`
    and each SSP as a series of calendar chunks rather than one file, so the
    pieces have to be stitched in time order.  Neighbouring chunks sometimes
    repeat the boundary month, so the join de-duplicates.
    """
    out_dir = OUT_ROOT / nc_paths[0].parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    vals_path = out_dir / f"{experiment}_values.npy"
    meta_path = out_dir / f"{experiment}_meta.json"
    # Require BOTH outputs: an interrupted run can leave the .npy behind without
    # the meta, and skipping on the .npy alone would then fail on the read.
    if vals_path.exists() and meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text())

    tas_parts, time_parts = [], []
    for p in sorted(nc_paths):
        t, ts = global_mean(p)
        if len(t) != len(ts):
            raise ValueError(f"{p.name}: {len(t)} values vs {len(ts)} timestamps")
        tas_parts.append(t)
        time_parts.append(ts)

    joined = pd.Series(np.concatenate(tas_parts),
                       index=time_parts[0].append(time_parts[1:]))
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()
    tas = joined.values.astype(np.float64)
    times = pd.DatetimeIndex(joined.index)

    co2 = co2_series(experiment)
    # The NetCDF and the CO2 pathway must cover the same months; trim to the
    # overlap rather than padding, and record the overlap in the meta.
    co2_times = _co2_timestamps(experiment)
    common = times.intersection(co2_times)
    if len(common) == 0:
        raise ValueError(f"{experiment}: no overlap between tas and CO2 timelines")
    tas_i = times.get_indexer(common)
    co2_i = co2_times.get_indexer(common)

    values = np.stack([tas[tas_i], co2[co2_i]], axis=1).astype(np.float32)
    np.save(vals_path, values)
    np.save(out_dir / f"{experiment}_timestamps.npy", np.asarray(common))

    pi = (common >= PI_START) & (common <= PI_END)
    meta = {
        "model": nc_paths[0].parent.name,
        "experiment": experiment,
        "n_months": int(len(values)),
        "n_chunks": len(nc_paths),
        "start": str(common[0].date()),
        "end": str(common[-1].date()),
        "pi_mean_K": float(np.nanmean(values[pi, 0])) if pi.any() else None,
        "tas_mean_K": float(np.nanmean(values[:, 0])),
        "co2_source": "CNRM-ESM2-1 pathways (data/processed/co2_*_monthly.npy)",
        "nc_files": [p.name for p in sorted(nc_paths)],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not NC_ROOT.exists():
        raise SystemExit(f"no NetCDF found at {NC_ROOT} — run download_cmip6_tas.py first")

    models = args.models or sorted(p.name for p in NC_ROOT.iterdir() if p.is_dir())
    if not models:
        raise SystemExit(f"no model sub-directories under {NC_ROOT}")

    done = []
    for model in models:
        model_dir = NC_ROOT / model
        ncs = sorted(model_dir.glob("tas_Amon_*.nc"))
        if not ncs:
            print(f"!! {model}: no tas_Amon_*.nc")
            continue

        # Group chunks by experiment.  Match on the delimited token rather than
        # a regex so a model name containing digits/dashes cannot confuse it.
        groups: dict[str, list[Path]] = {e: [] for e in EXPERIMENTS}
        for nc in ncs:
            hit = [e for e in EXPERIMENTS if f"_{e}_" in nc.name]
            if len(hit) != 1:
                print(f"  skip {nc.name}: cannot identify the experiment")
                continue
            groups[hit[0]].append(nc)

        print(f"\n=== {model} ({len(ncs)} files, "
              f"{sum(1 for v in groups.values() if v)}/{len(EXPERIMENTS)} experiments) ===")
        for experiment in EXPERIMENTS:
            paths = groups[experiment]
            if not paths:
                print(f"  !! {experiment}: no files")
                continue
            meta = process_experiment(paths, experiment, overwrite=args.overwrite)
            meta["n_chunks"] = len(paths)
            print(f"  {experiment:11s} {meta['n_months']:5d} months "
                  f"({len(paths)} chunk(s))  {meta['start']} → {meta['end']}  "
                  f"mean {meta['tas_mean_K']:.2f} K")
            done.append(meta)

            # Refresh the continuous historical+scenario timeline for this model.
            hist = OUT_ROOT / model / "historical_values.npy"
            if experiment != "historical" and hist.exists():
                hv = np.load(hist)
                ht = np.load(OUT_ROOT / model / "historical_timestamps.npy", allow_pickle=True)
                sv = np.load(OUT_ROOT / model / f"{experiment}_values.npy")
                st = np.load(OUT_ROOT / model / f"{experiment}_timestamps.npy", allow_pickle=True)
                np.save(OUT_ROOT / model / f"continuous_{experiment}_values.npy",
                        np.concatenate([hv, sv], axis=0))
                np.save(OUT_ROOT / model / f"continuous_{experiment}_timestamps.npy",
                        np.concatenate([ht, st], axis=0))

    (OUT_ROOT).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "index.json").write_text(json.dumps(done, indent=2))
    print(f"\n{len(done)} series -> {OUT_ROOT}")

if __name__ == "__main__":
    main()
