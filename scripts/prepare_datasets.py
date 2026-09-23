#!/usr/bin/env python3
"""Unified MTTGNet dataset preparation — A/B/C/D + wind/mslp/tcc/tp features.

One netCDF pass over /data_public/ERA5 computes, for BOTH aggregations
(area-weighted = C/D, arithmetic = A/B), a full **feature bank** saved to
``data/processed/features/``:

  * global + 3-latitude-band means × {u10, v10, speed, mslp, tisr, tcc, tp}
    (tp ×1000 → mm; no log transform)
  * global {KE, wind-stress} (weekly/monthly SST experiments)
  * mslp mid-latitude minus polar **gradient** (baroclinicity proxy)
  * t2m / tisr **anomalies** (fixed 1981-2010 daily climatology)
  * co2, nino34 (global scalars, shared across weightings)

Datasets assembled by column list:
  A = arithmetic bank + anomaly target;  B = arithmetic bank + raw target
  C = area-weighted bank + anomaly target;  D = area-weighted bank + raw target

Phase 1 variants (G0=default / G1 / B1 / B2 / B3) are built for ALL of A/B/C/D.
Column order: [target, co2, <wind block>, mslp, tisr, nino34, tcc, <extras…>]
=> co2_index is ALWAYS 1 (CO₂ fixed right after the target), target_index = 0.

Writes ``data/processed/weather_variants.json`` with each variant's co2_index /
num_features / column order for the training pipeline.

Usage (PyCharm: edit DATASETS / WIND_VARIANTS / EXTRA_COLUMNS, Ctrl+Shift+F10):
    python scripts/prepare_datasets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import (  # noqa: E402
    LEAP_FREE,
    compute_t2m_anomaly,
    delete_6hour,
    leap_free_timestamps,
    require_length,
    verify_dataset_alignment,
)

# ═══════════════════════════════════════════════════════════════════
# 配置区 — PyCharm 直接改这里，然后 Ctrl+Shift+F10 运行
# ═══════════════════════════════════════════════════════════════════
DATASETS = "A,B,C,D"        # 要生成的数据集（A/B 算术，C/D 面积加权）
WIND_VARIANTS = ["G1", "B1", "B2", "B3"]   # 风特征变体（G0 = 默认数据集本身）
                                            # 空列表 = 只建默认数据集
EXTRA_COLUMNS = ["tp", "tp_tropics", "tp_midlat", "tp_polar",
                 "mslp_gradient", "tcc_tropics", "tcc_midlat", "tcc_polar"]
                                            # 附加列（在 base 之后，co2_index 不变）
                                            # 空列表 = 不加
                                            # tp* 已 ×1000 转成 mm（见 loader）
DRY_RUN = False             # True = 只预览，不写文件
# ═══════════════════════════════════════════════════════════════════

# dataset_id -> (target kind, source kind)
DATASET_SPEC = {
    "A": ("anomaly", "arithmetic"),
    "B": ("raw", "arithmetic"),
    "C": ("anomaly", "area_weighted"),
    "D": ("raw", "area_weighted"),
}
ERA5_NETCDF = Path("/data_public/ERA5")
OUT_DIR = PROJECT_ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR = OUT_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# ── Latitude bands (3 纬度带) ──
LAT_BANDS = [
    ("tropics", [(-30.0, 30.0)]),
    ("midlat", [(-60.0, -30.0), (30.0, 60.0)]),
    ("polar", [(-90.0, -60.0), (60.0, 90.0)]),
]
BAND_NAMES = [b[0] for b in LAT_BANDS]

# ── Wind variant column blocks (abstract keys; weighting suffix applied per
#    dataset: "" for C/D area-weighted, "_arith" for A/B arithmetic). ──
_B = BAND_NAMES
WIND_VARIANTS_DEF = {
    "G0": ["u10", "v10"],
    "G1": ["u10", "v10", "speed"],
    "B1": [f"u10_{b}" for b in _B] + [f"v10_{b}" for b in _B],
    "B2": [f"u10_{b}" for b in _B] + [f"v10_{b}" for b in _B] + ["speed"],
    "B3": ([f"u10_{b}" for b in _B] + [f"v10_{b}" for b in _B]
           + [f"speed_{b}" for b in _B]),
}
# 风块之后的尾列。CO₂ 固定在 index=1（紧跟目标列），所以 co2_index 恒为 1。
BASE_TRAIL = ["mslp", "tisr", "nino34", "tcc"]

# 面积加权路径的变量文件夹（tp 可能缺失 → 降级为 0 并警告）
GRID_FOLDERS = {
    "t2m": "ERA5_2m_temperature",
    "u10": "ERA5_10m_u_component_of_wind",
    "v10": "ERA5_10m_v_component_of_wind",
    "mslp": "ERA5_mean_sea_level_pressure",
    "tisr": "ERA5_toa_incident_solar_radiation",
    "tcc": "ERA5_total_cloud_cover",
    "tp": "ERA5_total_precipitation",
}


# ═══════════════════════════════════════════════════════════════════
# Feature-key helpers
# ═══════════════════════════════════════════════════════════════════

def _wkey(key: str, weight: str) -> str:
    """Apply weight suffix ("" for area, "_arith" for arithmetic)."""
    if key in ("co2", "nino34"):
        return key                      # global scalars: shared
    return key + weight


def all_feature_keys() -> list[str]:
    """All feature-bank keys (both weightings)."""
    keys = []
    for ws in ("", "_arith"):
        for k in ("t2m", "mslp", "tisr", "tcc", "tp"):
            keys += [f"{k}{ws}"] + [f"{k}_{b}{ws}" for b in BAND_NAMES]
        for k in ("u10", "v10", "speed"):
            keys += [f"{k}{ws}"] + [f"{k}_{b}{ws}" for b in BAND_NAMES]
        keys += [f"ke{ws}", f"stress_u{ws}", f"stress_v{ws}", f"mslp_gradient{ws}"]
    keys += ["t2m_anomaly", "t2m_anomaly_arith", "tisr_anomaly", "tisr_anomaly_arith"]
    keys += ["co2", "nino34"]
    return keys


# ═══════════════════════════════════════════════════════════════════
# Grid helpers
# ═══════════════════════════════════════════════════════════════════

def _extract_var(ds):
    dv = [v for v in ds.data_vars
          if v not in ("lat", "lon", "latitude", "longitude", "time", "valid_time")]
    return ds[dv[0]].values


def _extract_lats(ds) -> np.ndarray:
    ln = "latitude" if "latitude" in ds.coords else "lat"
    return ds[ln].values


def _weighted_mean(arr: np.ndarray, lats: np.ndarray,
                   mask: np.ndarray | None = None,
                   weight: str = "area") -> np.ndarray:
    """Per-timestep mean over (masked) lats.

    weight: "area" → cos(lat) weighting (C/D); "arith" → equal weights (A/B).
    arr: [T, Nlat, Nlon] (or [Nlat, Nlon] → T=1).  Returns [T] (or scalar).
    """
    if weight == "area":
        w = np.abs(np.cos(np.deg2rad(lats)))[:, None]
    else:
        w = np.ones((len(lats), 1))
    if mask is not None:
        w = w[mask]
        arr = arr[..., mask, :]
    return (arr * w).sum(axis=(-2, -1)) / (w.sum() * arr.shape[-1])


def _band_masks(lats: np.ndarray) -> list[tuple[str, np.ndarray]]:
    masks = []
    for name, ranges in LAT_BANDS:
        m = np.zeros(len(lats), dtype=bool)
        for lo, hi in ranges:
            m |= (lats >= lo) & (lats <= hi)
        masks.append((name, m))
    return masks


def _band_mask(lats: np.ndarray, name: str) -> np.ndarray:
    return dict(_band_masks(lats))[name]


def _load_nino34(n: int) -> np.ndarray:
    p = OUT_DIR / "nino34_6hourly.npy"
    if p.exists():
        arr = np.load(p)
        return arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)), "edge")
    print("  WARNING: Niño3.4 missing, using zeros")
    return np.zeros(n, dtype=np.float32)


def _load_co2(n: int) -> np.ndarray:
    for cand in (PROJECT_ROOT / "data/ERA5_global" / "co2_new_6hourly.npy",
                 ERA5_NETCDF / "co2_new_6hourly.npy"):
        if cand.exists():
            co2 = delete_6hour(np.load(cand))
            require_length(len(co2), "CO2")
            return co2[:n]
    raise FileNotFoundError(f"CO2 source missing: {PROJECT_ROOT / 'data/ERA5_global/co2_new_6hourly.npy'}")


# ═══════════════════════════════════════════════════════════════════
# One netCDF pass → full feature bank (both weightings)
# ═══════════════════════════════════════════════════════════════════

def load_grid_features() -> tuple:
    """Read /data_public/ERA5 once, compute ALL features, save the bank.

    Returns (feature dict, leap-free timestamps).
    """
    if not ERA5_NETCDF.exists():
        raise SystemExit(
            f"/data_public/ERA5 not found. C/D and all band/derived features "
            f"require the raw netCDF (server only)."
        )
    import xarray as xr

    print("\n  [grid] loading /data_public/ERA5/ (one pass, both weightings) ...")
    # 只累积"网格派生"的键；co2/nino34/异常键单独加载/计算（见下），
    # 否则 np.concatenate(空列表) 会抛 ValueError。
    _acc_exclude = {"co2", "nino34", "t2m_anomaly", "t2m_anomaly_arith",
                    "tisr_anomaly", "tisr_anomaly_arith"}
    acc = {f: [] for f in all_feature_keys() if f not in _acc_exclude}
    lats = None
    tp_seen = False
    skipped_months = []

    for year in range(1980, 2025):
        for month in range(1, 13):
            fps = {k: ERA5_NETCDF / folder / f"{year}_{month:02d}.nc"
                   for k, folder in GRID_FOLDERS.items()}
            if not all(fps[k].exists() for k in fps if k != "tp"):
                # A core variable missing → the whole month is dropped.  This
                # used to be SILENT — the gap later surfaces as a confusing
                # length/alignment error in require_length().  Log it so a
                # missing file is traceable at the source.
                missing = [k for k in fps if k != "tp" and not fps[k].exists()]
                print(f"  WARNING: {year}-{month:02d} skipped — missing "
                      f"core variable(s): {', '.join(missing)}")
                skipped_months.append(f"{year}-{month:02d}")
                continue
            tp_file_ok = fps["tp"].exists()

            fields = {}
            for k in ("t2m", "u10", "v10", "mslp", "tisr", "tcc"):
                ds = xr.open_dataset(fps[k])
                if lats is None:
                    lats = _extract_lats(ds)
                arr = _extract_var(ds)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                fields[k] = arr
                ds.close()
            if tp_file_ok:
                ds = xr.open_dataset(fps["tp"])
                arr = _extract_var(ds)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                # 服务器 tp 是"1 小时降水"却放在 6h 网格（GRIB fc/accum, step=1h）
                # → 先 ×6 转成 6 小时累计，再 ×1000 转 mm（用户选择：只缩放，不做 log1p）
                fields["tp"] = arr * 1000.0 * 6.0
                tp_seen = True
                ds.close()
            else:
                fields["tp"] = np.zeros_like(fields["t2m"])

            for ws in ("", "_arith"):
                # non-wind spatial vars: global + 3 bands
                for k in ("t2m", "mslp", "tisr", "tcc", "tp"):
                    acc[f"{k}{ws}"].append(_weighted_mean(fields[k], lats, None, "area" if ws == "" else "arith"))
                    for bname, bmask in _band_masks(lats):
                        acc[f"{k}_{bname}{ws}"].append(
                            _weighted_mean(fields[k], lats, bmask, "area" if ws == "" else "arith"))
                # wind: u/v/speed global + 3 bands, KE/stress global
                u, v = fields["u10"], fields["v10"]
                s = np.sqrt(u ** 2 + v ** 2)
                for feat, arr in (("u10", u), ("v10", v), ("speed", s)):
                    acc[f"{feat}{ws}"].append(_weighted_mean(arr, lats, None, "area" if ws == "" else "arith"))
                    for bname, bmask in _band_masks(lats):
                        acc[f"{feat}_{bname}{ws}"].append(
                            _weighted_mean(arr, lats, bmask, "area" if ws == "" else "arith"))
                acc[f"ke{ws}"].append(_weighted_mean(0.5 * s ** 2, lats, None, "area" if ws == "" else "arith"))
                acc[f"stress_u{ws}"].append(_weighted_mean(u * np.abs(u), lats, None, "area" if ws == "" else "arith"))
                acc[f"stress_v{ws}"].append(_weighted_mean(v * np.abs(v), lats, None, "area" if ws == "" else "arith"))
                # mslp gradient = midlat - polar
                acc[f"mslp_gradient{ws}"].append(
                    _weighted_mean(fields["mslp"], lats, _band_mask(lats, "midlat"), "area" if ws == "" else "arith")
                    - _weighted_mean(fields["mslp"], lats, _band_mask(lats, "polar"), "area" if ws == "" else "arith"))
        print(f"    year {year} done, {len(np.concatenate(acc['t2m']))} steps", flush=True)

    if not tp_seen:
        print("  WARNING: total_precipitation folder not found — tp features are zeros.")
    if skipped_months:
        print(f"  WARNING: {len(skipped_months)} month(s) skipped due to missing "
              f"core variables: {', '.join(skipped_months[:10])}"
              f"{' …' if len(skipped_months) > 10 else ''}")
    if not acc["t2m"]:
        raise SystemExit(
            "No netCDF data was loaded — check /data_public/ERA5 folder/file "
            "names against GRID_FOLDERS (e.g. ERA5_2m_temperature/YYYY_MM.nc)."
        )

    # ── Concatenate → leap-free → align → anomalies ──
    data = {}
    for k in acc:
        data[k] = delete_6hour(np.concatenate(acc[k]))
    data["co2"] = _load_co2(LEAP_FREE)
    data["nino34"] = _load_nino34(LEAP_FREE)
    require_length(min(len(v) for v in data.values()), "feature bank")
    n = LEAP_FREE
    for k in data:
        data[k] = data[k][:n]
    ts = leap_free_timestamps(n)
    data["t2m_anomaly"] = compute_t2m_anomaly(data["t2m"], ts)
    data["t2m_anomaly_arith"] = compute_t2m_anomaly(data["t2m_arith"], ts)
    data["tisr_anomaly"] = compute_t2m_anomaly(data["tisr"], ts)
    data["tisr_anomaly_arith"] = compute_t2m_anomaly(data["tisr_arith"], ts)

    # ── Persist feature bank (netCDF read happens once) ──
    for k, arr in data.items():
        np.save(FEATURE_DIR / f"{k}.npy", np.asarray(arr))
    np.save(FEATURE_DIR / "timestamps.npy", ts.astype(str).to_numpy())
    print(f"  [bank] saved {len(data)} features -> {FEATURE_DIR}")
    return data, ts


# ═══════════════════════════════════════════════════════════════════
# Assembly + save
# ═══════════════════════════════════════════════════════════════════

def build_dataset(dataset_id: str, bank: dict, wind_block: list[str],
                  extras: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Assemble [T, N] + column names for a dataset.

    A/B use the "_arith" feature keys (equal-weight mean), C/D the area keys.
    Column order: [target, co2, *wind_block, mslp, tisr, nino34, tcc, *extras].
    CO₂ is FIXED at index 1 (right after the target) for EVERY variant, so
    co2_index is always 1 and no per-variant bookkeeping is needed.
    """
    target_kind, source = DATASET_SPEC[dataset_id]
    ws = "_arith" if source == "arithmetic" else ""
    target = bank[f"t2m_anomaly{ws}"] if target_kind == "anomaly" else bank[f"t2m{ws}"]
    co2 = bank["co2"]
    wind = [bank[_wkey(k, ws)] for k in wind_block]
    trail = [bank[_wkey(k, ws)] for k in BASE_TRAIL]
    extra = [bank[_wkey(k, ws)] for k in extras]
    values = np.column_stack([target, co2] + wind + trail + extra).astype(np.float32)
    target_name = "t2m_anomaly" if target_kind == "anomaly" else "t2m_raw"
    names = np.array([target_name, "co2"] + wind_block + BASE_TRAIL + extras)
    return values, names


def save_dataset(dataset_id: str, values: np.ndarray, names: np.ndarray, ts,
                 suffix: str = "") -> None:
    tag = f"{dataset_id}{suffix}"
    np.save(OUT_DIR / f"weather_values_{tag}.npy", values)
    np.save(OUT_DIR / f"weather_timestamps_{tag}.npy", ts.astype(str).to_numpy())
    np.save(OUT_DIR / f"weather_variable_names_{tag}.npy", names)
    co2_idx = int(np.where(np.char.startswith(names.astype(str), "co2"))[0][0])
    print(f"\n  Dataset {tag}: {values.shape}  [co2_index={co2_idx}]")
    for i, nm in enumerate(names):
        print(f"    [{i}] {nm:12s}: mean={values[:, i].mean():.2f}, "
              f"std={values[:, i].std():.2f}")
    verify_dataset_alignment(values, ts, tag)


# ═══════════════════════════════════════════════════════════════════
# Public entry
# ═══════════════════════════════════════════════════════════════════

def run_datasets(dataset_ids, wind_variants=None, extras=None,
                 dry_run: bool = False) -> None:
    ids = [d.upper() for d in dataset_ids]
    wind_variants = wind_variants or []
    extras = extras or []
    for d in ids:
        if d not in DATASET_SPEC:
            raise SystemExit(f"Unknown dataset {d!r}; choose from {list(DATASET_SPEC)}")
    for v in wind_variants:
        if v not in WIND_VARIANTS_DEF:
            raise SystemExit(f"Unknown wind variant {v!r}; choose from {list(WIND_VARIANTS_DEF)}")
    for k in extras:
        if k not in all_feature_keys():
            raise SystemExit(f"Unknown extra feature {k!r}")

    if dry_run:
        print(f"\nDry run — datasets: {ids}, wind variants: {wind_variants}, extras: {extras}")
        return

    bank, ts = load_grid_features()

    meta = {}
    for d in ids:
        # G0 = lean default (8 columns, global u/v); extras in a SEPARATE
        # "_ext" dataset so the baseline isn't bloated.
        g0 = WIND_VARIANTS_DEF["G0"]
        save_dataset(d, *build_dataset(d, bank, g0, []), ts)
        meta[d] = {"values_path": f"data/processed/weather_values_{d}.npy",
                   "timestamps_path": f"data/processed/weather_timestamps_{d}.npy",
                   "co2_index": 1, "target_index": 0,
                   "num_features": 1 + 1 + len(g0) + len(BASE_TRAIL),
                   "columns": [str(n) for n in build_dataset(d, bank, g0, [])[1]]}

        for v in wind_variants:
            block = WIND_VARIANTS_DEF[v]
            values, names = build_dataset(d, bank, block, [])
            save_dataset(d, values, names, ts, suffix=f"_{v}")
            meta[f"{d}_{v}"] = {"values_path": f"data/processed/weather_values_{d}_{v}.npy",
                                "timestamps_path": f"data/processed/weather_timestamps_{d}_{v}.npy",
                                "co2_index": 1, "target_index": 0,
                                "num_features": len(names),
                                "columns": [str(n) for n in names]}

        if extras:
            values, names = build_dataset(d, bank, g0, extras)
            save_dataset(d, values, names, ts, suffix="_ext")
            meta[f"{d}_ext"] = {"values_path": f"data/processed/weather_values_{d}_ext.npy",
                                "timestamps_path": f"data/processed/weather_timestamps_{d}_ext.npy",
                                "co2_index": 1, "target_index": 0,
                                "num_features": len(names),
                                "columns": [str(n) for n in names]}

    with open(OUT_DIR / "weather_variants.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  Variant metadata -> {OUT_DIR / 'weather_variants.json'}")
    for tag, m in meta.items():
        print(f"    {tag:10s} co2_index={m['co2_index']:<3d} num_features={m['num_features']}")
    print("\nDone.")


def main() -> None:
    p = argparse.ArgumentParser(description="Build MTTGNet datasets + wind variants.")
    p.add_argument("--datasets", default=DATASETS)
    p.add_argument("--wind-variants", default=",".join(WIND_VARIANTS))
    p.add_argument("--no-wind-variants", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    variants = [] if args.no_wind_variants else [v for v in args.wind_variants.split(",") if v]
    run_datasets([x.strip() for x in args.datasets.split(",") if x.strip()],
                 wind_variants=variants, extras=EXTRA_COLUMNS, dry_run=args.dry_run)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        run_datasets([s.strip() for s in DATASETS.split(",") if s.strip()],
                     wind_variants=WIND_VARIANTS, extras=EXTRA_COLUMNS, dry_run=DRY_RUN)
