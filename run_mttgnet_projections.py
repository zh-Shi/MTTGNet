#!/usr/bin/env python3
"""MTTGNet SSP projections (CMIP6-only CO2), with physical calibration.

Pipeline:
  1. Train on 1850–2014  (CMIP6 historical CO2 only)
  2. Project 2015–2100 with SSP CO2 pathways via autoregressive rollout
  3. Validate 2015–2026 against held-out HadCRUT5 observations
  4. Physically calibrate the century-scale response: the monthly 1-step GRU
     under-fits the slow CO2 trend (signal-to-noise ~0.006 K/month vs
     ~0.1-0.2 K of monthly variability) AND predicts the anomaly level too
     low (z-scored 1-step MSE shrinks forecasts toward the training mean).
     The model's entire linear CO2 response is therefore replaced by the one
     fit on the 1850-2014 HadCRUT5 record, keeping only its short-term
     residual around that response:
         pred_cal(t) = pred_raw(t) + [a_obs*ln(CO2(t)) + b_obs]
                                     - [a_model*ln(CO2(t)) + b_model]
     (emulator-style, cf. MAGICC-lite / FaIR).  This anchors BOTH the level
     and the century-scale slope to the instrumental record.  Raw and
     calibrated curves are both saved for transparency.

The processed .npy ends at 2014-12 (CMIP6 historical CO2 boundary).
SSP CO2 .npy files cover 2015-01 → 2100-12  — seamless transition.

Usage:
    python run_mttgnet_projections.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.mttgnet import MTTGNetWrapper
from src.scaler import StandardScaler
from src.seed import seed_everything
from src.time_utils import calendar_features

# `_check_forwarded` is the guard that makes a missing wrapper argument fail loudly
# instead of being swallowed by `**kwargs`; it lives in scripts/run_multiseed.py and
# is reused here so both entry points enforce the same invariant.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from run_multiseed import _check_forwarded                        # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
# Overridable so the same pipeline can produce the century projections from a
# different recipe.  This exists for paper item A2: the manuscript quotes a
# single-step R^2 from a two-variable monthly model trained through 2000
# (`cpx_mttgnet_H5_co2`) and the century rollout from this script's own model
# (`mttgnet_hadcrut5`, trained through 1990), i.e. two different runs presented as
# one.  Pointing PROJECTIONS_CONFIG at the H5 recipe regenerates the projections
# from the model that has the quoted skill, so one run backs both numbers:
#
#   PROJECTIONS_CONFIG=configs/cpx_mttgnet_H5_co2.yaml \
#   PROJECTIONS_SEED=123 python run_mttgnet_projections.py
#
# PROJECTIONS_SEED is needed whenever the model was trained by
# `scripts/run_multiseed.py`, which writes to `outputs/{stem}_s{seed}/`; without it
# the script would look for `outputs/{stem}/checkpoints/`, which does not exist for
# those runs.
_CONFIG_NAME = os.environ.get("PROJECTIONS_CONFIG", "configs/example_monthly.yaml")
CONFIG_PATH = PROJECT_ROOT / _CONFIG_NAME
_STEM = Path(_CONFIG_NAME).stem
_SEED = os.environ.get("PROJECTIONS_SEED", "").strip()
_RUN_NAME = f"{_STEM}_s{_SEED}" if _SEED else _STEM
CHECKPOINT = PROJECT_ROOT / f"outputs/{_RUN_NAME}/checkpoints/best_model.pt"
OUT_DIR = PROJECT_ROOT / f"outputs/{_RUN_NAME}/projections"

CO2_LO, CO2_HI = 280.0, 1200.0  # fixed [0,1] CO2 range used by the Trainer


def _to_unit(x: np.ndarray) -> np.ndarray:
    """Raw ppm → fixed [0,1] range (matches src/trainer.py)."""
    return (x - CO2_LO) / (CO2_HI - CO2_LO)


def _load_hadcrut5_raw() -> pd.DataFrame:
    """Load full HadCRUT5 (1850–2026) for validation beyond 2014."""
    p = (PROJECT_ROOT / "data/HadCRUT5/"
         "HadCRUT.5.1.0.0.analysis.summary_series.global.monthly.csv")
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["Time"].str.strip())
    df = df.rename(columns={"Anomaly (deg C)": "anomaly"})
    df = df.set_index("timestamp").sort_index()
    df.index = df.index + pd.offsets.Day(14)
    return df


# ---------------------------------------------------------------------------
# autoregressive rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def autoregressive_project(
    model: MTTGNetWrapper,
    origin_idx: int,
    co2_future: np.ndarray,              # [T_future] CO2 ppm per month
    values_all: np.ndarray,              # [T_all, N] unscaled (target °C, CO2 ppm)
    ts: pd.DatetimeIndex,                # training timestamps (1850-02 → 2014-12)
    scaler: StandardScaler,
    device: torch.device,
    *,
    L: int = 60,
    H: int = 1,
    future_co2_ppm: np.ndarray | None = None,
):
    """Autoregressive 1-step rollout over the future CO2 path.

    ``origin_idx`` is the index of the last *known* month (2014-12).
    Prediction covers months [origin_idx+1, origin_idx+T_future].

    The model for mttgnet_hadcrut5.yaml is a DIRECT-HEAD GRU: H=1, no
    anchors, no memory, no shift-encoder.  We must therefore keep K=0
    anchors (else the forward flips into the untrained fusion+decoder
    path), and feed CO2 in the SAME [0,1] scale the Trainer used — not
    z-scored.
    """
    N = values_all.shape[1]
    # Number of leading columns the scaler was fitted on; any trailing columns
    # (the constant hour sin/cos channels added by dataset.time_encode) are
    # passed through unscaled, exactly as the PeriodicAnchorDataset does.
    n_scaled = (int(np.asarray(scaler.mean).shape[0])
                if getattr(scaler, "mean", None) is not None else N)
    TGT, CO2 = 0, 1
    T = len(co2_future)

    recent = values_all[origin_idx - L + 1 : origin_idx + 1].copy()   # [L, N]
    preds = []
    step = 0
    while step < T:
        hs = min(H, T - step)                                          # H=1 here

        # Input: z-score every column, then overwrite CO2 with fixed [0,1]
        # (the Trainer's exact pipeline).  Feeding z-scored CO2 here would
        # show the GRU a +3..+13σ "feature" the model never saw in training.
        x_np = np.array(recent, dtype=np.float32)
        x_np[:, :n_scaled] = scaler.transform(recent[:, :n_scaled])
        x_np[:, CO2] = _to_unit(recent[:, CO2])
        x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)

        # Calendar for the target month.  use_calendar=false in the config,
        # so the model zeroes it — but compute real future dates anyway so
        # the code stays correct if the config ever enables it (no clamping
        # to ts[-1], which froze every 2015+ month at Dec 2014).
        cal = np.zeros((hs, 5), dtype=np.float32)
        for h in range(hs):
            fut_date = ts[0] + pd.DateOffset(months=int(origin_idx + step + h + 1))
            cal[h] = calendar_features(fut_date, h + 1, H)
        cal_t = torch.from_numpy(cal).unsqueeze(0).to(device)

        # Known future forcing for the target steps of THIS block.  Only models
        # built with use_future_co2 consume it; passing None leaves every other
        # configuration untouched.
        fut_co2_t = None
        if future_co2_ppm is not None:
            seg = np.asarray(future_co2_ppm[step:step + hs], dtype=np.float32)
            fut_co2_t = torch.from_numpy(_to_unit(seg)).reshape(1, hs, 1).to(device)

        # No anchors — keep the direct head active (C2 fix).
        d_anc_t = torch.zeros(1, hs, 0, N, device=device)
        d_mask_t = torch.zeros(1, hs, 0, dtype=torch.bool, device=device)
        d_gap_t = torch.zeros(1, hs, 0, device=device)
        y_anc_t = torch.zeros(1, hs, 0, N, device=device)
        y_mask_t = torch.zeros(1, hs, 0, dtype=torch.bool, device=device)
        y_gap_t = torch.zeros(1, hs, 0, device=device)

        out = model(
            x_recent=x_t,
            daily_anchor=d_anc_t, yearly_anchor=y_anc_t,
            daily_mask=d_mask_t, yearly_mask=y_mask_t,
            future_calendar=cal_t,
            daily_gaps=d_gap_t, yearly_gaps=y_gap_t,
            future_co2=fut_co2_t,
        )
        pred_std = out["prediction"].cpu().numpy()[0, :hs]

        # inverse standardisation → °C anomaly
        anomaly = scaler.inverse_target(pred_std, target_index=TGT)
        preds.extend(anomaly.tolist())

        # slide window: drop oldest month, append prediction + future CO2
        for h in range(hs):
            new = recent[-1].copy()
            new[TGT] = anomaly[h]
            new[CO2] = co2_future[step + h]
            recent = np.concatenate([recent[1:], new[np.newaxis, :]], axis=0)
        step += hs

    return np.array(preds[:T])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    seed_everything(123)

    print("=" * 64)
    print("  MTTGNet  SSP Projections  (CMIP6 CO2 only)")
    print("=" * 64)

    # ---- 1. load processed data (1850–2014) ------------------------------
    cfg = load_config(CONFIG_PATH)
    L, H = cfg["dataset"]["recent_length"], cfg["dataset"]["horizon"]

    values = np.load(PROJECT_ROOT / cfg["data"]["values_path"]).astype(np.float32)
    ts_raw = np.load(PROJECT_ROOT / cfg["data"]["timestamps_path"], allow_pickle=True)
    ts = pd.DatetimeIndex(pd.to_datetime(ts_raw))
    print(f"Training data: {len(ts)} months  ({ts[0].date()} → {ts[-1].date()})")

    # ---- 2. load full HadCRUT5 for 2015+ validation -----------------------
    had_raw = _load_hadcrut5_raw()
    print(f"Full HadCRUT5 : {len(had_raw)} months  "
          f"({had_raw.index[0].date()} → {had_raw.index[-1].date()})")

    # ---- 3. scaler (fitted on train split only — matches the Trainer) ----
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    scaler = StandardScaler().fit(values[ts <= train_end])

    # The Trainer's PeriodicAnchorDataset appends hour sin/cos when
    # dataset.time_encode is true, so the trained model expects
    # num_features + 2 input columns; without them the checkpoint's
    # node_emb / input_proj / variable_embedding shapes do not match and
    # load_state_dict fails with a size mismatch.  HadCRUT5 is monthly and
    # every timestamp is at 00:00 UTC, so the two extra channels are the
    # constants (0, 1).  They are appended AFTER z-scoring — the Trainer
    # scales only the physical variables — so the scaler must keep seeing
    # the first `n_raw` columns only.
    n_raw = values.shape[1]
    if cfg["dataset"].get("time_encode", False):
        hours = np.asarray(ts.hour, dtype=np.float32)
        values_model = np.concatenate([
            values,
            np.sin(2 * np.pi * hours / 24.0).reshape(-1, 1).astype(np.float32),
            np.cos(2 * np.pi * hours / 24.0).reshape(-1, 1).astype(np.float32),
        ], axis=1)
        print(f"  time_encode : +2 channels -> {values_model.shape[1]} model features")
    else:
        values_model = values

    # ---- 4. model ---------------------------------------------------------
    # The argument list below must mirror EVERY switch in the config's
    # `model_params`, because the wrapper ends in `**kwargs`: a switch this call
    # forgets is swallowed silently, the model is built with the wrong shape, and
    # the failure only surfaces as a `load_state_dict` size mismatch later.  That
    # is exactly what happened when PROJECTIONS_CONFIG was first pointed at the H5
    # recipe, which sets `use_multiscale_co2: true` -> `_co2_dim = 3` ->
    # `shift_encoder.encoder.0.weight` of shape [96, 99] against the [96, 97] this
    # list built.  `_check_forwarded` (borrowed from scripts/run_multiseed.py)
    # now raises if any wrapper parameter is missing from `kw`, so the next such
    # omission fails loudly and immediately instead of at load time.
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    mc = cfg["model_params"]
    ds = cfg["dataset"]
    co2_idx = cfg["data"].get("co2_index", 1)
    kw = dict(
        num_features=values_model.shape[1],
        hidden_dim=mc["hidden_dim"],
        horizon=H, seq_len=L,
        calendar_dim=mc.get("calendar_dim", 5),
        dropout=mc.get("dropout", 0.1),
        memory_slots=mc.get("memory_slots", 128),
        mem_top_k=mc.get("mem_top_k", 8),
        num_gru_layers=mc.get("num_gru_layers", 2),
        use_gate=mc.get("use_gate", True),
        use_time_bias=mc.get("use_time_bias", True),
        use_memory_horizon_gate=mc.get("use_memory_horizon_gate", False),
        memory_gate_init=mc.get("memory_gate_init", -3.0),
        use_memory=mc.get("use_memory", True),
        use_variable_encoder=mc.get("use_variable_encoder", True),
        use_shift_encoder=mc.get("use_shift_encoder", True),
        use_var_gnn=mc.get("use_var_gnn", True),
        use_calendar=mc.get("use_calendar", True),
        temporal_encoder=mc.get("temporal_encoder", "gru"),
        var_encoder_type=mc.get("var_encoder_type", "attention"),
        gat_version=mc.get("gat_version", "gat"),
        co2_index=co2_idx,
        # CO2 arrives already in [0,1] (see _to_unit above), which is the Trainer's
        # convention: src/trainer.py line 379 writes (raw-280)/(1200-280) as the
        # feature, and the model recovers ppm with co2_min/co2_max.  These MUST
        # therefore be the real bounds, not an identity [0,1] pair.  The identity
        # pair this script used to pass is equivalent for the single-scale CO2
        # level (both yield the same [0,1] value) which is why it went unnoticed,
        # but it is wrong for `use_multiscale_co2`, whose first feature is
        # log(ppm/co2_ref): with min/max = 0/1 that becomes log(unit/284) on a
        # number below 1, and its trend/anomaly features are off by the ppm span.
        co2_mean=mc.get("co2_mean", 0.0), co2_std=mc.get("co2_std", 1.0),
        co2_min=mc.get("co2_min", 280.0), co2_max=mc.get("co2_max", 1200.0),
        # ---- everything else the wrapper accepts, read from the config ----
        probabilistic=mc.get("probabilistic", False),
        sigma_floor=mc.get("sigma_floor", 1e-3),
        use_mono_reg=mc.get("use_mono_reg", False),
        mono_weight=mc.get("mono_weight", 0.1),
        mono_delta=mc.get("mono_delta", 0.05),
        mono_margin=mc.get("mono_margin", 0.0),
        use_analog_memory=mc.get("use_analog_memory", False),
        analog_top_k=mc.get("analog_top_k", 8),
        analog_temperature=mc.get("analog_temperature", 1.0),
        use_multiscale_co2=mc.get("use_multiscale_co2", False),
        co2_anchor_a=mc.get("co2_anchor_a", 0.0),
        co2_ref=mc.get("co2_ref", 284.0),
        use_short_residual=mc.get("use_short_residual", False),
        short_residual_steps=mc.get("short_residual_steps", 8),
        use_co2_timebias=mc.get("use_co2_timebias", False),
        target_index=cfg.get("data", {}).get("target_index", 0),
        decoder_type=mc.get("decoder_type", "calendar_mlp"),
        time_bias_scale=mc.get("time_bias_scale", 1.0),
        shift_bias_scale=mc.get("shift_bias_scale", 1.0),
        use_persist_blend=mc.get("use_persist_blend", False),
        persist_blend_steps=mc.get("persist_blend_steps", 8),
        lead_anchor_scale=mc.get("lead_anchor_scale", False),
        anchor_scale_min=mc.get("anchor_scale_min", 0.3),
        use_dual_forecast=mc.get("use_dual_forecast", False),
        dual_short_steps=mc.get("dual_short_steps", 8),
        use_multirate=mc.get("use_multirate", False),
        multirate_pool=mc.get("multirate_pool", 4),
        use_multiband=mc.get("use_multiband", False),
        band_pool1=mc.get("band_pool1", 2),
        band_pool2=mc.get("band_pool2", 8),
        use_future_co2=ds.get("use_future_co2", False),
    )
    _check_forwarded(kw)
    model = MTTGNetWrapper(**kw).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        raise RuntimeError(
            f"Checkpoint missing {len(missing)} keys — the model built here "
            f"({var_encoder_type}={mc.get('var_encoder_type')!r}) does not match "
            f"the checkpoint. Refusing to project on a partial load."
        )
    if unexpected:
        print(f"  ⚠ Unexpected keys: {len(unexpected)} (ignored)")
    model.eval()
    print(f"Model loaded : {CHECKPOINT}")
    print(f"  best val loss = {ckpt.get('best_val_loss', '?'):.6f}")
    print(f"  L={L}  H={H}  {device}\n")

    # ---- 5. projection origin (last training month) ----------------------
    origin_idx = len(ts) - 1
    print(f"Origin: idx={origin_idx}  {ts[-1].date()}")

    # SSP CO2 files: 2015-01 → 2100-12  (1032 months each)
    SSP = {"126": "SSP1-2.6", "245": "SSP2-4.5",
           "370": "SSP3-7.0", "585": "SSP5-8.5"}
    results = {}
    co2_futures = {}

    for key, label in SSP.items():
        print(f"\n--- {label} ---")
        co2_ssp = np.load(
            PROJECT_ROOT / f"data/processed/co2_ssp{key}_monthly.npy"
        ).astype(np.float32)
        co2_futures[key] = co2_ssp

        pred = autoregressive_project(
            model, origin_idx, co2_ssp, values_model, ts, scaler, device, L=L, H=H)
        results[key] = pred
        print(f"  2015 mean {pred[0]:+.3f} °C | 2100 mean {pred[-1]:+.3f} °C")

    # ---- 5.5 physically-calibrated century-scale CO2 response -----------
    # The monthly direct-head GRU under-fits the slow CO2 trend: at a 1-step
    # horizon the trend is ~0.006 K/month against ~0.1-0.2 K of monthly noise,
    # so MSE training assigns the model's capacity to persistence (a_model ≈
    # 0.12 K/ln(CO2) instead of the ~3.5 K/ln(CO2) seen in the record).  It
    # also predicts the anomaly *level* too low (z-scored 1-step MSE shrinks
    # forecasts toward the training mean: 2015 is forecast at ~+0.34 K vs
    # ~+0.7 K observed).  We therefore replace the model's entire linear CO2
    # response with the one fit on the 1850-2014 HadCRUT5 record
    # (emulator-style, cf. MAGICC-lite / FaIR), keeping only the model's
    # short-term / inter-annual residual around that response:
    #     pred_cal(t) = pred_raw(t) + [a_obs*ln(CO2(t)) + b_obs]
    #                                  - [a_model*ln(CO2(t)) + b_model]
    # This anchors BOTH the level (2015-2026 ≈ observed) and the century-scale
    # slope to the instrumental record.
    TGT_IDX = cfg["data"]["target_index"]
    CO2_IDX = cfg["data"]["co2_index"]
    lnc_obs = np.log(values[:, CO2_IDX])
    a_obs, b_obs = np.polyfit(lnc_obs, values[:, TGT_IDX], 1)
    # model's own linear response, pooled across all SSP trajectories
    lnc_pool = np.concatenate([np.log(co2_futures[k]) for k in SSP])
    pred_pool = np.concatenate([results[k] for k in SSP])
    a_model, b_model = np.polyfit(lnc_pool, pred_pool, 1)
    cal = {
        k: (results[k]
            + (a_obs * np.log(co2_futures[k]) + b_obs)
            - (a_model * np.log(co2_futures[k]) + b_model))
        for k in SSP
    }
    print("\n  Calibration (level+slope anchored to observed 1850-2014):")
    print(f"    observed : a={a_obs:.3f} K/ln(CO2)  b={b_obs:.3f} degC")
    print(f"    raw GRU  : a={a_model:.3f} K/ln(CO2)  b={b_model:.3f} degC")
    for k, label in SSP.items():
        print(f"  {label:9s} 2100: raw {results[k][-1]:+.2f} °C -> "
              f"calibrated {cal[k][-1]:+.2f} °C")

    # ---- 6. compare 2015–2026 against HadCRUT5 ---------------------------
    # SSP files start at 2015-01; prediction month i → date = 2015-01-15 + i.
    # Build month-DAY-15 anchors explicitly: freq="MS" alone would snapshot to
    # month-START (2015-02-01...), misaligning every validation month by one.
    n_future = len(results["126"])
    proj_dates = pd.date_range("2015-01-01", periods=n_future, freq="MS") + pd.Timedelta(days=14)
    had_2015 = had_raw["anomaly"].reindex(proj_dates, method="nearest")

    print(f"\n{'=' * 64}")
    print("  VALIDATION  (2015–2026 vs HadCRUT5)")
    print(f"{'=' * 64}")

    # all SSPs are nearly identical in 2015–2026, pick SSP2-4.5 as reference
    # IMPORTANT: reindex(method="nearest") FILLS every proj_date with the
    # nearest HadCRUT5 month — dates past the record (2026-06 → 2100-12)
    # collapse onto the last observation, so had_2015.dropna() spans the whole
    # century and the "RMSE" silently compares the projection against a flat
    # 2026-05 line.  Restrict the window to months with a REAL observation
    # (the actual overlap of the HadCRUT5 record with proj_dates).
    n_val = int(((had_raw.index >= proj_dates[0]) &
                 (had_raw.index <= proj_dates[-1])).sum())
    n_val = min(n_val, len(results["245"]))
    ref_obs = had_2015.values[:n_val]
    mask = ~np.isnan(ref_obs)
    rmse_raw = float(np.sqrt(np.mean((results["245"][:n_val][mask]
                                      - ref_obs[mask]) ** 2)))
    rmse_cal = float(np.sqrt(np.mean((cal["245"][:n_val][mask]
                                      - ref_obs[mask]) ** 2)))
    bias_raw = float(np.mean(results["245"][:n_val][mask] - ref_obs[mask]))
    bias_cal = float(np.mean(cal["245"][:n_val][mask] - ref_obs[mask]))
    print(f"  RMSE (SSP2-4.5 vs HadCRUT5, {n_val} months): "
          f"raw {rmse_raw:.4f} (bias {bias_raw:+.3f}) | "
          f"calibrated {rmse_cal:.4f} (bias {bias_cal:+.3f})")

    # ---- 7. warming summary -----------------------------------------------
    print(f"\n{'=' * 64}")
    print("  2081–2100 WARMING  (vs 1850–1900 pre-industrial)")
    print(f"{'=' * 64}")

    pi_mask = (ts >= pd.Timestamp("1850-01-15")) & (ts <= pd.Timestamp("1900-12-15"))
    pi_mean = float(values[pi_mask, 0].mean())

    for key, label in SSP.items():
        p = cal[key]
        end20 = float(np.mean(p[-240:])) if len(p) >= 240 else float(p[-1])
        raw_p = results[key]
        raw_end20 = (float(np.mean(raw_p[-240:])) if len(raw_p) >= 240
                     else float(raw_p[-1]))
        print(f"  {label:12s}  {end20:+.3f} degC  "
              f"(calibrated warming: {end20 - pi_mean:+.2f} degC)  "
              f"[raw GRU: {raw_end20 - pi_mean:+.2f} degC]")

    # ---- 8. save ----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save = {
        "proj_dates": np.array([str(d.date()) for d in proj_dates]),
        "hadcrut5_2015": had_2015.values,
        "pi_mean": pi_mean,
        "calibration": {"a_obs": a_obs, "b_obs": b_obs,
                        "a_model": a_model, "b_model": b_model},
    }
    for k in SSP:
        save[f"ssp{k}"] = results[k]          # raw GRU autoregressive rollout
        save[f"ssp{k}_cal"] = cal[k]          # physically-calibrated emulator
    np.savez(OUT_DIR / "projections.npz", **save)
    print(f"\nSaved: {OUT_DIR / 'projections.npz'}")

    print("Done.")


if __name__ == "__main__":
    main()
