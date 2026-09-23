#!/usr/bin/env python3
"""一键启动训练 — 用当前 src/trainer + src/mttgnet 管线训练单个配置。

用法:
    python run_model.py                          # 用下方 CONFIG
    python run_model.py path/to/config.yaml      # 或命令行指定配置

旧版此文件调度到已删除的 scripts/train_weather.py / train_explore.py 等，
现已改为统一走 src.trainer.Trainer（与 run_experiments.py 相同管线）。
"""

from __future__ import annotations

import sys
import gc
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import numpy as np
import torch

from src.config import load_config
from src.trainer import Trainer, anchor_forward
from src.mttgnet import MTTGNetWrapper
from src.seed import seed_everything

# ═══════════════════════════════════════════════════════════════
# 改这里！选择要跑的模型配置
# ═══════════════════════════════════════════════════════════════
CONFIG = "configs/example_h28.yaml"
# ═══════════════════════════════════════════════════════════════


def run_one(config_path: str, warmstart: str | None = None) -> None:
    cfg = load_config(Path(__file__).resolve().parent / config_path)
    seed_everything(cfg.get("seed", 233))

    trainer = Trainer(cfg)
    device = trainer.device
    m = cfg.get("model_params", {})
    ds = cfg["dataset"]
    model = MTTGNetWrapper(
        num_features=trainer.num_features,
        hidden_dim=m.get("hidden_dim", 96),
        horizon=ds["horizon"],
        seq_len=ds["recent_length"],
        dropout=m.get("dropout", 0.15),
        memory_slots=m.get("memory_slots", 128),
        mem_top_k=m.get("mem_top_k", 8),
        num_gru_layers=m.get("num_gru_layers", 3),
        use_gate=m.get("use_gate", True),
        use_time_bias=m.get("use_time_bias", True),
        use_memory=m.get("use_memory", True),
        use_variable_encoder=m.get("use_variable_encoder", True),
        use_shift_encoder=m.get("use_shift_encoder", True),
        use_var_gnn=m.get("use_var_gnn", True),
        use_calendar=m.get("use_calendar", True),
        temporal_encoder=m.get("temporal_encoder", "gru"),
        var_encoder_type=m.get("var_encoder_type", "attention"),
        gat_version=m.get("gat_version", "gat"),
        co2_index=cfg.get("data", {}).get("co2_index", 1),
        # CO2 is already [0,1] normalized by the trainer (280–1200 ppm).
        # Identity pass-through in shift_encoder.
        co2_mean=0.0, co2_std=1.0, co2_min=0.0, co2_max=1.0,
        # ── innovations (config-driven; default off) ──
        probabilistic=m.get("probabilistic", False),
        sigma_floor=m.get("sigma_floor", 1e-3),
        use_mono_reg=m.get("use_mono_reg", False),
        mono_weight=m.get("mono_weight", 0.1),
        mono_delta=m.get("mono_delta", 0.05),
        mono_margin=m.get("mono_margin", 0.0),
        use_co2_timebias=m.get("use_co2_timebias", False),
        use_short_residual=m.get("use_short_residual", False),
        target_index=cfg.get("data", {}).get("target_index", 0),
        short_residual_steps=m.get("short_residual_steps", 8),
        decoder_type=m.get("decoder_type", "calendar_mlp"),
        use_persist_blend=m.get("use_persist_blend", False),
        persist_blend_steps=m.get("persist_blend_steps", 8),
        lead_anchor_scale=m.get("lead_anchor_scale", False),
        anchor_scale_min=m.get("anchor_scale_min", 0.3),
        use_dual_forecast=m.get("use_dual_forecast", False),
        dual_short_steps=m.get("dual_short_steps", 8),
        use_multirate=m.get("use_multirate", False),
        multirate_pool=m.get("multirate_pool", 4),
        use_multiband=m.get("use_multiband", False),
        band_pool1=m.get("band_pool1", 2),
        band_pool2=m.get("band_pool2", 8),
        time_bias_scale=m.get("time_bias_scale", 1.0),
        shift_bias_scale=m.get("shift_bias_scale", 1.0),
        use_analog_memory=m.get("use_analog_memory", False),
        analog_top_k=m.get("analog_top_k", 8),
        analog_temperature=m.get("analog_temperature", 1.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Config: {config_path} | Params: {n_params:,} | Device: {device}")

    # Warm-start: load the shared (H-independent) weights from a trained checkpoint
    # (e.g. the strong H=28 model) with strict=False, skipping horizon-specific heads.
    if warmstart and os.path.exists(warmstart):
        import torch as _t
        _ck = _t.load(warmstart, map_location=device, weights_only=True)
        _sd = model.state_dict()
        # keep only keys present in BOTH with the SAME shape; drop horizon-dependent
        # (direct_head / query_encoder.horizon_embedding, [28,H] vs [120,H]) which would
        # raise a size-mismatch RuntimeError even with strict=False.
        _filt = {k: v for k, v in _ck["model_state_dict"].items()
                 if k in _sd and v.shape == _sd[k].shape}
        _miss, _unexp = model.load_state_dict(_filt, strict=False)
        print(f"[warmstart] shared weights: {len(_filt)}/{len(_ck['model_state_dict'])} loaded"
              f" (skipped {len(_ck['model_state_dict']) - len(_filt)} H-dependent/mismatched); -> {warmstart}")

    t0 = time.time()
    best_val, train_losses, val_losses = trainer.train(model, anchor_forward)
    print(f"Train time: {(time.time() - t0) / 60:.1f} min")

    pred, target = trainer.evaluate(model, anchor_forward)
    trainer.save_results(pred, target, n_params=n_params, best_val_loss=best_val,
                         train_losses=train_losses, val_losses=val_losses)

    overall_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    print(f"✓ RMSE={overall_rmse:.4f}")

    try:
        trainer.cleanup()
    except Exception:
        pass
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("config", nargs="?", default=CONFIG)
    _ap.add_argument("--warmstart", default=None,
                     help="trained checkpoint .pt to warm-start from (loads shared weights, strict=False)")
    _a = _ap.parse_args()
    run_one(_a.config, _a.warmstart)
