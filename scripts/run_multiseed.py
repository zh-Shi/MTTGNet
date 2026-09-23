#!/usr/bin/env python3
"""Run multi-seed training for an arbitrary MTTGNet config (incl. innovations).

Each seed trains the same config with a different random seed and saves to
``outputs/{cfg_name}_s{seed}/``; a mean/std summary is aggregated to
``outputs/{cfg_name}_multiseed_summary.json``.

Usage (remote):
    python scripts/run_multiseed.py configs/cpx_mttgnet_D_prob.yaml \
        --seeds 123,42,789,456,1024,2024
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import load_config
from src.trainer import Trainer, anchor_forward
from src.mttgnet import MTTGNetWrapper
from src.seed import seed_everything

DEFAULT_SEEDS = [123, 42, 789, 456, 1024, 2024]


def _check_forwarded(kw: dict) -> None:
    """Every MTTGNetWrapper parameter must be forwarded by build_model.

    `MTTGNetWrapper.__init__` ends in `**kwargs`, so a parameter that this
    function forgets to pass is swallowed silently: a config saying
    `use_X: true` trains a model without X and the run is reported under X's
    name.  That has produced three false results in this project's history
    (04_audit §9.3: the multi-rate flags; then use_future_co2; then
    use_memory_horizon_gate).  Failing loudly is cheaper than finding out later.
    """
    import inspect
    accepted = set(inspect.signature(MTTGNetWrapper.__init__).parameters)
    accepted -= {"self", "kwargs"}
    gap = accepted - set(kw)
    if gap:
        raise RuntimeError(
            f"build_model() does not forward these MTTGNetWrapper parameters: "
            f"{sorted(gap)} — a config that sets them would be silently ignored. "
            f"Add them to the call in scripts/run_multiseed.py (04_audit §9.3).")


def build_model(cfg, trainer) -> MTTGNetWrapper:
    m = cfg.get("model_params", {})
    ds = cfg["dataset"]
    kw = dict(
        num_features=trainer.num_features,
        hidden_dim=m.get("hidden_dim", 96),
        horizon=ds["horizon"], seq_len=ds["recent_length"],
        calendar_dim=m.get("calendar_dim", 5),
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
        co2_mean=0.0, co2_std=1.0,
        # ── innovations (config-driven) ──
        probabilistic=m.get("probabilistic", False),
        sigma_floor=m.get("sigma_floor", 1e-3),
        use_mono_reg=m.get("use_mono_reg", False),
        mono_weight=m.get("mono_weight", 0.1),
        mono_delta=m.get("mono_delta", 0.05),
        mono_margin=m.get("mono_margin", 0.0),
        use_analog_memory=m.get("use_analog_memory", False),
        analog_top_k=m.get("analog_top_k", 8),
        analog_temperature=m.get("analog_temperature", 1.0),
        use_multiscale_co2=m.get("use_multiscale_co2", False),
        co2_anchor_a=m.get("co2_anchor_a", 0.0),
        co2_ref=m.get("co2_ref", 284.0),
        co2_min=m.get("co2_min", 280.0), co2_max=m.get("co2_max", 1200.0),
        use_short_residual=m.get("use_short_residual", False),
        short_residual_steps=m.get("short_residual_steps", 8),
        use_co2_timebias=m.get("use_co2_timebias", False),
        target_index=cfg.get("data", {}).get("target_index", 0),
        decoder_type=m.get("decoder_type", "calendar_mlp"),
        time_bias_scale=m.get("time_bias_scale", 1.0),
        shift_bias_scale=m.get("shift_bias_scale", 1.0),
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
        use_memory_horizon_gate=m.get("use_memory_horizon_gate", False),
        # The M1 gate-open experiment sets this to 0.0 (sigmoid(0)=0.5) so the
        # retrieved prototypes actually mix.  It MUST be forwarded: the wrapper
        # ends in **kwargs, so omitting it here makes `_check_forwarded` raise for
        # every config (the wrapper signature gained the parameter when the M1
        # patch landed, but the kw dict did not) -- i.e. the whole multiseed
        # runner stops working, not just the gate-open arm.
        memory_gate_init=m.get("memory_gate_init", -3.0),
        # Lives under `dataset` because the Trainer reads it there to decide
        # whether to emit future_co2 in the batch; the model must be built with
        # the same switch or the batch supplies an input the model ignores.
        use_future_co2=ds.get("use_future_co2", False),
    )
    _check_forwarded(kw)
    return MTTGNetWrapper(**kw).to(trainer.device)


def run_seed(cfg: dict, cfg_name: str, seed: int) -> dict:
    cfg = dict(cfg)
    cfg["seed"] = seed
    out_root = PROJECT_ROOT / "outputs" / f"{cfg_name}_s{seed}"
    cfg["output"] = {
        "checkpoint_dir": str(out_root / "checkpoints"),
        "result_dir": str(out_root / "results"),
    }
    seed_everything(seed)
    trainer = Trainer(cfg)
    device = trainer.device
    model = build_model(cfg, trainer)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    best_val, *_ = trainer.train(model, anchor_forward)
    pred, target = trainer.evaluate(model, anchor_forward)
    metrics = trainer.save_results(
        pred, target, n_params=n_params, best_val_loss=best_val)
    overall_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    print(f"  seed {seed}: RMSE={overall_rmse:.4f} R2={metrics['overall_r2']:.4f} "
          f"params={n_params} ({ (time.time()-t0)/60:.1f} min)")
    try:
        trainer.cleanup()
    except Exception:
        pass
    return {"name": cfg_name, "seed": seed,
            "rmse": overall_rmse, "r2": metrics["overall_r2"],
            "params": n_params, "time_min": round((time.time() - t0) / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="config yaml path")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    args = ap.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    base_cfg = load_config(cfg_path)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg_name = cfg_path.stem  # e.g. cpx_mttgnet_D_prob

    print(f"Multi-seed: {args.config}  seeds={seeds}")
    results = [r for r in (run_seed(base_cfg, cfg_name, s) for s in seeds) if r]

    rmse = np.array([r["rmse"] for r in results])
    summary = {
        "config": args.config,
        "seeds": seeds,
        "rmse_mean": float(rmse.mean()),
        "rmse_std": float(rmse.std()),
        "rmse_min": float(rmse.min()),
        "rmse_per_seed": [round(r["rmse"], 6) for r in results],
        "n_params": results[0]["params"],
    }
    out = PROJECT_ROOT / "outputs" / f"{cfg_name}_multiseed_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n════ Summary ({len(results)} seeds) ════")
    print(f"  RMSE = {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}"
          f"  (min {summary['rmse_min']:.4f}, params {summary['n_params']})")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
