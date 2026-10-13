#!/usr/bin/env python3
"""Monthly conditional temperature modeling with prescribed CO2 pathways.

Training membership and temporal masks precede normalization. All target windows
remain in their split; the shared rollout retrieves only origin-known history.
Legacy fitted log-CO2 slopes are descriptive associations, not causal climate
sensitivity. The bounded Discussion study uses --train-only, a matched input
control, and a separate FP32 rollout worker instead of these legacy diagnostics.

Research-suite entry point: bash scripts/run_cmip6_discussion.sh (not exported).
Public inference entry point: python scripts/project_cmip6.py --run-dir <run>.
Standalone LOSO remains available through --holdout; it is not part of that suite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_config                       # noqa: E402
from src.dataset import PeriodicAnchorDataset, make_origin_indices  # noqa: E402
from src.mttgnet import MTTGNetWrapper                   # noqa: E402
from src.validation_protocol import PROTOCOL_VERSION, scenario_windows, training_mask
from src.scaler import StandardScaler                     # noqa: E402
from src.seed import seed_everything                      # noqa: E402
from src.trainer import Trainer, anchor_forward           # noqa: E402


def load_series(spec: dict) -> tuple[np.ndarray, pd.DatetimeIndex]:
    values_p, ts_p = ROOT / spec["values"], ROOT / spec["timestamps"]
    for p in (values_p, ts_p):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}\n"
                f"  build it with:  python scripts/prepare_cmip6_tas.py\n"
                f"  (or sync data/processed/cmip6_tas/ from the machine that ran it)")
    values = np.load(values_p).astype(np.float32)
    # allow_pickle belongs to np.load — the timestamps are saved as an object
    # array, so it must go there, not to pd.to_datetime.
    stamps = np.load(ts_p, allow_pickle=True)
    ts = pd.DatetimeIndex(pd.to_datetime(stamps))
    return values, ts


def monthly_climatology(values: np.ndarray, ts: pd.DatetimeIndex, target_idx: int) -> np.ndarray:
    """Mean target by calendar month over the historical record.

    Used only for de-seasonalisation.  Fitting it on the historical period keeps
    the transform free of future information relative to the scenarios.
    """
    s = pd.Series(values[:, target_idx], index=ts)
    return s.groupby(s.index.month).mean().reindex(range(1, 13)).values.astype(np.float64)


def deseasonalise(values: np.ndarray, ts: pd.DatetimeIndex, target_idx: int,
                  clim: np.ndarray) -> np.ndarray:
    """Subtract the monthly climatology from the target column (in place-safe copy)."""
    out = values.copy()
    out[:, target_idx] = values[:, target_idx] - clim[ts.month.values - 1]
    return out


def build_split_datasets(cfg: dict, scaler: StandardScaler, *, holdout: str | None = None,
                         clim: np.ndarray | None = None):
    """One PeriodicAnchorDataset per scenario per split.

    Every scenario contributes its own origins inside each time window.  With
    `holdout`, that scenario is excluded from train/val and instead becomes the
    entire test set — the leave-one-scenario-out generalisation test.  With
    `clim`, the target is de-seasonalised before scaling.
    """
    dcfg = cfg["dataset"]
    L, H = int(dcfg["recent_length"]), int(dcfg["horizon"])
    co2_idx = cfg["data"].get("co2_index")
    tgt_idx = int(cfg["data"]["target_index"])
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    test_end = pd.Timestamp(cfg["split"]["test_end"])

    sets: dict[str, list] = {"train": [], "val": [], "test": []}
    report: list[tuple[str, dict]] = []

    for spec in dcfg["scenarios"]:
        raw, ts = load_series(spec)
        if clim is not None:
            raw = deseasonalise(raw, ts, tgt_idx, clim)
        scaled = scaler.transform(raw).astype(np.float32)
        if co2_idx is not None:
            # Same fixed [0,1] convention the Trainer uses (280-1200 ppm), so a
            # scenario CO2 value is never an out-of-distribution z-score.
            scaled[:, co2_idx] = (raw[:, co2_idx] - 280.0) / (1200.0 - 280.0)

        common = dict(
            values=scaled, timestamps=ts, recent_length=L, horizon=H,
            daily_anchors=dcfg.get("daily_anchors", 0),
            yearly_anchors=dcfg.get("yearly_anchors", 3),
            target_index=cfg["data"]["target_index"],
            time_encode=bool(dcfg.get("time_encode", False)),
            # Emits the CO2 at every target step when the future-forcing input
            # is enabled (see dataset.use_future_co2 in the config).
            co2_index=co2_idx if dcfg.get("use_future_co2") else None,
        )

        windows = scenario_windows(cfg, spec['name'], ts, holdout)

        counts = {}
        for split, win in windows.items():
            if win is None:
                counts[split] = 0
                continue
            start, end, step = win
            origins = make_origin_indices(ts, start, end, L, H, step)
            counts[split] = int(len(origins))
            if len(origins):
                sets[split].append(
                    PeriodicAnchorDataset(origin_indices=origins, **common))
        report.append((spec["name"], counts))

    return sets, report


def build_model(cfg: dict, num_features: int) -> MTTGNetWrapper:
    from src.fusion_config import record_fusion_config
    record_fusion_config(cfg)
    m, d = cfg["model_params"], cfg["dataset"]
    # 2026-10: arch=clean dispatch.  Clean checkpoints carry the mttgnet_test
    # module layout; loading them into the legacy wrapper with strict=False
    # would silently train a random-initialised model.  The clean model also
    # needs the REAL ppm range to reconstruct CO2 for the counterfactual
    # reference — the data pipeline scales CO2 to [0,1] over 280-1200 ppm, so
    # the identity range (0/1) the legacy path uses would be wrong there.
    if m.get("arch", "mttgnet") == "clean":
        from src.mttgnet_test import MTTGNetTestWrapper
        return MTTGNetTestWrapper(
            num_features=num_features,
            hidden_dim=m["hidden_dim"],
            horizon=d["horizon"],
            seq_len=d["recent_length"],
            dropout=m.get("dropout", 0.1),
            memory_slots=m.get("memory_slots", 128),
            mem_top_k=m.get("mem_top_k", 8),
            num_gru_layers=m.get("num_gru_layers", 2),
            use_gate=m.get("use_gate", True),
            use_time_bias=m.get("use_time_bias", True),
            use_memory=m.get("use_memory", True),
            use_variable_encoder=m.get("use_variable_encoder", True),
            use_shift_encoder=m.get("use_shift_encoder", True),
            use_var_gnn=m.get("use_var_gnn", True),
            use_calendar=m.get("use_calendar", True),
            var_encoder_type=m.get("var_encoder_type", "gat"),
            gat_version=m.get("gat_version", "gat"),
            query_refine=m.get("query_refine", True),
            mem_clean=m.get("mem_clean", True),
            decoder_bypass=m.get("decoder_bypass", False),
            time_bias_mode=m.get("time_bias_mode", "additive"),
            co2_mode=m.get("co2_mode", "counterfactual"),
            use_multiscale_co2=m.get("use_multiscale_co2", False),
            co2_anchor_a=m.get("co2_anchor_a", 0.0),
            co2_index=cfg["data"].get("co2_index", 1),
            co2_mean=0.0, co2_std=1.0,
            co2_min=m.get("co2_min", 280.0), co2_max=m.get("co2_max", 1200.0),
            # Opt-in known-future-forcing input; the same key drives the dataset's
            # co2_index above, so model and data can never disagree.
            use_future_co2=d.get("use_future_co2", False),
            co2_attention_modulation=m.get('co2_attention_modulation',True),
            co2_additive_bias=m.get('co2_additive_bias',True),
            fusion_sources=m['fusion_sources'],
        )
    return MTTGNetWrapper(
        num_features=num_features,
        hidden_dim=m["hidden_dim"],
        horizon=d["horizon"],
        seq_len=d["recent_length"],
        dropout=m.get("dropout", 0.1),
        memory_slots=m.get("memory_slots", 128),
        mem_top_k=m.get("mem_top_k", 8),
        num_gru_layers=m.get("num_gru_layers", 2),
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
        co2_index=cfg["data"].get("co2_index", 1),
        # x_recent already carries CO2 in fixed [0,1] (see build_split_datasets).
        co2_mean=0.0, co2_std=1.0, co2_min=0.0, co2_max=1.0,
        # Opt-in known-future-forcing input; the same key drives the dataset's
        # co2_index above, so model and data can never disagree.
        use_future_co2=d.get("use_future_co2", False),
    )


def sensitivity_diagnostic(model, trainer, cfg, *, holdout=None, clim=None,
                           save_dir: Path | None = None,
                           trained: set[str] | None = None) -> dict:
    """Roll the trained model over each SSP CO2 path and fit T ~ a*ln(CO2).

    The slope summarizes a path-dependent temperature/CO2 association. It is
    not an isolated forcing effect or a climate-sensitivity estimate. Compare
    trajectory errors separately from this descriptive legacy diagnostic.
    """
    from src.scenario_rollout import autoregressive_project

    dcfg = cfg["dataset"]
    L, H = int(dcfg["recent_length"]), int(dcfg["horizon"])
    co2_idx = cfg["data"].get("co2_index")

    hist_spec = next(s for s in dcfg["scenarios"] if s["name"] == "historical")
    hist_v, hist_t = load_series(hist_spec)
    tgt_idx = int(cfg["data"]["target_index"])
    if clim is not None:
        hist_v = deseasonalise(hist_v, hist_t, tgt_idx, clim)
    scaler = trainer.scaler
    device = trainer.device
    model.eval()

    out: dict = {"a_model": None, "per_ssp": {}, "a_cmip6": {}}
    lnc_all, pred_all = [], []

    paths = {s["name"]: s for s in dcfg["scenarios"] if s["name"] != "historical"}
    for name, spec in paths.items():
        if holdout is not None and name != holdout:
            continue          # in leave-one-out mode, score only the held-out one
        ssp_v, ssp_t = load_series(spec)
        if clim is not None:
            ssp_v = deseasonalise(ssp_v, ssp_t, tgt_idx, clim)
        co2_path = ssp_v[:, co2_idx].astype(np.float64)

        pred = autoregressive_project(
            model, len(hist_v) - 1, co2_path, hist_v, hist_t, scaler, device,
            L=L, H=H,
            # The scenario's CO2 pathway is known by construction, so hand the
            # model the forcing at each target step instead of making it infer
            # the pathway from the look-back window.
            future_co2_ppm=co2_path if dcfg.get("use_future_co2") else None,
            daily_anchors=int(dcfg.get("daily_anchors", 0)),
            yearly_anchors=int(dcfg.get("yearly_anchors", 3)))

        a_m, b_m = np.polyfit(np.log(co2_path), pred, 1)
        a_c, b_c = np.polyfit(np.log(co2_path), ssp_v[:, tgt_idx].astype(np.float64), 1)

        # ── projection skill on the quantity the paper actually reports ────
        # `a` above is a path-dependent diagnostic (it swings with the shape of
        # each scenario's CO2 path), so the 2081-2100 warming is measured
        # directly.  Both sides use the same pre-industrial reference, so the
        # difference is a genuine model error, not a baseline artefact.
        fut = ((ssp_t >= pd.Timestamp("2081-01-15"))
               & (ssp_t <= pd.Timestamp("2100-12-15")))
        pi = ((hist_t >= pd.Timestamp("1850-01-15"))
              & (hist_t <= pd.Timestamp("1900-12-15")))
        pi_mean = float(hist_v[pi, tgt_idx].mean())
        warm_model = float(np.asarray(pred)[fut].mean() - pi_mean)
        warm_truth = float(ssp_v[fut, tgt_idx].mean() - pi_mean)

        out["per_ssp"][name] = {
            "rollout_rmse_K": float(np.sqrt(np.mean((pred - ssp_v[:, tgt_idx]) ** 2))),
            "a_model": float(a_m), "b_model": float(b_m),
            "a_cmip6": float(a_c), "b_cmip6": float(b_c),
            "n_months": int(len(pred)),
            "warming_model": warm_model,
            "warming_cmip6": warm_truth,
            "bias": warm_model - warm_truth,
        }
        lnc_all.append(np.log(co2_path))
        pred_all.append(pred)

        # Persist the trajectory itself: the ensemble and observational-constraint
        # steps need each member's projected path, not just its fitted slope.
        if save_dir is not None:
            np.save(save_dir / f"proj_{name}.npy", pred.astype(np.float32))
            np.save(save_dir / f"proj_co2_{name}.npy", co2_path.astype(np.float32))
            np.save(save_dir / f"proj_dates_{name}.npy",
                    np.asarray(ssp_t, dtype="datetime64[ns]"))
            if clim is None:
                np.save(save_dir / f"proj_truth_{name}.npy",
                        ssp_v[:, tgt_idx].astype(np.float32))

    if lnc_all:
        a_m, b_m = np.polyfit(np.concatenate(lnc_all), np.concatenate(pred_all), 1)
        out["a_model"] = float(a_m)
        out["b_model"] = float(b_m)
        out["a_per_doubling"] = float(a_m * np.log(2))

    # ── projection summary ────────────────────────────────────────────────
    # The spread ratio is the headline number for this experiment: a model that
    # has learned the forced response must reproduce the ACROSS-SCENARIO spread,
    # not just each scenario's level.  A ratio near 1 means the scenarios are
    # separated correctly; a small ratio is the "compression" failure.
    def _spread(names):
        wm = [out["per_ssp"][n]["warming_model"] for n in names]
        wt = [out["per_ssp"][n]["warming_cmip6"] for n in names]
        b = [out["per_ssp"][n]["bias"] for n in names]
        sm, st = max(wm) - min(wm), max(wt) - min(wt)
        return {
            "scenarios": list(names),
            "spread_model": float(sm), "spread_cmip6": float(st),
            "spread_ratio": float(sm / st) if st else None,
            "mean_bias": float(np.mean(b)),
            "max_abs_bias": float(np.max(np.abs(b))),
            "bias_sign_changes": bool(min(b) < 0 < max(b)),
        }

    if len(out["per_ssp"]) >= 2:
        all_names = sorted(out["per_ssp"])
        out["projection"] = _spread(all_names)
        # A ratio that mixes scenarios the model was trained on with scenarios it
        # never saw is not interpretable: the in-sample points drag it toward the
        # truth.  Report the held-out subset separately whenever there are at
        # least two of them, because that is the number the paper can quote.
        unsel = [n for n in all_names if trained is None or n not in trained]
        out["projection"]["trained_scenarios"] = sorted(trained or [])
        if len(unsel) >= 2:
            out["projection"]["heldout"] = _spread(unsel)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/example_century.yaml"))
    ap.add_argument("--holdout", default=None,
                    help="scenario name to hold out entirely (leave-one-scenario-out)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="override dataset.horizon (direct multi-step). The H=1 default "
                         "makes the forced trend ~4%% of the forecast noise, so MSE "
                         "training discards it; H=120 raises that signal-to-noise ~11x.")
    ap.add_argument("--deseasonalize", action="store_true",
                    help="subtract the historical monthly climatology from the target "
                         "before training. NOTE: this does NOT improve the forced "
                         "trend's signal-to-noise — the seasonal cycle is predictable "
                         "signal, not noise, so removing it scales trend and noise "
                         "together (S/N stays ~0.44 at H=120). It is offered only as a "
                         "capacity experiment: with the cycle removed the network need "
                         "not spend parameters fitting it.")
    ap.add_argument("--model", default=None,
                    help="CMIP6 source_id to train on; rewrites the scenario paths to "
                         "data/processed/cmip6_tas/<model>/ so one config serves every ESM")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg.seed, so one configuration can be repeated "
                         "under several seeds for a spread estimate")
    ap.add_argument("--tag", default=None,
                    help="extra suffix for the auto-generated output directory, e.g. "
                         "--tag h600split keeps control runs from overwriting the "
                         "main ones in outputs/")
    ap.add_argument("--split-train", default=None,
                    help="comma-separated scenario names for split.train (forces "
                         "scenario mode; overrides the config)")
    ap.add_argument("--split-val", default=None, help="... for split.val")
    ap.add_argument("--split-test", default=None, help="... for split.test")
    ap.add_argument("--out", default=None, help="override the result directory")
    ap.add_argument('--temperature-history', choices=['full', 'masked'], default='full')
    ap.add_argument('--train-only', action='store_true', help='Save weights/scaler/curves; separate rollout worker evaluates them')
    ap.add_argument('--max-training-seconds', type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.temperature_history == 'masked' and cfg['data']['target_index'] != 0:
        raise ValueError('The Discussion temperature-history control requires target_index=0')
    if args.max_training_seconds is not None:
        cfg['training']['max_training_seconds'] = args.max_training_seconds
    cfg['discussion'] = {'temperature_history': args.temperature_history,
                         'independent_scenario_validation': False if not args.holdout else True}
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.horizon is not None:
        cfg["dataset"]["horizon"] = int(args.horizon)
    if any((args.split_train, args.split_val, args.split_test)):
        cfg["split"]["mode"] = "scenario"
        for key, val in (("train", args.split_train), ("val", args.split_val),
                         ("test", args.split_test)):
            if val:
                cfg["split"][key] = [s.strip() for s in val.split(",") if s.strip()]

    model_name = args.model
    if model_name is None:                      # infer from the config's paths
        first = cfg["dataset"]["scenarios"][0]["values"]
        model_name = first.split("cmip6_tas/")[1].split("/")[0]
    else:
        base = f"data/processed/cmip6_tas/{model_name}"
        for spec in cfg["dataset"]["scenarios"]:
            spec["values"] = f"{base}/{spec['name']}_values.npy"
            spec["timestamps"] = f"{base}/{spec['name']}_timestamps.npy"

    seed_everything(cfg["seed"])

    tag = f"cmip6_pure_ml_{model_name}_h{cfg['dataset']['horizon']}_s{cfg['seed']}"
    if args.deseasonalize:
        tag += "_ds"
    if args.holdout:
        tag += f"_ho_{args.holdout}"
    if args.tag:
        tag += f"_{args.tag}"
    tag += "_independent_v2"
    out_dir = Path(args.out) if args.out else ROOT / "outputs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # Every ESM / holdout / seed must own its checkpoint, including early-stop
    # reloads. A shared config output path would silently overwrite other folds.
    cfg['output'] = {'checkpoint_dir': str(out_dir / 'checkpoints'),
                     'result_dir': str(out_dir / 'results')}

    # ── scaler: fit on the historical period only ──────────────────────────
    # The Trainer fits on its primary series' train slice, which for a
    # continuous historical+SSP file would leak future standardisation stats.
    # Refit on historical and hand the result back so forward transform and
    # inverse transform stay consistent.  With --deseasonalize the climatology is
    # taken from the historical period too, so nothing future leaks in.
    dcfg = cfg["dataset"]
    hist_spec = next(s for s in dcfg["scenarios"] if s["name"] == "historical")
    hist_v, hist_ts = load_series(hist_spec)
    tgt_idx = int(cfg["data"]["target_index"])
    clim = None
    if args.deseasonalize:
        hist_mask = training_mask(cfg, "historical", hist_ts, args.holdout)
        if not hist_mask.any():
            raise ValueError("Historical climatology requires historical training data")
        clim = monthly_climatology(hist_v[hist_mask], hist_ts[hist_mask], tgt_idx)
        hist_v = deseasonalise(hist_v, hist_ts, tgt_idx, clim)

    # Fit only records allowed by the training membership and time boundary.
    fit_scope = dcfg.get('scaler_fit', 'train')
    if fit_scope not in ('train', 'historical'):
        raise ValueError('scaler_fit must be train or historical')
    parts, fit_records = [], {}
    for spec in dcfg['scenarios']:
        if fit_scope == 'historical' and spec['name'] != 'historical':
            continue
        sv, st = load_series(spec)
        mask = training_mask(cfg, spec['name'], st, args.holdout)
        if not mask.any():
            continue
        if clim is not None:
            sv = deseasonalise(sv, st, tgt_idx, clim)
        parts.append(sv[mask])
        fit_records[spec['name']] = {'count': int(mask.sum()), 'first': str(st[mask][0]), 'last': str(st[mask][-1])}
    if not parts:
        raise ValueError('No training data available to fit normalization')
    scaler = StandardScaler().fit(np.concatenate(parts, axis=0))
    np.savez(out_dir / 'scaler.npz', mean=scaler.mean, std=scaler.std)
    n_fit = sum(len(p) for p in parts)
    protocol = {'version': PROTOCOL_VERSION, 'holdout': args.holdout,
                'split': cfg['split'], 'normalization_records': fit_records,
                'scaler_mean': scaler.mean.tolist(), 'scaler_std': scaler.std.tolist(),
                'rollout_daily_anchors': int(dcfg.get('daily_anchors', 0)),
                'rollout_yearly_anchors': int(dcfg.get('yearly_anchors', 3))}
    (out_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (out_dir / 'resolved_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')

    print(f"{'='*64}\n  CMIP6 pure-ML projection — {'holdout: ' + args.holdout if args.holdout else 'all scenarios'}\n{'='*64}")
    print(f"  model          : {hist_spec['values'].split('/')[-2]}")
    print(f"  horizon        : {cfg['dataset']['horizon']} months")
    print(f"  de-seasonalised: {clim is not None}")
    print(f"  scaler fit     : {fit_scope} ({n_fit} months)")

    # Which scenarios contributed training data INSIDE the 2081-2100 projection
    # window.  Only those make a projection in-sample; in time mode the training
    # targets stop at split.train_end (2060) so none of them do, even though
    # every scenario was seen earlier in its own history.
    if cfg["split"].get("mode") == "scenario":
        trained_in_window = set(cfg["split"].get("train", []) or [])
    else:
        trained_in_window = set()
    print(f"  trained in 2081-2100 window: "
          f"{sorted(trained_in_window) if trained_in_window else '(none)'}")

    # ── datasets ───────────────────────────────────────────────────────────
    sets, report = build_split_datasets(cfg, scaler, holdout=args.holdout, clim=clim)
    for name, counts in report:
        print(f"  {name:11s} train={counts['train']:6d}  val={counts['val']:6d}  test={counts['test']:6d}")
    if not sets["train"]:
        raise SystemExit("no training origins — check split windows and scenario coverage")

    from torch.utils.data import ConcatDataset, DataLoader
    tcfg = cfg["training"]
    loaders = {}
    for split in ("train", "val", "test"):
        if not sets[split]:
            loaders[split] = None
            continue
        ds = ConcatDataset(sets[split])
        loaders[split] = DataLoader(
            ds, batch_size=tcfg["batch_size"], shuffle=(split == "train"),
            num_workers=tcfg.get("num_workers", 0),
            pin_memory=tcfg.get("pin_memory", False),
            persistent_workers=False)
        print(f"  {split:5s} loader: {len(ds)} windows")

    # ── Trainer supplies the training recipe only ─────────────────────────
    # `_skip_data_init` stops it from building its own single-series datasets;
    # we hand it everything `train`/`evaluate`/`save_results` read back:
    # num_features, horizon, target_index, use_time_encode, scaler, loaders.
    trainer = Trainer(cfg, _skip_data_init=True)
    trainer.scaler = scaler
    trainer.num_features = sets["train"][0].num_features
    trainer.horizon = int(dcfg["horizon"])
    trainer.target_index = int(cfg["data"]["target_index"])
    trainer.use_time_encode = bool(dcfg.get("time_encode", False))
    if loaders["val"] is None:
        raise SystemExit(
            "\n  No validation origins.\n"
            "  Trainer.train() always reads loaders['val'] for early stopping; at\n"
            "  long horizons the time split leaves val empty (a 600-month target\n"
            "  started in 2090 would end in 2140).\n"
            "  Fix: pass a config that splits by SCENARIO, e.g.\n"
            "      --config configs/cmip6_pure_ml_h600.yaml\n"
            "  or set `split.mode: scenario` with explicit split.train/val/test\n"
            "  scenario lists (see experiments/cmip6/DESIGN.md §4).")
    trainer.loaders["train"] = loaders["train"]
    trainer.loaders["val"] = loaders["val"]
    if loaders["test"] is not None:
        trainer.loaders["test"] = loaders["test"]

    model = build_model(cfg, trainer.num_features)
    model = model.to(trainer.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params    : {n_params:,}   device: {trainer.device}")

    # ── train ──────────────────────────────────────────────────────────────
    t0 = time.time()
    from src.temperature_history import masked_anchor_forward
    forward = masked_anchor_forward if args.temperature_history == 'masked' else anchor_forward
    if args.temperature_history == 'masked' and not args.train_only:
        raise ValueError('Masked history requires --train-only and the matching Discussion rollout worker')
    best_val, train_losses, val_losses = trainer.train(model, forward)
    train_min = (time.time() - t0) / 60
    print(f"\n  best val loss {best_val:.6f}   train time {train_min:.1f} min"
          f"   epochs {len(val_losses)}/{tcfg.get('epochs', '?')}"
          f" (patience {tcfg.get('patience', '?')})")

    # ── test-set evaluation (in-distribution windows) ─────────────────────
    summary: dict = {"n_params": n_params, "best_val_loss": best_val,
                     'temperature_history': args.temperature_history,
                     'training_stop_reason': trainer.training_stop_reason,
                     "train_time_min": train_min, "holdout": args.holdout,
                     "config": str(args.config),
                     "protocol": protocol,
                     # Provenance for the CO2-pathway 2x2 (paper section V-G,
                     # option 3).  Without these the four arms are distinguishable
                     # only by their output directory name, so a mis-wired or
                     # silently-ignored key would be undetectable from the
                     # artifact itself -- which is exactly the failure mode the
                     # rest of this project's audits kept hitting.
                     "model": model_name,
                     "seed": int(cfg["seed"]),
                     "tag": args.tag,
                     "use_future_co2": bool(dcfg.get("use_future_co2", False)),
                     "use_shift_encoder": bool(
                         cfg["model_params"].get("use_shift_encoder", True)),
                     "horizon": int(dcfg["horizon"]),
                     "recent_length": int(dcfg["recent_length"]),
                     "deseasonalized": bool(clim is not None),
                     # Recorded so a run's normalisation is auditable from the
                     # summary alone; the scaler scope measurably moves both
                     # convergence and the projected scenario spread.
                     "scaler_fit": fit_scope,
                     "epochs_cap": int(tcfg.get("epochs", 0)),
                     "epochs_run": len(val_losses),
                     "patience": int(tcfg.get("patience", 0)),
                     "train_origins": {n: c["train"] for n, c in report},
                     "val_origins": {n: c["val"] for n, c in report},
                     "test_origins": {n: c["test"] for n, c in report}}
    # Keep the curves: without them an early-stopped run is indistinguishable
    # from a converged one in the summary.
    if train_losses:
        np.save(out_dir / "train_losses.npy", np.asarray(train_losses))
        np.save(out_dir / "val_losses.npy", np.asarray(val_losses))
    if args.train_only:
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('Training artifacts saved:', out_dir, flush=True)
        trainer.cleanup()
        return
    if loaders["test"] is not None:
        pred, target = trainer.evaluate(model, anchor_forward)
        trainer.save_results(pred, target, n_params=n_params, best_val_loss=best_val, result_dir_override=out_dir / "results")
        rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        r2 = float(1 - ((pred - target) ** 2).sum() / ((target - target.mean()) ** 2).sum())
        summary.update({"test_rmse": rmse, "test_r2": r2,
                        "test_n": int(len(target))})
        print(f"  test RMSE {rmse:.4f} K   R2 {r2:.4f}   (n={len(target)})")

    # ── the number that matters: learned CO2 sensitivity ──────────────────
    diag = sensitivity_diagnostic(model, trainer, cfg, holdout=args.holdout, clim=clim,
                                  save_dir=out_dir, trained=trained_in_window)
    summary["sensitivity"] = diag
    print(f"\n{'='*72}\n  CENTURY PROJECTION: 2081-2100 warming vs this ESM's own\n{'='*72}")
    print(f"  {'scenario':10s} {'model':>8s} {'CMIP6':>8s} {'bias':>8s}   "
          f"{'a_model':>8s} {'a_cmip6':>8s}")
    for name, d in diag["per_ssp"].items():
        print(f"  {name:10s} {d['warming_model']:8.2f} {d['warming_cmip6']:8.2f} "
              f"{d['bias']:+8.2f}   {d['a_model']:8.3f} {d['a_cmip6']:8.3f}")
    proj = diag.get("projection")

    def _spread_line(tag, d):
        print(f"  {tag:16s}: model {d['spread_model']:5.2f} K   "
              f"CMIP6 {d['spread_cmip6']:5.2f} K   ratio {d['spread_ratio']:.2f}"
              f"   |  bias mean {d['mean_bias']:+.2f}  max|bias| {d['max_abs_bias']:.2f}"
              f"   sign changes: {d['bias_sign_changes']}")

    if proj:
        print("\n  scenario spread over " + (
            "ALL scenarios" if not proj.get("heldout") else
            "all / held-out scenarios only"))
        _spread_line("all scenarios", proj)
        if proj.get("heldout"):
            _spread_line("held-out only", proj["heldout"])
            print("      ^ the held-out line is the quotable one: a ratio that "
                  "mixes scenarios\n        the model trained on with ones it "
                  "never saw is not interpretable.")
    if diag["a_model"] is not None:
        print(f"\n  pooled a_model  = {diag['a_model']:.3f} K/ln(CO2)"
              f"  ({diag['a_per_doubling']:.2f} K per doubling)")
        print("  reference       : observed HadCRUT5 3.459; observation-only MTTGNet 0.113;"
              " the per-scenario a_cmip6 column above is this ESM's own value")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  saved -> {out_dir / 'summary.json'}")
    trainer.cleanup()


if __name__ == "__main__":
    main()
