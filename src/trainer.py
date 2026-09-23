"""Unified training engine for all MTTGNet models.

Replaces duplicated logic across train_weather.py, train_explore.py,
train_pipeline.py, and train_tslib.py.

The key abstraction is ``forward_fn(model, batch) -> dict`` —
each model type provides its own, the engine handles everything else.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .dataset import PeriodicAnchorDataset, make_origin_indices
from .losses import MultiHorizonForecastLoss, ShashLoss, DBLoss
from .metrics import compute_all_metrics, compute_correlation, horizon_metrics
from .scaler import StandardScaler
from .seed import seed_everything


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_root_path(p) -> Path:
    """Resolve a (possibly relative) config path against the project root.

    The YAML configs use paths like ``data/processed/...`` and
    ``outputs/...``.  Resolving them against the project root (not the CWD)
    makes training reproducible from any working directory — important because
    the project is run on a remote server where the CWD is not guaranteed.
    """
    p = Path(p)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically.

    Without this, each worker seeds itself from PID+time, making shuffling
    non-reproducible when ``num_workers > 0``.
    """
    # Use the global seed (set by seed_everything before Trainer init) plus
    # the worker id so each worker gets a unique but deterministic sequence.
    import random
    worker_seed = (int(torch.initial_seed()) + worker_id) % (2 ** 32 - 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _write_summary(path: Path, *, n_params: int, best_val_loss: float,
                   overall_rmse: float, overall_r2: float,
                   rmse_h: np.ndarray, corr_h: np.ndarray,
                   horizon: int) -> None:
    """Write unified summary.txt."""
    with open(path, "w") as f:
        f.write(f"n_params: {n_params}\n")
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        f.write(f"overall_rmse: {overall_rmse:.6f}\n")
        f.write(f"overall_r2: {overall_r2:.6f}\n")
        for h in range(horizon):
            f.write(f"step_{h+1}_rmse: {rmse_h[h]:.6f}\n")
            f.write(f"step_{h+1}_corr: {corr_h[h]:.6f}\n")


# ═══════════════════════════════════════════════════════════════════
# Shared training / evaluation loops
# ═══════════════════════════════════════════════════════════════════

def run_epoch(
    model: nn.Module,
    loader: Iterable,
    forward_fn: Callable[[nn.Module, dict], dict[str, torch.Tensor]],
    criterion,
    device: torch.device,
    optimizer: "torch.optim.Optimizer | None" = None,
    gradient_clip: float | None = None,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    short_lead_steps: int = 0,
    short_lead_weight: float = 0.5,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    scaler = None
    if training and use_amp and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")          # torch >= 2.x (recommended)
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler()           # fallback for older torch

    totals = {"loss": 0.0}
    count = 0
    accum_counter = 0

    for batch in tqdm(loader, leave=False):
        # Handle both dict batches and tuple batches (TSLib)
        if isinstance(batch, (tuple, list)):
            batch = {"_tslib": [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]}
        else:
            batch = move_batch_to_device(batch, device)

        if "_tslib" in batch:
            batch_size = batch["_tslib"][0].shape[0]  # TSLib: x tensor first dim
        elif isinstance(batch.get("target"), torch.Tensor):
            batch_size = batch["target"].shape[0]
        else:
            batch_size = 1  # fallback

        if training and accum_counter == 0:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            ctx = torch.amp.autocast("cuda") if (scaler is not None) else \
                  torch.no_grad() if not training else torch.enable_grad()
            with ctx:
                output = forward_fn(model, batch)
                if isinstance(criterion, ShashLoss):
                    # SHASH head: the model returns (mu, sigma, gamma, tau)
                    # parameters; the loss consumes the full parameter tensor.
                    loss_dict = criterion(output["dist_params"], _extract_target(batch))
                else:
                    loss_dict = criterion(output["prediction"], _extract_target(batch))
                if isinstance(loss_dict, torch.Tensor):
                    loss_dict = {"loss": loss_dict}
                # Auxiliary losses exposed by the model (e.g. the CO2
                # monotonicity regularisation).  The model has already applied
                # its weight; we simply add the (scalarised) term to the loss.
                aux = output.get("aux_loss")
                if aux is not None:
                    aux = aux.mean()
                    loss_dict["loss"] = loss_dict["loss"] + aux
                    loss_dict["aux_loss"] = aux.detach()

            # Short-lead multi-task: force the model to fit the first few steps
            # (anti-starvation under long-horizon training) via an extra MSE term.
            if short_lead_steps > 0:
                _p = output["prediction"]
                _t = _extract_target(batch)
                if _p.shape[1] > short_lead_steps:
                    # cast target to the model-output dtype (Float under AMP → Half) to
                    # avoid "Found dtype Float but expected Half" during backward.
                    _sl = torch.nn.functional.mse_loss(
                        _p[:, :short_lead_steps], _t[:, :short_lead_steps].to(_p.dtype))
                    loss_dict["loss"] = loss_dict["loss"] + \
                        short_lead_weight * _sl.to(loss_dict["loss"].dtype)
                    loss_dict["short_lead_loss"] = _sl.detach()

            # NaN/Inf guard — covers both training and validation.
            # In training: reset grads + accum, skip the batch.
            # In validation: skip the batch so NaN doesn't poison totals.
            loss_val = loss_dict["loss"] / grad_accum_steps
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                continue

            if training:
                if scaler is not None:
                    scaler.scale(loss_val).backward()
                else:
                    loss_val.backward()

                accum_counter += 1
                if accum_counter >= grad_accum_steps:
                    _optimizer_step(model, optimizer, scaler, gradient_clip)
                    accum_counter = 0

        for key in loss_dict:
            totals.setdefault(key, 0.0)
            totals[key] += float(loss_dict[key].detach().cpu()) * batch_size
        count += batch_size

    if training and accum_counter > 0:
        _optimizer_step(model, optimizer, scaler, gradient_clip)
        optimizer.zero_grad(set_to_none=True)

    if count == 0:
        # Every batch was NaN/Inf-skipped (or the loader was empty).  Return a
        # loud inf instead of {} — callers index result["loss"] directly and
        # would crash on a missing key.  inf is never < best_val, so the
        # early-stop comparison stays safe.
        return {"loss": float("inf")}
    return {k: v / count for k, v in totals.items()}


def _extract_target(batch: dict) -> torch.Tensor:
    """Extract target tensor from batch (handles both dict and TSLib tuple batches)."""
    if "_tslib" in batch:
        return batch["_tslib"][1]  # (x, y) tuple
    return batch["target"]


def _optimizer_step(model, optimizer, scaler, gradient_clip):
    """Safe optimizer step with optional gradient clipping and scaler."""
    if gradient_clip is not None:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: Iterable,
    forward_fn: Callable[[nn.Module, dict], dict[str, torch.Tensor]],
    device: torch.device,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []

    for batch in tqdm(loader, leave=False):
        if isinstance(batch, (tuple, list)):
            batch = {"_tslib": [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]}
        else:
            batch = move_batch_to_device(batch, device)

        ctx = torch.amp.autocast("cuda") if (use_amp and device.type == "cuda") else torch.no_grad()
        with ctx:
            output = forward_fn(model, batch)

        # IMPORTANT: under AMP the model output can be float16.  Casting to
        # float32 before numpy prevents (a) precision loss at temperature
        # magnitudes (fp16 ulp ~0.125 K at 288 K, which alone inflates RMSE
        # by ~0.05 K), and (b) float16 overflow in metric reduction (which
        # produced impossible Corr > 1 values).  Metrics must run in fp32+.
        preds.append(output["prediction"].float().cpu().numpy())
        targets.append(_extract_target(batch).float().cpu().numpy())

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


@torch.no_grad()
def collect_probabilistic_predictions(
    model: nn.Module,
    loader: Iterable,
    forward_fn: Callable[[nn.Module, dict], dict[str, torch.Tensor]],
    device: torch.device,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like ``collect_predictions`` but also returns SHASH ``dist_params``.

    Returns ``(mu, target, dist_params)`` where ``mu`` is the point/location
    forecast [N, H], ``target`` the observed window [N, H] and ``dist_params``
    the [N, H, 4] SHASH parameters, all in *normalised* space.  Callers should
    inverse-transform via ``Trainer.evaluate_probabilistic``.
    """
    model.eval()
    preds, targets, dists = [], [], []

    for batch in tqdm(loader, leave=False):
        if isinstance(batch, (tuple, list)):
            batch = {"_tslib": [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]}
        else:
            batch = move_batch_to_device(batch, device)

        ctx = torch.amp.autocast("cuda") if (use_amp and device.type == "cuda") else torch.no_grad()
        with ctx:
            output = forward_fn(model, batch)

        dp = output.get("dist_params")
        if dp is None:
            raise RuntimeError(
                "Model did not return dist_params — build it with "
                "probabilistic=True for probabilistic evaluation.")
        # Cast to float32 — see collect_predictions for why (AMP emits fp16,
        # which destroys metric precision and overflows the correlation).
        preds.append(output["prediction"].float().cpu().numpy())
        targets.append(_extract_target(batch).float().cpu().numpy())
        dists.append(dp.float().cpu().numpy())

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (np.concatenate(preds, axis=0),
            np.concatenate(targets, axis=0),
            np.concatenate(dists, axis=0))


# ═══════════════════════════════════════════════════════════════════
# Trainer class
# ═══════════════════════════════════════════════════════════════════

class Trainer:
    """Unified trainer — handles data loading, training, evaluation, saving."""

    def __init__(self, config: dict, *, device: torch.device | None = None,
                 _skip_data_init: bool = False):
        self.config = config
        self.device = device or self._resolve_device()

        self.num_features: int = 0
        self.horizon: int = 0
        self.target_index: int = 0
        self.scaler: StandardScaler | None = None
        self.loaders: dict[str, DataLoader] = {}

        if not _skip_data_init:
            self._init_data()

    # ── device ──

    def _resolve_device(self) -> torch.device:
        d = self.config.get("device", "cuda")
        if d == "cuda" and not torch.cuda.is_available():
            d = "cpu"
        return torch.device(d)

    # ── data ──

    def _init_data(self) -> None:
        """Load data, fit scaler, create dataloaders."""
        cfg = self.config
        root = Path(__file__).resolve().parent.parent

        def _resolve(p: str) -> Path:
            p = Path(p)
            return p if p.is_absolute() else root / p

        values = np.load(_resolve(cfg["data"]["values_path"])).astype(np.float32)
        ts_raw = np.load(_resolve(cfg["data"]["timestamps_path"]), allow_pickle=True)
        timestamps = pd.to_datetime(ts_raw)

        self.num_features = values.shape[1]
        dataset_cfg = cfg["dataset"]
        self.horizon = dataset_cfg["horizon"]
        self.target_index = cfg["data"]["target_index"]

        # Scaler
        train_end = pd.Timestamp(cfg["split"]["train_end"])
        self.scaler = StandardScaler().fit(values[timestamps <= train_end])
        values_s = self.scaler.transform(values).astype(np.float32)
        # Fix CO2 column with [0,1] normalization using a fixed range
        # (280–1200 ppm).  Z-scoring to training stats makes test-period
        # CO2 look like +5σ outliers, collapsing the GRU output.
        co2_idx = cfg["data"].get("co2_index")
        if co2_idx is not None:
            co2_raw = values[:, co2_idx].astype(np.float32)
            values_s[:, co2_idx] = (co2_raw - 280.0) / (1200.0 - 280.0)

        # Splits
        train_origins = make_origin_indices(
            timestamps, timestamps[0], cfg["split"]["train_end"],
            dataset_cfg["recent_length"], self.horizon, dataset_cfg["train_step_size"])
        val_start = train_end + pd.Timedelta(hours=6)
        val_origins = make_origin_indices(
            timestamps, val_start, cfg["split"]["val_end"],
            dataset_cfg["recent_length"], self.horizon, dataset_cfg["eval_step_size"])
        test_start = pd.Timestamp(cfg["split"]["val_end"]) + pd.Timedelta(hours=6)
        test_origins = make_origin_indices(
            timestamps, test_start, cfg["split"]["test_end"],
            dataset_cfg["recent_length"], self.horizon, dataset_cfg["eval_step_size"])

        # Dataset
        self.use_time_encode: bool = dataset_cfg.get("time_encode", False)
        common = dict(values=values_s, timestamps=timestamps,
                      recent_length=dataset_cfg["recent_length"],
                      horizon=self.horizon,
                      daily_anchors=dataset_cfg.get("daily_anchors", 7),
                      yearly_anchors=dataset_cfg.get("yearly_anchors", 3),
                      target_index=self.target_index,
                      time_encode=self.use_time_encode,
                      # Only when use_future_co2 is on: the dataset then also
                      # emits the CO2 at every target step.  Default None keeps
                      # every existing config byte-identical.
                      co2_index=co2_idx if dataset_cfg.get("use_future_co2") else None)
        train_ds = PeriodicAnchorDataset(origin_indices=train_origins, **common)
        val_ds = PeriodicAnchorDataset(origin_indices=val_origins, **common)
        test_ds = PeriodicAnchorDataset(origin_indices=test_origins, **common)

        # num_features may have changed due to time encoding
        self.num_features = train_ds.num_features

        # DataLoaders
        train_cfg = cfg["training"]
        nw = train_cfg.get("num_workers", 0)
        pin = train_cfg.get("pin_memory", False)

        # persistent_workers=False: workers die after each epoch, preventing
        # fd exhaustion across trials. The ~1s recreation cost per epoch is
        # negligible compared to "Too many open files" crashes.
        self.loaders["train"] = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                                           shuffle=True, num_workers=nw, pin_memory=pin,
                                           persistent_workers=False,
                                           worker_init_fn=_worker_init_fn)
        self.loaders["val"] = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                                         shuffle=False, num_workers=nw, pin_memory=pin,
                                         persistent_workers=False,
                                         worker_init_fn=_worker_init_fn)
        self.loaders["test"] = DataLoader(test_ds, batch_size=train_cfg["batch_size"],
                                          shuffle=False, num_workers=nw, pin_memory=pin,
                                          persistent_workers=False,
                                          worker_init_fn=_worker_init_fn)

    def cleanup(self) -> None:
        """Explicitly shut down DataLoader workers to free file descriptors.

        Uses ``multiprocessing.active_children()`` to kill any remaining
        worker processes.  Also tries ``torch.multiprocessing`` queue cleanup
        in case PyTorch manages its own process pool.
        """
        import gc, multiprocessing

        # 1. Nuke DataLoader references — break any iterator cycles
        for name in list(self.loaders.keys()):
            loader = self.loaders.pop(name)
            loader._iterator = None
            del loader

        # 2. Trigger DataLoader iterator __del__ via aggressive GC
        gc.collect(); gc.collect()

        # 3. Kill every remaining child process (PyTorch workers + anything else)
        for child in multiprocessing.active_children():
            try:
                child.terminate()
                child.join(timeout=3)
            except Exception:
                pass

    # ── training ──

    def train(self, model: nn.Module,
              forward_fn: Callable[[nn.Module, dict], dict[str, torch.Tensor]],
              *, loss_fn=None, optimizer=None,
              train_loader: DataLoader | None = None,
              val_loader: DataLoader | None = None) -> tuple[float, list[float], list[float]]:
        """Train model, return best val loss."""
        train_cfg = self.config["training"]
        device = self.device

        train_loader = train_loader or self.loaders["train"]
        val_loader = val_loader or self.loaders["val"]

        if loss_fn is None:
            loss_cfg = self.config["loss"]
            loss_type = loss_cfg.get("type", "")
            if loss_type == "shash":
                # SHASH negative log-likelihood for the probabilistic head.
                # The model must be built with probabilistic=True so that
                # forward returns dist_params.
                loss_fn = ShashLoss().to(device)
            elif loss_type == "mse":
                _mse = nn.MSELoss().to(device)
                def _mse_wrapper(pred, target):
                    return {"loss": _mse(pred, target)}
                loss_fn = _mse_wrapper
            elif loss_type == "db":
                # DBLoss: EMA trend/seasonal decomposition, L2(season)+L1(trend),
                # scale-aligned — for long-horizon / seasonal-dominated targets.
                loss_fn = DBLoss(
                    delta_weight=loss_cfg.get("delta_weight", 0.0),
                ).to(device)
            else:
                loss_fn = MultiHorizonForecastLoss(
                    horizon=self.horizon,
                    delta_weight=loss_cfg.get("delta_weight", 0.0),
                    horizon_weight_power=loss_cfg.get("horizon_weight_power", 0.0),
                    horizon_weight_max=loss_cfg.get("horizon_weight_max", 1.0),
                    extreme_weight=loss_cfg.get("extreme_weight", 0.0),
                    extreme_threshold=loss_cfg.get("extreme_threshold", 1.5),
                ).to(device)

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg["learning_rate"],
                weight_decay=train_cfg["weight_decay"],
            )

        use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"
        grad_accum = train_cfg.get("grad_accum_steps", 1)
        patience_val = train_cfg.get("patience", 12)
        epochs = train_cfg["epochs"]
        grad_clip = train_cfg.get("gradient_clip")
        lr_decay = train_cfg.get("lr_decay", 0)
        lr_min = train_cfg.get("lr_min", 0)

        checkpoint_dir = _project_root_path(self.config["output"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(1, epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]

            # Train
            train_result = run_epoch(
                model, train_loader, forward_fn, loss_fn, device,
                optimizer=optimizer, gradient_clip=grad_clip,
                use_amp=use_amp, grad_accum_steps=grad_accum,
                short_lead_steps=int(self.config.get("loss", {}).get("short_lead_loss_steps", 0) or 0),
                short_lead_weight=float(self.config.get("loss", {}).get("short_lead_loss_weight", 0.5)),
            )
            # Val
            val_result = run_epoch(
                model, val_loader, forward_fn, loss_fn, device,
                optimizer=None, use_amp=use_amp,
            )

            train_losses.append(float(train_result["loss"]))
            val_losses.append(float(val_result["loss"]))

            tqdm.write(f"Epoch {epoch:03d} | train_loss={train_result['loss']:.6f} "
                       f"| val_loss={val_result['loss']:.6f} | lr={current_lr:.2e}")

            # Rebuild the analog memory bank from the train set so the keys
            # co-evolve with the encoder.  Runs in eval mode, no gradient, no
            # leakage (train origins only).  Controlled by
            # training.update_analog_memory / training.analog_bank_size.
            if train_cfg.get("update_analog_memory", False) and hasattr(
                    model, "build_analog_bank"):
                model.build_analog_bank(
                    train_loader, forward_fn, device,
                    max_entries=train_cfg.get("analog_bank_size", 20000))

            # min_delta: only count as improvement if val_loss drops by
            # more than this fraction of best_val.  Prevents tiny random
            # oscillations ("spiral" overfitting) from resetting patience.
            min_delta = train_cfg.get("early_stop_min_delta", 0.0)
            improved = val_result["loss"] < best_val * (1.0 - min_delta)

            if improved:
                best_val = val_result["loss"]
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": self.config,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                }, checkpoint_dir / "best_model.pt")
            else:
                patience_counter += 1
                # ── plateau lr decay ──
                if lr_decay > 0:
                    for pg in optimizer.param_groups:
                        pg["lr"] = max(lr_min, pg["lr"] - lr_decay)

            if patience_counter >= patience_val:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break

            if device.type == "cuda" and epoch % 5 == 0:
                torch.cuda.empty_cache()

        # Load best
        ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Best val loss: {best_val:.6f}")
        return best_val, train_losses, val_losses

    # ── evaluation ──

    def evaluate(self, model: nn.Module,
                 forward_fn: Callable[[nn.Module, dict], dict[str, torch.Tensor]],
                 loader: DataLoader | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (pred_orig, target_orig) after inverse transform."""
        loader = loader or self.loaders["test"]
        use_amp = self.config["training"].get("use_amp", False) and self.device.type == "cuda"
        pred, target = collect_predictions(model, loader, forward_fn, self.device, use_amp=use_amp)
        assert self.scaler is not None
        return (self.scaler.inverse_target(pred, self.target_index),
                self.scaler.inverse_target(target, self.target_index))

    def evaluate_probabilistic(
        self, model: nn.Module,
        forward_fn: Callable[[nn.Module, dict], dict[str, torch.Tensor]],
        loader: DataLoader | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mu_orig, dist_params_orig, target_orig) for SHASH models.

        ``mu_orig`` is the point/location forecast, ``dist_params_orig`` the
        [N, H, 4] SHASH parameters in the original temperature scale (mu/sigma
        rescaled, gamma/tau invariant), and ``target_orig`` the observed
        window.  Suitable for CRPS / interval-coverage evaluation.
        """
        loader = loader or self.loaders["test"]
        use_amp = self.config["training"].get("use_amp", False) and self.device.type == "cuda"
        mu, target, dist = collect_probabilistic_predictions(
            model, loader, forward_fn, self.device, use_amp=use_amp)
        assert self.scaler is not None
        mu_orig = self.scaler.inverse_target(mu, self.target_index)
        target_orig = self.scaler.inverse_target(target, self.target_index)
        dist_orig = self.scaler.inverse_dist_params(dist, self.target_index)
        return mu_orig, dist_orig, target_orig

    # ── save ──

    def save_results(self, pred: np.ndarray, target: np.ndarray, *,
                     n_params: int, best_val_loss: float,
                     train_losses: list | None = None,
                     val_losses: list | None = None,
                     result_dir_override: Path | None = None) -> dict:
        """Save .npy + summary.txt, return metrics dict."""
        result_dir = result_dir_override or _project_root_path(self.config["output"]["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)

        np.save(result_dir / "test_predict.npy", pred)
        np.save(result_dir / "test_target.npy", target)

        hm = horizon_metrics(pred, target)
        corr = compute_correlation(pred, target)
        all_m = compute_all_metrics(pred, target)
        np.save(result_dir / "test_rmse_per_horizon.npy", hm["rmse"])
        np.save(result_dir / "test_mae_per_horizon.npy", hm["mae"])

        # Extended metrics
        for key, arr in all_m.items():
            np.save(result_dir / f"test_{key}_per_horizon.npy", arr)

        # Loss history
        if train_losses:
            np.save(result_dir / "train_losses.npy", np.array(train_losses))
        if val_losses:
            np.save(result_dir / "val_losses.npy", np.array(val_losses))

        # Scaler
        if self.scaler is not None:
            scaler_dir = Path(__file__).resolve().parent.parent / "data/scalers"
            scaler_dir.mkdir(parents=True, exist_ok=True)
            np.save(scaler_dir / "weather_mean.npy", self.scaler.mean)
            np.save(scaler_dir / "weather_std.npy", self.scaler.std)

        overall_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        # Pooled R² (consistent with pooled RMSE) — previously the mean of
        # per-horizon R², which disagreed with run_experiments' pooled R².
        _ss_tot = float(((target - target.mean()) ** 2).sum())
        overall_r2 = float(1.0 - ((pred - target) ** 2).sum() / _ss_tot) if _ss_tot > 0 else 0.0

        report = {}
        if self.horizon >= 120:
            report = {"6h": 0, "1d": 3, "3d": min(11, self.horizon - 1),
                      "7d": min(27, self.horizon - 1), "15d": min(59, self.horizon - 1),
                      "30d": self.horizon - 1}
        elif self.horizon >= 28:
            report = {"6h": 0, "1d": 3, "3d": min(11, self.horizon - 1),
                      "7d": min(27, self.horizon - 1)}
        else:
            step = min(4, self.horizon) if self.horizon >= 4 else self.horizon
            report = {f"step_{i+1}": i for i in range(self.horizon) if i < step or i >= self.horizon - 2}
            if self.horizon > step + 2:
                report["..."] = step
        print(f"\nResults:")
        for name, step in report.items():
            if name == "..." or step < self.horizon:
                print(f"  {name}: RMSE={hm['rmse'][step]:.4f}, Corr={corr[step]:.4f}")
        print(f"  Overall: RMSE={overall_rmse:.4f}, R²={overall_r2:.4f}")

        _write_summary(result_dir / "summary.txt",
                       n_params=n_params, best_val_loss=best_val_loss,
                       overall_rmse=overall_rmse, overall_r2=overall_r2,
                       rmse_h=hm["rmse"], corr_h=corr, horizon=self.horizon)

        print(f"Saved to {result_dir}")
        return {"rmse": hm["rmse"], "corr": corr, "mae": hm["mae"],
                "overall_rmse": overall_rmse, "overall_r2": overall_r2}


# ═══════════════════════════════════════════════════════════════════
# Forward-function catalog
# ═══════════════════════════════════════════════════════════════════

def anchor_forward(model, batch):
    return model(
        x_recent=batch["x_recent"], daily_anchor=batch["daily_anchor"],
        yearly_anchor=batch["yearly_anchor"], daily_mask=batch["daily_mask"],
        yearly_mask=batch["yearly_mask"], future_calendar=batch["future_calendar"],
        daily_gaps=batch.get("daily_gaps"), yearly_gaps=batch.get("yearly_gaps"),
        # Present only when the dataset was built with co2_index set; None is
        # the no-op default for every existing config.
        future_co2=batch.get("future_co2"))


def simple_forward(model, batch):
    return model(x_recent=batch["x_recent"], future_calendar=batch["future_calendar"])


def trend_res_forward(model, batch):
    return model(x_recent=batch["x_recent"], future_calendar=batch["future_calendar"],
                 global_time_idx=batch.get("origin_index"))


def dlinear_forward(model, batch):
    return model(x_recent=batch["x_recent"])


def tslib_forward(model, batch):
    x, y = batch["_tslib"]
    out = model(x, None, None, None)
    if isinstance(out, tuple):
        out = out[0]
    return {"prediction": out[:, :, 0]}
