"""Forecast loss functions with extreme-event awareness."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class MultiHorizonForecastLoss(nn.Module):
    """SmoothL1 loss with horizon weighting, delta consistency, and
    extreme-event amplification.

    Parameters
    ----------
    horizon : int
        Number of forecast steps.
    delta_weight : float
        Weight of the first-difference (trend) consistency term.
    horizon_weight_power : float
        Power-law exponent for horizon weight growth.
        1.0 = linear, 2.0 = quadratic, 0.5 = sqrt.
        0 to disable (uniform weights).
    horizon_weight_max : float
        Maximum weight at the final horizon step.
        Default 4.0 → 7d is 4× more heavily weighted than 6h.
    extreme_weight : float
        Extra multiplicative weight for samples with |target| > threshold.
        Set to 0 to disable.
    extreme_threshold : float
        σ threshold for "extreme" (default 1.5, top ~7% of Gaussian).
    extreme_temperature : float
        Temperature for the sigmoid soft-threshold (default 0.3).
        Lower → sharper transition between normal/extreme.
    """

    def __init__(
        self,
        horizon: int,
        delta_weight: float = 0.1,
        horizon_weight_power: float = 2.0,
        horizon_weight_max: float = 4.0,
        extreme_weight: float = 2.0,
        extreme_threshold: float = 1.5,
        extreme_temperature: float = 0.3,
    ) -> None:
        super().__init__()

        if horizon_weight_power > 0:
            # Power-law: weight[t] = 1 + (max-1) * (t/(H-1))^power
            t = torch.linspace(0, 1, horizon)
            weights = 1.0 + (horizon_weight_max - 1.0) * (t ** horizon_weight_power)
        else:
            weights = torch.ones(horizon)
        self.register_buffer("horizon_weights", weights)

        self.delta_weight = delta_weight
        self.extreme_weight = extreme_weight
        self.extreme_threshold = extreme_threshold
        self.extreme_temperature = extreme_temperature

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shape mismatch")

        # Base element-wise loss (robust to outliers)
        element_loss = F.smooth_l1_loss(prediction, target, reduction="none")

        # ── Extreme-event sample weighting ──
        if self.extreme_weight > 0:
            # Per-sample mean absolute target → soft extreme score
            sample_amplitude = target.abs().mean(dim=1)          # [B]
            # Soft threshold: sigmoid((|t| - threshold) / temperature)
            # Gives ~0 for normal, ~1 for extreme samples
            extreme_score = torch.sigmoid(
                (sample_amplitude - self.extreme_threshold) / self.extreme_temperature
            )                                                     # [B]
            # Blend: weight = 1 + (extreme_weight - 1) * extreme_score
            sample_weight = (
                1.0
                + (self.extreme_weight - 1.0) * extreme_score
            ).unsqueeze(1)                                        # [B, 1]
        else:
            sample_weight = 1.0

        # Horizon-weighted mean (with extreme amplification)
        horizon_w = self.horizon_weights.unsqueeze(0)             # [1, H]
        forecast_loss = torch.mean(
            element_loss * horizon_w * sample_weight
        )

        # ── Delta (trend) consistency ──
        if prediction.shape[1] > 1:
            pred_delta = prediction[:, 1:] - prediction[:, :-1]
            true_delta = target[:, 1:] - target[:, :-1]
            delta_loss = F.smooth_l1_loss(pred_delta, true_delta)
        else:
            delta_loss = prediction.new_tensor(0.0)

        total_loss = forecast_loss + self.delta_weight * delta_loss

        return {
            "loss": total_loss,
            "forecast_loss": forecast_loss,
            "delta_loss": delta_loss,
        }


class ShashLoss(nn.Module):
    """Negative log-likelihood for a SHASH (sinh-arcsinh) distribution head.

    The SHASH distribution (Jones & Pewsey 2009) is parameterised by
        X = mu + sigma * sinh( (Z + gamma) / tau ),   Z ~ N(0, 1),
    where mu is the location, sigma > 0 the scale, gamma the skewness and
    tau > 0 the tail index (tau < 1 gives heavy tails / climate tail risk,
    tau > 1 gives lighter tails).  The head outputs ``[B, H, 4]``
    (mu, sigma, gamma, tau); sigma/tau are already positive via softplus in
    the head, but we clamp defensively.

    ``prediction`` is expected to be the model's ``dist_params`` tensor; the
    ``target`` is the observed future window ``[B, H]``.

    The same formula drives ``experiments/emulator_enhance/shash_emulator.py``
    (``shash_nll``); keeping a single implementation here lets the main model
    train with the identical objective as the standalone emulator.
    """

    def __init__(self):
        super().__init__()

    def forward(self, dist_params: torch.Tensor,
                target: torch.Tensor) -> dict[str, torch.Tensor]:
        if dist_params.shape[:-1] != target.shape:
            raise ValueError(
                f"dist_params shape {tuple(dist_params.shape)} != "
                f"target shape {tuple(target.shape)}")
        mu = dist_params[..., 0]
        sigma = torch.clamp(dist_params[..., 1], min=1e-4)
        gamma = dist_params[..., 2]
        tau = torch.clamp(dist_params[..., 3], min=1e-4)

        z = tau * torch.asinh((target - mu) / sigma) - gamma
        nll = (0.5 * z * z
               + 0.5 * math.log(2.0 * math.pi)
               - torch.log(tau) + torch.log(sigma)
               + 0.5 * torch.log1p(((target - mu) / sigma) ** 2))
        nll = nll.mean()
        return {"loss": nll, "nll": nll}


class DBLoss(nn.Module):
    """Decomposition-based loss (DBLoss, NeurIPS 2025).

    Decompose each forecast window into trend + seasonal components via a causal
    EMA over the horizon (trend = running EMA, seasonal = x - trend), penalise the
    seasonal component with an L2 loss and the trend with an L1 loss, and scale-align
    the two so neither dominates.  Useful for long-horizon / seasonal-dominated series
    (global mean temperature), where treating seasonal and trend errors uniformly
    under-weights the structure.
    """

    def __init__(self, delta_weight: float = 0.0, **kwargs) -> None:
        super().__init__()
        self.delta_weight = delta_weight

    def _ema(self, x: torch.Tensor, alpha: float = 0.25) -> torch.Tensor:
        # causal EMA along the horizon dimension (trend = smoothed path)
        out = x.clone()
        for t in range(1, x.shape[1]):
            out[:, t] = alpha * x[:, t] + (1.0 - alpha) * out[:, t - 1]
        return out

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> dict[str, torch.Tensor]:
        p_trend = self._ema(prediction)
        t_trend = self._ema(target)
        p_season = prediction - p_trend
        t_season = target - t_trend

        season_loss = F.mse_loss(p_season, t_season)
        trend_loss = F.l1_loss(p_trend, t_trend)

        # Scale-align (stop-gradient): match the seasonal loss magnitude to the
        # trend so the dominant (seasonal) component doesn't swallow the trend.
        scale = ((t_trend.detach().abs().mean() + 1e-8)
                 / (t_season.detach().abs().mean() + 1e-8)).clamp(0.1, 10.0)
        total = season_loss * scale + trend_loss

        # Optional first-difference (trend-consistency) term, following the main loss.
        if self.delta_weight > 0 and prediction.shape[1] > 1:
            total = total + self.delta_weight * F.smooth_l1_loss(
                prediction[:, 1:] - prediction[:, :-1],
                target[:, 1:] - target[:, :-1])

        return {"loss": total, "season_loss": season_loss, "trend_loss": trend_loss,
                "scale": scale.detach()}
