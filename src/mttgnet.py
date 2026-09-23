"""MTTGNet Dual Memory Architecture — original v5.

Core innovation:
  Dual Memory = Dynamic Prototype Memory + Periodic Anchor Attention
  Gate-controlled fusion adaptively combines:
    (1) Abstract weather patterns learned via prototype memory slots
    (2) Concrete historical observations retrieved via periodic anchors

Architecture:
  1. Joint Temporal Encoder   — 3-layer GRU on full feature vector
  2. Variable Encoder          — Cross-variable self-attention on anchor nodes
  3. Calendar Query Encoder    — Calendar features + horizon position
  4. Periodic Anchor Attention — Daily + yearly anchor cross-attention
                                  + similarity + time-distance + Gate
  5. Dynamic Memory Bank       — Learnable prototype retrieval
  6. Multi-Source Gate Fusion  — temporal/periodic/memory 3-source fusion
  7. Calendar-Conditioned Head — Per-horizon MLP with calendar awareness
"""

from __future__ import annotations
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Dynamic Memory Bank
# ═══════════════════════════════════════════════════════════════════

class DynamicMemory(nn.Module):
    def __init__(self, d_model, num_slots=128, top_k=8, temperature=1.0,
                 gate_init_bias=-3.0):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.memory = nn.Parameter(torch.empty(num_slots, d_model))
        nn.init.trunc_normal_(self.memory, std=0.02)
        self.query_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.Sigmoid())
        # Init the gate to ≈0 so memory starts as a small RESIDUAL corrector
        # (sigmoid(-3)≈0.05).  A random gate (≈0.5) lets untrained prototypes
        # perturb the context from the start, which is a common source of a
        # *false negative* in ablations.  The model learns to use memory only if
        # it helps; otherwise it can leave the gate near 0.
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, float(gate_init_bias))
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, periodic_bias=None):
        B, D = x.shape
        q = self.query_proj(x)
        scores = F.cosine_similarity(q.unsqueeze(1), self.memory.unsqueeze(0), dim=-1)
        k = min(self.top_k, self.memory.shape[0])
        topk_scores, topk_idx = torch.topk(scores, k, dim=-1)
        w = F.softmax(topk_scores / self.temperature, dim=-1)
        retrieved = (w.unsqueeze(-1) * self.memory[topk_idx]).sum(dim=1)
        if periodic_bias is not None:
            gate_input = torch.cat([x, retrieved, periodic_bias], dim=-1)
        else:
            # No periodic context → zero the bias channel (same shape as
            # retrieved, matching the dimension the gate expects).
            gate_input = torch.cat([x, retrieved, torch.zeros_like(retrieved)], dim=-1)
        gate = self.gate(gate_input)
        fused = gate * retrieved + (1.0 - gate) * x
        return self.norm(x + self.out_proj(fused))


# ═══════════════════════════════════════════════════════════════════
# Calendar Query Encoder
# ═══════════════════════════════════════════════════════════════════

class CalendarQueryEncoder(nn.Module):
    def __init__(self, calendar_dim, hidden_dim, max_horizon):
        super().__init__()
        self.calendar_projection = nn.Sequential(
            nn.Linear(calendar_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.horizon_embedding = nn.Embedding(max_horizon, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, future_calendar):
        _, horizon, _ = future_calendar.shape
        horizon_ids = torch.arange(horizon, device=future_calendar.device)
        horizon_emb = self.horizon_embedding(horizon_ids).unsqueeze(0)
        query = self.calendar_projection(future_calendar) + horizon_emb
        return self.norm(query)


# ═══════════════════════════════════════════════════════════════════
# Distribution Shift Encoder
# ═══════════════════════════════════════════════════════════════════

class DistributionShiftEncoder(nn.Module):
    def __init__(self, hidden_dim, co2_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim + co2_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.time_mod = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pred_bias = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1))
        # CO2-conditioned anchor time-profile: a scalar per sample that shifts
        # the periodic-anchor attention toward RECENT history when the
        # background CO2 level is high (and toward distant history when low).
        # This makes CO2 structurally influence WHICH historical states are
        # retrieved (not just a constant offset), giving it real gradient signal.
        self.co2_timebias = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, temporal_context, co2_level=None):
        if co2_level is None:
            co2_level = temporal_context.new_zeros(temporal_context.shape[0], 1)
        shift_input = torch.cat([temporal_context, co2_level], dim=-1)
        shift_emb = self.encoder(shift_input)
        return {
            "shift_emb": shift_emb,
            "time_modulation": self.time_mod(shift_emb),
            "pred_bias": self.pred_bias(shift_emb).squeeze(-1),
            # tanh-bounded so the anchor-time-profile shift stays stable [-1,1]
            "co2_timebias": torch.tanh(self.co2_timebias(shift_emb).squeeze(-1)),
        }

    def monotonicity_regularization(self, temporal_context, co2_level,
                                    delta=0.05, margin=0.0):
        """Soft monotonicity penalty on the CO2 -> bias response.

        The ``pred_bias`` term is the model's learned CO2-driven warming
        contribution.  Physical consistency requires it to be non-decreasing
        in CO2 (higher CO2 must not produce a colder bias).  We enforce this
        with a one-point finite-difference check: for each sample, perturb the
        CO2 level by ``delta`` (in the [0,1] normalised scale, ~45 ppm for the
        default 280-1200 ppm range) and penalise any decrease exceeding
        ``margin``.

        Returns a scalar ``>= 0`` (mean violation over the batch) suitable for
        direct addition to the training loss.
        """
        base = self.forward(temporal_context, co2_level)["pred_bias"]
        up = self.forward(temporal_context, co2_level + delta)["pred_bias"]
        violation = torch.relu(base - up + margin)
        return violation.mean()


# ═══════════════════════════════════════════════════════════════════
# Periodic Anchor Attention
# ═══════════════════════════════════════════════════════════════════

class PeriodicAnchorAttention(nn.Module):
    def __init__(self, hidden_dim, dropout, use_gate=True, use_time_bias=True,
                 time_bias_scale=1.0):
        super().__init__()
        D = hidden_dim
        self.use_gate = use_gate
        self.use_time_bias = use_time_bias
        self.time_bias_scale = float(time_bias_scale)   # <1 weakens the time-distance term
        self.query_projection = nn.Linear(D, D)
        self.daily_key = nn.Linear(D, D)
        self.daily_value = nn.Linear(D, D)
        self.daily_relation = nn.Parameter(torch.randn(D) * 0.02)
        self.yearly_key = nn.Linear(D, D)
        self.yearly_value = nn.Linear(D, D)
        self.yearly_relation = nn.Parameter(torch.randn(D) * 0.02)
        if use_time_bias:
            self.time_encoder = nn.Sequential(
                nn.Linear(1, D // 2), nn.GELU(), nn.Linear(D // 2, D))
        self.similarity_proj = nn.Linear(D, D)
        if use_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(D * 3, D), nn.GELU(), nn.Linear(D, D), nn.Sigmoid())
        self.output_projection = nn.Linear(2 * D, D)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.feed_forward = nn.Sequential(
            nn.Linear(D, 4 * D), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * D, D))

    @staticmethod
    def _masked_softmax(scores, mask):
        scores = scores.clamp(-50.0, 50.0)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        valid_count = mask.sum(dim=-1, keepdim=True).to(weights.dtype)
        return weights * (valid_count > 0).to(weights.dtype)

    def _compute_similarity_bias(self, temporal_context, anchors):
        q = F.normalize(self.similarity_proj(temporal_context), dim=-1)
        k_norm = F.normalize(anchors, dim=-1)
        return torch.einsum('bd,bhkd->bhk', q, k_norm)

    def _aggregate(self, query, anchors, mask, key_layer, value_layer,
                   relation_embedding, time_gaps=None,
                   similarity_bias=None, shift_time_mod=None,
                   co2_timebias=None):
        B, H, K, D = anchors.shape
        q = self.query_projection(query)
        k = key_layer(anchors)
        v = value_layer(anchors)
        k = k + relation_embedding.view(1, 1, 1, -1)
        if self.use_time_bias and time_gaps is not None:
            # time_gaps may be [K] (fallback) or [B, H, K] (per-sample,
            # produced by PeriodicAnchorDataset) — the true time distance in
            # days between the horizon target and each anchor.
            log_days = torch.log1p(time_gaps).to(k.dtype).to(k.device)
            time_enc = self.time_encoder(log_days.unsqueeze(-1))
            if time_enc.ndim == 3:  # [K, 1, D] fallback → [1, 1, K, D]
                time_enc = time_enc.view(1, 1, K, D)
            if shift_time_mod is not None:
                mod = 1.0 + shift_time_mod  # [B, D]
                time_enc = time_enc * mod.unsqueeze(1).unsqueeze(1)  # [B, 1, K/1, D]
            # weaken the time-distance term (a strong time bias can overwhelm the
            # similarity score and collapse attention onto the nearest anchors).
            k = k + self.time_bias_scale * time_enc
        scores = torch.sum(q.unsqueeze(2) * k, dim=-1) / math.sqrt(D)
        if similarity_bias is not None:
            scores = scores + similarity_bias * math.sqrt(D) * 0.1
        # CO2-conditioned anchor time-profile (combined timebias + CO2): a
        # positive co2_timebias shifts attention toward RECENT anchors (small
        # log time-gap), a negative one toward distant history.  This is how
        # CO2 structurally biases which historical states are retrieved.
        if co2_timebias is not None and time_gaps is not None and time_gaps.ndim == 3:
            log_gap = torch.log1p(time_gaps).to(scores.dtype)
            log_gap = log_gap - log_gap.mean(dim=-1, keepdim=True)  # centre per (b,h)
            scores = scores - co2_timebias.unsqueeze(-1).unsqueeze(-1) * log_gap * 0.3
        weights = self._masked_softmax(scores, mask)
        return torch.sum(weights.unsqueeze(-1) * v, dim=2)

    def forward(self, query, daily_nodes, yearly_nodes,
                daily_mask, yearly_mask,
                temporal_context=None,
                daily_gaps=None, yearly_gaps=None,
                shift_time_mod=None, co2_timebias=None):
        B, H, D = query.shape
        # Daily
        daily_sim = None
        if temporal_context is not None:
            daily_sim = self._compute_similarity_bias(temporal_context, daily_nodes)
        daily_context = self._aggregate(
            query=query, anchors=daily_nodes, mask=daily_mask,
            key_layer=self.daily_key, value_layer=self.daily_value,
            relation_embedding=self.daily_relation,
            time_gaps=daily_gaps, similarity_bias=daily_sim,
            shift_time_mod=shift_time_mod, co2_timebias=co2_timebias)
        # Yearly
        yearly_sim = None
        if temporal_context is not None and yearly_nodes.shape[2] > 0:
            yearly_sim = self._compute_similarity_bias(temporal_context, yearly_nodes)
        yearly_context = self._aggregate(
            query=query, anchors=yearly_nodes, mask=yearly_mask,
            key_layer=self.yearly_key, value_layer=self.yearly_value,
            relation_embedding=self.yearly_relation,
            time_gaps=yearly_gaps, similarity_bias=yearly_sim,
            shift_time_mod=shift_time_mod, co2_timebias=co2_timebias)
        # Fuse
        periodic_context = self.output_projection(
            torch.cat([daily_context, yearly_context], dim=-1))
        # Fusion gate (Fig. 1: [query, daily_ctx, yearly_ctx] → σ → gate × update).
        # The gate + FFN refine the *periodic context* (c_periodic) that is
        # consumed downstream by the fusion gate and the decoder.  Previously
        # they refined ``query`` instead — and that refined query was discarded
        # by the caller, leaving gate_mlp/norm1/norm2/feed_forward (~110K params)
        # with no gradient (never trained) and making ``use_gate`` a no-op.
        if self.use_gate:
            gate_input = torch.cat([query, daily_context, yearly_context], dim=-1)
            gate_weight = self.gate_mlp(gate_input)
            periodic_context = self.norm1(
                periodic_context + gate_weight * self.dropout(periodic_context))
        else:
            periodic_context = self.norm1(
                periodic_context + self.dropout(periodic_context))
        periodic_context = self.norm2(
            periodic_context + self.feed_forward(periodic_context))
        return query, periodic_context


# ═══════════════════════════════════════════════════════════════════
# Anchor Variable Encoder
# ═══════════════════════════════════════════════════════════════════

class AnchorVariableEncoder(nn.Module):
    """Cross-variable encoder for anchor nodes.

    Supports two modes:
    - ``var_encoder_type="attention"`` (default): standard symmetric
      MultiheadAttention over variables — fast and stable.
    - ``var_encoder_type="gat"`` or ``"gat"``: asymmetric GAT/GAT
      attention (imported from ``.layers``).  Physically motivated —
      variable dependencies are directional (e.g. solar radiation drives
      temperature more than the reverse).

    Both modes produce a pooled [B, H, K, D] representation from [B, H, K, N]
    anchor inputs.
    """

    def __init__(self, num_features, hidden_dim, num_heads=4, dropout=0.1,
                 var_encoder_type="attention", gat_version="gat"):
        super().__init__()
        self.var_encoder_type = var_encoder_type

        if var_encoder_type in ("gat",):
            # Lazy import — layers.py is a heavy module, only load when needed
            from .layers import GATVariableEncoder
            self._gat_encoder = GATVariableEncoder(
                num_features, hidden_dim, num_heads, dropout, gat_version)
        else:
            self._gat_encoder = None

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.value_projection = nn.Linear(1, hidden_dim)
        self.variable_embedding = nn.Parameter(
            torch.randn(num_features, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * hidden_dim, hidden_dim))
        self.pool_score = nn.Linear(hidden_dim, 1)

    def forward(self, anchors):
        B, H, K, N = anchors.shape
        D = self.hidden_dim

        # ── GAT path (asymmetric, directional attention) ──
        if self._gat_encoder is not None:
            # GATVariableEncoder.forward(x) handles [..., N] → [..., D];
            # internally splits into heads, applies asymmetric attention,
            # and pools over variables.
            x = anchors.reshape(B * H, K, N)          # [B*H, K, N]
            encoded = self._gat_encoder(x)             # [B*H, K, D]
            return encoded.reshape(B, H, K, D)

        # ── Standard self-attention path (symmetric) ──
        x_flat = anchors.reshape(B * H, K, N, 1)
        nodes = self.value_projection(x_flat)          # Linear(1,D) → [B*H,K,N,D]
        nodes = nodes + self.variable_embedding.view(1, 1, N, D)
        M = B * H * K
        nodes = nodes.reshape(M, N, D)
        attended, _ = self.attention(nodes, nodes, nodes, need_weights=False)
        nodes = self.norm1(nodes + attended)
        nodes = self.norm2(nodes + self.feed_forward(nodes))
        pool_scores = self.pool_score(nodes).squeeze(-1)
        pool_weights = torch.softmax(pool_scores, dim=-1)
        pooled = (nodes * pool_weights.unsqueeze(-1)).sum(dim=1)
        return pooled.reshape(B, H, K, D)


# ═══════════════════════════════════════════════════════════════════
# Calendar-Conditioned Decoder
# ═══════════════════════════════════════════════════════════════════

class CalendarConditionedDecoder(nn.Module):
    def __init__(self, hidden_dim, horizon, calendar_dim=5, dropout=0.1):
        super().__init__()
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + calendar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, fused, periodic_ctx, future_calendar, temporal_context=None):
        decoder_input = torch.cat([fused, periodic_ctx, future_calendar], dim=-1)
        return self.context_fusion(decoder_input).squeeze(-1)


class LinearContextDecoder(nn.Module):
    """Baseline-style head: encoder context -> Linear(d, H).  This is exactly how
    the GRU/LSTM/Transformer/MLP baselines decode (ctx -> Linear -> all H steps)."""
    def __init__(self, hidden_dim, horizon, calendar_dim=5, dropout=0.1):
        super().__init__()
        self.head = nn.Linear(hidden_dim, horizon)
    def forward(self, fused, periodic_ctx, future_calendar, temporal_context=None):
        return self.head(temporal_context)


class LinearFusedDecoder(nn.Module):
    """Per-horizon Linear on the fused representation (no calendar)."""
    def __init__(self, hidden_dim, horizon, calendar_dim=5, dropout=0.1):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, fused, periodic_ctx, future_calendar, temporal_context=None):
        return self.head(fused).squeeze(-1)


class MLPNoCalDecoder(nn.Module):
    """Per-horizon MLP on [fused, periodic] — the current decoder minus the
    calendar channel (calendar is ~useless on anomaly / long horizon)."""
    def __init__(self, hidden_dim, horizon, calendar_dim=5, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))
    def forward(self, fused, periodic_ctx, future_calendar, temporal_context=None):
        return self.net(torch.cat([fused, periodic_ctx], dim=-1)).squeeze(-1)


class SoftmaxAttnDecoder(nn.Module):
    """Softmax-gated readout over the fused feature dims, then a scalar head."""
    def __init__(self, hidden_dim, horizon, calendar_dim=5, dropout=0.1):
        super().__init__()
        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, fused, periodic_ctx, future_calendar, temporal_context=None):
        g = torch.softmax(self.mix(fused), dim=-1)          # [B,H,D]
        readout = (g * fused).sum(-1, keepdim=True)         # [B,H,1]
        return self.head(fused + readout).squeeze(-1)


class HorizonCalDecoder(nn.Module):
    """Calendar-conditioned MLP with EXPLICIT lead-time awareness.

    The default CalendarConditionedDecoder never uses its ``horizon`` argument
    and only senses the lead indirectly via the calendar (doy/hour) + periodic
    context, while the temporal/memory branches are constant across the horizon.
    The shared per-horizon MLP therefore tends to emit similar outputs for every
    step, flattening the short- vs long-lead distinction.  This variant adds:
      (i) a per-step *horizon embedding* (discrete lead index),
      (ii) a continuous normalized *lead-time* feature,
      (iii) the *temporal context* expanded over the horizon, so the same MLP can
    specialise each lead — short leads lean near-persistence, long leads lean on
    the structural/seasonal forecast."""
    def __init__(self, hidden_dim, horizon, calendar_dim=5, dropout=0.1):
        super().__init__()
        self.horizon_embed = nn.Embedding(horizon, hidden_dim)
        # fused(D) + periodic(D) + temporal(D) + calendar(cal) + horizon_emb(D) + lead(1)
        in_dim = hidden_dim * 4 + calendar_dim + 1
        self.context_fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, fused, periodic_ctx, future_calendar, temporal_context=None):
        B, H, D = fused.shape
        dev = fused.device
        hids = torch.arange(H, device=dev).unsqueeze(0).expand(B, H)
        h_emb = self.horizon_embed(hids)                               # [B,H,D]
        lead = torch.linspace(0, 1, H, device=dev).view(1, H, 1).expand(B, H, 1)
        tctx = (temporal_context.unsqueeze(1).expand(B, H, D)
                if temporal_context is not None else torch.zeros_like(fused))
        decoder_input = torch.cat(
            [fused, periodic_ctx, tctx, future_calendar, h_emb, lead], dim=-1)
        return self.context_fusion(decoder_input).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# Multi-Source Gate Fusion
# ═══════════════════════════════════════════════════════════════════

class MultiSourceGateFusion(nn.Module):
    def __init__(self, hidden_dim, horizon, calendar_dim=5):
        super().__init__()
        self.gate_predictor = nn.Sequential(
            nn.Linear(calendar_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, temporal_ctx, periodic_ctx, memory_ctx, future_calendar):
        B, H, _ = future_calendar.shape
        horizon_pos = torch.linspace(0, 1, H, device=future_calendar.device)
        horizon_pos = horizon_pos.unsqueeze(0).unsqueeze(-1).expand(B, H, 1)
        gate_input = torch.cat([future_calendar, horizon_pos], dim=-1)
        gate_logits = self.gate_predictor(gate_input)
        gate = torch.softmax(gate_logits, dim=-1)
        fused = (gate[..., 0:1] * temporal_ctx +
                 gate[..., 1:2] * periodic_ctx +
                 gate[..., 2:3] * memory_ctx)
        return self.norm(fused), gate


class ShashHead(nn.Module):
    """SHASH (sinh-arcsinh) probabilistic head.

    Replaces the point-estimate decoder when ``probabilistic=True``: the fused
    multi-source context is mapped to four distribution parameters per horizon
    step, ``(mu, sigma, gamma, tau)`` (Jones & Pewsey 2009).  ``mu`` is the
    location forecast (identical role to the point prediction); ``sigma`` and
    ``tau`` are positivity-constrained so the negative log-likelihood
    (``ShashLoss``) is well-defined.  ``tau > 1`` by construction, giving the
    distribution a flexible tail (tau < 1 would be heavier-tailed than the
    sinh-arcsinh normal; we keep tau > 1 for numerical stability).
    """

    def __init__(self, hidden_dim, horizon, calendar_dim=5, dropout=0.1,
                 sigma_floor=1e-3):
        super().__init__()
        self.sigma_floor = float(sigma_floor)
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + calendar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4))

    def forward(self, fused, periodic_ctx, future_calendar):
        decoder_input = torch.cat([fused, periodic_ctx, future_calendar], dim=-1)
        p = self.context_fusion(decoder_input)          # [B, H, 4]
        mu = p[..., 0]
        sigma = F.softplus(p[..., 1]) + self.sigma_floor
        gamma = p[..., 2]
        tau = 1.0 + F.softplus(p[..., 3])               # tau > 1
        return mu, sigma, gamma, tau


class AnalogMemory(nn.Module):
    """Instance-based analog memory: retrieve real historical states.

    Classical "analog forecasting" (Lorenz, 1956) predicts a new state by
    looking up the most similar past states and inheriting their subsequent
    evolution.  This module instantiates that idea inside the network:

      * The bank holds ``(key, value)`` pairs for sampled **training** origins
        (never validation/test — no leakage):
          - key   = the GRU temporal context at that origin     [M, D]
          - value = the observed next-H target window           [M, H]
      * At inference the current temporal context retrieves the top-K most
        similar historical states by cosine similarity and forms a weighted
        analog forecast — "the last time the atmosphere looked like this,
        temperature did X over the next H steps".
      * A learned gate blends the analog forecast with the query context,
        letting the model down-weight analog contributions when no good match
        exists.

    ``build_bank`` is called by the Trainer once per epoch (in eval mode, no
    gradient) so the keys co-evolve with the encoder; the values are fixed
    observations.
    """

    def __init__(self, d_model, horizon, top_k=8, temperature=1.0):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.top_k = top_k
        self.temperature = temperature
        self.query_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model + horizon, d_model), nn.GELU(),
            nn.Linear(d_model, 1), nn.Sigmoid())
        # Bank buffers; empty until ``build_bank`` populates them.  They are
        # NON-persistent so they are excluded from state_dict: the bank is a
        # cache over real observations, not model weights, and its size varies
        # per run (persistent buffers would break strict checkpoint loading on
        # shape mismatch).  Callers must rebuild the bank (``build_bank`` or
        # the model's ``build_analog_bank``) after loading a checkpoint.
        self.register_buffer("keys", torch.zeros(0, d_model), persistent=False)
        self.register_buffer("values", torch.zeros(0, horizon), persistent=False)
        self.register_buffer("_n_bank", torch.zeros(1, dtype=torch.long),
                             persistent=False)

    @torch.no_grad()
    def build_bank(self, keys, values):
        """Replace the bank with new (keys, values).

        keys: [M, D] float array/tensor; values: [M, H] float array/tensor.
        """
        device = self.keys.device
        k = torch.as_tensor(keys, dtype=torch.float32, device=device)
        v = torch.as_tensor(values, dtype=torch.float32, device=device)
        if k.ndim == 2 and v.ndim == 2 and k.shape[0] == v.shape[0]:
            self.keys = k.contiguous()
            self.values = v.contiguous()
            self._n_bank = torch.tensor([k.shape[0]], dtype=torch.long, device=device)

    @property
    def bank_size(self) -> int:
        return int(self._n_bank.item())

    def forward(self, x):
        """x: [B, D] temporal context -> dict{analog, gate}.

        ``analog`` [B, H] is the top-K-weighted mean of the retrieved
        historical target windows (the classical analog forecast); ``gate``
        [B, 1] is the learned trust in that analog (0 = ignore, 1 = full
        pull toward the analog trajectory).  When the bank is empty (never
        built, or no valid origins) both are zero so the model degenerates to
        the base forecast.
        """
        if self.bank_size == 0:
            B = x.shape[0]
            return {"analog": torch.zeros(B, self.horizon, device=x.device),
                    "gate": torch.zeros(B, 1, device=x.device)}
        q = F.normalize(self.query_proj(x), dim=-1)          # [B, D]
        k = F.normalize(self.keys, dim=-1)                    # [M, D]
        scores = torch.mm(q, k.t())                           # [B, M]
        kk = min(self.top_k, self.bank_size)
        top_scores, top_idx = torch.topk(scores, kk, dim=-1)
        w = F.softmax(top_scores / self.temperature, dim=-1)
        analog = (w.unsqueeze(-1) * self.values[top_idx]).sum(dim=1)  # [B, H]
        g = self.gate(torch.cat([x, analog], dim=-1))          # [B, 1]
        return {"analog": analog, "gate": g}


# ═══════════════════════════════════════════════════════════════════
# Variable Graph Network (pre-GRU variable interaction)
# ═══════════════════════════════════════════════════════════════════

class VariableGNN(nn.Module):
    """Per-timestep GNN for variable interaction before GRU.

    At each time step t, N variables exchange information via a learned
    adjacency matrix A [N, N].  Output has same shape as input, but each
    variable's value is enriched by its graph neighbors.
    """
    def __init__(self, num_vars, d_model, dropout=0.1):
        super().__init__()
        # Node embeddings for adjacency learning
        self.node_emb1 = nn.Parameter(torch.randn(num_vars, d_model // 2) * 0.02)
        self.node_emb2 = nn.Parameter(torch.randn(num_vars, d_model // 2) * 0.02)
        # Per-timestep message passing: project scalar to d_model, mix, project back
        self.msg_in = nn.Linear(1, d_model)
        self.msg_out = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Learnable residual weight (starts small)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def _build_adj(self):
        aff = torch.sigmoid(torch.matmul(self.node_emb1, self.node_emb2.T))
        aff = aff * (1 - torch.eye(aff.shape[0], device=aff.device))
        return aff / (aff.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, x_recent):
        """x_recent: [B, L, N] → enhanced: [B, L, N]"""
        B, L, N = x_recent.shape
        adj = self._build_adj()                            # [N, N]

        # Per-timestep: [B, L, N] → [B*L, N, 1] → msg → [B*L, N, 1] → [B, L, N]
        x_flat = x_recent.reshape(B * L, N, 1)              # [B*L, N, 1]
        h = self.msg_in(x_flat)                              # [B*L, N, D]
        # Graph convolve: h_i = Σ_j A[i,j] * h_j
        h = torch.matmul(adj.unsqueeze(0), h)               # [B*L, N, D]
        h = self.norm(h)
        h = self.msg_out(self.dropout(h))                   # [B*L, N, 1]
        enhancement = h.reshape(B, L, N)                     # [B, L, N]

        return x_recent + self.alpha * enhancement


# ═══════════════════════════════════════════════════════════════════
# MTTGNet Main Model
# ═══════════════════════════════════════════════════════════════════

class MTTGNetModel(nn.Module):
    def __init__(self, num_vars, seq_len, horizon,
                 d_model=96, num_heads=4, dropout=0.15,
                 memory_slots=128, mem_top_k=8, calendar_dim=5,
                 num_gru_layers=3, use_memory_horizon_gate=False,
                 memory_gate_init=-3.0,
                 use_gate=True, use_time_bias=True,
                 use_memory=True, use_variable_encoder=True,
                 use_shift_encoder=True, use_var_gnn=True,
                 use_calendar=True, co2_index=None,
                 var_encoder_type="attention", gat_version="gat",
                 probabilistic=False, sigma_floor=1e-3,
                 use_mono_reg=False, mono_weight=0.1,
                 mono_delta=0.05, mono_margin=0.0,
                 use_co2_timebias=False,
                 use_short_residual=False, target_index=0,
                 short_residual_steps=8, decoder_type="calendar_mlp",
                 use_persist_blend=False, persist_blend_steps=8,
                 lead_anchor_scale=False, anchor_scale_min=0.3,
                 use_dual_forecast=False, dual_short_steps=8,
                 use_multirate=False, multirate_pool=4,
                 use_multiband=False, band_pool1=2, band_pool2=8,
                 time_bias_scale=1.0, shift_bias_scale=1.0,
                 use_analog_memory=False, analog_top_k=8,
                 analog_temperature=1.0,
                 use_multiscale_co2=False, co2_anchor_a=0.0, co2_ref=284.0,
                 **kwargs):
        super().__init__()
        D = d_model
        self.use_memory = use_memory
        self.use_memory_horizon_gate = use_memory_horizon_gate
        self.use_variable_encoder = use_variable_encoder
        self.use_shift_encoder = use_shift_encoder
        self.use_calendar = use_calendar
        self.time_bias_scale = float(time_bias_scale)
        self.shift_bias_scale = float(shift_bias_scale)
        self.probabilistic = probabilistic
        self.use_mono_reg = use_mono_reg
        self.mono_weight = float(mono_weight)
        self.mono_delta = float(mono_delta)
        self.mono_margin = float(mono_margin)
        self.use_co2_timebias = use_co2_timebias
        self.use_short_residual = use_short_residual
        self._target_index = int(target_index)
        self.short_residual_steps = int(short_residual_steps)
        self.use_persist_blend = use_persist_blend
        self.persist_blend_steps = int(persist_blend_steps)
        self.lead_anchor_scale = lead_anchor_scale
        self.anchor_scale_min = float(anchor_scale_min)
        self.use_dual_forecast = use_dual_forecast
        self.dual_short_steps = int(dual_short_steps)
        if use_dual_forecast:
            # per-horizon softmax over {short-residual branch, main decoder branch}
            self.dual_gate = nn.Linear(calendar_dim + 1, 2)
            self.dual_short_gate = nn.Parameter(torch.linspace(2.5, -5.0, horizon))
        self.use_multirate = use_multirate
        self.multirate_pool = int(multirate_pool)
        if use_multirate:
            # N-HiTS-flavoured coarse branch: pool fused over horizon -> coarse head ->
            # interpolate back; a lead-aware gate blends it with the fine main prediction
            # (coarse -> long range, fine -> short lead).
            self.coarse_head = nn.Linear(D, 1)
            self.multirate_gate = nn.Linear(calendar_dim + 1, 1)
        self.use_multiband = use_multiband
        self.band_pool1 = int(band_pool1)
        self.band_pool2 = int(band_pool2)
        if use_multiband:
            # N-HiTS multi-stack: two coarse branches at different rates (fine band -> short
            # lead, coarse band -> long lead) blended with the main prediction by a 3-way
            # per-horizon softmax.
            self.band_head1 = nn.Linear(D, 1)
            self.band_head2 = nn.Linear(D, 1)
            self.band_gate = nn.Linear(calendar_dim + 1, 3)  # learnable short-lead blend gate
        self.use_analog_memory = use_analog_memory
        self.sigma_floor = float(sigma_floor)
        self._co2_index = co2_index
        self.use_multiscale_co2 = use_multiscale_co2
        self.co2_anchor_a = float(co2_anchor_a)
        self.co2_ref = float(co2_ref)
        # Default: identity pass-through.  The Trainer [0,1]-normalises CO2
        # (280–1200 ppm) before it reaches the model, so the shift_encoder
        # expects CO2 ∈ [0, 1] by default.
        # Scripts that feed z-scored CO2 instead must pass the raw-ppm range
        # explicitly, e.g. co2_mean/co2_std = scaler stats and
        # co2_min/co2_max = 280/1200.
        self._co2_min = float(kwargs.pop('co2_min', 0.0))
        self._co2_max = float(kwargs.pop('co2_max', 1.0))
        # Optional de-standardisation: recover raw ppm (z-scored input) before
        # the fixed-range normalisation above.  Identity by default.
        self._co2_mean = float(kwargs.pop('co2_mean', 0.0))
        self._co2_std  = float(kwargs.pop('co2_std', 1.0))

        # ⓪ Variable GNN (pre-GRU variable interaction)
        if use_var_gnn:
            self.var_gnn = VariableGNN(num_vars, D, dropout)
        else:
            self.var_gnn = None

        # ① GRU Temporal Encoder
        self.input_proj = nn.Linear(num_vars, D)
        self.gru = nn.GRU(D, D, num_gru_layers, batch_first=True,
                          dropout=dropout if num_gru_layers > 1 else 0.0)
        self.temporal_norm = nn.LayerNorm(D)
        self.temporal_dropout = nn.Dropout(dropout)

        # ② Variable Encoder on anchors
        if use_variable_encoder:
            self.anchor_var_encoder = AnchorVariableEncoder(
                num_vars, D, num_heads, dropout,
                var_encoder_type=var_encoder_type,
                gat_version=gat_version)
        else:
            self.anchor_var_encoder = None

        # ③ Calendar Query Encoder
        self.query_encoder = CalendarQueryEncoder(calendar_dim, D, horizon)
        self.local_to_query = nn.Linear(D, D)

        # ④ Periodic Anchor Attention
        self.periodic_attention = PeriodicAnchorAttention(
            D, dropout, use_gate=use_gate, use_time_bias=use_time_bias,
            time_bias_scale=self.time_bias_scale)

        # ④b Distribution Shift Encoder
        self._co2_dim = 3 if use_multiscale_co2 else 1
        self.shift_encoder = DistributionShiftEncoder(D, self._co2_dim) if use_shift_encoder else None

        # ⑤ Dynamic Memory
        if use_memory:
            self.memory = DynamicMemory(D, memory_slots, mem_top_k,
                                        gate_init_bias=memory_gate_init)
        # Optional per-horizon gate on the memory term.  DynamicMemory returns a
        # single [B, D] vector that the forward pass broadcasts to every lead, so
        # the retrieved prototype is injected into the 30-day lead as strongly as
        # into the 6-hour one.  That is harmless at H=28 and measurably harmful at
        # H=120 (paper: memory helps at the 7-day configuration, hurts the 30-day
        # one).  This gate lets the model decay the memory contribution with lead;
        # it is a single scalar per horizon step, shared across samples, because
        # the lead dependence is a property of the horizon, not of the sample.
        # Initialised at sigmoid(0)=0.5 so the model can move it either way.
        self.memory_hgate = None
        if use_memory and use_memory_horizon_gate:
            self.memory_hgate = nn.Linear(1, 1)

        # ⑥ Multi-Source Gate Fusion (temporal / periodic / prototype-memory).
        # The analog memory blends in *target space* (Section ⑦), so the gate
        # stays 3-source.
        self.fusion = MultiSourceGateFusion(D, horizon, calendar_dim)

        # ⑦ Decoder — configurable head type (each is a small, self-contained
        # module taking (fused, periodic_ctx, future_calendar, temporal_context)).
        self.decoder_type = decoder_type
        if decoder_type == "linear_ctx":       # baseline-style: ctx -> Linear -> H
            self.decoder = LinearContextDecoder(D, horizon, calendar_dim, dropout)
        elif decoder_type == "linear_fused":   # per-horizon Linear on fused
            self.decoder = LinearFusedDecoder(D, horizon, calendar_dim, dropout)
        elif decoder_type == "mlp_nocal":      # MLP minus calendar channel
            self.decoder = MLPNoCalDecoder(D, horizon, calendar_dim, dropout)
        elif decoder_type == "softmax_attn":   # softmax-gated feature readout
            self.decoder = SoftmaxAttnDecoder(D, horizon, calendar_dim, dropout)
        elif decoder_type == "cal_mlp_h":      # calendar MLP + explicit horizon/lead encoding
            self.decoder = HorizonCalDecoder(D, horizon, calendar_dim, dropout)
        else:                                  # default: current calendar-conditioned MLP
            self.decoder = CalendarConditionedDecoder(D, horizon, calendar_dim, dropout)

        # ⑧ Direct prediction head — used when anchors / memory / shift are all
        #    disabled, so the fusion+decoder skeleton would only degrade the GRU
        #    output.  Matches the SimpleLSTM baseline: GRU → Linear → pred.
        self.direct_head = nn.Linear(D, horizon)

        # ⑧c Short-lead residual branch: pull the near-term steps toward the
        #    persistence forecast (the last observed target value).  Simple
        #    baselines are excellent at short lead precisely because they copy
        #    the recent value; the multi-horizon decoder underfits it.  A learned
        #    per-horizon gate blends persistence with the base forecast, decaying
        #    to ~0 at long lead so the structure is unchanged there.
        if use_short_residual:
            # gate logits decay fast over the horizon; HARD-MASKED to the first
            # short_residual_steps in forward so persistence never leaks into the
            # long lead (which must stay the structural forecast).
            self.short_res_gate = nn.Parameter(
                torch.linspace(2.5, -5.0, horizon))  # sigmoid: ~0.92 -> ~0.01

        # ⑧b SHASH probabilistic head (replaces the point decoder when
        #    probabilistic=True).  Returns (mu, sigma, gamma, tau) per horizon
        #    step; mu is reported as the point forecast for metric parity.
        if probabilistic:
            self.shash_head = ShashHead(D, horizon, calendar_dim, dropout,
                                        sigma_floor=sigma_floor)

        # ⑨ Analog memory (instance-level retrieval from the training set).
        #    The bank is populated by the Trainer (``build_analog_bank``) once
        #    per epoch from train-loader contexts in eval mode (no leakage).
        if use_analog_memory:
            self.analog_memory = AnalogMemory(
                D, horizon, top_k=analog_top_k, temperature=analog_temperature)

    def forward(self, x_recent, daily_anchor, yearly_anchor,
                daily_mask, yearly_mask, future_calendar,
                daily_gaps=None, yearly_gaps=None, co2_override_ppm=None):
        B, L, N = x_recent.shape
        H = future_calendar.shape[1]
        D = self.input_proj.out_features
        device = x_recent.device

        # Calendar ablation: zero-out calendar conditioning.  The query
        # encoder / gate / decoder still see the (zeroed) calendar tensor so
        # tensor shapes are unchanged — the network simply gets no calendar
        # information.  (Historically "calendar encoding" was claimed harmful
        # on C; this flag makes that ablation actually testable.)
        if not self.use_calendar:
            future_calendar = torch.zeros_like(future_calendar)

        # Capture the RAW CO₂ forcing BEFORE the GNN mixes variables.
        # The DistributionShiftEncoder must read the true CO₂ value — if it
        # reads the GNN-modified column, pred_bias becomes garbage and the
        # whole prediction (including 6h) is corrupted.
        co2_idx = self._co2_index
        if co2_idx is None:
            # Config did NOT enable CO₂ → disable forcing.  Do NOT silently
            # assume column 1: the Trainer only [0,1]-normalises CO₂ when the
            # config sets co2_index, so assuming column 1 here would feed a
            # z-scored column as if it were [0,1].
            co2_raw = None
        elif co2_idx >= N:
            import warnings
            warnings.warn(
                f"co2_index={co2_idx} >= num_vars={N}. "
                "CO₂ forcing will be disabled. "
                "Set co2_index correctly in the config (typically 1 for the "
                "8-variable ERA5 dataset, or None to disable).")
            co2_raw = None
        else:
            co2_raw = x_recent[:, -1, co2_idx:co2_idx + 1]
            co2_win = x_recent[:, :, co2_idx:co2_idx + 1]   # [B,L,1] for multiscale (pre-GNN)

        # Capture the last observed target value (z-space) BEFORE the VariableGNN
        # mixes variables — this is the persistence forecast used by the
        # short-lead residual branch.
        last_target = x_recent[:, -1, self._target_index] if self._target_index < N else None

        # ⓪ Variable GNN: enhance input with cross-variable graph info
        if self.var_gnn is not None:
            x_recent = self.var_gnn(x_recent)

        # ① GRU encoding
        x = self.input_proj(x_recent)
        _, hn = self.gru(x)
        temporal_context = self.temporal_dropout(self.temporal_norm(hn[-1]))

        # ② Variable Encoding on Anchors
        K_d = daily_anchor.shape[2]
        K_y = yearly_anchor.shape[2]
        if K_d + K_y > 0:
            all_anchors = torch.cat([daily_anchor, yearly_anchor], dim=2)
            if self.use_variable_encoder and self.anchor_var_encoder is not None:
                all_nodes = self.anchor_var_encoder(all_anchors)
            else:
                all_nodes = all_anchors.mean(dim=-1, keepdim=True)
                all_nodes = all_nodes.expand(-1, -1, -1, D)
            daily_nodes = all_nodes[:, :, :K_d, :]
            yearly_nodes = all_nodes[:, :, K_d:, :]
        else:
            daily_nodes = daily_anchor.new_zeros(B, H, 0, D)
            yearly_nodes = yearly_anchor.new_zeros(B, H, 0, D)

        # ③ Calendar Query Encoding
        query = self.query_encoder(future_calendar)
        query = query + self.local_to_query(temporal_context).unsqueeze(1)

        # ④b Distribution Shift Encoding
        # co2_raw is None when co2_index ≥ num_vars (CO₂ disabled); in that
        # case there is no CO₂ signal to encode — fall through to zero bias
        # instead of crashing on None arithmetic.
        if (self.use_shift_encoder and self.shift_encoder is not None
                and co2_raw is not None):
            # The Trainer feeds CO₂ pre-normalised to [0,1] (280–1200 ppm).
            # Identity params (default) pass it through as-is; a script that
            # feeds z-scored CO₂ would instead pass scaler mean/std and
            # 280/1200 here to recover the same [0,1] level.
            co2_ppm = co2_raw * (self._co2_max - self._co2_min) + self._co2_min   # [B,1] real ppm
            if co2_override_ppm is not None:
                co2_ppm = co2_override_ppm      # explicit ppm override (bypasses batch injection)
            if self.use_multiscale_co2:
                # Multiscale CO2 features: magnitude / trend / distribution -> [B,3]
                win = (co2_win * (self._co2_max - self._co2_min) + self._co2_min).squeeze(-1)   # [B,L] real ppm
                last = win[:, -1]
                logf = torch.log(last.clamp_min(1.0) / self.co2_ref)
                span = (self._co2_max - self._co2_min) + 1e-6
                trendf = (last - win[:, 0]) / span
                anomf = (last - win.mean(dim=-1)) / span
                co2_level = torch.stack([logf, trendf, anomf], dim=-1)        # [B,3]
            else:
                co2_level = (co2_ppm - self._co2_min) / (self._co2_max - self._co2_min)  # [B,1]
            shift_info = self.shift_encoder(temporal_context, co2_level)
            # Physically-anchored log-forcing contribution (observation-calibrated slope).
            # Added DIRECTLY to the prediction (not via pred_bias) so it is never
            # shrunk by shift_bias_scale or absorbed into the learned bias.
            anchor_K = (self.co2_anchor_a *
                        torch.log(co2_ppm.squeeze(-1).clamp_min(1.0) / self.co2_ref)) if self.co2_anchor_a else None
        else:
            shift_info = {"time_modulation": None,
                          "pred_bias": torch.zeros(B, device=device),
                          "co2_timebias": None}
            anchor_K = None

        # CO₂ monotonicity regularisation: penalise any learned CO₂→bias
        # response that decreases with rising CO₂.  Computed only during
        # training (finite-difference perturbation of the shift encoder),
        # exposed as ``aux_loss`` for the Trainer to add to the total.
        mono_loss = None
        if (self.use_mono_reg and self.training and self.use_shift_encoder
                and self.shift_encoder is not None and co2_raw is not None):
            mono_loss = self.mono_weight * self.shift_encoder.monotonicity_regularization(
                temporal_context, co2_level,
                delta=self.mono_delta, margin=self.mono_margin)

        # ④ Periodic Anchor Attention
        # Prefer the dataset-computed per-sample anchor gaps (days).  Fallback
        # to slot-index gaps only for direct calls that don't provide them.
        if daily_gaps is None:
            daily_gaps = torch.arange(1, K_d + 1, device=device, dtype=torch.float32)
        else:
            daily_gaps = daily_gaps.to(device=device, dtype=torch.float32)
        if yearly_gaps is None:
            yearly_gaps = torch.arange(1, K_y + 1, device=device, dtype=torch.float32) * 365.0
        else:
            yearly_gaps = yearly_gaps.to(device=device, dtype=torch.float32)

        query, periodic_ctx = self.periodic_attention(
            query=query, daily_nodes=daily_nodes, yearly_nodes=yearly_nodes,
            daily_mask=daily_mask, yearly_mask=yearly_mask,
            temporal_context=temporal_context,
            daily_gaps=daily_gaps, yearly_gaps=yearly_gaps,
            shift_time_mod=shift_info["time_modulation"],
            co2_timebias=(shift_info.get("co2_timebias")
                          if self.use_co2_timebias else None))

        # ⑤ Dynamic Memory
        if self.use_memory:
            periodic_bias = periodic_ctx.mean(dim=1)
            memory_out = self.memory(temporal_context, periodic_bias)
        else:
            memory_out = torch.zeros_like(temporal_context)

        # ⑥ Multi-Source Fusion
        temporal_expanded = temporal_context.unsqueeze(1).expand(B, H, D)
        memory_expanded = memory_out.unsqueeze(1).expand(B, H, D)
        if self.memory_hgate is not None:
            _lead = torch.arange(H, device=device).float().view(1, H, 1) \
                / max(H - 1, 1)
            memory_expanded = memory_expanded * torch.sigmoid(
                self.memory_hgate(_lead))
        # Lead-conditioned anchor scaling: de-emphasise geophysical anchors at short
        # leads (persistence-dominated) and ramp back up toward long leads.
        periodic_fusion = periodic_ctx
        if self.lead_anchor_scale and H > 1:
            _hs = torch.arange(H, device=device).float()
            _sc = (self.anchor_scale_min
                   + (1.0 - self.anchor_scale_min) * _hs / (H - 1))   # [H]
            periodic_fusion = periodic_ctx * _sc.view(1, -1, 1)
        fused, fusion_gate = self.fusion(
            temporal_ctx=temporal_expanded,
            periodic_ctx=periodic_fusion,
            memory_ctx=memory_expanded,
            future_calendar=future_calendar)

        # ⑦ Decode + shift bias
        # When anchors, memory, and shift are all absent, the fusion+decoder
        # skeleton degrades the GRU signal (dead input channels, fixed gates).
        # Use the direct head instead — matches SimpleLSTM baseline.
        has_anchors = (K_d + K_y) > 0
        use_direct = not has_anchors and not self.use_memory and not self.use_shift_encoder

        if self.probabilistic:
            # SHASH distribution head: (mu, sigma, gamma, tau) per horizon step.
            mu, sigma, gamma, tau = self.shash_head(
                fused, periodic_ctx, future_calendar)
            prediction = mu + shift_info["pred_bias"].unsqueeze(1)
        elif use_direct:
            prediction = self.direct_head(temporal_context)  # [B, H]
        else:
            prediction = self.decoder(fused, periodic_ctx, future_calendar,
                                      temporal_context)
            prediction = prediction + self.shift_bias_scale * shift_info["pred_bias"].unsqueeze(1)

        # N-HiTS-flavoured multi-rate coarse branch: pool fused over horizon windows,
        # forecast coarsely, interpolate back to H, and blend via a lead-aware gate
        # (coarse -> long range, fine -> short lead).
        if self.use_multirate and fused is not None and fused.shape[1] == prediction.shape[1]:
            Hp = prediction.shape[1]                     # actual forecast horizon (e.g. 120)
            R = max(1, self.multirate_pool)
            _cf = F.avg_pool1d(fused.transpose(1, 2), kernel_size=R, stride=R,
                               ceil_mode=True).transpose(1, 2)          # [B, nc, D]
            _cp = self.coarse_head(_cf).squeeze(-1)                      # [B, nc]
            _cu = F.interpolate(_cp.unsqueeze(1), size=Hp, mode="linear",
                                align_corners=False).squeeze(1)          # [B, Hp]
            _hn = (torch.arange(Hp, device=device).float() / max(Hp - 1, 1)).view(1, Hp, 1)
            _gc = torch.cat([future_calendar[:, :Hp], _hn.expand(B, Hp, 1)], dim=-1)  # [B,Hp,cal+1]
            _g = torch.sigmoid(self.multirate_gate(_gc)).squeeze(-1)     # [B,Hp]
            prediction = (1.0 - _g) * prediction + _g * _cu              # [B,Hp]

        # N-HiTS multi-stack: two coarse branches at different rates (fine band -> short
        # lead, coarse band -> long lead), blended with the main (fine) prediction by a
        # 3-way per-horizon softmax gate.
        if self.use_multiband and fused is not None and fused.shape[1] == prediction.shape[1]:
            Hp = prediction.shape[1]
            def _band(pool, head):
                R = max(1, pool)
                _bf = F.avg_pool1d(fused.transpose(1, 2), kernel_size=R, stride=R,
                                   ceil_mode=True).transpose(1, 2)          # [B, nc, D]
                _bp = head(_bf).squeeze(-1)                                  # [B, nc]
                return F.interpolate(_bp.unsqueeze(1), size=Hp, mode="linear",
                                     align_corners=False).squeeze(1)         # [B, Hp]
            _b1 = _band(self.band_pool1, self.band_head1)
            _b2 = _band(self.band_pool2, self.band_head2)
            _hn = (torch.arange(Hp, device=device).float() / max(Hp - 1, 1)).view(1, Hp, 1)
            _bc = torch.cat([future_calendar[:, :Hp], _hn.expand(B, Hp, 1)], dim=-1)
            _w = torch.softmax(self.band_gate(_bc), dim=-1)                  # [B,Hp,3]
            prediction = (_w[..., 0] * prediction + _w[..., 1] * _b1
                          + _w[..., 2] * _b2)                                # [B,Hp]

        # ⑧c Dual-branch forecast: a SHORT-lead branch (short-residual blend toward the last
        # observed value) vs the LONG-range decoder, combined by a per-horizon softmax gate.
        # Lets the model route short leads through a persistence-friendly branch and long leads
        # through the structured decoder, learning the split (avoids hard-coding which wins).
        if (self.use_dual_forecast and last_target is not None):
            pred_long = prediction                                     # [B,H] decoder (+shift bias)
            k = min(self.dual_short_steps, H)
            _mh = torch.zeros(H, device=device); _mh[:k] = 1.0
            _gh = torch.sigmoid(self.dual_short_gate) * _mh            # [H] learned, masked to short lead
            pred_short = pred_long + _gh.unsqueeze(0) * (
                last_target.unsqueeze(1) - pred_long)                  # [B,H]
            _hn = (torch.arange(H, device=device).float() / max(H - 1, 1)).view(1, H, 1)  # [1,H,1]
            _cond = torch.cat([future_calendar, _hn.expand(B, H, 1)], dim=-1)              # [B,H,cal+1]
            _w = torch.softmax(self.dual_gate(_cond), dim=-1)          # [B,H,2]
            prediction = _w[..., 0] * pred_short + _w[..., 1] * pred_long   # [B,H]

        # ⑧b Short-lead persistence blend: deterministic hard-decay gate that pulls the
        # first few steps toward the last observed value (persistence).  Fixes short-lead
        # starvation / cycle-pull; hard-zero beyond persist_blend_steps so long leads are
        # untouched (unlike the learnable short-residual gate, this is stable).
        if (self.use_persist_blend and last_target is not None):
            k = min(self.persist_blend_steps, H)
            _hb = torch.arange(H, device=device).float()
            _gb = torch.clamp(1.0 - _hb / max(k, 1), min=0.0)         # [H] 1 -> 0 linear
            prediction = prediction + _gb.unsqueeze(0) * (
                last_target.unsqueeze(1) - prediction)

        # ⑨ Short-lead residual: blend the forecast toward the persistence
        # forecast (last observed target) with a learned, HARD-MASKED gate over
        # the first short_residual_steps (a few days).  This materially improves
        # the short lead (simple baselines are strong there because they copy the
        # recent value); the mask confines the effect to the short lead.  The
        # learned gate is tuned per step (decaying init), and the mask bounds its
        # reach so the long lead is only mildly affected.
        if (self.use_short_residual and last_target is not None):
            k = min(self.short_residual_steps, H)
            gate = torch.sigmoid(self.short_res_gate)                 # [H]
            mask = torch.zeros(H, device=device); mask[:k] = 1.0
            gate = gate * mask
            prediction = prediction + gate.unsqueeze(0) * (
                last_target.unsqueeze(1) - prediction)

        # Analog memory: pull the point/mu forecast toward the retrieved
        # historical trajectory by the learned gate.  ``prediction`` lives in
        # target (temperature) space; the analog forecast is the mean of the
        # observed next-H windows of the top-K similar training states.
        analog_gate = None
        if self.use_analog_memory:
            am = self.analog_memory(temporal_context)
            analog_forecast = am["analog"]                    # [B, H]
            analog_gate = am["gate"]                          # [B, 1]
            prediction = prediction + analog_gate * (analog_forecast - prediction)

        # Physically-anchored CO2 log-forcing term — added AFTER the short-
        # residual / analog blends so it is not diluted by the persistence pull
        # and every horizon step carries the physics-based CO2 response.
        if anchor_K is not None:
            prediction = prediction + anchor_K.unsqueeze(1)

        out = {
            "prediction": prediction,
            "temporal_context": temporal_context,
            "periodic_context": periodic_ctx,
            "memory_out": memory_out,
            "fusion_gate": fusion_gate,
            "shift_bias": shift_info["pred_bias"],
            "co2_anchor_K": anchor_K if anchor_K is not None else torch.zeros(B, device=device),
            "aux_loss": mono_loss,
        }
        if self.probabilistic:
            out["dist_params"] = torch.stack([prediction, sigma, gamma, tau], dim=-1)
            out["shash_heads"] = (sigma, gamma, tau)
        if analog_gate is not None:
            out["analog_gate"] = analog_gate
            # Expose the analog forecast for downstream gate/interpretability
            # analysis (diagnostics, not used for training).
            out["analog_forecast"] = analog_forecast
        return out

    @torch.no_grad()
    def build_analog_bank(self, loader, forward_fn, device, max_entries=20000):
        """Populate the analog memory bank from training origins (no leakage).

        Runs the model in eval mode over the train loader, collecting the
        GRU temporal context (key) and the observed next-H target window
        (value) for each origin.  Called by the Trainer once per epoch so the
        keys co-evolve with the encoder; values are fixed observations and are
        never drawn from the validation/test splits.
        """
        if not self.use_analog_memory or not hasattr(self, "analog_memory"):
            return
        was_training = self.training
        self.eval()
        keys, values = [], []
        total = 0
        for batch in loader:
            dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
            out = forward_fn(self, dev)
            # .float(): analog keys/values must not be fp16 (AMP emits fp16).
            keys.append(out["temporal_context"].detach().float().cpu().numpy())
            values.append(dev["target"].detach().float().cpu().numpy())
            total += keys[-1].shape[0]
            if total >= max_entries:
                break
        self.train(was_training)
        keys = np.concatenate(keys, axis=0)[:max_entries]
        values = np.concatenate(values, axis=0)[:max_entries]
        self.analog_memory.build_bank(keys, values)


# ═══════════════════════════════════════════════════════════════════
# Wrapper
# ═══════════════════════════════════════════════════════════════════

class MTTGNetWrapper(nn.Module):
    def __init__(self, num_features, hidden_dim=96, horizon=28, seq_len=28,
                 dropout=0.15, memory_slots=128, mem_top_k=8, calendar_dim=5,
                 num_gru_layers=3, use_gate=True, use_time_bias=True,
                 use_memory_horizon_gate=False,
                 memory_gate_init=-3.0,
                 use_memory=True, use_variable_encoder=True,
                 use_shift_encoder=True, use_var_gnn=True,
                 use_calendar=True, temporal_encoder="gru",
                 var_encoder_type="attention", gat_version="gat",
                 probabilistic=False, sigma_floor=1e-3,
                 use_mono_reg=False, mono_weight=0.1,
                 mono_delta=0.05, mono_margin=0.0,
                 use_co2_timebias=False,
                 use_short_residual=False, target_index=0,
                 short_residual_steps=8, decoder_type="calendar_mlp",
                 use_persist_blend=False, persist_blend_steps=8,
                 lead_anchor_scale=False, anchor_scale_min=0.3,
                 use_dual_forecast=False, dual_short_steps=8,
                 use_multirate=False, multirate_pool=4,
                 use_multiband=False, band_pool1=2, band_pool2=8,
                 time_bias_scale=1.0, shift_bias_scale=1.0,
                 use_analog_memory=False, analog_top_k=8,
                 analog_temperature=1.0,
                 use_multiscale_co2=False, co2_anchor_a=0.0, co2_ref=284.0,
                 use_future_co2=False,
                 **kwargs):
        super().__init__()
        self.use_future_co2 = bool(use_future_co2)
        if self.use_future_co2 and not use_calendar:
            # The inner model zeroes future_calendar wholesale when calendar is
            # ablated, which would silently zero the CO2 pathway too.  Refuse the
            # combination rather than produce a run labelled future_co2 that
            # never sees one.
            raise ValueError(
                "use_future_co2=True requires use_calendar=True: the CO2 pathway "
                "travels inside the per-horizon calendar vector, which the "
                "calendar ablation zeroes.")
        if self.use_future_co2:
            # The CO2 level at each horizon step is appended to the calendar
            # vector, so the per-horizon query and every decoder that reads
            # calendar_dim (gate, multirate, dual, residual) become CO2-aware
            # without further changes.
            calendar_dim = int(calendar_dim) + 1
        self._co2_calendar_index = -1 if self.use_future_co2 else None
        self.use_multiscale_co2 = use_multiscale_co2
        self.co2_anchor_a = float(co2_anchor_a)
        self.co2_ref = float(co2_ref)
        # v5-restored is deliberately GRU-locked.  The previous code accepted
        # ``temporal_encoder`` and silently ignored it, so encoder-variant runs
        # produced five byte-identical GRU models labelled LSTM/TCN/Mamba/MLP.
        if temporal_encoder not in (None, "gru"):
            raise NotImplementedError(
                f"temporal_encoder={temporal_encoder!r} is not implemented in "
                "the v5-restored MTTGNet (GRU-only). Pass 'gru' or omit the "
                "argument; don't rely on a silently-ignored flag."
            )
        self.model = MTTGNetModel(
            num_vars=num_features, seq_len=seq_len, horizon=horizon,
            d_model=hidden_dim, dropout=dropout,
            memory_slots=memory_slots, mem_top_k=mem_top_k,
            calendar_dim=calendar_dim, num_gru_layers=num_gru_layers,
            use_memory_horizon_gate=use_memory_horizon_gate,
            memory_gate_init=memory_gate_init,
            use_gate=use_gate, use_time_bias=use_time_bias,
            use_memory=use_memory,
            use_variable_encoder=use_variable_encoder,
            use_shift_encoder=use_shift_encoder,
            use_var_gnn=use_var_gnn,
            use_calendar=use_calendar,
            var_encoder_type=var_encoder_type,
            gat_version=gat_version,
            probabilistic=probabilistic, sigma_floor=sigma_floor,
            use_mono_reg=use_mono_reg, mono_weight=mono_weight,
            mono_delta=mono_delta, mono_margin=mono_margin,
            use_co2_timebias=use_co2_timebias,
            use_multiscale_co2=use_multiscale_co2,
            co2_anchor_a=co2_anchor_a, co2_ref=co2_ref,
            use_short_residual=use_short_residual,
            target_index=target_index,
            short_residual_steps=short_residual_steps,
            decoder_type=decoder_type,
            time_bias_scale=time_bias_scale, shift_bias_scale=shift_bias_scale,
            use_analog_memory=use_analog_memory,
            analog_top_k=analog_top_k, analog_temperature=analog_temperature,
            use_persist_blend=use_persist_blend, persist_blend_steps=persist_blend_steps,
            lead_anchor_scale=lead_anchor_scale, anchor_scale_min=anchor_scale_min,
            use_dual_forecast=use_dual_forecast, dual_short_steps=dual_short_steps,
            use_multirate=use_multirate, multirate_pool=multirate_pool,
            use_multiband=use_multiband, band_pool1=band_pool1, band_pool2=band_pool2,
            **kwargs)

    def forward(self, x_recent, daily_anchor=None, yearly_anchor=None,
                daily_mask=None, yearly_mask=None, future_calendar=None,
                daily_gaps=None, yearly_gaps=None, co2_override_ppm=None,
                future_co2=None, **kwargs):
        B, L, N = x_recent.shape
        H = future_calendar.shape[1] if future_calendar is not None else 28
        device = x_recent.device
        if daily_anchor is None:
            daily_anchor = torch.zeros(B, H, 0, N, device=device)
        if yearly_anchor is None:
            yearly_anchor = torch.zeros(B, H, 0, N, device=device)
        if daily_mask is None:
            daily_mask = torch.zeros(B, H, 0, dtype=torch.bool, device=device)
        if yearly_mask is None:
            yearly_mask = torch.zeros(B, H, 0, dtype=torch.bool, device=device)
        if future_calendar is None:
            future_calendar = torch.zeros(B, H, 5, device=device)
        if self.use_future_co2:
            if future_co2 is None:
                raise ValueError(
                    "use_future_co2=True but the batch carried no future_co2. "
                    "Set dataset.co2_index in the config so "
                    "PeriodicAnchorDataset emits it (and pass it through in "
                    "anchor_forward / the rollout).")
            future_calendar = torch.cat(
                [future_calendar,
                 future_co2.reshape(B, H, 1).to(future_calendar.dtype)], dim=-1)
        return self.model(
            x_recent=x_recent, daily_anchor=daily_anchor,
            yearly_anchor=yearly_anchor, daily_mask=daily_mask,
            yearly_mask=yearly_mask, future_calendar=future_calendar,
            daily_gaps=daily_gaps, yearly_gaps=yearly_gaps,
            co2_override_ppm=co2_override_ppm)
