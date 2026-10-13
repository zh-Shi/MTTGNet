"""Paper-faithful ("clean") MTTGNet variant — the code following the paper, not the reverse.

Motivation (2026-09 code audit): five places where src/mttgnet.py's semantics
drifted from what paper.txt describes.  This module re-implements each drift so
that the CODE matches the PAPER's claim, as a separately trained experimental
model (``arch: clean``).  src/mttgnet.py is left byte-compatible with the
already-trained headline results; nothing here changes those numbers.

Deviations from src/mttgnet.py, each tied to the paper claim it restores:

1. Query refinement is real (paper Eq. 3).
   Legacy: the fusion gate rescales the periodic context in place,
   ``c' = LN(c + g ⊙ c)`` — the query is returned untouched (the original
   query-refinement output was discarded by the caller, so the gate was
   rewired; see the comment in legacy PeriodicAnchorAttention.forward).
   Clean: ``g = σ(MLP_gate([q, c_D, c_Y]))``;  ``q' = LN(q + g ⊙ c_periodic)``
   followed by the same feed-forward sub-layer, and the REFINED QUERY is what
   the multi-source fusion consumes as the periodic source.  Paper Eq. (3)
   plus Sec. III-F as written leave q' dangling; we interpret the refined
   query as the periodic source so the gate actually controls how much
   periodic information enters the forecast.

2. Memory gate == memory off (paper's "fusion gate initialised so the
   retrieved prototypes enter at weight σ(-3)≈0.05").
   Legacy DynamicMemory: ``LN(x + out_proj(gate·r + (1-gate)·x))`` — at
   gate=0 the block is still a residual adapter ``LN(x + W x)``.
   Clean: ``x + gate · out_proj(retrieved)`` — gate=0 is an exact identity,
   so the prototype-memory ablation measures prototype retrieval and nothing
   else.  Same parameter count as legacy (the (1-gate)·x term carries none).

3. No decoder bypass (paper Sec. III-F: the decoder maps the FUSED
   representation).  Legacy CalendarConditionedDecoder concatenates the raw
   periodic context next to the fused vector, so the periodic source reaches
   the decoder even when its fusion weight is zero.  Clean decoder input is
   ``[fused, calendar]`` only.  ``decoder_bypass: true`` restores the legacy
   wiring for an A/B comparison.

4. Time-distance bias as written in paper Eq. (2): an additive scalar term
   ``MLP_time(log(1+Δt))`` on the attention score.  Legacy injects a vector
   encoding of the gap into the key before the dot product (a
   query-dependent bias).  ``time_bias_mode: key`` selects the legacy form.

5. CO₂ bias branch measures CO₂ (paper Sec. III-E: "b_shift ... captures the
   slow background trend").  Legacy pred_bias = f(h, c) with the whole
   temporal context as input, so the branch can fit state information that
   has nothing to do with CO₂.  Two clean modes:
     - ``counterfactual``: pred_bias = f(h, c) − f(h, c_ref), zero at the
       reference level (280 ppm) — the incremental response conditional on
       the atmospheric state.
     - ``softplus_log``: pred_bias = softplus(g(h)) · ln(C / 280), monotone
       in CO₂ BY CONSTRUCTION (the claim the soft penalty could not support).
   ``learned`` keeps the legacy behaviour.

6. Optional ``gnn_dynamic: true``: per-timestep state-dependent adjacency
   A_t = σ(P1(x_t) P2(x_t)ᵀ) — the paper's "per-timestep graph convolution"
   made literal (legacy adjacency is globally fixed).

Everything else (anchors, masks, true time gaps, calendar query, variable
encoder, multi-source fusion, short-lead objective in the Trainer, future-CO₂
pathway) is reused unchanged from src/mttgnet.py.  Legacy experimental
branches (multirate / multiband / dual / persist-blend / short-residual /
analog / SHASH) are NOT carried over and fail loudly if a config asks for
them.

Standalone smoke test (no training data needed):
    python src/mttgnet_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mttgnet import (
    AnchorVariableEncoder,
    CalendarQueryEncoder,
    DistributionShiftEncoder,
    DynamicMemory,
    MultiSourceGateFusion,
    VariableGNN,
)


# ═══════════════════════════════════════════════════════════════════
# 1+4. Periodic attention: query refinement (Eq. 3) + additive time bias (Eq. 2)
# ═══════════════════════════════════════════════════════════════════

class RefinePeriodicAttention(nn.Module):
    """Wraps the legacy PeriodicAnchorAttention modules, with two switches.

    time_bias_mode:
        "additive"  — paper Eq. (2): scores += MLP_time(log(1+Δt)); a scalar,
                      query-independent time-distance term.
        "key"       — legacy: a vector encoding of the gap is added to the key
                      before the dot product (query-dependent bias).

    query_refine:
        True   — paper Eq. (3): the gate refines the QUERY and the refined
                 query is the periodic source consumed by the fusion gate.
        False  — the periodic context itself is the source (legacy output
                 contract); the legacy in-place rescale gate is NOT applied.
    """

    def __init__(self, hidden_dim, dropout, use_gate=True, use_time_bias=True,
                 time_bias_scale=1.0, time_bias_mode="additive",
                 query_refine=True):
        super().__init__()
        self.time_bias_mode = time_bias_mode
        self.query_refine = query_refine
        self.attn = nn.Module()  # namespace so module names match legacy 1:1
        D = hidden_dim
        a = self.attn
        a.use_gate = use_gate
        a.use_time_bias = use_time_bias
        a.time_bias_scale = float(time_bias_scale)
        a.query_projection = nn.Linear(D, D)
        a.daily_key = nn.Linear(D, D)
        a.daily_value = nn.Linear(D, D)
        a.daily_relation = nn.Parameter(torch.randn(D) * 0.02)
        a.yearly_key = nn.Linear(D, D)
        a.yearly_value = nn.Linear(D, D)
        a.yearly_relation = nn.Parameter(torch.randn(D) * 0.02)
        if use_time_bias:
            a.time_encoder = nn.Sequential(          # legacy: vector form (key mode)
                nn.Linear(1, D // 2), nn.GELU(), nn.Linear(D // 2, D))
            self.time_score_mlp = nn.Sequential(     # clean: scalar form (additive mode)
                nn.Linear(1, D // 2), nn.GELU(), nn.Linear(D // 2, 1))
        a.similarity_proj = nn.Linear(D, D)
        if use_gate:
            a.gate_mlp = nn.Sequential(
                nn.Linear(D * 3, D), nn.GELU(), nn.Linear(D, D), nn.Sigmoid())
        a.output_projection = nn.Linear(2 * D, D)
        a.dropout = nn.Dropout(dropout)
        a.norm1 = nn.LayerNorm(D)
        a.norm2 = nn.LayerNorm(D)
        a.feed_forward = nn.Sequential(
            nn.Linear(D, 4 * D), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * D, D))
        self.a = a

    @staticmethod
    def _masked_softmax(scores, mask):
        scores = scores.clamp(-50.0, 50.0)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        valid = mask.sum(dim=-1, keepdim=True).to(weights.dtype)
        return weights * (valid > 0).to(weights.dtype)

    def _aggregate(self, query, anchors, mask, key_layer, value_layer,
                   relation_embedding, time_gaps=None, similarity_bias=None,
                   shift_time_mod=None, co2_timebias=None):
        a = self.a
        B, H, K, D = anchors.shape
        q = a.query_projection(query)
        k = key_layer(anchors) + relation_embedding.view(1, 1, 1, -1)
        v = value_layer(anchors)
        scores = torch.sum(q.unsqueeze(2) * k, dim=-1) / math.sqrt(D)
        if a.use_time_bias and time_gaps is not None:
            log_days = torch.log1p(time_gaps).to(scores.dtype).to(scores.device)
            if self.time_bias_mode == "additive":
                # Paper Eq. (2): additive scalar time-distance term.
                # tb: [K] (fallback gaps) or [B,H,K] (per-sample gaps); the
                # CO₂ modulation is a per-sample scalar, reshaped to [B,1,1]
                # so it broadcasts over the horizon and anchor dims of both.
                tb = self.time_score_mlp(log_days.unsqueeze(-1)).squeeze(-1)
                if tb.ndim == 1:
                    tb = tb.view(1, 1, -1)
                if shift_time_mod is not None:
                    tb = tb * (1.0 + shift_time_mod.mean(dim=-1)).view(-1, 1, 1)
                scores = scores + a.time_bias_scale * tb.to(scores.dtype)
            else:  # "key": legacy vector injection (query-dependent)
                time_enc = a.time_encoder(log_days.unsqueeze(-1))
                if time_enc.ndim == 3:
                    time_enc = time_enc.view(1, 1, K, D)
                if shift_time_mod is not None:
                    mod = 1.0 + shift_time_mod
                    time_enc = time_enc * mod.unsqueeze(1).unsqueeze(1)
                k = k + a.time_bias_scale * time_enc
                scores = torch.sum(q.unsqueeze(2) * k, dim=-1) / math.sqrt(D)
        if similarity_bias is not None:
            scores = scores + similarity_bias * math.sqrt(D) * 0.1
        if co2_timebias is not None and time_gaps is not None and time_gaps.ndim == 3:
            log_gap = torch.log1p(time_gaps).to(scores.dtype)
            log_gap = log_gap - log_gap.mean(dim=-1, keepdim=True)
            scores = scores - co2_timebias.unsqueeze(-1).unsqueeze(-1) * log_gap * 0.3
        weights = self._masked_softmax(scores, mask)
        return torch.sum(weights.unsqueeze(-1) * v, dim=2)

    def _similarity(self, temporal_context, anchors):
        a = self.a
        q = F.normalize(a.similarity_proj(temporal_context), dim=-1)
        k_norm = F.normalize(anchors, dim=-1)
        return torch.einsum('bd,bhkd->bhk', q, k_norm)

    def forward(self, query, daily_nodes, yearly_nodes, daily_mask, yearly_mask,
                temporal_context=None, daily_gaps=None, yearly_gaps=None,
                shift_time_mod=None, co2_timebias=None):
        a = self.a
        daily_sim = (self._similarity(temporal_context, daily_nodes)
                     if temporal_context is not None else None)
        daily_context = self._aggregate(
            query=query, anchors=daily_nodes, mask=daily_mask,
            key_layer=a.daily_key, value_layer=a.daily_value,
            relation_embedding=a.daily_relation, time_gaps=daily_gaps,
            similarity_bias=daily_sim, shift_time_mod=shift_time_mod,
            co2_timebias=co2_timebias)
        yearly_sim = None
        if temporal_context is not None and yearly_nodes.shape[2] > 0:
            yearly_sim = self._similarity(temporal_context, yearly_nodes)
        yearly_context = self._aggregate(
            query=query, anchors=yearly_nodes, mask=yearly_mask,
            key_layer=a.yearly_key, value_layer=a.yearly_value,
            relation_embedding=a.yearly_relation, time_gaps=yearly_gaps,
            similarity_bias=yearly_sim, shift_time_mod=shift_time_mod,
            co2_timebias=co2_timebias)

        c_periodic = a.output_projection(
            torch.cat([daily_context, yearly_context], dim=-1))

        if not self.query_refine:
            # Legacy output contract: the periodic context is the source; no
            # in-place rescale gate (that legacy mechanism is not carried over).
            c_periodic = a.norm2(c_periodic + a.feed_forward(c_periodic))
            return query, c_periodic

        # Paper Eq. (3): the gate refines the QUERY, and the refined query is
        # what the fusion consumes as the periodic source.
        if a.use_gate:
            g = a.gate_mlp(torch.cat([query, daily_context, yearly_context], dim=-1))
            q_refined = a.norm1(query + g * a.dropout(c_periodic))
        else:
            q_refined = a.norm1(query + a.dropout(c_periodic))
        q_refined = a.norm2(q_refined + a.feed_forward(q_refined))
        return q_refined, q_refined


# ═══════════════════════════════════════════════════════════════════
# 2. Clean prototype memory: gate == 0  ⇒  exact identity
# ═══════════════════════════════════════════════════════════════════

class CleanDynamicMemory(nn.Module):
    """x + gate · out_proj(retrieved).

    Same slots / query projection / gate network / out projection as legacy
    DynamicMemory (same parameter count), but the block is an exact identity
    when the gate is zero — so "the memory contributes nothing" and "the gate
    is closed" are the same statement, as the paper's wording assumes.
    """

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
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, float(gate_init_bias))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, periodic_bias=None):
        q = self.query_proj(x)
        scores = F.cosine_similarity(q.unsqueeze(1), self.memory.unsqueeze(0), dim=-1)
        k = min(self.top_k, self.memory.shape[0])
        topk_scores, topk_idx = torch.topk(scores, k, dim=-1)
        w = F.softmax(topk_scores / self.temperature, dim=-1)
        retrieved = (w.unsqueeze(-1) * self.memory[topk_idx]).sum(dim=1)
        if periodic_bias is None:
            periodic_bias = torch.zeros_like(retrieved)
        gate = self.gate(torch.cat([x, retrieved, periodic_bias], dim=-1))
        return x + gate * self.out_proj(retrieved)


# ═══════════════════════════════════════════════════════════════════
# 5. CO₂-conditioned bias: counterfactual / monotone-by-construction
# ═══════════════════════════════════════════════════════════════════

class CleanShiftEncoder(nn.Module):
    """Wraps the legacy DistributionShiftEncoder and replaces pred_bias.

    co2_mode:
        "counterfactual" — pred_bias(h, c) = f(h, c) − f(h, c_ref); zero at
            the reference level, so the branch is the INCREMENTAL response to
            CO₂ conditional on the atmospheric state.  The soft monotonicity
            penalty applies to the increment (equivalent to f itself).
        "softplus_log"   — pred_bias = softplus(g(h)) · ln(C / C_ref),
            non-decreasing in CO₂ by construction; no penalty needed.
        "learned"        — legacy f(h, c) unchanged.
    time_modulation / co2_timebias always come from the legacy encoder (the
    CO₂-state conditioning of the anchor attention is unchanged).
    """

    def __init__(self, hidden_dim, co2_dim=1, co2_mode="counterfactual",
                 co2_ref_ppm=280.0):
        super().__init__()
        if co2_mode not in ("counterfactual", "softplus_log", "learned"):
            raise ValueError(f"unknown co2_mode: {co2_mode!r}")
        if co2_mode != "learned" and co2_dim != 1:
            raise ValueError(
                "co2_mode counterfactual/softplus_log require the scalar CO₂ "
                "level (use_multiscale_co2: false).")
        self.enc = DistributionShiftEncoder(hidden_dim, co2_dim)
        self.co2_mode = co2_mode
        self.co2_ref_ppm = float(co2_ref_ppm)
        if co2_mode == "softplus_log":
            self.slope_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1))

    def forward(self, temporal_context, co2_level, co2_ppm=None,
                co2_level_ref=None):
        out = self.enc(temporal_context, co2_level)
        if self.co2_mode == "learned":
            return out
        if self.co2_mode == "counterfactual":
            if co2_level_ref is None:
                raise ValueError(
                    "counterfactual mode needs co2_level_ref (the normalised "
                    "reference level, computed by the model from co2_ref_ppm).")
            out["pred_bias"] = out["pred_bias"] - self.enc(
                temporal_context, co2_level_ref)["pred_bias"]
            return out
        # softplus_log: monotone by construction
        slope = F.softplus(self.slope_head(temporal_context)).squeeze(-1)  # [B]
        log_forcing = torch.log(co2_ppm.squeeze(-1).clamp_min(1.0)
                                / self.co2_ref_ppm)                        # [B]
        out["pred_bias"] = slope * log_forcing
        return out

    def monotonicity_regularization(self, temporal_context, co2_level,
                                    co2_ppm=None, co2_level_ref=None,
                                    delta=0.05, margin=0.0):
        if self.co2_mode == "softplus_log":
            return torch.zeros((), device=temporal_context.device)
        base = self.forward(temporal_context, co2_level, co2_ppm,
                            co2_level_ref)["pred_bias"]
        up = self.forward(temporal_context, co2_level + delta, co2_ppm,
                          co2_level_ref)["pred_bias"]
        return torch.relu(base - up + margin).mean()


# ═══════════════════════════════════════════════════════════════════
# 6. Optional: per-timestep state-dependent variable graph
# ═══════════════════════════════════════════════════════════════════

class DynamicVariableGNN(nn.Module):
    """A_t = σ(P1(x_t) P2(x_t)ᵀ) per timestep — the "per-timestep graph
    convolution with learned adjacency" of the paper, literally.  Otherwise
    mirrors legacy VariableGNN (message passing → LayerNorm → scalar head,
    learnable residual weight)."""

    def __init__(self, num_vars, d_model, dropout=0.1):
        super().__init__()
        self.msg_in = nn.Linear(1, d_model)
        self.adj_p1 = nn.Linear(d_model, d_model // 2)
        self.adj_p2 = nn.Linear(d_model, d_model // 2)
        self.msg_out = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_recent):
        B, L, N = x_recent.shape
        x_flat = x_recent.reshape(B * L, N, 1)
        h = self.msg_in(x_flat)                                   # [B*L, N, D]
        e1 = self.adj_p1(h)
        e2 = self.adj_p2(h)
        aff = torch.sigmoid(torch.bmm(e1, e2.transpose(1, 2)))    # [B*L, N, N]
        aff = aff * (1 - torch.eye(N, device=aff.device).unsqueeze(0))
        adj = aff / (aff.sum(dim=-1, keepdim=True) + 1e-8)
        h = torch.bmm(adj, h)
        h = self.norm(h)
        out = self.msg_out(self.dropout(h)).reshape(B, L, N)
        return x_recent + self.alpha * out


# ═══════════════════════════════════════════════════════════════════
# 3. Decoder without the periodic bypass
# ═══════════════════════════════════════════════════════════════════

class CleanDecoder(nn.Module):
    """Per-horizon MLP on [fused, calendar] (paper Sec. III-F).

    decoder_bypass=True appends the periodic context, restoring the legacy
    wiring for A/B comparison; False is the paper-faithful default.
    """

    def __init__(self, hidden_dim, horizon, calendar_dim=5, dropout=0.1,
                 bypass=False):
        super().__init__()
        self.bypass = bypass
        in_dim = hidden_dim * (2 if bypass else 1) + calendar_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, fused, periodic_ctx, future_calendar, temporal_context=None):
        parts = [fused] + ([periodic_ctx] if self.bypass else []) + [future_calendar]
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════════

class MTTGNetTestModel(nn.Module):
    def __init__(self, num_vars, seq_len, horizon,
                 d_model=96, num_heads=4, dropout=0.15,
                 memory_slots=128, mem_top_k=8, calendar_dim=5,
                 num_gru_layers=2, use_gate=True, use_time_bias=True,
                 use_memory=True, use_variable_encoder=True,
                 use_shift_encoder=True, use_var_gnn=True,
                 use_calendar=True, co2_index=None,
                 var_encoder_type="gat", gat_version="gat",
                 time_bias_mode="additive", query_refine=True,
                 mem_clean=True, decoder_bypass=False,
                 co2_mode="counterfactual", gnn_dynamic=False,
                 co2_ref_ppm=280.0,
                 use_mono_reg=False, mono_weight=0.1,
                 mono_delta=0.05, mono_margin=0.0,
                 use_co2_timebias=False, target_index=0,
                 time_bias_scale=1.0, shift_bias_scale=1.0,
                 memory_gate_init=-3.0,
                 co2_min=280.0, co2_max=1200.0,
                 use_multiscale_co2=False, **kwargs):
        super().__init__()
        if use_multiscale_co2:
            raise NotImplementedError(
                "mttgnet_test: use_multiscale_co2 is not supported (the "
                "counterfactual reference level is undefined for the 3-dim "
                "feature). Use the scalar pathway.")
        if co2_mode != "learned" and not use_shift_encoder:
            # Ablation rows legitimately drop the whole bias branch; the mode
            # switch is then moot (inherited from the base package).  Normalise
            # instead of refusing — the no_shift row is exactly this case.
            co2_mode = "learned"
        # CO₂ arrives [0,1]-normalised over 280–1200 ppm from the Trainer; the
        # counterfactual/log modes need real ppm, so the range must be real.
        # Discriminator: the identity range has max=1, real ppm has max=1200.
        if co2_mode != "learned" and co2_max <= 500.0:
            raise ValueError(
                f"co2_mode={co2_mode!r} needs co2_min/co2_max = 280/1200 "
                f"(ppm), got {co2_min}/{co2_max}.")
        self.use_memory = use_memory
        self.use_variable_encoder = use_variable_encoder
        self.use_shift_encoder = use_shift_encoder
        self.co2_attention_modulation = bool(kwargs.pop('co2_attention_modulation',True))
        self.co2_additive_bias = bool(kwargs.pop('co2_additive_bias',True))
        from src.fusion_config import resolve_fusion_sources
        fusion_sources = resolve_fusion_sources(use_memory, kwargs.pop('fusion_sources', None))
        self.use_calendar = use_calendar
        self.use_co2_timebias = use_co2_timebias
        self.probabilistic = False
        self.use_mono_reg = use_mono_reg
        self.mono_weight = float(mono_weight)
        self.mono_delta = float(mono_delta)
        self.mono_margin = float(mono_margin)
        self.shift_bias_scale = float(shift_bias_scale)
        self._target_index = int(target_index)
        self._co2_index = co2_index
        self._co2_min = float(co2_min)
        self._co2_max = float(co2_max)
        self.co2_mode = co2_mode
        self.co2_ref_ppm = float(co2_ref_ppm)

        # ⓪ Variable graph
        if use_var_gnn:
            self.var_gnn = (DynamicVariableGNN(num_vars, d_model, dropout)
                            if gnn_dynamic else VariableGNN(num_vars, d_model, dropout))
        else:
            self.var_gnn = None

        # ① GRU temporal encoder
        self.input_proj = nn.Linear(num_vars, d_model)
        self.gru = nn.GRU(d_model, d_model, num_gru_layers, batch_first=True,
                          dropout=dropout if num_gru_layers > 1 else 0.0)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_dropout = nn.Dropout(dropout)

        # ② Variable encoder on anchors
        self.anchor_var_encoder = (AnchorVariableEncoder(
            num_vars, d_model, num_heads, dropout,
            var_encoder_type=var_encoder_type, gat_version=gat_version)
            if use_variable_encoder else None)

        # ③ Calendar query encoder (unchanged)
        self.query_encoder = CalendarQueryEncoder(calendar_dim, d_model, horizon)
        self.local_to_query = nn.Linear(d_model, d_model)

        # ④ Periodic attention (clean: Eq.2 + Eq.3)
        self.periodic_attention = RefinePeriodicAttention(
            d_model, dropout, use_gate=use_gate, use_time_bias=use_time_bias,
            time_bias_scale=time_bias_scale, time_bias_mode=time_bias_mode,
            query_refine=query_refine)

        # ④b CO₂ shift encoder (clean modes)
        self.shift_encoder = (CleanShiftEncoder(
            d_model, co2_dim=1, co2_mode=co2_mode, co2_ref_ppm=co2_ref_ppm)
            if use_shift_encoder else None)

        # ⑤ Prototype memory (clean: gate==0 ⇒ identity)
        if use_memory:
            mem_cls = CleanDynamicMemory if mem_clean else DynamicMemory
            self.memory = mem_cls(d_model, memory_slots, mem_top_k,
                                  gate_init_bias=memory_gate_init)

        # ⑥ Original three-output gate; memory-free runs supply a zero third source.
        # Explicit fusion_sources=2 remains available as a separate ablation.
        self.fusion = MultiSourceGateFusion(d_model, horizon, calendar_dim, num_sources=fusion_sources)

        # ⑦ Decoder (clean: no periodic bypass)
        self.decoder = CleanDecoder(d_model, horizon, calendar_dim, dropout,
                                    bypass=decoder_bypass)

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, x_recent, daily_anchor, yearly_anchor,
                daily_mask, yearly_mask, future_calendar,
                daily_gaps=None, yearly_gaps=None, co2_override_ppm=None):
        B, L, N = x_recent.shape
        H = future_calendar.shape[1]
        D = self.input_proj.out_features
        device = x_recent.device

        if not self.use_calendar:
            future_calendar = torch.zeros_like(future_calendar)

        co2_idx = self._co2_index
        if co2_idx is None or co2_idx >= N:
            co2_raw = None
        else:
            co2_raw = x_recent[:, -1, co2_idx:co2_idx + 1]

        if self.var_gnn is not None:
            x_recent = self.var_gnn(x_recent)

        x = self.input_proj(x_recent)
        _, hn = self.gru(x)
        temporal_context = self.temporal_dropout(self.temporal_norm(hn[-1]))

        K_d = daily_anchor.shape[2]
        K_y = yearly_anchor.shape[2]
        if K_d + K_y > 0:
            all_anchors = torch.cat([daily_anchor, yearly_anchor], dim=2)
            if self.anchor_var_encoder is not None:
                all_nodes = self.anchor_var_encoder(all_anchors)
            else:
                all_nodes = all_anchors.mean(dim=-1, keepdim=True).expand(-1, -1, -1, D)
            daily_nodes = all_nodes[:, :, :K_d, :]
            yearly_nodes = all_nodes[:, :, K_d:, :]
        else:
            daily_nodes = daily_anchor.new_zeros(B, H, 0, D)
            yearly_nodes = yearly_anchor.new_zeros(B, H, 0, D)

        query = self.query_encoder(future_calendar)
        query = query + self.local_to_query(temporal_context).unsqueeze(1)

        # CO₂ level (real ppm) and normalised level
        co2_ppm = None
        co2_level = None
        co2_level_ref = None
        if (self.use_shift_encoder and self.shift_encoder is not None
                and co2_raw is not None):
            co2_ppm = co2_raw * (self._co2_max - self._co2_min) + self._co2_min
            if co2_override_ppm is not None:
                co2_ppm = co2_override_ppm.reshape(B, 1).to(co2_raw.dtype)
            co2_level = ((co2_ppm - self._co2_min)
                         / (self._co2_max - self._co2_min)).clamp(0.0, 1.5)
            ref_ppm = torch.full_like(co2_ppm, self.co2_ref_ppm)
            co2_level_ref = ((ref_ppm - self._co2_min)
                             / (self._co2_max - self._co2_min))
            shift_info = self.shift_encoder(temporal_context, co2_level,
                                            co2_ppm=co2_ppm,
                                            co2_level_ref=co2_level_ref)
        else:
            shift_info = {"time_modulation": None,
                          "pred_bias": torch.zeros(B, device=device),
                          "co2_timebias": None}

        if not self.co2_attention_modulation:
            shift_info['time_modulation'] = None
            shift_info['co2_timebias'] = None
        if not self.co2_additive_bias:
            shift_info['pred_bias'] = torch.zeros_like(shift_info['pred_bias'])
        mono_loss = None
        if (self.use_mono_reg and self.training
                and self.shift_encoder is not None and co2_level is not None):
            mono_loss = self.mono_weight * self.shift_encoder.monotonicity_regularization(
                temporal_context, co2_level, co2_ppm=co2_ppm,
                co2_level_ref=co2_level_ref,
                delta=self.mono_delta, margin=self.mono_margin)

        if daily_gaps is None:
            daily_gaps = torch.arange(1, K_d + 1, device=device, dtype=torch.float32)
        else:
            daily_gaps = daily_gaps.to(device=device, dtype=torch.float32)
        if yearly_gaps is None:
            yearly_gaps = torch.arange(1, K_y + 1, device=device,
                                       dtype=torch.float32) * 365.0
        else:
            yearly_gaps = yearly_gaps.to(device=device, dtype=torch.float32)

        query, periodic_source = self.periodic_attention(
            query=query, daily_nodes=daily_nodes, yearly_nodes=yearly_nodes,
            daily_mask=daily_mask, yearly_mask=yearly_mask,
            temporal_context=temporal_context,
            daily_gaps=daily_gaps, yearly_gaps=yearly_gaps,
            shift_time_mod=shift_info["time_modulation"],
            co2_timebias=(shift_info.get("co2_timebias")
                          if self.use_co2_timebias else None))

        if self.use_memory:
            memory_out = self.memory(temporal_context,
                                     periodic_source.mean(dim=1))
        else:
            memory_out = torch.zeros_like(temporal_context)

        temporal_expanded = temporal_context.unsqueeze(1).expand(B, H, D)
        memory_expanded = (memory_out.unsqueeze(1).expand(B, H, D)
                           if self.fusion.num_sources == 3 else None)
        fused, fusion_gate = self.fusion(
            temporal_ctx=temporal_expanded,
            periodic_ctx=periodic_source,
            memory_ctx=memory_expanded,
            future_calendar=future_calendar)

        prediction = self.decoder(fused, periodic_source, future_calendar,
                                  temporal_context)
        prediction = prediction + self.shift_bias_scale * shift_info["pred_bias"].unsqueeze(1)

        return {
            "prediction": prediction,
            "temporal_context": temporal_context,
            "periodic_context": periodic_source,
            "memory_out": memory_out,
            "fusion_gate": fusion_gate,
            "shift_bias": shift_info["pred_bias"],
            "co2_anchor_K": torch.zeros(B, device=device),
            "aux_loss": mono_loss,
        }


# ═══════════════════════════════════════════════════════════════════
# Wrapper (drop-in for MTTGNetWrapper under run_multiseed's arch: clean)
# ═══════════════════════════════════════════════════════════════════

_FORBIDDEN = {
    "probabilistic": "SHASH head",
    "use_analog_memory": "analog memory",
    "use_multirate": "multi-rate decoder",
    "use_multiband": "multi-band decoder",
    "use_dual_forecast": "dual forecast",
    "use_persist_blend": "persistence blend",
    "use_short_residual": "short-lead residual",
    "lead_anchor_scale": "lead anchor scaling",
    "use_memory_horizon_gate": "memory horizon gate",
}


class MTTGNetTestWrapper(nn.Module):
    """Same constructor surface as MTTGNetWrapper (so run_multiseed's
    build_model kw dict forwards unchanged) plus the clean-architecture
    switches.  Legacy experimental branches fail loudly instead of being
    silently ignored."""

    def __init__(self, num_features, hidden_dim=96, horizon=28, seq_len=28,
                 dropout=0.15, memory_slots=128, mem_top_k=8, calendar_dim=5,
                 num_gru_layers=2, use_gate=True, use_time_bias=True,
                 use_memory_horizon_gate=False,
                 memory_gate_init=-3.0,
                 use_memory=True, use_variable_encoder=True,
                 use_shift_encoder=True, use_var_gnn=True,
                 use_calendar=True, temporal_encoder="gru",
                 var_encoder_type="gat", gat_version="gat",
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
                 time_bias_mode="additive", query_refine=True,
                 mem_clean=True, decoder_bypass=False,
                 co2_mode="counterfactual", gnn_dynamic=False,
                 co2_ref_ppm=280.0,
                 **kwargs):
        super().__init__()
        if temporal_encoder not in (None, "gru"):
            raise NotImplementedError(f"temporal_encoder={temporal_encoder!r}: GRU only.")
        if decoder_type != "calendar_mlp":
            raise NotImplementedError(f"decoder_type={decoder_type!r}: clean model has one decoder.")
        if co2_anchor_a:
            raise NotImplementedError(
                "co2_anchor_a (observation-anchored log forcing) is a legacy "
                "path; use co2_mode='softplus_log' for a monotone-by-"
                "construction learned slope.")
        # NOTE: no locals()-based guard here — dict/set comprehensions have
        # their own scope and locals() inside them does NOT see __init__'s
        # arguments (Python 3 scoping; that silently broke the check once).
        flags = {
            "probabilistic": probabilistic,
            "use_analog_memory": use_analog_memory,
            "use_multirate": use_multirate,
            "use_multiband": use_multiband,
            "use_dual_forecast": use_dual_forecast,
            "use_persist_blend": use_persist_blend,
            "use_short_residual": use_short_residual,
            "lead_anchor_scale": lead_anchor_scale,
            "use_memory_horizon_gate": use_memory_horizon_gate,
        }
        enabled = {k: _FORBIDDEN[k] for k, v in flags.items() if v}
        if enabled:
            raise NotImplementedError(
                f"mttgnet_test does not carry the legacy branches: {enabled}. "
                "Set them to false / use src.mttgnet.MTTGNetWrapper.")
        self.use_future_co2 = bool(use_future_co2)
        if self.use_future_co2 and not use_calendar:
            raise ValueError("use_future_co2=True requires use_calendar=True.")
        if self.use_future_co2:
            calendar_dim = int(calendar_dim) + 1

        self.model = MTTGNetTestModel(
            num_vars=num_features, seq_len=seq_len, horizon=horizon,
            d_model=hidden_dim, dropout=dropout,
            memory_slots=memory_slots, mem_top_k=mem_top_k,
            calendar_dim=calendar_dim, num_gru_layers=num_gru_layers,
            use_gate=use_gate, use_time_bias=use_time_bias,
            use_memory=use_memory,
            use_variable_encoder=use_variable_encoder,
            use_shift_encoder=use_shift_encoder,
            use_var_gnn=use_var_gnn,
            use_calendar=use_calendar,
            co2_index=kwargs.pop("co2_index", 1),
            var_encoder_type=var_encoder_type, gat_version=gat_version,
            time_bias_mode=time_bias_mode, query_refine=query_refine,
            mem_clean=mem_clean, decoder_bypass=decoder_bypass,
            co2_mode=co2_mode, gnn_dynamic=gnn_dynamic,
            co2_ref_ppm=co2_ref_ppm,
            use_mono_reg=use_mono_reg, mono_weight=mono_weight,
            mono_delta=mono_delta, mono_margin=mono_margin,
            use_co2_timebias=use_co2_timebias,
            target_index=target_index,
            time_bias_scale=time_bias_scale,
            shift_bias_scale=shift_bias_scale,
            memory_gate_init=memory_gate_init,
            co2_min=kwargs.pop("co2_min", 280.0),
            co2_max=kwargs.pop("co2_max", 1200.0),
            use_multiscale_co2=use_multiscale_co2,
            co2_attention_modulation=kwargs.pop('co2_attention_modulation',True),
            co2_additive_bias=kwargs.pop('co2_additive_bias',True),
            fusion_sources=kwargs.pop('fusion_sources', None))

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
                raise ValueError("use_future_co2=True but the batch carried no future_co2.")
            future_calendar = torch.cat(
                [future_calendar,
                 future_co2.reshape(B, H, 1).to(future_calendar.dtype)], dim=-1)
        return self.model(
            x_recent=x_recent, daily_anchor=daily_anchor,
            yearly_anchor=yearly_anchor, daily_mask=daily_mask,
            yearly_mask=yearly_mask, future_calendar=future_calendar,
            daily_gaps=daily_gaps, yearly_gaps=yearly_gaps,
            co2_override_ppm=co2_override_ppm)


# ═══════════════════════════════════════════════════════════════════
# Smoke test (CPU, no data): python src/mttgnet_test.py
# ═══════════════════════════════════════════════════════════════════

def _run_selftest() -> None:
    torch.manual_seed(0)
    B, L, N, H, Kd, Ky = 4, 24, 8, 8, 3, 2
    kw = dict(num_features=N, hidden_dim=32, horizon=H, seq_len=L,
              memory_slots=16, mem_top_k=4, num_gru_layers=2,
              co2_index=1, var_encoder_type="attention")
    batch = dict(
        x_recent=torch.randn(B, L, N),
        daily_anchor=torch.randn(B, H, Kd, N),
        yearly_anchor=torch.randn(B, H, Ky, N),
        daily_mask=torch.ones(B, H, Kd, dtype=torch.bool),
        yearly_mask=torch.ones(B, H, Ky, dtype=torch.bool),
        future_calendar=torch.randn(B, H, 5),
        daily_gaps=torch.rand(B, H, Kd) * 2 + 0.5,
        yearly_gaps=torch.rand(B, H, Ky) * 30 + 340.0,
    )

    print("=== mttgnet_test self-test ===")
    # 1. forward + backward, both co2 modes
    for mode in ("counterfactual", "softplus_log"):
        m = MTTGNetTestWrapper(**kw, co2_mode=mode, use_mono_reg=(mode == "counterfactual"))
        out = m(**batch)
        assert out["prediction"].shape == (B, H)
        loss = out["prediction"].square().mean()
        if out["aux_loss"] is not None:
            loss = loss + out["aux_loss"]
        loss.backward()
        n_par = sum(p.numel() for p in m.parameters())
        print(f"  [{mode:15s}] forward/backward OK, params={n_par}, "
              f"shift_bias mean={out['shift_bias'].mean():+.4f}")

    # 2. clean memory: gate → 0 ⇒ exact identity
    mem = CleanDynamicMemory(32, num_slots=16, top_k=4, gate_init_bias=-30.0)
    x = torch.randn(5, 32)
    delta = (mem(x) - x).abs().max()
    assert delta < 1e-5, f"clean memory not an identity at gate→0: {delta}"
    print(f"  [memory identity] max |out - x| at gate→0 = {delta:.2e}  ✓")

    # 3. counterfactual: through the model, bias at the 280 ppm reference is zero
    m = MTTGNetTestWrapper(**kw, co2_mode="counterfactual").eval()
    with torch.no_grad():
        b_ref = dict(batch)
        b_ref["x_recent"] = batch["x_recent"].clone()
        b_ref["x_recent"][:, :, 1] = 0.0     # 0.0 == 280 ppm on the [0,1] scale
        bias = m(**b_ref)["shift_bias"]
    assert bias.abs().max() < 1e-5, f"counterfactual bias not zero at reference: {bias.abs().max()}"
    print(f"  [counterfactual] |shift_bias(C=280ppm)| max = {bias.abs().max():.2e}  ✓")

    # 4. softplus_log: bias non-decreasing in CO₂ for a FIXED temporal context.
    #    (Varying the input column would also change h, and the whole-network
    #    output is not guaranteed monotone — the by-construction claim is
    #    about the bias branch at fixed state, so override ppm instead.)
    m = MTTGNetTestWrapper(**kw, co2_mode="softplus_log").eval()
    with torch.no_grad():
        p_low = m(**batch, co2_override_ppm=torch.full((B,), 372.0))["shift_bias"]
        p_high = m(**batch, co2_override_ppm=torch.full((B,), 1108.0))["shift_bias"]
    assert (p_high >= p_low - 1e-6).all(), \
        f"softplus_log not monotone at fixed h: {p_low.tolist()} -> {p_high.tolist()}"
    print(f"  [softplus_log] bias at fixed h: C=372ppm → {p_low.mean():+.4f}, "
          f"C=1108ppm → {p_high.mean():+.4f}  ✓")

    # 5. forbidden branches fail loudly
    try:
        MTTGNetTestWrapper(**kw, use_multirate=True)
        raise AssertionError("use_multirate should have raised")
    except NotImplementedError:
        print("  [guards] legacy branches refuse to start silently  ✓")

    print("=== all self-tests passed ===")


if __name__ == "__main__":
    _run_selftest()
