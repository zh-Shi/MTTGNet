from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


# ==================== Causal Conv1d (pure PyTorch, zero cuDNN) ====================

class CausalConv1d(nn.Module):
    """Causal dilated 1D convolution — pure PyTorch, no cuDNN dependency.

    Uses index‑gather + F.linear instead of nn.Conv1d so that even a broken
    cuDNN installation (missing nvrtc.so) cannot degrade performance.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = dilation * (kernel_size - 1)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[1] * self.kernel_size
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] contiguous  →  [B, O, L]"""
        B, C, L = x.shape
        K, d = self.kernel_size, self.dilation
        P = self.pad  # d * (K - 1)

        if not x.is_contiguous():
            x = x.contiguous()

        # ── causal left-pad ──
        x = F.pad(x, (P, 0))                              # [B, C, L+P]

        # ── build gather indices ──
        t = torch.arange(L, device=x.device)               # [L]
        offsets = d * torch.arange(K, device=x.device)     # [K]: 0, d, 2d, …
        idx = (t.unsqueeze(1) + P - offsets.unsqueeze(0))  # [L, K]
        idx = idx.clamp(0, x.shape[2] - 1)

        # ── gather + linear ──
        gathered = x[:, :, idx]                             # [B, C, L, K]
        gathered = gathered.permute(0, 2, 1, 3)             # [B, L, C, K]
        gathered = gathered.reshape(B * L, C * K)           # [B*L, C*K]

        w = self.weight.reshape(self.weight.shape[0], -1)   # [O, C*K]
        out = F.linear(gathered, w, self.bias)              # [B*L, O]
        return out.view(B, L, -1).transpose(1, 2)           # [B, O, L]


# ==================== Non-Causal Dilated Conv1d (pure PyTorch) ====================

class DilatedConv1d(nn.Module):
    """Non-causal dilated 1D convolution — pure PyTorch, no cuDNN.

    Uses both-sides padding (unlike ``CausalConv1d``) so every output
    position sees ``kernel_size`` neighbours spaced by ``dilation``.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = dilation  # both-sides padding preserves length for k=odd

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[1] * self.kernel_size
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L]  →  [B, O, L]"""
        B, C, L = x.shape
        K, d, P = self.kernel_size, self.dilation, self.pad

        if not x.is_contiguous():
            x = x.contiguous()
        x = F.pad(x, (P, P))                                 # both sides

        t = torch.arange(L, device=x.device)                  # [L]
        offsets = d * torch.arange(K, device=x.device)        # [K]: 0, d, 2d, …
        idx = (t.unsqueeze(1) + P - offsets.unsqueeze(0))     # [L, K]

        gathered = x[:, :, idx]                                # [B, C, L, K]
        gathered = gathered.permute(0, 2, 1, 3)                # [B, L, C, K]
        gathered = gathered.reshape(B * L, C * K)

        w = self.weight.reshape(self.weight.shape[0], -1)      # [O, C*K]
        out = F.linear(gathered, w, self.bias)                 # [B*L, O]
        return out.view(B, L, -1).transpose(1, 2)              # [B, O, L]


# ==================== TCN Residual Block ====================

class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(
            hidden_dim, hidden_dim, kernel_size, dilation,
        )
        self.conv2 = CausalConv1d(
            hidden_dim, hidden_dim, kernel_size, dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = out + residual

        out = out.transpose(1, 2)              # [B, L, D]
        out = self.norm(out)                   # LayerNorm over last dim
        return out.transpose(1, 2)             # [B, D, L]


# ==================== Local Temporal Encoder ====================

class LocalTemporalEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(num_features, hidden_dim)

        self.blocks = nn.ModuleList([
            TemporalResidualBlock(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dilation=2 ** layer_idx,
                dropout=dropout,
            )
            for layer_idx in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, N]  →  [B, D]"""
        x = self.input_projection(x)             # [B, L, D]
        x = x.transpose(1, 2).contiguous()        # [B, D, L]

        for block in self.blocks:
            x = block(x)

        context = x[:, :, -1]                    # [B, D]
        return self.output_norm(context)


# ==================== GRU Temporal Encoder ====================

class GRUTemporalEncoder(nn.Module):
    """GRU-based temporal encoder for capturing long-range dependencies.

    Replaces the TCN with a stacked GRU that processes the sequence
    and returns the final hidden state as the local context vector.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(
            num_features,
            hidden_dim,
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, L, N]
        return: [B, D]
        """
        x = self.input_projection(x)  # [B, L, D]

        # GRU returns: output [B, L, D*dirs], h_n [layers*dirs, B, D]
        _, h_n = self.gru(x)

        # Take the last layer's final hidden state
        context = h_n[-1]  # [B, D]

        return self.output_norm(context)


# ==================== 2D Variable-Time Convolution (VTC) Encoder ====================

class VTC2DEncoder(nn.Module):
    """2D Variable-Time Convolution encoder from the MTTGNet paper.

    Instead of 1D causal convolutions on the time axis only, 2D-VTC
    applies 2D convolutions on the [time × feature] plane, learning
    joint spatio-temporal patterns across variables and time steps.

    Simple implementation: treat [B, L, D] as a 2D image [B, 1, L, D],
    apply Conv2d with causal padding in time, then aggregate.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(num_features, hidden_dim)
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(
                VTC2DBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    is_first=(i == 0),
                )
            )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, L, N]
        return: [B, D]
        """
        x = self.input_projection(x)  # [B, L, D]

        for block in self.blocks:
            x = block(x)  # [B, L, D]

        # Take last time step
        context = x[:, -1, :]  # [B, D]
        return self.output_norm(context)


class VTC2DBlock(nn.Module):
    """Single 2D-VTC residual block — pure PyTorch, zero cuDNN.

    Applies two 2D convolutions (causal in time, sliding in feature)
    using index‑gather + F.linear instead of nn.Conv2d.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        is_first: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.hidden_dim = hidden_dim
        self.pad_time = dilation * (kernel_size - 1)

        # conv1: 1 → hidden_dim
        self.w1 = nn.Parameter(
            torch.empty(hidden_dim, 1, kernel_size, kernel_size))
        self.b1 = nn.Parameter(torch.empty(hidden_dim))

        # conv2: hidden_dim → hidden_dim
        self.w2 = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, kernel_size, kernel_size))
        self.b2 = nn.Parameter(torch.empty(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w in (self.w1, self.w2):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        for b in (self.b1, self.b2):
            fan_in = b.shape[0] * self.kernel_size * self.kernel_size
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(b, -bound, bound)

    @staticmethod
    def _conv2d_pure(
        x: torch.Tensor,          # [B, C_in, L_pad, D_in]
        weight: torch.Tensor,    # [C_out, C_in, Kt, Kf]
        bias: torch.Tensor,      # [C_out]
        dt: int,                 # dilation (time axis)
        pad_time: int,           # causal padding already applied
    ) -> torch.Tensor:
        """Pure-PyTorch 2D conv: causal-dilated in time, sliding in feature."""
        B, C_in, _, D_in = x.shape
        C_out, _, Kt, Kf = weight.shape
        L_out = x.shape[2] - pad_time
        D_out = D_in - Kf + 1

        # gather dilated time positions
        t = torch.arange(L_out, device=x.device)               # [L_out]
        offsets_t = dt * torch.arange(Kt, device=x.device)     # [Kt]: 0, dt, 2dt, …
        time_idx = (t.unsqueeze(1) + pad_time - offsets_t.unsqueeze(0))  # [L_out, Kt]

        x_time = x[:, :, time_idx, :]                           # [B, C_in, L_out, Kt, D_in]

        # unfold feature dim
        x_both = x_time.unfold(-1, Kf, 1)                      # [B, C_in, L_out, Kt, D_out, Kf]

        # permute + reshape for linear
        x_both = x_both.permute(0, 2, 4, 1, 3, 5)              # [B, L_out, D_out, C_in, Kt, Kf]
        x_both = x_both.reshape(B, L_out * D_out, C_in * Kt * Kf)

        w = weight.reshape(C_out, C_in * Kt * Kf)
        out = F.linear(x_both, w, bias)                         # [B, L_out*D_out, C_out]
        return out.view(B, L_out, D_out, C_out).permute(0, 3, 1, 2)  # [B, C_out, L_out, D_out]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D]  →  [B, L, D]"""
        B, L, D = x.shape
        residual = x
        pt = self.pad_time
        dt = self.dilation

        # ── conv1 ──
        x_2d = x.unsqueeze(1)                                   # [B, 1, L, D]
        x_padded = F.pad(x_2d, (0, 0, pt, 0))                   # causal in time
        out = self._conv2d_pure(x_padded, self.w1, self.b1, dt, pt)
        out = F.gelu(out)
        out = self.dropout(out)
        out = F.pad(out, (0, 0, pt, 0))                         # pad again for conv2

        # ── conv2 ──
        out = self._conv2d_pure(out, self.w2, self.b2, dt, pt)
        out = F.gelu(out)
        out = self.dropout(out)
        out = out.mean(dim=-1)                                   # [B, C, L]  pool features

        # match residual shape
        out = out.transpose(1, 2)                                # [B, L, D_out]
        if out.shape[1] != L:
            out = F.interpolate(out.transpose(1, 2), size=L,
                                mode='linear', align_corners=False).transpose(1, 2)
        if out.shape[-1] != D:
            out = F.pad(out, (0, D - out.shape[-1]))

        out = out + residual
        return self.norm(out)


# ==================== Mamba / SSM Temporal Encoder ====================


class SSMTemporalEncoder(nn.Module):
    """Selective State Space Model (Mamba-style) for temporal encoding.

    Uses a diagonal SSM with parallel associative scan for O(L) complexity.
    Falls back gracefully if the official mamba-ssm package is not installed.

    Key idea: instead of attention (O(L²)) or convolution (O(L·K)),
    SSM maintains a hidden state that evolves linearly over time with
    input-dependent gating. This is particularly suited to weather
    sequences where dynamics evolve continuously.

    Reference: Mamba (Gu & Dao, 2023), S4 (Gu et al., 2022)
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int = 4,
        dropout: float = 0.1,
        d_state: int = 16,
        expand: int = 2,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(num_features, hidden_dim)
        self.num_layers = num_layers

        self.blocks = nn.ModuleList(
            [
                SSMBlock(
                    d_model=hidden_dim,
                    d_state=d_state,
                    d_conv=kernel_size,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, N]
        return: [B, D]
        """
        x = self.input_projection(x)  # [B, L, D]

        for block in self.blocks:
            x = block(x)  # [B, L, D]

        context = x[:, -1, :]  # [B, D]
        return self.output_norm(context)


class SSMBlock(nn.Module):
    """Single SSM block with conv + selective scan.

    Architecture (Mamba-style):
      input → LayerNorm → Linear(x, z) → Conv1d → SiLU → SSM scan → gate(z) → output

    The depthwise conv provides local context before the global SSM scan.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        # Input projections: x → (x, z) where z is the gate
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Depthwise 1D conv for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,  # "same" padding
        )

        # SSM: x → (dt, B, C) parameters
        # dt: input-dependent time step [B, L, inner]
        # B:  input-dependent input projection [B, L, state]
        # C:  input-dependent output projection [B, L, state]
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)  # [dt, B, C]

        # Learnable diagonal state matrix A
        # Initialize so eigenvalues are negative real (stable) with varied time scales
        A_init = -torch.arange(1, d_state + 1, dtype=torch.float32)  # [state]
        self.A_log = nn.Parameter(torch.log(-A_init))  # log of positive A_mag

        # Skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _selective_scan(
        self,
        u: torch.Tensor,     # [B, L, D_inner]
        delta: torch.Tensor, # [B, L, D_inner]
        A: torch.Tensor,     # [D_inner, State]
        B: torch.Tensor,     # [B, L, State]
        C: torch.Tensor,     # [B, L, State]
        D: torch.Tensor,     # [D_inner]
    ) -> torch.Tensor:
        """Parallel selective scan via Blelloch-style prefix sum.

        The recurrence is:
          h_t = A_bar_t ⊙ h_{t-1} + B_bar_t ⊙ u_t
          y_t = C_t · h_t + D ⊙ u_t

        This is computed in O(log L) parallel steps using the semigroup
        (a, b) ∘ (a', b') = (a'⊙a,  a'⊙b + b')  where ∘ means "apply
        left then right".

        Note: B [B,L,State] and C [B,L,State] are broadcast over the D_inner
        dimension, matching the S6/Mamba architecture where parameters are
        shared across inner channels.
        """
        B_, L, din = u.shape
        state = self.d_state

        # A_bar: [B, L, din, state] — per-channel time-varying diagonal
        A_bar = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )

        # B_bar: broadcast B [B,L,State] over din dim
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, din, state]
        Bx = B_bar * u.unsqueeze(-1)                   # [B, L, din, state]

        # ── Parallel associative scan (doubling method) ──
        # Each "element" is the transform h → a⊙h + b.
        # Composition: (a₁,b₁) ∘ (a₂,b₂) = (a₂·a₁,  a₂·b₁ + b₂)
        # After scan, position t stores prefix 0..t, so h_t = b[t] (h₀=0).

        a = A_bar  # [B, L, din, state]
        b = Bx     # [B, L, din, state]

        # Index mask for positions that have a valid predecessor
        idx = torch.arange(L, device=u.device).view(1, -1, 1, 1)

        for k in range(L.bit_length()):
            step = 1 << k

            # Shift: a_prev[t] = a[t - step] (identity where t < step)
            a_prev = torch.roll(a, shifts=step, dims=1)
            b_prev = torch.roll(b, shifts=step, dims=1)

            # Compose: (a[t], b[t]) ∘ (a_prev[t], b_prev[t])
            #          = (a[t] * a_prev[t],  a[t] * b_prev[t] + b[t])
            a_new = a * a_prev
            b_new = a * b_prev + b

            # Only update positions t ≥ step (valid predecessor exists)
            mask = idx >= step
            a = torch.where(mask, a_new, a)
            b = torch.where(mask, b_new, b)

        # b[t] is now the cumulative hidden state h_t
        h = b  # [B, L, din, state]

        # Output: y_t = C_t · h_t + D ⊙ u_t
        y = (C.unsqueeze(2) * h).sum(dim=-1)  # [B, L, din]
        y = y + D.unsqueeze(0).unsqueeze(0) * u

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        return: [B, L, D]
        """
        B, L, D = x.shape
        residual = x

        # LayerNorm + input projection
        normed = self.norm(x)
        projected = self.in_proj(normed)  # [B, L, 2*D_inner]
        x_ssm, z_gate = projected.chunk(2, dim=-1)  # Each [B, L, D_inner]

        # Depthwise conv for local context
        x_conv = x_ssm.transpose(1, 2)  # [B, D_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal (remove future padding)
        x_conv = F.silu(x_conv.transpose(1, 2))  # [B, L, D_inner]

        # SSM parameters
        ssm_params = self.x_proj(x_conv)  # [B, L, 2*state + 1]
        delta, B_ssm, C_ssm = ssm_params.split(
            [1, self.d_state, self.d_state], dim=-1
        )
        # delta: [B, L, 1], B_ssm: [B, L, state], C_ssm: [B, L, state]

        # Softplus ensures delta > 0
        delta = F.softplus(delta.squeeze(-1))  # [B, L]
        # Expand delta to [B, L, D_inner] for per-channel modulation
        delta = delta.unsqueeze(-1).expand(B, L, self.d_inner)

        # A parameter: [D_inner, State]
        A = -torch.exp(self.A_log)  # [state]
        # Expand A to [D_inner, State]
        # For simplicity, broadcast state dim
        A_expanded = A.unsqueeze(0).expand(self.d_inner, -1)  # [D_inner, State]

        # Apply SSM
        x_ssm_out = self._selective_scan(
            u=x_conv,
            delta=delta,
            A=A_expanded,
            B=B_ssm,
            C=C_ssm,
            D=self.D,
        )  # [B, L, D_inner]

        # Gate with z (SiLU gating, like Mamba)
        z_gated = F.silu(z_gate)  # [B, L, D_inner]
        x_out = x_ssm_out * z_gated  # [B, L, D_inner]

        # Output projection
        x_out = self.out_proj(self.dropout(x_out))  # [B, L, D]

        return x_out + residual


# ==================== LSTM Temporal Encoder ====================

class LSTMTemporalEncoder(nn.Module):
    """LSTM-based temporal encoder for capturing long-range dependencies.

    Uses a stacked LSTM to process the sequence and returns the final
    hidden state as the local context vector.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(
            num_features,
            hidden_dim,
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, L, N]
        return: [B, D]
        """
        x = self.input_projection(x)  # [B, L, D]

        # LSTM returns: output [B, L, D], (h_n, c_n)
        _, (h_n, _) = self.lstm(x)

        # Take the last layer's final hidden state
        context = h_n[-1]  # [B, D]

        return self.output_norm(context)


# ==================== Transformer Temporal Encoder ====================

class TransformerTemporalEncoder(nn.Module):
    """Standard Transformer encoder for temporal modeling.

    Uses sinusoidal positional encoding + N transformer encoder layers.
    The final hidden state (or mean-pooled) is used as context vector.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(num_features, hidden_dim)

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_len, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, L, N]
        return: [B, D]
        """
        B, L, _ = x.shape

        x = self.input_projection(x)  # [B, L, D]

        # Add positional encoding
        x = x + self.pos_encoding[:, :L, :]

        # Transformer encoder
        x = self.transformer(x)  # [B, L, D]

        # Take last time step as context
        context = x[:, -1, :]  # [B, D]

        return self.output_norm(context)


# ==================== PatchTST Temporal Encoder ====================

class PatchTSTEncoder(nn.Module):
    """PatchTST-style temporal encoder.

    Divides the input sequence into patches (subseries), projects each
    patch to hidden_dim, then applies transformer over patches. This
    captures both local patterns (within patches) and global patterns
    (across patches).

    Key design: patch_len controls local granularity, stride controls
    overlap. Fewer patches = more efficient but less granular.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        patch_len: int = 12,     # each patch = 12 steps (3 days @ 6h)
        stride: int = 6,         # 50% overlap
        max_patches: int = 64,
    ) -> None:
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride

        # Project each patch: [patch_len * num_features] → hidden_dim
        self.patch_projection = nn.Linear(
            patch_len * num_features,
            hidden_dim,
        )

        # Learnable patch position encoding
        self.patch_pos = nn.Parameter(
            torch.randn(1, max_patches, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Divide [B, L, N] into patches [B, N_patches, patch_len*N]."""
        B, L, N = x.shape

        # Unfold along time dimension to get patches
        # x: [B, L, N] → [B, N_patches, patch_len, N]
        patches = x.unfold(1, self.patch_len, self.stride)  # [B, N, patch_len, N]
        N_patches = patches.shape[1]

        # Flatten each patch
        patches = patches.reshape(B, N_patches, self.patch_len * N)

        return patches

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, L, N]
        return: [B, D]
        """
        B = x.shape[0]

        # Patchify input
        patches = self._patchify(x)  # [B, N_patches, patch_len*N]
        N_patches = patches.shape[1]

        # Project patches
        patch_embeds = self.patch_projection(patches)  # [B, N_patches, D]

        # Add patch position encoding
        patch_embeds = patch_embeds + self.patch_pos[:, :N_patches, :]

        # Transformer over patches
        encoded = self.transformer(patch_embeds)  # [B, N_patches, D]

        # Mean pool over patches for context
        context = encoded.mean(dim=1)  # [B, D]

        return self.output_norm(context)


# ==================== TCN Horizon Decoder ====================

class TCNHorizonDecoder(nn.Module):
    """Causal TCN decoder over the horizon dimension.

    Unlike the pointwise MLP head which predicts each horizon step
    independently, this decoder applies causal 1D convs along the
    horizon axis.  Step h can see steps 0..h−1, allowing long‑horizon
    predictions to build on short‑horizon context.

    Uses ``CausalConv1d`` (pure PyTorch, zero cuDNN) internally.
    """

    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_norm = nn.LayerNorm(hidden_dim)

        self.convs = nn.ModuleList([
            CausalConv1d(hidden_dim, hidden_dim, kernel_size, 2 ** i)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """query: [B, H, D]  →  [B, H]"""
        x = self.input_norm(query).transpose(1, 2)          # [B, D, H]

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x)                                      # causal pad + linear
            x = F.gelu(x)
            x = self.dropout(x)
            x = x + residual
            x = norm(x.transpose(1, 2)).transpose(1, 2)     # LayerNorm

        x = x.transpose(1, 2)                                # [B, H, D]
        return self.output_proj(x).squeeze(-1)               # [B, H]


# ==================== Temporal Encoder Factory ====================

def build_temporal_encoder(
    encoder_type: str,
    num_features: int,
    hidden_dim: int,
    num_layers: int = 2,
    kernel_size: int = 3,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory for temporal encoders.

    Args:
        encoder_type: 'tcn', 'gru', or 'lstm'
        num_features: number of input variables
        hidden_dim: hidden dimension
        num_layers: number of TCN blocks or RNN layers
        kernel_size: TCN kernel size (ignored for RNN)
        dropout: dropout rate

    Returns:
        Temporal encoder module
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "tcn":
        return LocalTemporalEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    elif encoder_type == "gru":
        return GRUTemporalEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "lstm":
        return LSTMTemporalEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif encoder_type == "vtc2d":
        return VTC2DEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    elif encoder_type == "transformer":
        return TransformerTemporalEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=4,
            dropout=dropout,
        )
    elif encoder_type == "patchtst":
        return PatchTSTEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=4,
            dropout=dropout,
        )
    elif encoder_type in ("mamba", "ssm"):
        return SSMTemporalEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown temporal encoder type: {encoder_type}. "
            f"Supported: 'tcn', 'gru', 'lstm', 'vtc2d', 'transformer', 'patchtst', 'mamba'"
        )


# ==================== Variable Interaction Encoder ====================

class VariableInteractionEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads"
            )

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.value_projection = nn.Linear(
            1,
            hidden_dim,
        )

        self.variable_embedding = nn.Parameter(
            torch.randn(
                num_features,
                hidden_dim,
            ) * 0.02
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(
                hidden_dim,
                4 * hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                4 * hidden_dim,
                hidden_dim,
            ),
        )

        self.pool_score = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [...,N]
        return: [...,D]
        """
        if x.shape[-1] != self.num_features:
            raise ValueError(
                "Last dimension must be num_features"
            )

        leading_shape = x.shape[:-1]

        x_flat = x.reshape(
            -1,
            self.num_features,
            1,
        )

        nodes = self.value_projection(x_flat)
        nodes = (
            nodes
            + self.variable_embedding.unsqueeze(0)
        )

        attended, _ = self.attention(
            nodes,
            nodes,
            nodes,
            need_weights=False,
        )

        nodes = self.norm1(nodes + attended)
        nodes = self.norm2(
            nodes + self.feed_forward(nodes)
        )

        scores = self.pool_score(
            nodes
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        pooled = torch.sum(
            nodes * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled.reshape(
            *leading_shape,
            self.hidden_dim,
        )


# ==================== GAT Variable Interaction Encoder ====================

class GATVariableEncoder(nn.Module):
    """Graph Attention Network (GAT / GAT) for variable interaction.

    Unlike the standard self-attention encoder, GAT uses asymmetric
    attention with LeakyReLU and a learnable attention vector, producing
    directional influence weights α_ij (variable j → variable i).

    This reflects the physical reality that variable dependencies are
    often asymmetric (e.g. solar radiation drives temperature more
    than temperature drives solar radiation).
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        gat_version: str = "gat",
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads"
            )

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.gat_version = "gat"

        self.value_projection = nn.Linear(1, hidden_dim)

        self.variable_embedding = nn.Parameter(
            torch.randn(num_features, hidden_dim) * 0.02
        )

        # GAT: linear projection for source and target
        self.W_src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Single asymmetric GAT implementation (GAT-style: a^T LeakyReLU(W[h_i||h_j])).
        # Unified under the name "gat"; the historical v1/v2 version split is removed.
        self.attn_lin = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn = nn.Parameter(
            torch.randn(1, num_heads, self.head_dim) * 0.02
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.pool_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # Analysis hooks: cache the last attention/pool weights so the
        # variable-interaction (D4) analysis can read the *trained* GAT
        # attention instead of re-deriving weights from the untrained
        # symmetric self-attention path.
        self._last_alpha = None
        self._last_pool = None

    def _gat_attention(
        self,
        nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GAT/GAT attention over variables.

        Args:
            nodes: [M, N, D] — M=batch*horizon*anchors, N=num_features, D=hidden_dim

        Returns:
            attended: [M, N, D]
        """
        M, N, D = nodes.shape

        # Source and target projections
        h_src = self.W_src(nodes)  # [M, N, D] — each node as source (sends info)
        h_dst = self.W_dst(nodes)  # [M, N, D] — each node as target (receives info)

        # Split into heads
        h_src = h_src.view(M, N, self.num_heads, self.head_dim)  # [M, N, H, d]
        h_dst = h_dst.view(M, N, self.num_heads, self.head_dim)  # [M, N, H, d]

        # Expand for all pairs (i,j): target i attends to source j
        h_dst_exp = h_dst.unsqueeze(2).expand(M, N, N, self.num_heads, self.head_dim)  # [M, N, N, H, d]
        h_src_exp = h_src.unsqueeze(1).expand(M, N, N, self.num_heads, self.head_dim)  # [M, N, N, H, d]

        # GAT (asymmetric): score e_ij = a^T LeakyReLU(W[h_i || h_j]) — unified single implementation
        concat = torch.cat([h_dst_exp, h_src_exp], dim=-1)  # [M, N, N, H, 2d]
        concat_flat = concat.reshape(M * N * N, self.num_heads, 2 * self.head_dim)
        concat_flat = concat_flat.reshape(M * N * N, self.hidden_dim * 2)
        transformed = self.attn_lin(concat_flat)  # [M*N*N, D]
        transformed = transformed.view(M, N, N, self.num_heads, self.head_dim)
        e = (self.attn * F.leaky_relu(transformed, 0.2)).sum(dim=-1)  # [M, N, N, H]

        # GAT is asymmetric: e_ij ≠ e_ji due to concat order [h_i, h_j] and LeakyReLU

        # Softmax over sources j for each target i.  A single LeakyReLU has
        # already been applied above (GAT: on e; GAT: on the transformed
        # concat).  Applying it again here would squash negative scores twice.
        alpha = torch.softmax(e, dim=2)  # [M, N, N, H]
        alpha = self.dropout(alpha)

        # Aggregate: h_i' = Σ_j α_ij * W_src(h_j)
        attended = (alpha.unsqueeze(-1) * h_src_exp).sum(dim=2)  # [M, N, H, d]
        attended = attended.reshape(M, N, D)  # [M, N, D]

        self._last_alpha = alpha.detach()  # [M, N, N, heads]
        return attended

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [..., N]
        return: [..., D]
        """
        if x.shape[-1] != self.num_features:
            raise ValueError(
                "Last dimension must be num_features"
            )

        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.num_features, 1)
        nodes = self.value_projection(x_flat)
        nodes = nodes + self.variable_embedding.unsqueeze(0)

        # GAT attention (asymmetric)
        attended = self._gat_attention(nodes)

        nodes = self.norm1(nodes + attended)
        nodes = self.norm2(nodes + self.feed_forward(nodes))

        scores = self.pool_score(nodes).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(nodes * weights.unsqueeze(-1), dim=1)

        self._last_pool = weights.detach()  # [M, N]
        return pooled.reshape(*leading_shape, self.hidden_dim)


# ==================== Variable Encoder Factory ====================

def build_variable_encoder(
    encoder_type: str,
    num_features: int,
    hidden_dim: int,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory for variable interaction encoders.

    Args:
        encoder_type: 'self_attn', 'gat', or 'gat'
        num_features: number of variables
        hidden_dim: hidden dimension
        num_heads: number of attention heads
        dropout: dropout rate

    Returns:
        Variable interaction encoder module
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "self_attn":
        return VariableInteractionEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif encoder_type in ("gat",):
        return GATVariableEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            gat_version=encoder_type,
        )
    elif encoder_type == "graphsage":
        return GraphSAGEEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
    elif encoder_type == "dynamic":
        return DynamicGraphEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif encoder_type == "none":
        # Ablation: no variable interaction, just project each variable
        return NoVariableEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
        )
    else:
        raise ValueError(
            f"Unknown variable encoder type: {encoder_type}. "
            f"Supported: 'self_attn', 'gat', 'gat', 'graphsage', 'dynamic', 'none'"
        )


class NoVariableEncoder(nn.Module):
    """No variable interaction — just project and pool (for ablation)."""
    def __init__(self, num_features: int, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(1, hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.num_features = num_features
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.num_features, 1)
        nodes = self.projection(x_flat)
        scores = self.pool_score(nodes).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(nodes * weights.unsqueeze(-1), dim=1)
        return pooled.reshape(*leading_shape, self.hidden_dim)


# ==================== GraphSAGE Variable Encoder ====================

class GraphSAGEEncoder(nn.Module):
    """GraphSAGE mean-pooling encoder for variable interactions.

    Each variable aggregates information from all other variables via
    learnable mean-pooling (optionally with learned adjacency weights).
    Multiple layers allow higher-order variable interactions.

    This implements the core message-passing idea from the paper's
    GraphSAGE module but adapted for the per-sample variable graph.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        learn_adj: bool = True,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.value_projection = nn.Linear(1, hidden_dim)
        self.variable_embedding = nn.Parameter(
            torch.randn(num_features, hidden_dim) * 0.02
        )

        # Learnable adjacency matrix (initialized near-uniform)
        if learn_adj:
            self.adj_raw = nn.Parameter(
                torch.randn(num_features, num_features) * 0.1
            )
        else:
            # Fixed fully-connected graph (excluding self-loops)
            adj = torch.ones(num_features, num_features)
            adj.fill_diagonal_(0)
            self.register_buffer("adj_raw", adj)

        self.learn_adj = learn_adj

        # SAGE layers: W·MEAN(neighbors) + B·self
        self.sage_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.self_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.pool_score = nn.Linear(hidden_dim, 1)

    def _get_adjacency(self, device: torch.device) -> torch.Tensor:
        """Get normalized adjacency matrix [N, N]."""
        if self.learn_adj:
            # Softplus ensures non-negative, then normalize
            adj = F.softplus(self.adj_raw)
        else:
            adj = self.adj_raw.float()

        # Row normalization: D^{-1} A (mean pooling)
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj / row_sum

        return adj_norm.to(device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [..., N]
        return: [..., D]
        """
        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.num_features, 1)
        M = x_flat.shape[0]

        nodes = self.value_projection(x_flat)  # [M, N, D]
        nodes = nodes + self.variable_embedding.unsqueeze(0)  # [M, N, D]

        adj = self._get_adjacency(nodes.device)  # [N, N]

        for i in range(self.num_layers):
            # Mean aggregation from neighbors: A @ nodes
            neighbor_msg = torch.matmul(
                adj.unsqueeze(0), nodes
            )  # [M, N, D]

            # Update: SAGE(neighbors) + self_connection
            updated = (
                self.sage_layers[i](neighbor_msg)
                + self.self_layers[i](nodes)
            )
            updated = F.gelu(updated)
            updated = self.dropout(updated)
            nodes = self.norms[i](nodes + updated)

        # Weighted pool over variables
        scores = self.pool_score(nodes).squeeze(-1)  # [M, N]
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(nodes * weights.unsqueeze(-1), dim=1)  # [M, D]

        return pooled.reshape(*leading_shape, self.hidden_dim)


# ==================== Dynamic Graph Variable Encoder ====================

class DynamicGraphEncoder(nn.Module):
    """Dynamic graph encoder — learns per-sample adjacency from node features.

    Unlike fixed or globally-learned adjacency, this encoder computes
    edge weights dynamically from the actual variable values in each
    sample. This allows the graph structure to adapt to different
    meteorological conditions (e.g., different coupling strengths
    during storms vs calm periods).

    Implementation: adjacency = softmax(MLP(node_features) / temperature)
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.temperature = temperature

        self.value_projection = nn.Linear(1, hidden_dim)
        self.variable_embedding = nn.Parameter(
            torch.randn(num_features, hidden_dim) * 0.02
        )

        # Edge computation: concat(node_i, node_j) → edge weight
        self.edge_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )

        # Value projection for message passing
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.pool_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [..., N]
        return: [..., D]
        """
        leading_shape = x.shape[:-1]
        M = 1
        for s in leading_shape:
            M *= s
        N = self.num_features

        x_flat = x.reshape(M, N, 1)
        nodes = self.value_projection(x_flat)  # [M, N, D]
        nodes = nodes + self.variable_embedding.unsqueeze(0)

        # Compute per-sample dynamic adjacency
        # For each pair (i,j): edge_ij = edge_net(concat(node_i, node_j))
        node_i = nodes.unsqueeze(2).expand(M, N, N, self.hidden_dim)  # [M, N, N, D]
        node_j = nodes.unsqueeze(1).expand(M, N, N, self.hidden_dim)  # [M, N, N, D]
        edge_input = torch.cat([node_i, node_j], dim=-1)  # [M, N, N, 2D]
        edge_logits = self.edge_net(edge_input)  # [M, N, N, H]

        # Softmax over sources j for each target i
        adj = torch.softmax(edge_logits / self.temperature, dim=2)  # [M, N, N, H]
        adj = self.dropout(adj)

        # Message passing with multi-head adjacency
        v = self.W_v(nodes).view(M, N, self.num_heads, self.hidden_dim // self.num_heads)
        v_expand = v.unsqueeze(1)  # [M, 1, N, H, d]

        messages = (adj.unsqueeze(-1) * v_expand).sum(dim=2)  # [M, N, H, d]
        messages = messages.reshape(M, N, self.hidden_dim)  # [M, N, D]

        nodes = self.norm1(nodes + messages)
        nodes = self.norm2(nodes + self.feed_forward(nodes))

        scores = self.pool_score(nodes).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(nodes * weights.unsqueeze(-1), dim=1)

        return pooled.reshape(*leading_shape, self.hidden_dim)


# ==================== Climate Encoder (CMIP Interface) ====================

class ClimateEncoder(nn.Module):
    """Climate scenario encoder — stub for future CMIP integration.

    Per protocol §8: model MUST retain a ClimateEncoder(). During current
    training (ERA5 only), this module is initialized but inactive. In the
    future, it will encode CMIP climate variables and influence the hidden
    state via FiLM, cross-attention, or gating.

    Design:
      - Input: climate variables (CO2, aerosols, etc.) per time step
      - Output: climate_embedding [B, D] — a conditioning vector
      - Integration: FiLM(gamma, beta) = climate_embedding → modulate hidden
    """

    def __init__(
        self,
        num_climate_vars: int = 3,  # e.g., CO2, aerosol, solar
        hidden_dim: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_climate_vars, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM parameters: gamma (scale) and beta (shift)
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention: climate as key/value, weather hidden as query
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        climate_vars: torch.Tensor,
        mode: str = "film",
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            climate_vars: [B, C] or [B, T, C] climate variables
            mode: "film" | "cross_attn" | "gate" — how to integrate

        Returns:
            dict with "embedding" and integration parameters
        """
        if climate_vars.ndim == 3:
            # Average over time dimension
            climate_vars = climate_vars.mean(dim=1)

        embedding = self.encoder(climate_vars)  # [B, D]

        return {
            "embedding": embedding,
            "film_gamma": self.film_gamma(embedding),
            "film_beta": self.film_beta(embedding),
        }


# ==================== Future Query Encoder ====================

class FutureQueryEncoder(nn.Module):
    def __init__(
        self,
        calendar_dim: int,
        hidden_dim: int,
        max_horizon: int,
    ) -> None:
        super().__init__()

        self.calendar_projection = nn.Sequential(
            nn.Linear(
                calendar_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.horizon_embedding = nn.Embedding(
            max_horizon,
            hidden_dim,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        future_calendar: torch.Tensor,
    ) -> torch.Tensor:
        """
        future_calendar: [B,H,F]
        return: [B,H,D]
        """
        _, horizon, _ = future_calendar.shape

        horizon_ids = torch.arange(
            horizon,
            device=future_calendar.device,
        )

        horizon_embedding = (
            self.horizon_embedding(horizon_ids)
            .unsqueeze(0)
        )

        query = (
            self.calendar_projection(
                future_calendar
            )
            + horizon_embedding
        )

        return self.norm(query)


# ==================== Relation-Aware Periodic Attention ====================

class RelationAwarePeriodicAttention(nn.Module):
    """Periodic anchor attention with optional gating and time-distance bias.

    Args:
        hidden_dim: hidden dimension
        dropout: dropout rate
        use_gate: if True, use learned gate to control periodic info flow
        use_time_bias: if True, add learned time-distance bias to anchor scores
        max_anchors: max number of anchors (for time-bias embedding)
        return_contexts: if True, return (query, daily_ctx, yearly_ctx) separately
            instead of fusing them internally. Used for branch-gated fusion.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        use_gate: bool = False,
        use_time_bias: bool = False,
        max_anchors: int = 10,
        return_contexts: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_gate = use_gate
        self.use_time_bias = use_time_bias
        self.return_contexts = return_contexts

        self.query_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.daily_key = nn.Linear(
            hidden_dim,
            hidden_dim,
        )
        self.daily_value = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.yearly_key = nn.Linear(
            hidden_dim,
            hidden_dim,
        )
        self.yearly_value = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.daily_relation = nn.Parameter(
            torch.randn(hidden_dim) * 0.02
        )

        self.yearly_relation = nn.Parameter(
            torch.randn(hidden_dim) * 0.02
        )

        # Time-distance bias: learned bias for each anchor position
        if use_time_bias:
            self.daily_time_bias = nn.Parameter(
                torch.randn(max_anchors, hidden_dim) * 0.02
            )
            self.yearly_time_bias = nn.Parameter(
                torch.randn(max_anchors, hidden_dim) * 0.02
            )

        # Gated fusion: learned gate to control periodic information flow
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

        self.output_projection = nn.Linear(
            2 * hidden_dim,
            hidden_dim,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(
                hidden_dim,
                4 * hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                4 * hidden_dim,
                hidden_dim,
            ),
        )

    @staticmethod
    def _masked_softmax(
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Clamp scores to prevent overflow in softmax exp()
        # float32: exp(89) ≈ Inf, so we clamp to [-50, 50] for safety
        scores = scores.clamp(-50.0, 50.0)

        # Replace masked positions with a very negative value
        scores = scores.masked_fill(
            ~mask,
            -1e9,  # Safe: exp(-1e9) ≈ 0, avoids finfo.min overflow issues
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        valid_count = mask.sum(
            dim=-1,
            keepdim=True,
        ).to(weights.dtype)

        # Zero out rows with no valid anchors to avoid NaN from all-masked softmax
        return weights * (valid_count > 0).to(weights.dtype)

    def _aggregate(
        self,
        query: torch.Tensor,
        anchors: torch.Tensor,
        mask: torch.Tensor,
        key_layer: nn.Linear,
        value_layer: nn.Linear,
        relation_embedding: torch.Tensor,
        time_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query:   [B,H,D]
        anchors: [B,H,K,D]
        mask:    [B,H,K]
        time_bias: [K, D] or None — learned time-distance bias
        """
        q = self.query_projection(query)
        k = key_layer(anchors)
        v = value_layer(anchors)

        k = (
            k
            + relation_embedding.view(
                1, 1, 1, -1
            )
        )

        # Add time-distance bias if enabled
        if time_bias is not None:
            k = k + time_bias.view(1, 1, -1, self.hidden_dim)

        scores = torch.sum(
            q.unsqueeze(2) * k,
            dim=-1,
        ) / math.sqrt(self.hidden_dim)

        weights = self._masked_softmax(
            scores,
            mask,
        )

        return torch.sum(
            weights.unsqueeze(-1) * v,
            dim=2,
        )

    def forward(
        self,
        query: torch.Tensor,
        daily_nodes: torch.Tensor,
        yearly_nodes: torch.Tensor,
        daily_mask: torch.Tensor,
        yearly_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, H, K_d, _ = daily_nodes.shape
        K_y = yearly_nodes.shape[2]

        # Prepare time bias tensors
        daily_tb = None
        yearly_tb = None
        if self.use_time_bias:
            daily_tb = self.daily_time_bias[:K_d]  # [K_d, D]
            yearly_tb = self.yearly_time_bias[:K_y]  # [K_y, D]

        daily_context = self._aggregate(
            query=query,
            anchors=daily_nodes,
            mask=daily_mask,
            key_layer=self.daily_key,
            value_layer=self.daily_value,
            relation_embedding=self.daily_relation,
            time_bias=daily_tb,
        )

        yearly_context = self._aggregate(
            query=query,
            anchors=yearly_nodes,
            mask=yearly_mask,
            key_layer=self.yearly_key,
            value_layer=self.yearly_value,
            relation_embedding=self.yearly_relation,
            time_bias=yearly_tb,
        )

        update = self.output_projection(
            torch.cat(
                [
                    daily_context,
                    yearly_context,
                ],
                dim=-1,
            )
        )

        # Gated fusion: gate controls how much periodic info flows in
        if self.use_gate:
            gate_input = torch.cat([query, daily_context, yearly_context], dim=-1)
            gate_weight = self.gate(gate_input)  # [B, H, D]
            query = self.norm1(
                query + gate_weight * self.dropout(update)
            )
        else:
            query = self.norm1(
                query + self.dropout(update)
            )

        query = self.norm2(
            query + self.feed_forward(query)
        )

        if self.return_contexts:
            # Return query and both context vectors for branch-gated fusion
            return query, daily_context, yearly_context

        return query
