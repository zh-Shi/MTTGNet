from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .time_utils import (
    calendar_features,
    safe_shift_year,
)


class PeriodicAnchorDataset(Dataset):
    """Dataset that provides recent windows + periodic anchor retrieval.

    Optimized v2: anchor indices and calendar features are pre-computed at
    __init__ time and stored as numpy int32/float32 arrays.  ``__getitem__``
    performs only numpy array indexing — **zero** dict lookups, zero pandas
    operations, zero anchor-time arithmetic per sample.

    For each origin time step, this yields:
      - x_recent:       [L, N] — recent observation window
      - daily_anchor:   [H, K_d, N] — daily periodic anchor values
      - yearly_anchor:  [H, K_y, N] — yearly periodic anchor values
      - daily_mask:     [H, K_d] — daily anchor validity
      - yearly_mask:    [H, K_y] — yearly anchor validity
      - future_calendar: [H, F] — calendar features for each horizon step
      - target:         [H] — target values (temperature anomaly)
    """

    def __init__(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        origin_indices: np.ndarray,
        recent_length: int,
        horizon: int,
        daily_anchors: int,
        yearly_anchors: int,
        target_index: int,
        time_encode: bool = False,
        co2_index: int | None = None,
    ) -> None:
        super().__init__()

        if values.ndim != 2:
            raise ValueError("values must have shape [T,N]")

        self.values = values.astype(np.float32)
        self.timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))

        if len(self.values) != len(self.timestamps):
            raise ValueError("values and timestamps length mismatch")
        if not self.timestamps.is_monotonic_increasing:
            raise ValueError("timestamps must be sorted monotonically")
        if self.timestamps.has_duplicates:
            raise ValueError("timestamps contain duplicates")

        # ── Time encoding: append hour sin/cos to feature dim ──
        if time_encode:
            # DatetimeIndex.hour may return an ndarray (modern pandas) or an
            # Index; np.asarray handles both.  `.values` fails on the former.
            hours = np.asarray(self.timestamps.hour, dtype=np.float32)
            hour_sin = np.sin(2 * np.pi * hours / 24.0).reshape(-1, 1)
            hour_cos = np.cos(2 * np.pi * hours / 24.0).reshape(-1, 1)
            self.values = np.concatenate(
                [self.values, hour_sin, hour_cos], axis=1).astype(np.float32)

        self.origin_indices = np.asarray(origin_indices, dtype=np.int64)
        self.recent_length = int(recent_length)
        self.horizon = int(horizon)
        self.daily_anchors = int(daily_anchors)
        self.yearly_anchors = int(yearly_anchors)
        self.target_index = int(target_index)
        # When set, __getitem__ also returns the CO2 level at EVERY horizon step
        # of the target window.  Scenario CO2 is a known exogenous input (that is
        # what a scenario *is*), so this leaks no target information; it is what
        # lets the decoder know the forcing at each lead instead of having to
        # infer the pathway from the look-back window.
        self.co2_index = None if co2_index is None else int(co2_index)
        self.num_features = self.values.shape[1]

        if target_index < 0 or target_index >= self.num_features:
            raise ValueError(
                f"target_index ({target_index}) out of range "
                f"[0, {self.num_features - 1}]"
            )

        # ── Filter valid origins ──
        valid_origins = []
        for origin_idx in self.origin_indices:
            if origin_idx - self.recent_length + 1 < 0:
                continue
            if origin_idx + self.horizon >= len(self.values):
                continue
            valid_origins.append(origin_idx)

        self.origin_indices = np.asarray(valid_origins, dtype=np.int64)

        if len(self.origin_indices) == 0:
            raise ValueError(
                "No valid origins found. Check split ranges, "
                "recent_length, and horizon settings."
            )

        # ── Fast int64-ns timestamp → index mapping ──
        self._ts_ns = np.array(
            [t.value for t in self.timestamps], dtype=np.int64
        )
        self._ts_to_idx = {int(v): i for i, v in enumerate(self._ts_ns)}

        # ── Pre-compute anchor indices & calendar features ──
        self._precompute()

    # -----------------------------------------------------------------
    # Pre-computation (run once at init)
    # -----------------------------------------------------------------

    def _precompute(self) -> None:
        """Pre-compute all anchor indices and calendar features.

        This is *the* key optimisation: instead of doing dict lookups +
        pandas arithmetic inside ``__getitem__`` (called millions of times
        during training), we pay the cost once here and store results as
        plain numpy arrays.  ``__getitem__`` then reduces to pure slicing.
        """
        n = len(self.origin_indices)
        H = self.horizon
        Kd = self.daily_anchors
        Ky = self.yearly_anchors
        ts_ns = self._ts_ns
        ts_pd = self.timestamps
        lut = self._ts_to_idx

        NS_PER_HOUR = 3_600_000_000_000
        NS_PER_DAY = 24 * NS_PER_HOUR

        # Pre-allocate
        self._daily_idx = np.full((n, H, Kd), -1, dtype=np.int32)
        self._yearly_idx = np.full((n, H, Ky), -1, dtype=np.int32)
        # Real time gap (in days) between the horizon target and each anchor.
        # The model's PeriodicAnchorAttention encodes these as time-distance
        # bias; hardcoding a fixed 1..K_d arange in the model was wrong for
        # horizons > 1 day (the first daily anchor sits at ceil(h/4) days
        # before the target, which grows with the horizon).
        self._daily_gaps = np.zeros((n, H, Kd), dtype=np.float32)
        self._yearly_gaps = np.zeros((n, H, Ky), dtype=np.float32)
        self._calendar = np.zeros((n, H, 5), dtype=np.float32)

        for i in range(n):
            origin_idx = int(self.origin_indices[i])
            origin_ns = ts_ns[origin_idx]

            for h in range(1, H + 1):
                target_idx = origin_idx + h
                target_ns = ts_ns[target_idx]
                target_pd = ts_pd[target_idx]

                # ── Daily anchors ──
                horizon_hours = (target_ns - origin_ns) / NS_PER_HOUR
                horizon_days = horizon_hours / 24.0
                first_lag = max(1, math.ceil(float(horizon_days)))

                for k in range(Kd):
                    offset_ns = (first_lag + k) * NS_PER_DAY
                    anchor_ns = target_ns - offset_ns
                    idx = lut.get(anchor_ns, -1)
                    if idx >= 0 and idx <= origin_idx:
                        self._daily_idx[i, h - 1, k] = idx
                        self._daily_gaps[i, h - 1, k] = (
                            (target_ns - ts_ns[idx]) / NS_PER_DAY
                        )

                # ── Yearly anchors ──
                for k in range(1, Ky + 1):
                    anchor_pd = safe_shift_year(target_pd, years=k)
                    anchor_ns = anchor_pd.value
                    idx = lut.get(anchor_ns, -1)
                    if idx >= 0 and idx <= origin_idx:
                        self._yearly_idx[i, h - 1, k - 1] = idx
                        self._yearly_gaps[i, h - 1, k - 1] = (
                            (target_ns - ts_ns[idx]) / NS_PER_DAY
                        )

                # ── Calendar features ──
                self._calendar[i, h - 1] = calendar_features(
                    target_pd, h, H
                )

    # -----------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.origin_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        origin_idx = int(self.origin_indices[item])

        # ── Recent window (single contiguous slice) ──
        recent_start = origin_idx - self.recent_length + 1
        x_recent = torch.from_numpy(
            self.values[recent_start : origin_idx + 1].copy()
        )

        # ── Daily anchors from pre-computed index table ──
        daily_idx = self._daily_idx[item]                     # [H, K_d]
        daily_mask = daily_idx >= 0
        daily_anchor = np.zeros(
            (self.horizon, self.daily_anchors, self.num_features),
            dtype=np.float32,
        )
        valid_d = daily_idx[daily_mask]
        if valid_d.size > 0:
            daily_anchor[daily_mask] = self.values[valid_d]

        # ── Yearly anchors from pre-computed index table ──
        yearly_idx = self._yearly_idx[item]                   # [H, K_y]
        yearly_mask = yearly_idx >= 0
        yearly_anchor = np.zeros(
            (self.horizon, self.yearly_anchors, self.num_features),
            dtype=np.float32,
        )
        valid_y = yearly_idx[yearly_mask]
        if valid_y.size > 0:
            yearly_anchor[yearly_mask] = self.values[valid_y]

        # ── Target ──
        target = self.values[
            origin_idx + 1 : origin_idx + 1 + self.horizon,
            self.target_index,
        ].copy()

        item = {
            "x_recent": x_recent,
            "daily_anchor": torch.from_numpy(daily_anchor),
            "yearly_anchor": torch.from_numpy(yearly_anchor),
            "daily_mask": torch.from_numpy(daily_mask),
            "yearly_mask": torch.from_numpy(yearly_mask),
            # True time distance (days) between the target and each anchor,
            # used by PeriodicAnchorAttention for the time-distance bias.
            "daily_gaps": torch.from_numpy(self._daily_gaps[item].copy()),
            "yearly_gaps": torch.from_numpy(self._yearly_gaps[item].copy()),
            "future_calendar": torch.from_numpy(self._calendar[item].copy()),
            "target": torch.from_numpy(target),
            "origin_index": torch.tensor(origin_idx, dtype=torch.long),
        }

        if self.co2_index is not None:
            # CO2 at each of the H TARGET steps (not the origin), already in the
            # fixed [0,1] scale the Trainer applies to that column.
            item["future_co2"] = torch.from_numpy(
                self.values[
                    origin_idx + 1: origin_idx + 1 + self.horizon, self.co2_index
                ].copy()).unsqueeze(-1)                       # [H, 1]

        return item


# ═══════════════════════════════════════════════════════════════════════
# Origin index generation
# ═══════════════════════════════════════════════════════════════════════


def make_origin_indices(
    timestamps: np.ndarray,
    split_start: "str | pd.Timestamp",
    split_end: "str | pd.Timestamp",
    recent_length: int,
    horizon: int,
    step_size: int,
) -> np.ndarray:
    """Generate valid origin indices for a time split.

    An origin is a time step that can serve as the forecast origin:
    it has enough preceding steps (recent_length) and following steps (horizon).

    Args:
        timestamps: full timestamp array
        split_start: start of the split (inclusive)
        split_end: end of the split (inclusive)
        recent_length: number of preceding time steps needed
        horizon: number of future time steps needed
        step_size: stride between consecutive origins

    Returns:
        Array of valid origin indices
    """
    time_index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    start = pd.Timestamp(split_start)
    end = pd.Timestamp(split_end)

    candidates = np.flatnonzero((time_index >= start) & (time_index <= end))

    valid = []
    for idx in candidates[::step_size]:
        if idx - recent_length + 1 < 0:
            continue
        if idx + horizon >= len(time_index):
            continue
        # Ensure ALL horizon steps stay within the split boundary (no leakage).
        # Targets cover [idx+1, idx+horizon], so the LAST target step is
        # idx+horizon — checking idx+horizon-1 let the final step spill into
        # the next split (e.g. a train origin predicting past train_end).
        if time_index[idx + horizon] > end:
            continue
        valid.append(idx)

    return np.asarray(valid, dtype=np.int64)
