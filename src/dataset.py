from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .validation_protocol import anchor_index_tables

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
        anchor_cache_dir: str | None = None,
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
        self._anchor_cache_dir = Path(anchor_cache_dir) if anchor_cache_dir else None
        if not self._load_anchor_cache():
            self._precompute()
            self._save_anchor_cache()

    _CACHE_ARRAYS = ('_daily_idx', '_yearly_idx', '_daily_gaps', '_yearly_gaps', '_calendar')

    def _cache_folder(self):
        if self._anchor_cache_dir is None:
            return None
        digest = hashlib.sha256(self._ts_ns.tobytes()+self.origin_indices.tobytes())
        digest.update(str((self.horizon, self.daily_anchors, self.yearly_anchors)).encode())
        for name in ('dataset.py', 'validation_protocol.py', 'time_utils.py'):
            digest.update((Path(__file__).parent/name).read_bytes())
        return self._anchor_cache_dir/digest.hexdigest()

    def _load_anchor_cache(self):
        folder = self._cache_folder()
        if folder is None or not (folder/'complete.json').exists():
            return False
        record = json.loads((folder/'complete.json').read_text())
        for name in self._CACHE_ARRAYS:
            path = folder/f'{name}.npy'
            if hashlib.sha256(path.read_bytes()).hexdigest() != record[name]['sha256']:
                raise ValueError('Anchor cache corrupted: '+str(path))
            array = np.load(path, mmap_mode='r')
            if list(array.shape) != record[name]['shape'] or str(array.dtype) != record[name]['dtype']:
                raise ValueError('Anchor cache shape or dtype mismatch')
            setattr(self, name, array)
        return True

    def _save_anchor_cache(self):
        folder = self._cache_folder()
        if folder is None:
            return
        folder.mkdir(parents=True, exist_ok=True)
        record = {}
        for name in self._CACHE_ARRAYS:
            array = getattr(self, name); path = folder/f'{name}.npy'
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, array)
            temporary.replace(path)
            record[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                shape=list(array.shape), dtype=str(array.dtype))
        temporary = folder/'complete.tmp'
        temporary.write_text(json.dumps(record))
        temporary.replace(folder/'complete.json')

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

            tables = anchor_index_tables(ts_pd, origin_idx, ts_pd[origin_idx+1:origin_idx+H+1], Kd, Ky, lookup=lut)
            self._daily_idx[i] = tables['daily_indices']
            self._yearly_idx[i] = tables['yearly_indices']
            self._daily_gaps[i] = tables['daily_gaps']
            self._yearly_gaps[i] = tables['yearly_gaps']

            for h in range(1, H + 1):
                target_idx = origin_idx + h
                target_ns = ts_ns[target_idx]
                target_pd = ts_pd[target_idx]

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
    from .validation_protocol import make_origin_indices as shared_origins
    return shared_origins(timestamps, split_start, split_end, recent_length, horizon, step_size)
