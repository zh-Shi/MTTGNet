import numpy as np
import pandas as pd

from src.dataset import PeriodicAnchorDataset


def test_future_target_change() -> None:
    timestamps = pd.date_range(
        "2018-01-01",
        periods=4 * 365 * 7,
        freq="6h",
    )

    values = np.random.randn(
        len(timestamps), 8
    ).astype(np.float32)

    origin_idx = 4 * 365 * 5

    args = {
        "timestamps": timestamps,
        "origin_indices": np.asarray([origin_idx]),
        "recent_length": 120,
        "horizon": 28,
        "daily_anchors": 7,
        "yearly_anchors": 3,
        "target_index": 0,
    }

    sample_a = PeriodicAnchorDataset(
        values=values.copy(), **args
    )[0]

    changed = values.copy()
    changed[origin_idx + 1 : origin_idx + 29, 0] += 1000.0

    sample_b = PeriodicAnchorDataset(
        values=changed, **args
    )[0]

    assert np.allclose(
        sample_a["x_recent"].numpy(),
        sample_b["x_recent"].numpy(),
    )

    assert np.allclose(
        sample_a["daily_anchor"].numpy(),
        sample_b["daily_anchor"].numpy(),
    )

    assert np.allclose(
        sample_a["yearly_anchor"].numpy(),
        sample_b["yearly_anchor"].numpy(),
    )

    assert not np.allclose(
        sample_a["target"].numpy(),
        sample_b["target"].numpy(),
    )
