"""Batch-independence test for the current MTTGNet pipeline.

A model that fuses information across samples (e.g. a leaky normalisation,
a cross-sample top-k, or shared state mutated in-place) would give different
predictions for sample 0 depending on whether it is scored alone or inside a
batch.  This guards the core promise of the MTTGNet design: every sample
has its own temporal graph, anchors and memory, so scoring one sample at a
time must equal scoring it as part of a batch.
"""

import torch

from src.mttgnet import MTTGNetWrapper


def make_batch(batch_size: int, *, L: int = 16, H: int = 4, N: int = 4,
               K_d: int = 2, K_y: int = 1) -> dict[str, torch.Tensor]:
    return {
        "x_recent": torch.randn(batch_size, L, N),
        "daily_anchor": torch.randn(batch_size, H, K_d, N),
        "yearly_anchor": torch.randn(batch_size, H, K_y, N),
        "daily_mask": torch.ones(batch_size, H, K_d, dtype=torch.bool),
        "yearly_mask": torch.ones(batch_size, H, K_y, dtype=torch.bool),
        "future_calendar": torch.randn(batch_size, H, 5),
    }


def test_batch_independence() -> None:
    torch.manual_seed(1)

    model = MTTGNetWrapper(
        num_features=4,
        hidden_dim=16,
        horizon=4,
        seq_len=16,
        dropout=0.0,
        memory_slots=8,
        mem_top_k=4,
        num_gru_layers=1,
        temporal_encoder="gru",
        var_encoder_type="gat",
    )

    model.eval()
    batch = make_batch(4)
    single = {key: value[0:1] for key, value in batch.items()}

    with torch.no_grad():
        pred_batch = model(**batch)["prediction"]
        pred_single = model(**single)["prediction"]

    assert pred_batch.shape[0] == 4 and pred_single.shape[0] == 1
    assert torch.allclose(pred_single[0], pred_batch[0], atol=1e-5)
