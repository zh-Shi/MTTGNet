"""MTTGNet — Dual Memory Architecture for Weather Forecasting.

Core innovation: Dynamic Prototype Memory + Periodic Anchor Attention
with gate-controlled multi-source fusion.

Core modules:
  - mttgnet.py:  MTTGNet model (main)
  - layers.py:   Temporal encoders (GRU, LSTM, TCN, Mamba, Transformer, PatchTST)
  - dataset.py:  PeriodicAnchorDataset with historical anchor retrieval
  - trainer.py:  Unified training engine with AMP and gradient accumulation
  - losses.py:   MultiHorizonForecastLoss with extreme-event weighting
  - metrics.py:  Extended evaluation metrics (RMSE, MAE, NSE, KGE, IoA, etc.)
  - scaler.py:   StandardScaler for data normalization
  - config.py:   YAML config loader
  - time_utils.py: Anchor time computation and calendar features
  - seed.py:     Deterministic seeding
"""

__version__ = "1.0.0"
