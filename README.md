# MTTGNet — Multi-Timescale Graph-Temporal Network for Global Mean Surface Temperature

A geophysically-grounded graph-temporal model for **global mean surface temperature
(GMST)**, used at two timescales:

* **weather scale (7 and 30 days)** — deterministic multi-step forecasting on raw
  6-hourly ERA5, with an explicit encoding of the physics a forecast needs at each
  lead time: orbital and diurnal phase (periodic anchors), directed atmospheric
  coupling (variable interaction graph), and the CO₂ background state;
* **century scale** — the same architecture trained on CMIP6 to represent
  multi-year forced change, then driven by SSP CO₂ pathways to project 2081–2100
  warming out of sample.

This repository contains the model, the training/evaluation pipeline, the data
preparation scripts and the tests. It deliberately does **not** contain data,
figures, trained checkpoints or the experiment scripts behind the paper; three
example configs are included so the code runs as-is.

---

## 1. Install

```bash
pip install -r requirements.txt
```

The reported runs used **Python 3.10, PyTorch 2.1.2+cu121, one NVIDIA GPU**. The
H=28 configuration needs roughly 4 GB of GPU memory at batch size 64. Everything
except training runs on CPU.

## 2. Data format

Training reads two NumPy files per dataset, named by the config:

| key | shape | dtype | meaning |
|---|---|---|---|
| `data.values_path` | `[T, N]` | `float32` | one row per timestep; **column 0 = the target**, column 1 = CO₂ |
| `data.timestamps_path` | `[T]` | `datetime64` | the matching time index |

The column contract is fixed: `data.target_index = 0` and `data.co2_index = 1`, CO₂
always immediately after the target. The remaining `N-2` columns are the atmospheric
variables (the paper's Dataset D uses eight: the T2m target plus U10, V10, MSLP,
SSRD, Niño3.4, TCC and CO₂). With `dataset.time_encode: true`, two sinusoidal
time-of-day channels are appended internally, so temporal modules see
`num_features + 2` channels.

The papered series are area-weighted global means at 6-hourly resolution with the
split `train 1980–2018 / val 2019–2020 / test 2021–2024`, but the loader only
requires the column contract above — point `values_path` at your own series and it
will train.

**Building the series from raw reanalysis.** `scripts/prepare_datasets.py` does one
netCDF pass over ERA5 and writes the feature bank plus the assembled A/B/C/D
datasets; `scripts/prepare_hadcrut5.py`, `scripts/prepare_cmip6_tas.py`,
`scripts/process_cmip6_co2.py` and `scripts/download_cmip6_tas.py` do the same for
the HadCRUT5 record and the CMIP6 tas/CO₂ pathways. They need `xarray` and
`netCDF4` (see `requirements.txt`) and your own raw-data access, so they are here
for completeness rather than for out-of-the-box use.

## 3. Quick start

```bash
# train the 7-day configuration (writes outputs/example_h28/...)
python run_model.py configs/example_h28.yaml

# 3-seed run with an aggregated summary
python scripts/run_multiseed.py configs/example_h28.yaml --seeds 123,42,789

# 30-day configuration
python run_model.py configs/example_h120.yaml

# century-scale SSP projection from a trained monthly model
python run_mttgnet_projections.py
```

`run_model.py` writes `checkpoints/best_model.pt` and a set of result arrays under
`result_dir`, including `test_rmse_per_horizon.npy`, `test_predict.npy` and
`test_target.npy`.

> **When you report an RMSE**, use the pooled definition
> `sqrt(mean_h(rmse_h**2))` over `test_rmse_per_horizon.npy`. The flat RMSE over all
> elements is a different, slightly smaller number; mixing the two is the easiest way
> to misquote a result from this codebase.

## 4. Model

`src/mttgnet.py` holds the whole model. The forward path is

```
x_recent ─┬─ VariableGNN ── GRU encoder ──┐
          │                               ├─ MultiSourceGateFusion ─ per-horizon decoder → prediction
anchors ──┴─ AnchorVariableEncoder (GAT) ─┤
              → PeriodicAnchorAttention ──┤
              → DistributionShiftEncoder ─┘   (CO₂ climate-state modulation)
              DynamicMemory ──────────────────┘  (prototype bank, residual branch)
```

| module | class | role |
|---|---|---|
| variable-interaction graph | `mttgnet.VariableGNN` | directed coupling between atmospheric variables |
| temporal encoder | `torch.nn.GRU` | stacked GRU over the look-back window |
| anchor encoder | `mttgnet.AnchorVariableEncoder` | GAT over the retrieved historical anchors |
| periodic attention | `mttgnet.PeriodicAnchorAttention` | phase-locked retrieval at the target's calendar position |
| CO₂ modulation | `mttgnet.DistributionShiftEncoder` | log-forcing climate-state modulation + scalar bias |
| prototype memory | `mttgnet.DynamicMemory` | learned bank of climate-state prototypes (residual branch) |
| fusion | `mttgnet.MultiSourceGateFusion` | horizon-conditioned softmax over the sources |
| decoder | `mttgnet.CalendarConditionedDecoder` and siblings | one scalar prediction per horizon step |
| multi-rate branch | `MTTGNetModel` (coarse head + gate) | N-HiTS-style pooled coarse prediction |

Two design points the configs expose, worth knowing before ablating anything:

* **The prototype memory's fusion gate is initialised at `sigmoid(-3) ≈ 0.047`**
  (`DynamicMemory`; `memory_gate_init` on the wrapper). The retrieved prototypes
  therefore enter the residual branch at ~5 % weight, and in every checkpoint we
  inspected the gate stayed within ±0.005 of that value. Ablating this module
  therefore measures the extra residual branch much more than it measures retrieval.
  Set `memory_gate_init: 0.0` to put retrieval at half weight, and verify the gate in
  your own checkpoints.
* **A component ablation is only interpretable if every row shares one parameter
  base.** `use_multirate` changes `n_params` by 72; compare `n_params` in
  `summary.txt` across the rows you put in one table.

Every `model_params` key in a config is read by `MTTGNetWrapper.__init__`. That
constructor ends in `**kwargs`, so a key the runner forgets to forward is swallowed
silently — `scripts/run_multiseed.py` therefore asserts that **all** wrapper
parameters are passed and raises if one is missing. If you add a parameter, add it to
that call too.

## 5. Tests

```bash
python -m pytest tests/ -q
```

* `test_time_utils.py` — calendar and phase encoding,
* `test_batch_independence.py` — the variable graph does not leak across the batch,
* `test_future_leakage.py` — the split windows cannot see future targets.

## 6. Configuration reference

`configs/example_h28.yaml` (7-day), `configs/example_h120.yaml` (30-day) and
`configs/example_monthly.yaml` (the monthly two-variable recipe used for the
century-scale projection) are templates. The key groups are

| group | what it controls |
|---|---|
| `data` | `values_path`, `timestamps_path`, `target_index=0`, `co2_index=1` |
| `split` | `train_end` / `val_end` / `test_end` |
| `dataset` | `recent_length` (look-back steps), `horizon`, `daily_anchors`, `yearly_anchors`, `train_step_size`, `eval_step_size`, `time_encode` |
| `model_params` | every architectural switch listed in section 4 |
| `training` | optimiser, `epochs`, `patience`, AMP, workers |
| `loss` | `delta_weight`, `horizon_weight_power/max`, `short_lead_loss_steps/weight`, monotonicity and extreme weights |
| `output` | `checkpoint_dir`, `result_dir` |

`horizon` is in *steps* of the input resolution: 28 steps = 7 days and 120 steps =
30 days at 6-hourly sampling; 120 steps = 10 years at monthly sampling.

## 7. What is not here

Data, figures, trained checkpoints, the per-experiment configs, and the analysis and
plotting scripts behind the paper's tables. They are omitted deliberately: they are
bulky, they are tied to one machine's directory layout, and reproducing them means
re-running the model rather than reading a file. The pipeline above is the code that
produces them.

## 8. Citation

```bibtex
@article{shi2026mttgnet,
  title   = {MTTGNet: Multi-Timescale Graph-Temporal Modelling of Global Mean Surface Temperature},
  author  = {Shi, Zihao and Wang, Zhiguo and Chen, Ziwei},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
