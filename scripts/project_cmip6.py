#!/usr/bin/env python3
"""Run conditional monthly SSP projections from a saved training checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--advance', type=int, default=12, help='Monthly predictions committed per model call')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    import torch
    from experiments.cmip6.run_cmip6_pure_ml import build_model, load_series
    from src.fusion_config import config_for_saved_fusion
    from src.scaler import StandardScaler
    from src.scenario_rollout import autoregressive_project
    from src.temperature_history import TemperatureHistoryMaskedModel

    run = a.run_dir.resolve()
    saved = torch.load(run/'checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    cfg = config_for_saved_fusion(saved['config'], saved['model_state_dict'])
    d = cfg['dataset']
    if cfg.get('data', {}).get('target_index', 0) != 0 or cfg.get('data', {}).get('co2_index', 1) != 1:
        raise ValueError('Monthly projection requires [temperature, CO2] columns')
    if d.get('time_encode', False) or d.get('resolution') != '1m':
        raise ValueError('Use the two-variable monthly CMIP6 configuration without hour encodings')
    if not 1 <= a.advance <= d['horizon']:
        raise ValueError('Advance must be between 1 and the trained horizon')
    with np.load(run/'scaler.npz') as f:
        scaler = StandardScaler(mean=f['mean'], std=f['std'])
    protocol_path = run/'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        if not np.array_equal(scaler.mean, protocol['scaler_mean']) or not np.array_equal(scaler.std, protocol['scaler_std']):
            raise ValueError('Saved scaler and training protocol disagree')
        if any(pd.Timestamp(r['last']) > pd.Timestamp(cfg['split']['train_end']) for r in protocol['normalization_records'].values()):
            raise ValueError('Normalization includes values after the training period')
    model = build_model(cfg, 2)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model = model.to(a.device).eval()
    summary_path = run/'summary.json'
    kind = json.loads(summary_path.read_text(encoding='utf-8')).get('temperature_history', 'full') if summary_path.exists() else 'full'
    if kind not in ('full', 'masked'):
        raise ValueError('Unknown temperature-history setting')
    inference = TemperatureHistoryMaskedModel(model) if kind == 'masked' else model
    specs = d['scenarios']
    historical = next(s for s in specs if s['name'] == 'historical')
    history, hist_dates = load_series(historical)
    out = a.out or run/'projection'
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in specs:
        if spec['name'] == 'historical':
            continue
        raw, dates = load_series(spec)
        expected = pd.date_range(hist_dates[-1] + pd.DateOffset(months=1), periods=len(dates), freq=pd.DateOffset(months=1))
        if not pd.DatetimeIndex(dates).equals(expected):
            raise ValueError(f"{spec['name']}: scenario timestamps must follow historical months")
        pred = autoregressive_project(inference, len(history)-1, raw[:,1], history, hist_dates,
            scaler, torch.device(a.device), L=d['recent_length'], H=d['horizon'], advance=a.advance,
            future_co2_ppm=raw[:,1] if d.get('use_future_co2', False) else None,
            daily_anchors=d.get('daily_anchors',0), yearly_anchors=d.get('yearly_anchors',0))
        np.save(out/f"{spec['name']}_predict.npy", pred)
        np.save(out/f"{spec['name']}_timestamps.npy", np.asarray(dates, dtype='datetime64[ns]'))
        # Truth is used only after rollout, for scoring held-out annual means.
        years=pd.DatetimeIndex(dates).year
        annual_pred=pd.Series(pred,index=years).groupby(level=0).mean()
        annual_true=pd.Series(raw[:,0].astype(float),index=years).groupby(level=0).mean()
        mask=(annual_pred.index>=2081)&(annual_pred.index<=2100)
        error=(annual_pred-annual_true)[mask].to_numpy()
        results[spec['name']] = dict(months=len(pred), annual_rmse_2081_2100_K=float(np.sqrt(np.mean(error**2))) if len(error) else None,
            annual_bias_2081_2100_K=float(error.mean()) if len(error) else None)
    (out/'summary.json').write_text(json.dumps(dict(advance_months=a.advance, temperature_history=kind,
        inference='recursive temperature history; prescribed CO2; no future temperature feedback',scenarios=results), indent=2),encoding='utf-8')
    print(f'Saved {len(results)} scenario projections to {out}')


if __name__ == '__main__':
    main()
