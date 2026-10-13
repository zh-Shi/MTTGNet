"""Recursive CMIP6 rollout, independent of training-script imports."""
import numpy as np
import pandas as pd
import torch
from .time_utils import calendar_features


def _to_unit(co2):
    return (np.asarray(co2,dtype=np.float32)-280.0)/920.0


@torch.no_grad()
def autoregressive_project(model, origin_idx, co2_future, values_all, ts, scaler, device,
                           *, L=60, H=1, future_co2_ppm=None,
                           daily_anchors=0, yearly_anchors=0, advance=None):
    """Direct blocks with origin-known anchors from observations and earlier predictions.

    Future ESM temperatures are never read. The final partial block retains H
    query positions; only its available forcing/temperature steps are returned.
    Legacy observational models explicitly use zero anchors by default.
    ``advance`` commits only that many leading predictions per model call.
    The query still spans the training horizon H, including the complete
    available future forcing path; this supports receding-horizon inference.
    """
    from src.validation_protocol import anchor_index_tables
    if L < 1 or H < 1 or origin_idx < L-1:
        raise ValueError('Invalid rollout origin/window')
    advance = H if advance is None else advance
    if not isinstance(advance, (int, np.integer)) or not 1 <= advance <= H:
        raise ValueError('Rollout advance must be an integer between 1 and H')
    history = np.asarray(values_all[:origin_idx+1]).copy()
    history_ts = pd.DatetimeIndex(ts[:origin_idx+1])
    if len(history) != len(history_ts):
        raise ValueError('History/timestamp mismatch')
    n_scaled = len(scaler.mean); N = history.shape[1]
    TGT, CO2 = 0, 1
    # Match training preprocessing once, then retain predictions in model space.
    # A float32 Kelvin round-trip loses small anomalies (and NumPy 1.x/2.x
    # promote scalar arithmetic differently), so never re-normalize predictions.
    scaled = history.copy()
    scaled[:, :n_scaled] = scaler.transform(history[:, :n_scaled])
    scaled[:, CO2] = _to_unit(history[:, CO2])
    scaled = scaled.astype(np.float32)
    forcing = np.asarray(co2_future, dtype=np.float32)
    if future_co2_ppm is not None and len(future_co2_ppm) != len(forcing):
        raise ValueError('Future forcing length mismatch')
    preds = []
    while len(preds) < len(forcing):
        step = len(preds); hs = min(advance, len(forcing)-step)
        targets = pd.DatetimeIndex([history_ts[-1]+pd.DateOffset(months=h+1) for h in range(H)])
        tables = anchor_index_tables(history_ts, len(scaled)-1, targets, daily_anchors, yearly_anchors)
        args = {'x_recent': torch.from_numpy(scaled[-L:]).unsqueeze(0).to(device),
                'future_calendar': torch.from_numpy(np.stack([calendar_features(t,h+1,H) for h,t in enumerate(targets)])).unsqueeze(0).to(device)}
        for kind in ('daily','yearly'):
            idx = tables[kind+'_indices']; valid = idx >= 0
            values = np.zeros((*idx.shape,N),dtype=np.float32)
            values[valid] = scaled[idx[valid]]
            args[kind+'_anchor'] = torch.from_numpy(values).unsqueeze(0).to(device)
            args[kind+'_mask'] = torch.from_numpy(valid).unsqueeze(0).to(device)
            args[kind+'_gaps'] = torch.from_numpy(tables[kind+'_gaps']).unsqueeze(0).to(device)
        if future_co2_ppm is not None:
            available = min(H, len(forcing)-step)
            seg = np.asarray(future_co2_ppm[step:step+available], dtype=np.float32)
            seg = np.pad(seg,(0,H-available),mode='edge')
            args['future_co2'] = torch.from_numpy(_to_unit(seg)).reshape(1,H,1).to(device)
        pred_std = model(**args)['prediction'].cpu().numpy()[0,:hs]
        temperatures = scaler.inverse_target(pred_std.astype(np.float64),target_index=TGT)
        if not np.isfinite(temperatures).all():
            raise ValueError('Non-finite recursive projection')
        preds.extend(temperatures.tolist())
        appended = np.repeat(scaled[-1:],hs,axis=0)
        appended[:,TGT] = pred_std
        appended[:,CO2] = _to_unit(forcing[step:step+hs])
        scaled = np.concatenate([scaled,appended],axis=0)
        history_ts = history_ts.append(targets[:hs])
    return np.asarray(preds,dtype=np.float64)
