"""Preprocessing and aggregation rules shared by training and validation."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .time_utils import safe_shift_year

PROTOCOL_VERSION = 'independent-v2'


def make_origin_indices(timestamps, split_start, split_end, recent_length, horizon, step_size):
    ts = pd.DatetimeIndex(timestamps)
    if not ts.is_monotonic_increasing or ts.has_duplicates or step_size < 1:
        raise ValueError('Invalid timestamps or stride')
    candidates = np.flatnonzero((ts >= pd.Timestamp(split_start)) & (ts <= pd.Timestamp(split_end)))[::step_size]
    candidates = candidates[(candidates >= recent_length-1) & (candidates+horizon < len(ts))]
    return candidates[np.asarray(ts[candidates+horizon] <= pd.Timestamp(split_end))].astype(np.int64)


def scenario_windows(cfg, name, stamps, holdout=None):
    """Single source of truth for membership and time boundaries."""
    names = [s['name'] for s in cfg['dataset']['scenarios']]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate scenario names')
    if holdout is not None and holdout not in names:
        raise ValueError(f'Unknown holdout: {holdout}')
    split = cfg['split']; ds = cfg['dataset']; ts = pd.DatetimeIndex(stamps)
    mode = split.get('mode', 'time')
    if mode not in ('time', 'scenario'):
        raise ValueError(f'Unknown split mode: {mode}')
    windows = dict.fromkeys(('train', 'val', 'test'))
    if mode == 'scenario':
        if holdout:
            raise ValueError('Use scenario membership OR --holdout, not both')
        assigned = [n for part in windows for n in (split.get(part) or [])]
        if len(set(assigned)) != len(assigned) or not set(assigned) <= set(names):
            raise ValueError('Scenario membership overlaps or contains unknown names')
        if not assigned:
            raise ValueError('Scenario split needs explicit memberships')
        for part in windows:
            if name in (split.get(part) or []):
                windows[part] = (ts[0], ts[-1], ds['train_step_size'] if part == 'train' else ds['eval_step_size'])
        return windows
    bounds = [('train',ts[0],split['train_end']),('val',pd.Timestamp(split['train_end'])+pd.Timedelta(nanoseconds=1),split['val_end']),('test',pd.Timestamp(split['val_end'])+pd.Timedelta(nanoseconds=1),split['test_end'])]
    for part, start, end in bounds:
        if holdout and (name == holdout or part == 'test'):
            continue
        windows[part] = (pd.Timestamp(start),pd.Timestamp(end),ds['train_step_size'] if part == 'train' else ds['eval_step_size'])
    if name == holdout:
        windows['test'] = (ts[0],pd.Timestamp(split['test_end']),ds['eval_step_size'])
    return windows


def training_mask(cfg, name, stamps, holdout=None):
    win = scenario_windows(cfg,name,stamps,holdout)['train']
    ts = pd.DatetimeIndex(stamps)
    if win is None:
        return np.zeros(len(ts),dtype=bool)
    return np.asarray((ts >= win[0]) & (ts <= win[1]))


def anchor_index_tables(stamps, origin_index, targets, daily_anchors, yearly_anchors, lookup=None):
    """Origin-known candidate indices, identical in datasets and recursive rollout."""
    ts = pd.DatetimeIndex(stamps); targets = pd.DatetimeIndex(targets)
    origin = ts[origin_index]; lut = lookup if lookup is not None else {t.value:i for i,t in enumerate(ts)}
    result = {}
    for kind, count in [('daily',daily_anchors),('yearly',yearly_anchors)]:
        indices = np.full((len(targets),count),-1,dtype=np.int32)
        gaps = np.zeros(indices.shape,dtype=np.float32)
        for h,t in enumerate(targets):
            lag = max(1,math.ceil((t-origin)/pd.Timedelta(days=1)))
            for k in range(count):
                a = t-pd.Timedelta(days=lag+k) if kind == 'daily' else safe_shift_year(t,k+1)
                idx = lut.get(a.value,-1)
                if 0 <= idx <= origin_index:
                    indices[h,k] = idx; gaps[h,k] = (t-ts[idx])/pd.Timedelta(days=1)
        result[kind+'_indices'] = indices
        result[kind+'_gaps'] = gaps
    return result


def monthly_target_series(pred, target_times, lead=1):
    """Aggregate a fixed lead by its actual target month, preserving calendar gaps."""
    pred = np.asarray(pred); times = np.asarray(target_times)
    if pred.ndim != 2 or pred.shape != times.shape or not 1 <= lead <= pred.shape[1]:
        raise ValueError('Prediction/time shape or lead mismatch')
    idx = pd.DatetimeIndex(pd.to_datetime(times[:,lead-1]))
    return pd.Series(pred[:,lead-1],index=idx.to_period('M').astype(str)).groupby(level=0).mean()


def origin_persistence(series, target_indices):
    idx = np.asarray(target_indices)
    if idx.ndim != 2 or np.any(idx[:,0] < 1):
        raise ValueError('Invalid target index matrix')
    return np.broadcast_to(np.asarray(series)[idx[:,0]-1,None],idx.shape).copy()
