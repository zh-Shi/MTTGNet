"""Resolve new fusion configurations and recover saved checkpoint structure."""
from copy import deepcopy
from numbers import Integral


def resolve_fusion_sources(use_memory, fusion_sources=None):
    """Keep the original three-source gate; two-source fusion is an explicit control."""
    if fusion_sources is None:
        fusion_sources = 3
    if isinstance(fusion_sources, bool) or not isinstance(fusion_sources, Integral):
        raise ValueError('fusion_sources must be the integer 2 or 3')
    if fusion_sources not in (2, 3):
        raise ValueError('Fusion requires two or three sources')
    if use_memory and fusion_sources != 3:
        raise ValueError('Memory-enabled model requires three fusion sources')
    return int(fusion_sources)


def record_fusion_config(cfg, *, default_use_memory=True):
    """Record the effective clean-model setting in the config saved during training."""
    m = cfg.setdefault('model_params', {})
    if m.get('arch', 'mttgnet') == 'clean':
        m['fusion_sources'] = resolve_fusion_sources(
            m.get('use_memory', default_use_memory), m.get('fusion_sources'))
    elif m.get('fusion_sources') not in (None, 3):
        raise ValueError('Two-source fusion requires model_params.arch: clean')


def config_for_saved_fusion(cfg, state_dict):
    """Infer source count from weights, without converting or dropping parameters.

    Older checkpoint configs omitted fusion_sources. Reading the last gate
    layer preserves their exact three-source structure after defaults change.
    Explicit metadata that disagrees with the weights is an error.
    """
    resolved = deepcopy(cfg)
    gates = [value for key, value in state_dict.items()
             if key.endswith('fusion.gate_predictor.2.weight')]
    if not gates:
        return resolved  # Baseline model, with no fusion module.
    if len(gates) != 1:
        raise ValueError('Expected exactly one fusion gate in the checkpoint')
    count = int(gates[0].shape[0])
    m = resolved.setdefault('model_params', {})
    explicit = m.get('fusion_sources')
    if explicit is not None and explicit != count:
        raise ValueError(f'fusion_sources={explicit} disagrees with checkpoint gate width {count}')
    m['fusion_sources'] = resolve_fusion_sources(m.get('use_memory', True), count)
    return resolved
