"""Input-only temperature controls; labels and forcing are never changed."""


def mask_temperature_history(inputs, *, enabled=True, target_index=0):
    if not enabled:
        return inputs
    result = dict(inputs)
    for key in ('x_recent', 'daily_anchor', 'yearly_anchor'):
        value = inputs.get(key)
        if value is None:
            continue
        if value.ndim < 2 or not 0 <= target_index < value.shape[-1]:
            raise ValueError(f'{key}: invalid temperature channel')
        copy = value.clone() if hasattr(value, 'clone') else value.copy()
        copy[..., target_index] = 0
        result[key] = copy
    return result


def masked_anchor_forward(model, batch):
    from .trainer import anchor_forward
    return anchor_forward(model, mask_temperature_history(batch))


class TemperatureHistoryMaskedModel:
    """Inference adapter; checkpoint keys belong to the unwrapped base model."""
    def __init__(self, model):
        self.model = model

    def __call__(self, **inputs):
        return self.model(**mask_temperature_history(inputs))
