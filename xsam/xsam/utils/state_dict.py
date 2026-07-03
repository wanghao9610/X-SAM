from collections import OrderedDict


def _to_cpu(value):
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    return value


def merge_partial_state_dict_into_model(model, partial_state_dict):
    """Load partial weights into a model and return the full CPU state_dict.

    DeepSpeed checkpoints can omit frozen parameters. Loading with
    ``strict=False`` keeps those initialized or pretrained weights in the model.
    Use ``full_state_dict`` when available because XSamModel.state_dict()
    intentionally returns only trainable/export weights by default.
    """
    incompatible_keys = model.load_state_dict(partial_state_dict, strict=False)
    unexpected_keys = list(getattr(incompatible_keys, "unexpected_keys", []))
    if unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected_keys}")

    missing_keys = list(getattr(incompatible_keys, "missing_keys", []))
    if hasattr(model, "full_state_dict") and callable(model.full_state_dict):
        state_dict = model.full_state_dict()
    else:
        state_dict = model.state_dict()
    state_dict = OrderedDict((key, _to_cpu(value)) for key, value in state_dict.items())
    return state_dict, missing_keys
