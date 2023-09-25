from functools import partial

import torch.nn as nn

from toolbox.models.utils import build_kwargs_from_config

__all__ = ["build_act"]


# register activation function here
#REGISTERED_ACT_DICT: dict[str, type] = {
REGISTERED_ACT_DICT: dict = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    #"gelu": partial(nn.GELU, approximate="tanh"),
    "gelu": nn.GELU,
}


def build_act(name: str, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None