import numpy as np
import random
import torch


def dict2str(d, level=0, compact=True):
    out_str = []
    if compact:
        indents, newline, colon, comma = "." * level, "", "(", ")+"
        brackets = "", ""
    else:
        indents, newline, colon, comma = "  " * level, "\n", ": ", ","
        brackets = "{", "}"
    for i, (k, v) in enumerate(d.items()):
        line = indents + str(k) + colon
        if isinstance(v, str):
            line += v
        elif isinstance(v, float):
            line += f"{v:.3e}"
        elif isinstance(v, dict):
            line += brackets[0] + newline + dict2str(v, level + 1, compact=compact)
            line += indents + brackets[1]
        else:
            if compact and isinstance(v, (list, tuple)):
                line += "_".join(list(map(str, v)))
            else:
                line += str(v)
        if i != len(d) - 1:
            line += comma
        line += newline
        out_str.append(line)
    return "".join(out_str)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_param(param, obj_1, obj_2):
    def get(obj, attr):
        if hasattr(obj, "__getitem__"):
            return obj[attr]
        elif hasattr(obj, "__getattribute__"):
            return getattr(obj, attr)
        else:
            NotImplementedError("Not supported!")
    try:
        param = get(obj_1, param)
    except (KeyError, AttributeError):
        param = get(obj_2, param)
    return param


class ConfigDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, name):
        return self.get(name, None)
