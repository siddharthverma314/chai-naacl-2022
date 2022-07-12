from typing import Dict
from .loggable import Loggable
import torch


def consolidate_stats(items) -> Dict[str, float]:
    """Take a list of items and return their statistics (mean, std, min, max)"""

    if isinstance(items, torch.Tensor):
        items = items.cpu().detach()
    else:
        items = torch.tensor(items)

    if len(items) == 0:
        return {i: 0.0 for i in ["mean", "max", "min", "std"]}

    return {
        "mean": torch.mean(items),
        "max": torch.max(items),
        "min": torch.min(items),
        "std": torch.std(items),
    }


class Container(dict, Loggable):
    """A loggable dictionary"""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        Loggable.__init__(self)

    def log_collect(self):
        return {k: v for k, v in self.items() if isinstance(v, Loggable)}

    def log_local_hyperparams(self) -> Dict[str, object]:
        return {}

    def log_local_epoch(self) -> Dict[str, object]:
        return {}


class Hyperparams(dict, Loggable):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        Loggable.__init__(self)

    def log_collect(self):
        return {}

    def log_local_hyperparams(self) -> Dict[str, object]:
        return self

    def log_local_epoch(self) -> Dict[str, object]:
        return {}
