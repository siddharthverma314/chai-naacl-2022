from typing import List
from torch.nn import Module
from abc import ABCMeta
from gym.spaces import Space, Box, Discrete, Tuple, Dict, MultiBinary, MultiDiscrete
import numpy as np


def flatdim(space: Space) -> int:
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return len(space.nvec)
    else:
        raise NotImplementedError


class Transform(Module):
    def __init__(self, before_space: Space, after_space: Space):
        super().__init__()
        self.before_space = before_space
        self.after_space = after_space
        self.before_dim = flatdim(before_space)
        self.after_dim = flatdim(after_space)
