from typing import List
from .base import Transform
from collections import OrderedDict
from gym.spaces import Tuple, Dict, Space
from .base import flatdim
import numpy as np
import torch
import gym


def flatten_space(space: Space) -> Tuple:
    if isinstance(space, Tuple):
        return Tuple(sum([flatten_space(s).spaces for s in space.spaces], []))
    elif isinstance(space, Dict):
        return Tuple(sum([flatten_space(s).spaces for s in space.spaces.values()], []))
    else:
        return Tuple([space])


class Flatten(Transform):
    def __init__(self, space):
        super().__init__(space, flatten_space(space))

    @staticmethod
    def flatten(space, x):
        if isinstance(space, Tuple):
            return torch.cat(
                [Flatten.flatten(s, xp) for xp, s in zip(x, space.spaces)], dim=1
            )
        elif isinstance(space, Dict):
            return torch.cat(
                [Flatten.flatten(s, x[k]) for k, s in space.spaces.items()], dim=1
            )
        else:
            return x

    def forward(self, x):
        return self.flatten(self.before_space, x)


class Unflatten(Transform):
    def __init__(self, space: Space):
        super().__init__(flatten_space(space), space)

    @staticmethod
    def unflatten(space, x):
        if isinstance(space, Tuple):
            list_flattened = torch.split(x, list(map(flatdim, space.spaces)), dim=-1)
            list_unflattened = [
                Unflatten.unflatten(s, flattened)
                for flattened, s in zip(list_flattened, space.spaces)
            ]
            return tuple(list_unflattened)
        elif isinstance(space, Dict):
            list_flattened = torch.split(
                x, list(map(flatdim, space.spaces.values())), dim=-1
            )
            list_unflattened = [
                (key, Unflatten.unflatten(s, flattened))
                for flattened, (key, s) in zip(list_flattened, space.spaces.items())
            ]
            return OrderedDict(list_unflattened)
        else:
            return x

    def forward(self, x):
        return self.unflatten(self.after_space, x)
