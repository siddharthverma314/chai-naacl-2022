from .base import Transform
from torch.functional import F
import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict


def onehotdim(space: gym.Space) -> int:
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return int(sum([onehotdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([onehotdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return sum(space.nvec)
    else:
        raise NotImplementedError


def unonehotdim(space: gym.Space) -> int:
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, Tuple):
        return int(sum([unonehotdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([unonehotdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return len(space.nvec)
    else:
        raise NotImplementedError


def one_hot_space(space) -> gym.Space:
    if isinstance(space, Tuple):
        return Tuple([one_hot_space(s) for s in space.spaces])
    elif isinstance(space, Dict):
        return Dict({k: one_hot_space(v) for k, v in space.spaces.items()})
    elif isinstance(space, MultiDiscrete) or isinstance(space, Discrete):
        dim = onehotdim(space)
        return Box(np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))
    else:
        return space


class OneHot(Transform):
    def __init__(self, space):
        super().__init__(space, one_hot_space(space))

    @staticmethod
    def one_hot(space, x):
        if isinstance(space, list) or isinstance(space, tuple):
            return tuple([OneHot.one_hot(s, xp) for s, xp in zip(space, x)])
        elif isinstance(space, Tuple):
            return OneHot.one_hot(space.spaces, x)
        elif isinstance(space, Dict):
            return {k: OneHot.one_hot(s, x[k]) for k, s in space.spaces.items()}
        elif isinstance(space, MultiDiscrete):
            return torch.cat(
                OneHot.one_hot([Discrete(d) for d in space.nvec], x.split(1, dim=1)),
                dim=1,
            )
        elif isinstance(space, Discrete):
            return F.one_hot(x.squeeze(1).long(), space.n).float()
        else:
            return x

    def forward(self, x):
        return self.one_hot(self.before_space, x)


class UnOneHot(Transform):
    def __init__(self, space):
        super().__init__(one_hot_space(space), space)

    @staticmethod
    def un_one_hot(space, x):
        if isinstance(space, tuple) or isinstance(space, list):
            return tuple([UnOneHot.un_one_hot(s, xp) for s, xp in zip(space, x)])
        elif isinstance(space, Tuple):
            return UnOneHot.un_one_hot(space.spaces, x)
        elif isinstance(space, Dict):
            return {k: UnOneHot.un_one_hot(s, x[k]) for k, s in space.spaces.items()}
        elif isinstance(space, MultiDiscrete):
            return torch.cat(
                UnOneHot.un_one_hot(
                    [Discrete(d) for d in space.nvec],
                    x.split(tuple(space.nvec), dim=1),
                ),
                dim=1,
            )
        elif isinstance(space, Discrete):
            return torch.argmax(x, dim=1, keepdim=True)
        else:
            return x

    def forward(self, x):
        return self.un_one_hot(self.after_space, x)
