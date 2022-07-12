import numpy as np
import torch
from gym import spaces


class String(spaces.Space):
    def __init__(self):
        super().__init__(shape=None, dtype=str)
        self.chars = list("abcdefghijklmnopqrstuvwxyz ")

    def sample(self):
        # this isn't generating all of the output space
        length = np.random.randint(1, 50)
        return "".join([np.random.choice(self.chars) for _ in range(length)])

    def contains(self, obj):
        return isinstance(obj, str)

    def __repr__(self):
        return "String"


def create_random_space() -> spaces.Space:
    "Create a random space that might be nested. Mostly for testing."

    choice = np.random.randint(4)
    if choice == 0:
        dim = np.random.randint(5) + 1
        return spaces.Box(
            low=-5 * np.ones(dim, dtype=np.float32),
            high=5 * np.ones(dim, dtype=np.float32),
        )
    elif choice == 1:
        return spaces.Dict({"a": create_random_space(), "b": create_random_space()})
    elif choice == 2:
        return spaces.Discrete(np.random.randint(5) + 2)
    elif choice == 3:
        return spaces.MultiDiscrete(
            [np.random.randint(5) + 2 for _ in range(np.random.randint(10) + 1)]
        )
    else:
        raise NotImplementedError()


def sample(space, n, device):
    if isinstance(space, spaces.Box):
        low, high = map(torch.tensor, [space.low, space.high])
        s = torch.rand((n, len(low))) * (high - low) + low
        return s.to(device)
    elif isinstance(space, spaces.Discrete):
        s = torch.randint(space.n, (n, 1))
        return s.to(device)
    elif isinstance(space, spaces.MultiDiscrete):
        return sample(spaces.Tuple([spaces.Discrete(n) for n in space.nvec]), n, device)
    elif isinstance(space, spaces.Tuple):
        return torch.cat([sample(s, n, device) for s in space.spaces], dim=1)
    elif isinstance(space, spaces.Dict):
        return {k: sample(s, n, device) for k, s in space.spaces.items()}
    else:
        raise NotImplementedError


def unif_log_prob(space):
    if isinstance(space, spaces.Box):
        return -sum(np.log(space.high - space.low))
    elif isinstance(space, spaces.Discrete):
        return -np.log(space.n)
    elif isinstance(space, spaces.MultiDiscrete):
        return -sum(np.log(space.nvec))
    elif isinstance(space, spaces.Tuple):
        return sum([unif_log_prob(s) for s in space.spaces])
    elif isinstance(space, spaces.Dict):
        return sum([unif_log_prob(s) for s in space.spaces.values()])
    else:
        raise NotImplementedError


def replace_space(space, inp_space, out_space):
    """Replaces every occurence of `inp_space` with `out_space` if
    `out_space` is not None.

    """

    if isinstance(space, inp_space):
        return out_space
    elif isinstance(space, spaces.Dict):
        return spaces.Dict(
            {
                k: replace_space(v, inp_space, out_space)
                for k, v in space.spaces.items()
                if replace_space(v, inp_space, out_space) is not None
            }
        )
    else:
        return space


def apply_space(inp, inp_space, out_space, fn):
    if inp_space.__class__ != out_space.__class__:
        return fn(inp)
    elif isinstance(inp_space, spaces.Dict):
        return {
            k: apply_space(inp[k], inp_space[k], out_space[k], fn)
            for k in inp_space.spaces
        }
    else:
        return inp
