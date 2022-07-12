from .base import make_test_single_space, make_test_multi_space
from neural_chat.transforms import Flatten, Unflatten
from neural_chat.utils import create_random_space
from torch.nn import Sequential
from gym.spaces import Dict, Tuple


m = lambda space: Sequential(Flatten(space), Unflatten(space))
test_single_space = make_test_single_space(m)
test_multi_space = make_test_multi_space(m)


def test_flatten_space():
    for _ in range(100):
        space = create_random_space()
        print("SPACE", space)
        f = Flatten(space)
        assert not isinstance(f.after_space, Dict)
        if isinstance(f.after_space, Tuple):
            for space in f.after_space.spaces:
                assert not isinstance(space, Tuple)
                assert not isinstance(space, Dict)
