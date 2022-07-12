from .base import make_test_single_space, make_test_multi_space
from neural_chat.transforms import OneHotFlatten, UnOneHotUnflatten
from torch.nn import Sequential


m = lambda space: Sequential(OneHotFlatten(space), UnOneHotUnflatten(space))
test_single_space = make_test_single_space(m)
test_multi_space = make_test_multi_space(m)
