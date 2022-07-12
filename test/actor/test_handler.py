from neural_chat.actor.handlers import DiscreteHandler, MultiDiscreteHandler, BoxHandler
from gym.spaces import Discrete, MultiDiscrete, Box
import torch
import numpy as np


def do_test(handler, inp_size=100):
    inp = torch.rand((100, handler.input_size))

    sample = handler.sample(inp)
    prob = handler.log_prob(inp, sample)
    d_sample = handler.sample(inp, deterministic=True)

    assert sample.shape == (100, handler.output_size)
    assert d_sample.shape == (100, handler.output_size)
    assert prob.shape == (100, 1)


def test_discrete_handler():
    for _ in range(100):
        n = np.random.randint(1, 10)
        handler = DiscreteHandler(Discrete(n))
        do_test(handler, 100)
        do_test(handler, 1)


def test_multi_discrete_handler():
    for _ in range(100):
        n = np.random.randint(1, 10)
        handler = MultiDiscreteHandler(
            MultiDiscrete([np.random.randint(1, 10) for _ in range(n)])
        )
        do_test(handler, 100)
        do_test(handler, 1)


def test_box_handler():
    for _ in range(100):
        n = np.random.randint(1, 10)
        handler = BoxHandler(
            Box(np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32))
        )
        do_test(handler, 100)
        do_test(handler, 1)
