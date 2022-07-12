from neural_chat.critic import DoubleQCritic, DoubleVCritic
from neural_chat.utils import create_random_space, torchify
from gym.spaces import Box
import numpy as np
import torch


def test_integration():
    obs_spec = Box(
        low=np.zeros(10, dtype=np.float32), high=np.ones(10, dtype=np.float32)
    )
    act_spec = Box(low=np.zeros(3, dtype=np.float32), high=np.ones(3, dtype=np.float32))
    c1 = DoubleQCritic(obs_spec, act_spec, [60, 50])
    c2 = DoubleVCritic(obs_spec, [60, 50])
    obs = torch.rand((100, 10))
    act = torch.rand((100, 3))
    q = c1.forward(obs, act)
    assert q.shape == (100, 1)
    v = c2.forward(obs)
    assert v.shape == (100, 1)


def test_random_space():
    for _ in range(100):
        obs_spec = create_random_space()
        act_spec = create_random_space()
        print(obs_spec)
        print(act_spec)
        c1 = DoubleQCritic(obs_spec, act_spec, [60, 50])
        c2 = DoubleVCritic(obs_spec, [60, 50])
        obs = torchify(obs_spec.sample())
        act = torchify(act_spec.sample())
        c1.forward(obs, act)
        c2.forward(obs)
