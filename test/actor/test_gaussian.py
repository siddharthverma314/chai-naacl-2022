from neural_chat.actor import GaussianActor, TanhGaussianActor
from neural_chat.utils import create_random_space, torchify
from flatten_dict import flatten
from gym.spaces import Box
import torch
import numpy as np


def test_integration():
    obs_spec = Box(
        low=np.zeros(10, dtype=np.float32), high=np.ones(10, dtype=np.float32)
    )
    act_spec = Box(low=np.zeros(3, dtype=np.float32), high=np.ones(3, dtype=np.float32))
    actor = GaussianActor(obs_spec, act_spec, [60, 50])
    obs = torch.rand((100, 10))
    actions = actor.action(obs)
    assert actions.shape == (100, 3)


def test_random_space():
    for _ in range(100):
        obs_spec = create_random_space()
        act_spec = create_random_space()
        print(obs_spec)
        print(act_spec)
        actor = GaussianActor(obs_spec, act_spec, [60, 50])
        actor.action(torchify(obs_spec.sample()))


def test_log():
    obs_spec = Box(
        low=np.zeros(10, dtype=np.float32), high=np.ones(10, dtype=np.float32)
    )
    act_spec = Box(low=np.zeros(3, dtype=np.float32), high=np.ones(3, dtype=np.float32))
    actor = GaussianActor(obs_spec, act_spec, [60, 50])
    obs = torch.rand((100, 10))
    actor.action(obs)
    print(flatten(actor.log_hyperparams()).keys())
    print(flatten(actor.log_epoch()).keys())


def test_integration_tanh():
    obs_spec = Box(
        low=np.zeros(10, dtype=np.float32), high=np.ones(10, dtype=np.float32)
    )
    act_spec = Box(low=np.zeros(3, dtype=np.float32), high=np.ones(3, dtype=np.float32))
    actor = TanhGaussianActor(obs_spec, act_spec, [60, 50])
    obs = torch.rand((100, 10))
    actions = actor.action(obs)
    assert actions.shape == (100, 3)


def test_random_space_tanh():
    for _ in range(100):
        obs_spec = create_random_space()
        act_spec = create_random_space()
        print(obs_spec)
        print(act_spec)
        actor = TanhGaussianActor(obs_spec, act_spec, [60, 50])
        actor.action(torchify(obs_spec.sample()))


def test_log_tanh():
    obs_spec = Box(
        low=np.zeros(10, dtype=np.float32), high=np.ones(10, dtype=np.float32)
    )
    act_spec = Box(low=np.zeros(3, dtype=np.float32), high=np.ones(3, dtype=np.float32))
    actor = TanhGaussianActor(obs_spec, act_spec, [60, 50])
    obs = torch.rand((100, 10))
    actor.action(obs)
    print(flatten(actor.log_hyperparams()).keys())
    print(flatten(actor.log_epoch()).keys())
