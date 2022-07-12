from typing import List
import torch
from torch import nn
from gym import Space
from neural_chat.utils import MLP
from neural_chat.transforms import OneHotFlatten
from neural_chat.logger import simpleloggable


@simpleloggable
class DoubleVCritic(nn.Module):
    """Value network, employes double Q-learning."""

    def __init__(self, obs_spec: Space, hidden_dim: List[int]) -> None:
        nn.Module.__init__(self)

        self.obs_flat = OneHotFlatten(obs_spec)

        self.vf_1 = MLP(self.obs_flat.after_dim, hidden_dim, 1)
        self.vf_2 = MLP(self.obs_flat.after_dim, hidden_dim, 1)

    def double_v(self, obs):
        obs = self.obs_flat(obs)

        v1, v2 = self.vf_1(obs), self.vf_2(obs)
        self.log("val_1", v1)
        self.log("val_2", v2)
        return v1, v2

    def forward(self, obs):
        return torch.min(*self.double_v(obs))
