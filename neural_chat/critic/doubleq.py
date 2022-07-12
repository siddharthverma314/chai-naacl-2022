from typing import List
import torch
from torch import nn
from gym import Space
from neural_chat.utils import MLP
from neural_chat.transforms import OneHotFlatten
from neural_chat.logger import simpleloggable


@simpleloggable
class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(
        self, obs_spec: Space, action_spec: Space, hidden_dim: List[int]
    ) -> None:
        nn.Module.__init__(self)

        self.obs_flat = OneHotFlatten(obs_spec)
        self.act_flat = OneHotFlatten(action_spec)

        self.qf_1 = MLP(
            self.obs_flat.after_dim + self.act_flat.after_dim, hidden_dim, 1
        )
        self.qf_2 = MLP(
            self.obs_flat.after_dim + self.act_flat.after_dim, hidden_dim, 1
        )

    def double_q(self, obs, action):
        obs_action = torch.cat([self.obs_flat(obs), self.act_flat(action)], dim=-1)
        q1, q2 = self.qf_1(obs_action), self.qf_2(obs_action)
        self.log("qval_1", q1)
        self.log("qval_2", q2)
        return q1, q2

    def forward(self, obs, action):
        return torch.min(*self.double_q(obs, action))
