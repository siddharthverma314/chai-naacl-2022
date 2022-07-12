from typing import Tuple
from torch.nn import Module
import abc
import torch


class BaseActor(Module, metaclass=abc.ABCMeta):
    """Base class for all actors"""

    @abc.abstractmethod
    def action_with_log_prob(
        self, obs, deterministic=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        "Return either a stochastic or fixed action along with log_prob"

    @abc.abstractmethod
    def log_prob(self, obs, act) -> torch.Tensor:
        "Return the log prob of a given observation and action"

    def action(self, obs, deterministic=False) -> torch.Tensor:
        return self.action_with_log_prob(obs, deterministic)[0]

    def forward(self, obs) -> torch.Tensor:
        return self.action(obs, deterministic=False)
