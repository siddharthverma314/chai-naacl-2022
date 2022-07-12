from typing import Optional, Tuple, Dict
from neural_chat.logger.loggable import simpleloggable
from torch import nn
from neural_chat.actor import BaseActor
from neural_chat.transforms import TanhScaleTransform
import gym.spaces as sp
from .dict_actor import DictActor
import torch.distributions as pyd
import torch


class QHandler:
    def __init__(self, q_fn: nn.Module):
        super().__init__()
        self.q_fn = q_fn

    def _get_dist(
        self,
        obs: Dict[str, torch.Tensor],
        act: Dict[str, torch.Tensor],
        num_actions: int,
    ) -> pyd.Categorical:
        qs = self.q_fn(obs, act)
        qs = qs.reshape(qs.shape[0] // num_actions, num_actions)
        return pyd.Categorical(logits=qs.float())

    def sample_with_log_prob(
        self,
        obs: Dict[str, torch.Tensor],
        act: Dict[str, torch.Tensor],
        num_actions: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._get_dist(obs, act, num_actions)
        if deterministic:
            indices = dist.logits.argmax(dim=-1)
        else:
            indices = dist.sample()
        log_prob = dist.log_prob(indices)
        return indices, log_prob.unsqueeze(1)

    @staticmethod
    def index_tensor(
        v: torch.Tensor, indices: torch.Tensor, num_actions: int
    ) -> torch.Tensor:
        return v.reshape(v.shape[0] // num_actions, num_actions, *v.shape[1:])[
            torch.arange(len(indices)), indices
        ]

    def log_prob(self, obs, act, utterances, sample: torch.Tensor):
        return self._get_dist(obs, act, utterances).log_prob(sample).unsqueeze(1)


@simpleloggable
class CraigslistEMAQActor(BaseActor):
    def __init__(
        self,
        obs_spec: Dict,
        act_spec: Dict,
        q_fn: torch.nn.Module,
        price_low: float = 0,
        price_high: float = 3,
        delta_low: float = -3,
        delta_high: float = 3,
        **kwargs,
    ):
        super().__init__()
        torch.nn.Module.__init__(self)

        # make handler
        self.q_handler = QHandler(q_fn)

        # model
        self.model = DictActor(
            # add utterance to observation space
            sp.Dict({**obs_spec.spaces, "action": obs_spec["utterance"]}),
            # remove utterance from action space and add price gate
            sp.Dict(
                {
                    **{k: v for k, v in act_spec.spaces.items() if k != "utterance"},
                    "price_gate": act_spec.spaces["price"],
                }
            ),
            **kwargs,
        )

        # price transform
        self.delta_trans = TanhScaleTransform(delta_low, delta_high)
        self.price_trans = TanhScaleTransform(price_low, price_high)

    def action_with_log_prob(self, obs, deterministic=False):
        # separate out observations
        obs = obs.copy()
        actions = obs["actions"]
        del obs["actions"]

        # number of actions will get used to create the observation
        num_actions: int = actions.shape[1]

        # repeat observation num_actions times
        obs = {k: v.repeat_interleave(num_actions, dim=0) for k, v in obs.items()}

        # get price and action type from the model
        act_other, log_prob_other = self.model.action_with_log_prob(
            {
                **obs,
                "action": actions.flatten(start_dim=0, end_dim=1),
            },
            deterministic,
        )

        # price calculation
        delta = act_other["price"]
        gate = act_other["price_gate"]
        seller_pr = obs["seller_price"]
        del act_other["price_gate"]
        act_other["price"] = seller_pr + torch.sigmoid(gate) * self.delta_trans(delta)

        # fix action price and utterance
        act_other["price"] = self.price_trans(act_other["price"])
        act_other["utterance"] = actions.flatten(start_dim=0, end_dim=1)

        # log stuff
        self.log("delta", delta)
        self.log("gate", gate)
        self.log("price", act_other["price"])

        # get utterance
        indices, log_prob_utterance = self.q_handler.sample_with_log_prob(
            obs,
            act_other,
            num_actions,
            deterministic,
        )

        act = {
            k: self.q_handler.index_tensor(v, indices, num_actions)
            for k, v in act_other.items()
        }
        log_prob = self.q_handler.index_tensor(log_prob_other, indices, num_actions)

        return act, log_prob_utterance + log_prob

    def log_prob(self, *_, **__):
        raise NotImplementedError
