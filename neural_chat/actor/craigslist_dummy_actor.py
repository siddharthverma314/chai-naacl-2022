from typing import Dict
from neural_chat.logger.loggable import simpleloggable
from .base import BaseActor
from .craigslist_emaq_actor import QHandler
import torch


@simpleloggable
class CraigslistDummyActor(BaseActor):
    def __init__(
        self,
        q_fn: torch.nn.Module,
        price_low: float = 0,
        price_high: float = 3,
    ):
        super().__init__()
        torch.nn.Module.__init__(self)

        # HACK: include one parameter so that SAC is happy
        self._dont_use = torch.nn.Linear(1, 1)

        # make handler
        self.q_handler = QHandler(q_fn)
        self.price_low = price_low
        self.price_high = price_high

        # num prices per action
        self.prices_per_action = 1

    def _action_candidates(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "utterance": actions.flatten(start_dim=0, end_dim=1),
            "price": torch.clamp(
                obs["seller_price"]
                - (torch.rand(obs["seller_price"].shape) * 0.3).to(
                    obs["seller_price"].device
                ),
                self.price_low,
                self.price_high,
            ),
            "action_type": action_types.flatten().unsqueeze(1),
        }

    def action_with_log_prob(self, obs, deterministic=False):
        # separate out observations
        obs = obs.copy()
        actions = obs.pop("actions")
        action_types = obs.pop("action_types")

        # number of actions will get used to create the observation
        num_actions: int = actions.shape[1] * self.prices_per_action

        # repeat observation num_actions times
        obs = {k: v.repeat_interleave(num_actions, dim=0) for k, v in obs.items()}

        # get price and action type from the model
        act_candidates = self._action_candidates(obs, actions, action_types)

        # log stuff
        self.log("price", act_candidates["price"])

        # get utterance
        indices, log_prob_utterance = self.q_handler.sample_with_log_prob(
            obs,
            act_candidates,
            num_actions,
            deterministic,
        )

        return {
            k: self.q_handler.index_tensor(v, indices, num_actions)
            for k, v in act_candidates.items()
        }, log_prob_utterance

    def log_prob(self, *_, **__):
        raise NotImplementedError
