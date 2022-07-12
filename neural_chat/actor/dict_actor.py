from typing import List, Optional
from neural_chat.actor import BaseActor
from neural_chat.logger import simpleloggable
from neural_chat.transforms import OneHotFlatten, Flatten, Unflatten
from neural_chat.utils import MLP
from torch import nn
from gym import Space
from .handlers import BoxHandler, DiscreteHandler, MultiDiscreteHandler, MultiHandler


HANDLERS = {
    "Box": BoxHandler,
    "Discrete": DiscreteHandler,
    "MultiDiscrete": MultiDiscreteHandler,
}


@simpleloggable
class DictActor(BaseActor):
    def __init__(
        self,
        obs_spec: Space,
        act_spec: Space,
        hidden_dim: List[int],
        handlers: dict = HANDLERS,
        clip_log_prob: Optional[float] = None,
    ) -> None:
        BaseActor.__init__(self)
        nn.Module.__init__(self)

        self.obs_flat = OneHotFlatten(obs_spec)
        self.act_flat = Flatten(act_spec)
        self.act_unflat = Unflatten(act_spec)

        self.handler = MultiHandler(
            [
                handlers[space.__class__.__name__](space)
                for space in self.act_flat.after_space.spaces
            ]
        )

        self.policy = MLP(
            self.obs_flat.after_dim, hidden_dim, int(self.handler.input_size)
        )
        self.clip_log_prob = clip_log_prob

    def log_prob(self, obs, act):
        inp = self.policy(self.obs_flat(obs))
        return self.handler.log_prob(inp, self.act_flat(act))

    def action_with_log_prob(self, obs, deterministic=False):
        inp = self.policy(self.obs_flat(obs))
        self.log("raw_outputs", inp)
        act = self.handler.sample(inp, deterministic)
        log_prob = self.handler.log_prob(inp, act)
        if self.clip_log_prob is not None:
            log_prob = log_prob.clamp_min(self.clip_log_prob)
        return self.act_unflat(act), log_prob
