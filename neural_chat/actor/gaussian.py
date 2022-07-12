from typing import List, Tuple
from torch import nn, distributions as pyd
from gym import Space
from neural_chat.utils import MLP
from neural_chat.transforms import Flatten, Unflatten, TanhScaleTransform
from neural_chat.logger import simpleloggable
from .base import BaseActor


@simpleloggable
class GaussianActor(BaseActor):
    def __init__(
        self,
        obs_spec: Space,
        act_spec: Space,
        hidden_dim: List[int],
        _log_std_bounds: Tuple[float, float] = (-5, 2),
    ) -> None:
        BaseActor.__init__(self)
        nn.Module.__init__(self)

        self.obs_flat = Flatten(obs_spec)
        self.act_flat = Flatten(act_spec)
        self.act_unflat = Unflatten(act_spec)

        self.log_std_bounds = _log_std_bounds
        self.policy = MLP(
            self.obs_flat.after_dim, hidden_dim, self.act_flat.after_dim * 2
        )

    def _get_dist(self, obs):
        # get mu and log_std
        mu, log_std = self.policy(self.obs_flat(obs)).chunk(2, dim=-1)
        self.log("mu", mu)
        self.log("log_std_before", log_std)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = TanhScaleTransform(*self.log_std_bounds)(log_std)
        self.log("log_std_after", log_std)

        # make normal distribution
        dist = pyd.Normal(mu, log_std.exp())
        return dist

    def log_prob(self, obs, act):
        dist = self._get_dist(obs)
        return dist.log_prob(self.act_flat(act)).sum(-1, keepdim=True)

    def action_with_log_prob(self, obs, deterministic=False):
        dist = self._get_dist(obs)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        self.log("action", action)
        self.log("log_prob", log_prob)

        return self.act_unflat(action), log_prob


class TanhGaussianActor(GaussianActor):
    "Works well with MuJoCo environments"

    def log_prob(self, obs, act):
        # hack to prevent computing atanh(1.)
        act = self.act_flat(act)
        mask = (act > 0.999) + (act < 0.999)
        mask = 0.999 * mask.float()
        act = self.act_unflat((act * mask).atanh())

        return super().log_prob(obs, act)

    def action_with_log_prob(self, obs, deterministic=False):
        act, log_prob = super().action_with_log_prob(obs, deterministic)
        act = self.act_flat(act).tanh()
        self.log("action_tanh", act)
        return self.act_unflat(act), log_prob
