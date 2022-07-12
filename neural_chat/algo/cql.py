import torch
import gym

from neural_chat.logger import simpleloggable
from neural_chat.utils import sample, unif_log_prob
from neural_chat.transforms import Flatten
from neural_chat.utils import collate
from .sac import SAC
from torch.functional import F


@simpleloggable
class CQL(SAC):
    def __init__(
        self,
        action_space: gym.Space,
        _num_logexp_samples: int = 4,
        _uniform_scale: float = 0.1,
        _temperature: float = 1,
        *args,
        **kwargs
    ) -> None:

        super().__init__(*args, _act_dim=Flatten(action_space).dim, **kwargs)
        self.num_logexp_samples = _num_logexp_samples
        self.action_space = action_space
        self.temperature = _temperature
        self.uniform_scale = _uniform_scale

    def logsumexp(self, obs, next_obs):
        batch_size = len(obs[next(iter(obs.keys()))])

        # collect actions
        cat_obs = collate([obs for _ in range(self.num_logexp_samples)])
        cat_next_obs = collate([next_obs for _ in range(self.num_logexp_samples)])
        pol_a, pol_a_lp = self.actor.action_with_log_prob(cat_obs)
        pol_na, pol_na_lp = self.actor.action_with_log_prob(cat_next_obs)
        rand_a = sample(
            self.action_space, batch_size * self.num_logexp_samples, self.device
        )
        rand_lp = unif_log_prob(self.action_space)

        # collect all q values
        q1_r, q2_r = self.critic.double_q(cat_obs, rand_a)
        q1_a, q2_a = self.critic.double_q(cat_obs, pol_a)
        q1_na, q2_na = self.critic.double_q(cat_obs, pol_na)

        # concatenations
        cat_q1 = torch.cat(
            [q1_r - rand_lp, q1_a - pol_a_lp.detach(), q1_na - pol_na_lp.detach()],
            dim=1,
        )
        cat_q2 = torch.cat(
            [q2_r - rand_lp, q2_a - pol_a_lp.detach(), q2_na - pol_na_lp.detach()],
            dim=1,
        )

        # logsumexp
        return map(
            lambda q: torch.logsumexp(q / self.temperature, dim=1, keepdim=True),
            [cat_q1, cat_q2],
        )

    def update_critic(self, obs, act, rew, next_obs, done):
        with torch.no_grad():
            # compute next values
            next_act, next_log_prob = self.actor.action_with_log_prob(next_obs)
            next_Q = self.critic_target.forward(next_obs, next_act)

            # compute target Q
            next_V = next_Q - self.alpha * next_log_prob
            tar_Q = rew + (1.0 - done) * self.discount * next_V
            tar_Q = tar_Q.detach()

        # compute critic loss
        cur_Q1, cur_Q2 = self.critic.double_q(obs, act)
        cql_Q1, cql_Q2 = self.logsumexp(obs, next_obs)
        self.log("cql_q1", cql_Q1)
        self.log("cql_q2", cql_Q2)

        loss_Q1 = (
            F.mse_loss(cur_Q1, tar_Q)
            + ((cql_Q1 - cur_Q1.mean()) / self.temperature).mean()
        )
        loss_Q2 = (
            F.mse_loss(cur_Q2, tar_Q)
            + ((cql_Q2 - cur_Q2.mean()) / self.temperature).mean()
        )
        critic_loss = 0.5 * loss_Q1 + 0.5 * loss_Q2

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # logging
        self.log("critic/loss", critic_loss)
        self.log("critic/target_q", tar_Q)
        self.log("critic/q1", cur_Q1)
        self.log("critic/q2", cur_Q2)
