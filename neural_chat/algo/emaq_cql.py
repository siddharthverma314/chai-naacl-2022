from neural_chat.logger.loggable import simpleloggable
import torch
import torch.nn.functional as F
import gym.spaces as sp
from neural_chat.utils import collate
from neural_chat.utils import sample, unif_log_prob
from .emaq import EMAQ


@simpleloggable
class EMAQ_CQL(EMAQ):
    def __init__(
        self,
        action_space: sp.Dict,
        num_logexp_samples: int = 4,
        temperature: float = 1.0,
        cql_minq_version=2,
        cql_minq_weight=1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.cql_action_space = sp.Dict(dict(action_space.spaces))
        self.num_logexp_samples = num_logexp_samples
        self.temperature = temperature
        self.minq_version = cql_minq_version
        self.minq_weight = cql_minq_weight

    def update_actor_and_alpha(self, obs, cur_choices, **_):
        obs = {**obs, "actions": cur_choices}

        action, log_prob = self.actor.action_with_log_prob(obs)
        actor_Q = self.critic.forward(obs, action)

        # 1. "unbias" the Q values by subtracting the mean
        actor_Q -= actor_Q.min().detach()

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.log("actor/loss", actor_loss)
        self.log("actor/entropy", log_prob)
        self.log("pred_actions", action)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            alpha_loss = -(
                self.alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            self.log("alpha/loss", alpha_loss)
            self.log("alpha/value", self.alpha)

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def logsumexp(self, obs, cur_choices, next_obs, next_choices, q1_pred, q2_pred):
        batch_size = len(obs[next(iter(obs.keys()))])
        obs["actions"] = cur_choices
        next_obs["actions"] = next_choices

        # collect actions
        cat_obs = collate([obs for _ in range(self.num_logexp_samples)])
        cat_next_obs = collate([next_obs for _ in range(self.num_logexp_samples)])
        pol_a, pol_a_lp = self.actor.action_with_log_prob(cat_obs)
        pol_na, pol_na_lp = self.actor.action_with_log_prob(cat_next_obs)
        rand_a = sample(
            self.cql_action_space, batch_size * self.num_logexp_samples, self.device
        )
        rand_lp = unif_log_prob(self.cql_action_space)
        # price is none

        # collect all q values
        q1_r, q2_r = self.critic.double_q(cat_obs, rand_a)
        q1_a, q2_a = self.critic.double_q(cat_obs, pol_a)
        q1_na, q2_na = self.critic.double_q(cat_obs, pol_na)

        # reshape all values
        def reshape(tens):
            N, _ = tens.shape
            assert N % self.num_logexp_samples == 0
            return tens.view([N // self.num_logexp_samples, self.num_logexp_samples, 1])

        q1_r = reshape(q1_r)
        q2_r = reshape(q2_r)
        q1_a = reshape(q1_a)
        q2_a = reshape(q2_a)
        q1_na = reshape(q1_na)
        q2_na = reshape(q2_na)
        q1_pred = q1_pred.unsqueeze(1)
        q2_pred = q2_pred.unsqueeze(1)

        # concatenations
        if self.minq_version == 3:
            cat_q1 = torch.cat(
                [q1_r - rand_lp, q1_a - pol_a_lp.detach(), q1_na - pol_na_lp.detach()],
                dim=1,
            )
            cat_q2 = torch.cat(
                [q2_r - rand_lp, q2_a - pol_a_lp.detach(), q2_na - pol_na_lp.detach()],
                dim=1,
            )
        else:
            cat_q1 = torch.cat([q1_r, q1_pred, q1_a, q1_na], dim=1)
            cat_q2 = torch.cat([q2_r, q2_pred, q2_a, q2_na], dim=1)

        # logsumexp
        return map(
            lambda q: torch.logsumexp(q / self.temperature, dim=1, keepdim=True)
            * self.temperature,
            [cat_q1, cat_q2],
        )

    def update_critic(
        self, obs, act, rew, next_obs, done, cur_choices, next_choices, **_
    ):
        next_obs = {**next_obs, "actions": next_choices}
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
        cql_Q1, cql_Q2 = self.logsumexp(
            obs, cur_choices, next_obs, next_choices, cur_Q1, cur_Q2
        )
        self.log("cql_q1", cql_Q1)
        self.log("cql_q2", cql_Q2)

        cql_q1_loss = cql_Q1.mean() * self.minq_weight
        cql_q2_loss = cql_Q2.mean() * self.minq_weight

        loss_Q1 = F.mse_loss(cur_Q1, tar_Q) + cql_q1_loss
        loss_Q2 = F.mse_loss(cur_Q2, tar_Q) + cql_q2_loss
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
