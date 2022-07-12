from neural_chat.logger.loggable import simpleloggable
import torch
import torch.nn.functional as F
import gym.spaces as sp
from neural_chat.utils import collate
from neural_chat.utils import sample, unif_log_prob
from neural_chat.actor import CraigslistDummyActor
from neural_chat.critic import DoubleQCritic
from .sac import SAC


@simpleloggable
class EMAQ_NOA(SAC):
    def __init__(
        self,
        actor: CraigslistDummyActor,
        critic: DoubleQCritic,
        action_space: sp.Dict,
        num_logexp_samples: int = 4,
        temperature: float = 1.0,
        cql_minq_version=2,
        cql_minq_weight=0.0,
        **kwargs,
    ):
        super().__init__(actor=actor, critic=critic, **kwargs)
        self.actor = actor
        self.critic = critic
        self.cql_action_space = sp.Dict(dict(action_space.spaces))
        self.num_logexp_samples = num_logexp_samples
        self.temperature = temperature
        self.minq_version = cql_minq_version
        self.minq_weight = cql_minq_weight

    def logsumexp(
        self,
        obs,
        cur_choices,
        cur_choices_type,
        next_obs,
        next_choices,
        next_choices_type,
        q1_pred,
        q2_pred,
    ):
        batch_size = len(obs[next(iter(obs.keys()))])
        obs = {
            **next_obs,
            "actions": cur_choices,
            "action_types": cur_choices_type,
        }
        next_obs = {
            **next_obs,
            "actions": next_choices,
            "action_types": next_choices_type,
        }

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

    def update_actor_and_alpha(*args, **kwargs):
        # we don't need to do anything here
        pass

    def update_critic(
        self,
        obs,
        act,
        rew,
        next_obs,
        done,
        cur_choices,
        cur_choices_type,
        next_choices,
        next_choices_type,
        **_,
    ):
        next_obs = {
            **next_obs,
            "actions": next_choices,
            "action_types": next_choices_type,
        }
        with torch.no_grad():
            # compute next values
            next_act = self.actor.action(next_obs)
            next_Q = self.critic_target.forward(next_obs, next_act)

            # compute target Q
            next_V = next_Q
            tar_Q = rew + (1.0 - done) * self.discount * next_V
            tar_Q = tar_Q.detach()

        # compute critic loss
        cur_Q1, cur_Q2 = self.critic.double_q(obs, act)
        if self.minq_weight > 0:
            cql_Q1, cql_Q2 = self.logsumexp(
                obs,
                cur_choices,
                cur_choices_type,
                next_obs,
                next_choices,
                next_choices_type,
                cur_Q1,
                cur_Q2,
            )
            self.log("cql_q1", cql_Q1)
            self.log("cql_q2", cql_Q2)

            cql_q1_loss = cql_Q1.mean() * self.minq_weight
            cql_q2_loss = cql_Q2.mean() * self.minq_weight
        else:
            cql_q1_loss = cql_q2_loss = 0

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
