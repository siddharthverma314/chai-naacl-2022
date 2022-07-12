from neural_chat.logger.loggable import simpleloggable
import torch
import torch.nn.functional as F
import torch.distributions as pyd
from .sac import SAC


@simpleloggable
class BRAC(SAC):
    def __init__(
        self,
        _price_loss_weight: float = 1,
        _price_clamp_min: float = -10,
        _price_distribution=None,
        penalty_type="pr",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if _price_distribution is None:
            self._price_distribution = lambda x, _: pyd.Normal(0.8137537366214004, 1)
        elif _price_distribution == "adaptive":
            self._price_distribution = lambda x, _: pyd.Normal(
                0.69082594 + x * 0.20665981, 1
            )
        elif _price_distribution == "adaptive_history":
            self._price_distribution = lambda buyer, prev_seller: pyd.Normal(
                0.27577925 + 0.06040737 * buyer + 0.58641446 * prev_seller, 1
            )
        else:
            raise ValueError()
        self.penalty_type = penalty_type
        self._price_loss_weight = _price_loss_weight
        self._price_clamp_min = _price_clamp_min

    def update_actor_and_alpha(self, obs, cur_choices, **_):
        obs = {**obs, "actions": cur_choices}

        action, log_prob = self.actor.action_with_log_prob(obs)
        actor_Q = self.critic.forward(obs, action)

        # 1. "unbias" the Q values by subtracting the mean
        actor_Q -= actor_Q.min().detach()

        partner_price = obs["buyer_price"].detach()
        prev_seller_price = obs["seller_price"].detach()

        # 2. Compute price penalty
        if self.penalty_type == "pr":
            policy_penalty = (
                self._price_distribution(partner_price, prev_seller_price)
                .log_prob(action["price"])
                .clamp_min(self._price_clamp_min)
            ) * self._price_loss_weight
        else:
            policy_penalty = 0.0

        # 3. compute the raw actor loss
        raw_actor_loss = self.alpha.detach() * log_prob - actor_Q

        actor_loss = (raw_actor_loss - policy_penalty).mean()

        self.log("actor/loss", actor_loss)
        self.log("actor/raw_actor_loss", raw_actor_loss.mean())
        self.log("actor/entropy", log_prob)

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

    def update_critic(self, obs, act, rew, next_obs, done):
        partner_price = obs["buyer_price"].detach()
        prev_seller_price = obs["seller_price"].detach()
        if self.penalty_type == "vp":
            value_penalty = (
                self._price_distribution(partner_price, prev_seller_price)
                .log_prob(act["price"])
                .clamp_min(self._price_clamp_min)
            ) * self._price_loss_weight
        else:
            value_penalty = 0

        with torch.no_grad():
            # compute next values
            next_act, next_log_prob = self.actor.action_with_log_prob(next_obs)
            next_Q = self.critic_target.forward(next_obs, next_act)

            # compute target Q
            next_V = next_Q - self.alpha * next_log_prob
            tar_Q = rew + (1.0 - done) * self.discount * next_V - value_penalty
            tar_Q = tar_Q.detach()

        # compute critic loss
        cur_Q1, cur_Q2 = self.critic.double_q(obs, act)
        critic_loss = 0.5 * F.mse_loss(cur_Q1, tar_Q) + 0.5 * F.mse_loss(cur_Q2, tar_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # logging
        self.log("critic/loss", critic_loss)
        self.log("critic/target_q", tar_Q)
        self.log("critic/q1", cur_Q1)
        self.log("critic/q2", cur_Q2)
