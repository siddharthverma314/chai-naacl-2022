from neural_chat.logger.loggable import simpleloggable
import torch
import torch.nn.functional as F
import torch.distributions as pyd
from .sac import SAC


@simpleloggable
class EMAQ(SAC):
    def __init__(
        self,
        _price_loss_weight: float = 1,
        _price_clamp_min: float = -10,
        _price_distribution=None,
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

        # 2. Compute price loss and scale with actor_Q magnitude
        price_loss = (
            (
                self._price_distribution(partner_price, prev_seller_price)
                .log_prob(action["price"])
                .clamp_min(self._price_clamp_min)
            )
            * actor_Q.mean().detach()
            * self._price_loss_weight
        )

        # 3. compute the raw actor loss
        raw_actor_loss = self.alpha.detach() * log_prob - actor_Q

        # 4. subtract the price loss and profit
        actor_loss = (raw_actor_loss - price_loss).mean()

        self.log("actor/loss", actor_loss)
        self.log("actor/price_loss", price_loss.mean())
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

    def update_critic(self, obs, act, rew, next_obs, done, next_choices, **_):
        return super().update_critic(
            obs, act, rew, {**next_obs, "actions": next_choices}, done
        )
