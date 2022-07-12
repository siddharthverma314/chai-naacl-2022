import numpy as np
import torch
import torch.nn.functional as F
import copy

from neural_chat.actor import GaussianActor
from neural_chat.critic import DoubleQCritic
from neural_chat.logger import simpleloggable


@simpleloggable
class SAC:
    def __init__(
        self,
        actor: GaussianActor,
        critic: DoubleQCritic,
        _device: str,
        _target_entropy: float = 0.1,
        _critic_tau: float = 5e-3,
        _discount: float = 0.99,
        _init_temperature: float = 0.1,
        _learnable_temperature: bool = True,
        _actor_update_frequency: int = 1,
        _critic_target_update_frequency: int = 1,
        _optimizer_type: str = "adam",
        _alpha_lr: float = 3e-4,
        _actor_lr: float = 3e-4,
        _actor_weight_decay: float = 1e-2,
        _critic_lr: float = 3e-4,
    ) -> None:

        super().__init__()

        # set other parameters
        self.device = torch.device(_device)
        self.discount = _discount
        self.critic_tau = _critic_tau
        self.actor_update_frequency = _actor_update_frequency
        self.critic_target_update_frequency = _critic_target_update_frequency
        self.learnable_temperature = _learnable_temperature

        # instantiate actor and critic
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(critic)
        self.critic_target.eval()

        # instantiate log alpha
        self.log_alpha = torch.tensor(np.log(_init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = _target_entropy

        # optimizers
        if _optimizer_type == "adam":
            optimizer = torch.optim.Adam
        elif _optimizer_type == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError()

        self.actor_optimizer = optimizer(
            self.actor.parameters(), lr=_actor_lr, weight_decay=_actor_weight_decay
        )
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=_critic_lr)
        self.log_alpha_optimizer = optimizer([self.log_alpha], lr=_alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().to(self.device)

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

    def update_actor_and_alpha(self, obs, **kwargs):
        action, log_prob = self.actor.action_with_log_prob(obs)
        actor_Q = self.critic.forward(obs, action)
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

    def update(self, batch, step):
        "Done should not include the last step. Basically, infinite horizon task"
        self.log("batch_reward", batch["rew"])

        self.update_critic(**batch)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(**batch)

        if step % self.critic_target_update_frequency == 0:
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(
                    self.critic_tau * p.data + (1 - self.critic_tau) * tp.data
                )
