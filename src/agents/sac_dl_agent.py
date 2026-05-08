from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class SACAgentDivLearn(nn.Module):
    """
    Soft Actor-Critic with Div-Learn capabilities
    """
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_beta,
        make_beta_optimizer,
        make_encoder,
        make_encoder_optimizer,
        bc_ramp_scale,

        discount: float,
        target_update_rate: float,
        alpha: float,
        kappa,
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.beta = make_beta()

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.beta_optimizer = make_beta_optimizer(self.beta.parameters())

        encoder = make_encoder(observation_shape, action_dim)
        if encoder is not None:
            self.encoder = encoder.to(ptu.device)
            self.encoder_optimizer = make_encoder_optimizer(self.encoder.parameters())
            self.kappa = kappa
            self.encoder_update_freq = max(1, int(round(1 / self.kappa)))
            self._last_encoder_metrics: dict = {}
        else:
            self.encoder = None
            self.encoder_optimizer = None
            self._last_encoder_metrics = {}

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.bc_ramp_scale = bc_ramp_scale

        self.target_entropy = -action_dim / 2  # Heuristic value (|A| / 2) from the SAC paper.

    def get_action(self, observation: np.ndarray):
        """Greedy action for eval (tanh of Gaussian mean; matches common SAC eval)."""
        observation = ptu.from_numpy(np.asarray(observation))[None]
        with torch.no_grad():
            base = self.actor(observation).base_dist.base_dist
            action = torch.tanh(base.mean)
        return ptu.to_numpy(action[0])

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        q = self.critic.forward(observations, actions)
        with torch.no_grad():
            next_actions = self.actor(next_observations).rsample()
            next_qs = self.target_critic.forward(next_observations, next_actions)
            next_q = next_qs.min(dim=0).values
        targets = rewards + (1 - dones.float()) * self.discount * next_q
        loss = nn.functional.mse_loss(q, targets)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        step
    ):
        """

        Update the actor
        """
        actor_dists = self.actor.forward(observations)
        actor_actions = actor_dists.rsample()
        qs = self.critic.forward(observations, actor_actions)
        q = qs.min(dim=0).values
        q_loss = -q.mean()

        mses = nn.functional.mse_loss(actor_actions, actions)
        # bc_loss = self.alpha * (1/actions.shape[1]) * mses
        #TODO: UPDATE bc_loss to BE A MIXTURE OF STANDARD KL AND OUR MODEL ESTIMATE

        bc_logprob = -self.alpha * actor_dists.log_prob(actions).mean() #replafirst ced so that we are actually approximating reverse KL with this constraint
        if self.encoder is not None:
            z_policy = self.encoder(actor_actions)
            z_expert = self.encoder(actions)
            bc_encoder = (z_policy * z_expert).sum(dim=-1).mean()
            encoder_loss_weight = (step * self.kappa) / (self.bc_ramp_scale + step * self.kappa)
            bc_loss = ((1-encoder_loss_weight) * bc_logprob)  + (encoder_loss_weight * bc_encoder)
        else:
            bc_loss = bc_logprob



        log_probs = actor_dists.log_prob(actor_actions).mean()
        entropy_loss = self.beta.forward() * log_probs

        loss = q_loss + bc_loss + entropy_loss

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "total_loss": loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
            "entropy_loss": entropy_loss,
            "mse": mses.mean(),
        }

    def update_beta(
        self,
        observations: torch.Tensor,
    ):
        """
        Update the beta parameter using dual gradient descent.
        """
        actor_dists = self.actor(observations)
        actor_actions = actor_dists.rsample()
        log_probs = actor_dists.log_prob(actor_actions)

        loss = self.beta() * (-log_probs - self.target_entropy).detach().mean()

        self.beta_optimizer.zero_grad()
        loss.backward()
        self.beta_optimizer.step()

        return {
            "beta_loss": loss,
            "beta": self.beta(),
        }

    def update_encoder(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ): #TODO: ACCOUNT FOR SCHEDULING STUFF, HOW TO MAKE IT UPDATE ONLY EVERY SO OFTEN
        actor_dists = self.actor(observations)
        actor_actions = actor_dists.rsample()

        z_policy = self.encoder(actor_actions)
        z_expert = self.encoder(actions)

        dot = (z_policy * z_expert).sum(dim=-1)

        kl_target = -actor_dists.log_prob(actions).detach()

        loss = nn.functional.mse_loss(dot, kl_target)

        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        return {"encoder_loss": loss}

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):

        #TODO: ENCODER UPDATES EVERY SO OFTEN ACCOUNTING FOR RATE STUFF
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions, step)
        metrics_beta = self.update_beta(observations)
        #metrics_encoder = self.update_encoder(observations, actions)
        if self.encoder is not None and step % self.encoder_update_freq == 0:
            self._last_encoder_metrics = self.update_encoder(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"beta/{k}": v.item() for k, v in metrics_beta.items()},
            **{f"encoder/{k}": v.item() for k, v in self._last_encoder_metrics.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # Update target_critic using Polyak averaging with self.target_update_rate
        # In principle: self.target_critic = self.target_update_rate * self.critic + (1 - self.target_update_rate) * self.target_critic
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate) + param.data * self.target_update_rate
            )