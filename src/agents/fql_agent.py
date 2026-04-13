from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        noise = torch.randn(1, self.action_dim, device=ptu.device)
        action = self.onestep_actor(observation, noise)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        action = noise
        for i in range(self.flow_steps):
            t = torch.full((observation.shape[0], 1), i / self.flow_steps, device=observation.device)
            action = action + (1 / self.flow_steps) * self.bc_actor(observation, action, t)
        action = torch.clamp(action, -1, 1)
        return action

    @torch.compile
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
        # TODO(student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        with torch.no_grad():
            noise = torch.randn_like(actions)
            next_actions = self.onestep_actor(next_observations, noise)
            next_actions = torch.clamp(next_actions, -1, 1)
            next_qs = self.target_critic(next_observations, next_actions)
            # Use AVERAGE of ensemble (not min) per the tips
            next_q = next_qs.mean(dim=0)
            target_q = rewards + self.discount * (1 - dones.float()) * next_q

        q_values = self.critic(observations, actions)
        q = q_values.min(dim=0).values
        loss = nn.functional.mse_loss(q_values, target_q.unsqueeze(0).expand_as(q_values))

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        noise = torch.randn_like(actions)
        t = torch.rand(observations.shape[0], 1, device=observations.device)
        x_t = (1 - t) * noise + t * actions
        target_v = actions - noise
        pred_v = self.bc_actor(observations, x_t, t)
        loss = nn.functional.mse_loss(pred_v, target_v)

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        noise = torch.randn_like(actions)
        with torch.no_grad():
            bc_actions = self.get_bc_action(observations, noise)
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        onestep_actions = self.onestep_actor(observations, noise)
        distill_loss = nn.functional.mse_loss(onestep_actions, bc_actions)

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        onestep_actions_clipped = torch.clamp(onestep_actions, -1, 1)
        qs = self.critic(observations, onestep_actions_clipped)
        q = qs.mean(dim=0)
        q_loss = -q.mean()

        # Total loss.
        loss = self.alpha * distill_loss + q_loss

        # Additional metrics for logging.
        mse = nn.functional.mse_loss(onestep_actions, actions)

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            new_value = self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data
            target_param.data.copy_(new_value)

