from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from agents.base import Agent
from infrastructure import pytorch_util as ptu
from networks.rl_networks import EnsembleCritic, LogParam, Policy


class SACAgent(Agent):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool = False,
        n_layers: int = 3,
        layer_size: int = 256,
        discount: float = 0.99,
        tau: float = 0.005,
        learning_rate: float = 3e-4,
        init_temperature: float = 1.0,
        target_entropy: float = None,
        n_critics: int = 2,
    ):
        super().__init__()
        assert not discrete, "SACAgent only supports continuous action spaces."

        self.discount = discount
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy is not None else -ac_dim

        self.actor = Policy(
            ac_dim=ac_dim,
            ob_dim=ob_dim,
            discrete=False,
            n_layers=n_layers,
            layer_size=layer_size,
            use_tanh=True,
            state_dependent_std=True,
        )

        self.critic = EnsembleCritic(
            ob_dim=ob_dim,
            ac_dim=ac_dim,
            n_layers=n_layers,
            size=layer_size,
            n_ensembles=n_critics,
        )
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.log_alpha = LogParam(init_value=init_temperature)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=learning_rate)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = ptu.from_numpy(obs)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        action = self.actor(obs_t).sample()
        return ptu.to_numpy(action.squeeze(0))

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:

        # ------------------------------------------------------------------ #
        # Critic update                                                        #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            next_dist = self.actor(next_obs)
            next_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_actions)          # (B,)

            next_qs = self.target_critic(next_obs, next_actions)       # (n, B)
            next_q = next_qs.min(dim=0).values                         # (B,)

            alpha = self.log_alpha()
            target_q = rewards + self.discount * (1.0 - dones.float()) * (
                next_q - alpha * next_log_probs
            )

        qs = self.critic(obs, actions)                                  # (n, B)
        critic_loss = sum(F.mse_loss(qs[i], target_q) for i in range(qs.shape[0]))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------------------------------ #
        # Actor + temperature update (shared forward pass)                    #
        # ------------------------------------------------------------------ #
        dist = self.actor(obs)
        new_actions = dist.rsample()
        log_probs = dist.log_prob(new_actions)                          # (B,)

        qs_new = self.critic(obs, new_actions)
        q_new = qs_new.min(dim=0).values

        alpha = self.log_alpha()

        actor_loss = (alpha.detach() * log_probs - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ------------------------------------------------------------------ #
        # Soft target update                                                  #
        # ------------------------------------------------------------------ #
        ptu.soft_update(self.target_critic, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item(),
            "entropy": -log_probs.mean().item(),
        }
