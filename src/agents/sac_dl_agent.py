from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu
import torch.nn.functional as F

from typing import Callable, Optional, Sequence, Tuple, List


def _check(name: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        finite = t[torch.isfinite(t)]
        mn = finite.min().item() if finite.numel() else float("nan")
        mx = finite.max().item() if finite.numel() else float("nan")
        raise RuntimeError(
            f"non-finite in {name}: nan={n_nan} inf={n_inf} "
            f"finite_min={mn} finite_max={mx} shape={tuple(t.shape)}"
        )


def _safe_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo).clamp(lo, hi)


_ACT_EPS = 1e-6


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
        _check("critic.q", q)
        with torch.no_grad():
            next_dist = self.actor(next_observations)
            next_actions = next_dist.rsample()
            _check("critic.next_actions", next_actions)
            next_actions_safe = next_actions.clamp(-1.0 + _ACT_EPS, 1.0 - _ACT_EPS)
            next_log_prob_raw = next_dist.log_prob(next_actions_safe)
            next_log_prob = _safe_clamp(next_log_prob_raw, -50.0, 50.0)
            next_qs = self.target_critic.forward(next_observations, next_actions_safe)
            next_q = next_qs.min(dim=0).values - self.beta() * next_log_prob
        targets = rewards + (1 - dones.float()) * self.discount * next_q
        _check("critic.targets", targets)
        loss = nn.functional.mse_loss(q, targets)
        _check("critic.loss", loss)

        self.critic_optimizer.zero_grad()
        loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        _check("critic.grad_norm", critic_grad_norm)
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "grad_norm": critic_grad_norm,
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
        _check("actor.actor_actions", actor_actions)
        actor_actions_safe = actor_actions.clamp(-1.0 + _ACT_EPS, 1.0 - _ACT_EPS)
        qs = self.critic.forward(observations, actor_actions_safe)
        q = qs.min(dim=0).values
        q_loss = -q.mean()

        mses = nn.functional.mse_loss(actor_actions_safe, actions)
        # bc_loss = self.alpha * (1/actions.shape[1]) * mses
        #TODO: UPDATE bc_loss to BE A MIXTURE OF REVERSE KL AND OUR MODEL ESTIMATE
        actions_clipped = actions.clamp(-1.0 + _ACT_EPS, 1.0 - _ACT_EPS)
        bc_logprob_raw = actor_dists.log_prob(actions_clipped)
        _check("actor.bc_logprob_raw", bc_logprob_raw)
        bc_logprob = -self.alpha * _safe_clamp(bc_logprob_raw, -50.0, 50.0).mean()
        if self.encoder is not None:
            z_policy = self.encoder(actor_actions_safe)
            with torch.no_grad():
                z_expert = self.encoder(actions_clipped)
            _check("actor.z_policy", z_policy)
            _check("actor.z_expert", z_expert)
            sim = F.cosine_similarity(z_policy, z_expert, dim=-1, eps=1e-6)
            bc_encoder = -sim.mean()
            ##bc_encoder = (z_policy * z_expert).sum(dim=-1).mean()
            encoder_loss_weight = (step * self.kappa) / (self.bc_ramp_scale + step * self.kappa)
            bc_loss = ((1-encoder_loss_weight) * bc_logprob)  + (encoder_loss_weight * bc_encoder)
        else:
            bc_loss = bc_logprob



        log_probs_raw = actor_dists.log_prob(actor_actions_safe)
        _check("actor.log_probs_raw", log_probs_raw)
        log_probs = _safe_clamp(log_probs_raw, -50.0, 50.0).mean()
        entropy_loss = self.beta.forward() * log_probs

        loss = q_loss + bc_loss + entropy_loss
        _check("actor.loss", loss)

        self.actor_optimizer.zero_grad()
        loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        _check("actor.grad_norm", actor_grad_norm)
        self.actor_optimizer.step()

        return {
            "total_loss": loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
            "entropy_loss": entropy_loss,
            "mse": mses.mean(),
            "grad_norm": actor_grad_norm,
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
        actor_actions_safe = actor_actions.clamp(-1.0 + _ACT_EPS, 1.0 - _ACT_EPS)
        log_probs_raw = actor_dists.log_prob(actor_actions_safe)
        _check("beta.log_probs_raw", log_probs_raw)
        log_probs = _safe_clamp(log_probs_raw, -50.0, 50.0)

        loss = self.beta() * (-log_probs - self.target_entropy).detach().mean()
        _check("beta.loss", loss)

        self.beta_optimizer.zero_grad()
        loss.backward()
        beta_grad_norm = torch.nn.utils.clip_grad_norm_(self.beta.parameters(), 1.0)
        _check("beta.grad_norm", beta_grad_norm)
        self.beta_optimizer.step()

        return {
            "beta_loss": loss,
            "beta": self.beta(),
            "grad_norm": beta_grad_norm,
        }

    def update_encoder(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):

        actions_clipped = actions.clamp(-1.0 + _ACT_EPS, 1.0 - _ACT_EPS)
        actor_dists = self.actor(observations)
        actor_actions = actor_dists.rsample()
        actor_actions_safe = actor_actions.clamp(-1.0 + _ACT_EPS, 1.0 - _ACT_EPS)

        z_policy = self.encoder(actor_actions_safe)
        z_expert = self.encoder(actions_clipped)
        _check("encoder.z_policy", z_policy)
        _check("encoder.z_expert", z_expert)

        dot = (z_policy * z_expert).sum(dim=-1)
        sim = F.cosine_similarity(z_policy, z_expert, dim=-1, eps=1e-6)
        _check("encoder.sim", sim)

        # kl_target = (-actor_dists.log_prob(actions_clipped)).clamp(min=-50.0, max=50.0).detach()
        KL_SCALE = 20.0  # hyperparameter
        nll_raw = -actor_dists.log_prob(actions_clipped).detach()
        _check("encoder.nll_raw", nll_raw)
        nll = _safe_clamp(nll_raw, -1e6, 1e6)
        kl_target = torch.tanh(nll / KL_SCALE)

        loss = nn.functional.mse_loss(sim, kl_target)
        _check("encoder.loss", loss)

        self.encoder_optimizer.zero_grad()
        loss.backward()
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        _check("encoder.grad_norm", encoder_grad_norm)
        self.encoder_optimizer.step()
        return {"encoder_loss": loss, "grad_norm": encoder_grad_norm}

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