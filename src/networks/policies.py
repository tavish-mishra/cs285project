import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs_tensor = ptu.from_numpy(obs)
        distribution = self.forward(obs_tensor)
        action = ptu.to_numpy(distribution.sample())

        return action

    @torch.no_grad()
    def get_action_and_log_prob(self, obs: np.ndarray):
        """Returns (action, log_prob) both as numpy arrays."""
        obs_tensor = ptu.from_numpy(obs)
        distribution = self.forward(obs_tensor)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        if not self.discrete:
            log_prob = log_prob.sum(dim=-1)
        return ptu.to_numpy(action), ptu.to_numpy(log_prob)

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            return distributions.Categorical(logits=self.logits_net(obs))
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            return distributions.Normal(self.mean_net(obs), torch.exp(self.logstd))

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        log_probs_old: np.ndarray = None,
        kl_coef: float = 0.0,
    ) -> dict:
        """Implements the policy gradient actor update.

        If log_probs_old is provided, uses importance sampling (off-policy correction)
        and adds a KL divergence penalty scaled by kl_coef.
        """
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        distribution = self.forward(obs)
        log_probs_new = distribution.log_prob(actions)
        if not self.discrete:
            log_probs_new = log_probs_new.sum(dim=-1)

        if log_probs_old is not None:
            log_probs_old = ptu.from_numpy(log_probs_old)
            ratio = (log_probs_new - log_probs_old).exp()
            pg_loss = -(ratio * advantages).mean()
            kl = (log_probs_old - log_probs_new).mean()
            loss = pg_loss + kl_coef * kl
        else:
            pg_loss = (-log_probs_new * advantages).mean()
            kl = torch.tensor(0.0)
            loss = pg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
            "PG Loss": pg_loss.item(),
            "KL": kl.item(),
        }
