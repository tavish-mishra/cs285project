from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import tqdm

from agents.sac_dl_agent import SACAgentDivLearn
from infrastructure.replay_buffer import ReplayBuffer
from infrastructure.utils import EpisodeMonitor
from networks.rl_networks import Policy, EnsembleCritic, LogParam
from encoders.fcn_encoder import FCNEncoder
from encoders.transformer_encoder import TransformerEncoder


# Map gymnasium env IDs to default Minari dataset IDs.
# "medium" datasets come from a partially-trained SAC policy — good diversity
# for offline RL.  "expert" variants are also available.
DEFAULT_MINARI_DATASETS = {
    "Humanoid-v5": "mujoco/humanoid/medium-v0",
    "Walker2d-v5": "mujoco/walker2d/medium-v0",
    "Walker2d-v4": "mujoco/walker2d/medium-v0",
    "HalfCheetah-v5": "mujoco/half_cheetah/medium-v0",
    "Hopper-v5": "mujoco/hopper/medium-v0",
    "Ant-v5": "mujoco/ant/medium-v0",
}


def sac_dl_config(
    env_name: str,
    exp_name: Optional[str] = None,
    minari_dataset: Optional[str] = None,
    hidden_size: int = 512,
    num_layers: int = 6,
    learning_rate: float = 3e-4,
    discount: float = 0.99,
    target_update_rate: float = 0.005,
    alpha: float = 1.0,
    total_steps: int = 1_000_000,
    batch_size: int = 256,
    dataset_size: int = 100_000,
    buffer_size: int = 1_000_000,
    warmup_steps: int = 5_000,
    update_every: int = 1,

    encoder_type : str = 'fcn', #or fcn or transformer
    encoder_kwargs: dict = None,
    kappa: float = 0.1,#0.1 means update every 10 steps
    dl_base_lr : float = 1e-3,
    dl_lr_cap : float = 1e-2, #max effective LR after dividing by kappa
    bc_ramp_scale: float = 100,
    **kwargs,
):
    """
    TODO: Adding the following parameters and implementing make_divlearn

    Add following parameters: warm-up / DivLearn lr schedule kappa, dl_base_lr (divided by and clipped by kappa)
    """
    resolved_dataset = minari_dataset or DEFAULT_MINARI_DATASETS.get(env_name)
    if encoder_kwargs is None:
        if encoder_type == "fcn":
            encoder_kwargs = {'hidden_sizes': [256, 256], 'out_dim': 64}
        elif encoder_type == "transformer":
            encoder_kwargs = {'d_model':64, 'out_dim': 64,
                              'num_layers': 2, 'num_heads': 2}
        else:
            encoder_kwargs = {}


    def make_actor(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        return Policy(
            ac_dim=action_dim,
            ob_dim=int(np.prod(observation_shape)),
            discrete=False,
            n_layers=num_layers,
            layer_size=hidden_size,
            use_tanh=True,
            state_dependent_std=True,
        )

    def make_critic(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        return EnsembleCritic(
            ob_dim=int(np.prod(observation_shape)),
            ac_dim=action_dim,
            n_layers=num_layers,
            size=hidden_size,
            n_ensembles=2,
        )

    def make_beta() -> nn.Module:
        return LogParam()

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_encoder_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        encoder_learning_rate = min(dl_base_lr / kappa, dl_lr_cap)
        return torch.optim.Adam(params, lr=encoder_learning_rate) #TODO: kappa learning rate stuff

    def make_encoder(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        if encoder_type == None:
            return None
        obs_dim = int(np.prod(observation_shape))
        if encoder_type == 'fcn':
            hidden_sizes = encoder_kwargs['hidden_sizes']
            out_dim = encoder_kwargs['out_dim']
            return FCNEncoder(action_dim, obs_dim, hidden_sizes, out_dim)
        elif encoder_type == 'transformer':
            d_model = encoder_kwargs['d_model']
            out_dim = encoder_kwargs['out_dim']
            num_layers = encoder_kwargs['num_layers']
            num_heads = encoder_kwargs['num_heads']
            return TransformerEncoder(action_dim, obs_dim, d_model, out_dim, num_layers, num_heads)

    agent_kwargs = {
        "make_actor": make_actor,
        "make_actor_optimizer": make_optimizer,
        "make_critic": make_critic,
        "make_critic_optimizer": make_optimizer,
        "make_beta": make_beta,
        "make_beta_optimizer": make_optimizer,
        "discount": discount,
        "target_update_rate": target_update_rate,
        "alpha": alpha,
        'make_encoder': make_encoder,
        'make_encoder_optimizer': make_encoder_optimizer,
        "kappa": kappa,
        'bc_ramp_scale': bc_ramp_scale,
    }

    def make_agent(ob_dim: int, ac_dim: int, discrete: bool = False) -> SACAgentDivLearn:
        return SACAgentDivLearn(
            observation_shape=(ob_dim,),
            action_dim=ac_dim,
            **agent_kwargs,
        )

    def _load_minari_dataset() -> Tuple[gymnasium.Env, ReplayBuffer]:
        """Load a curated offline dataset from Minari."""
        import minari

        print(f"Loading Minari dataset: {resolved_dataset}")
        md = minari.load_dataset(resolved_dataset, download=True)
        env = EpisodeMonitor(md.recover_environment())

        total_steps = sum(ep.rewards.shape[0] for ep in md.iterate_episodes())
        dataset = ReplayBuffer(capacity=total_steps)

        for ep in md.iterate_episodes():
            obs = ep.observations[:-1]
            next_obs = ep.observations[1:]
            actions = ep.actions
            rewards = ep.rewards
            # Bellman bootstrap: only true environment termination stops bootstrapping.
            # Time-limit truncation is NOT terminal for infinite-horizon discounted RL (D4RL / Minari convention).
            terminations = ep.terminations.astype(np.float32)

            for i in range(len(rewards)):
                dataset.insert(
                    observation=obs[i].astype(np.float32),
                    action=actions[i].astype(np.float32),
                    reward=np.float32(rewards[i]),
                    next_observation=next_obs[i].astype(np.float32),
                    done=np.float32(terminations[i]),
                )

        print(f"Loaded {len(dataset)} transitions from {resolved_dataset}")
        return env, dataset

    def _collect_random_dataset() -> Tuple[gymnasium.Env, ReplayBuffer]:
        """Fallback: collect random-policy transitions."""
        env = EpisodeMonitor(gymnasium.make(env_name))
        ac_space = env.action_space

        dataset = ReplayBuffer(capacity=dataset_size)
        obs, _ = env.reset()
        for _ in tqdm.trange(dataset_size, desc=f"Collecting {dataset_size} random transitions"):
            action = ac_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            dataset.insert(
                observation=obs.astype(np.float32),
                action=np.asarray(action, dtype=np.float32),
                reward=np.float32(reward),
                next_observation=next_obs.astype(np.float32),
                done=np.float32(done),
            )
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        return env, dataset

    def make_env_and_dataset() -> Tuple[gymnasium.Env, ReplayBuffer]:
        if resolved_dataset is not None:
            return _load_minari_dataset()
        print(f"WARNING: No Minari dataset found for {env_name}, collecting random data.")
        return _collect_random_dataset()

    log_string = f"{exp_name or 'sacbc'}_{env_name}"

    config = {
        "agent": "sacbc",
        "agent_kwargs": agent_kwargs,
        "make_agent": make_agent,
        "make_env_and_dataset": make_env_and_dataset,
        "env_name": env_name,
        "log_name": log_string,
        "minari_dataset": resolved_dataset,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "dataset_size": dataset_size,
        "buffer_size": buffer_size,
        "warmup_steps": warmup_steps,
        "update_every": update_every,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        **kwargs,
    }

    return config
