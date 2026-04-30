"""Online training loop for off-policy agents on gymnasium tasks.

Usage (standalone):
    python run_online.py --env_name Walker2d-v4
    python run_online.py --env_name Humanoid-v5 --training_steps 1000000

Via the unified entry point:
    python run.py --mode online --env_name Humanoid-v5
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from typing import Any, Dict

import gymnasium
import numpy as np
import torch
import tqdm

from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import Logger, dump_log, setup_wandb
from infrastructure.replay_buffer import ReplayBuffer
from infrastructure.utils import EpisodeMonitor


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(agent, env_name: str, num_episodes: int, max_steps: int) -> Dict[str, float]:
    """Roll out the current policy greedily and return mean/std return."""
    eval_env = gymnasium.make(env_name)
    returns = []
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        while not done and steps < max_steps:
            with torch.no_grad():
                action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_return += reward
            steps += 1
        returns.append(ep_return)
    eval_env.close()
    return {
        "eval/mean_return": float(np.mean(returns)),
        "eval/std_return": float(np.std(returns)),
        "eval/min_return": float(np.min(returns)),
        "eval/max_return": float(np.max(returns)),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_online_training_loop(config: Dict[str, Any], train_logger: Logger, eval_logger: Logger, args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    env_name = config["env_name"]
    env = EpisodeMonitor(gymnasium.make(env_name))
    obs_space = env.observation_space
    ac_space = env.action_space

    ob_dim = int(np.prod(obs_space.shape))
    discrete = isinstance(ac_space, gymnasium.spaces.Discrete)
    ac_dim = ac_space.n if discrete else int(np.prod(ac_space.shape))

    ep_len = getattr(env.spec, "max_episode_steps", None) or config.get("ep_len", 1000)

    # Build agent
    agent = config["make_agent"](ob_dim=ob_dim, ac_dim=ac_dim, discrete=discrete)

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=config.get("buffer_size", 1_000_000))

    # ------------------------------------------------------------------
    # Warm-up: fill buffer with random actions before any gradient steps
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    warmup_steps = config.get("warmup_steps", 5_000)
    for _ in tqdm.trange(warmup_steps, desc="Warm-up", dynamic_ncols=True):
        raw_action = ac_space.sample()
        action = np.array([raw_action]) if discrete and np.array(raw_action).ndim == 0 else np.array(raw_action, dtype=np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(raw_action)
        done = terminated or truncated
        replay_buffer.insert(
            observation=obs.astype(np.float32),
            action=action,
            reward=np.float32(reward),
            next_observation=next_obs.astype(np.float32),
            done=np.bool_(done),
        )
        obs = env.reset()[0].astype(np.float32) if done else next_obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Main loop: collect one transition → update agent
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    obs = obs.astype(np.float32)

    batch_size = config.get("batch_size", 256)
    update_every = config.get("update_every", 1)   # gradient steps per env step

    for step in tqdm.trange(config["training_steps"] + 1, desc="Training", dynamic_ncols=True):

        # --- Collect one transition ---
        with torch.no_grad():
            action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(
            int(action) if discrete else action
        )
        done = terminated or truncated
        next_obs = next_obs.astype(np.float32)

        replay_buffer.insert(
            observation=obs,
            action=action if not discrete else np.array(action, dtype=np.float32),
            reward=np.float32(reward),
            next_observation=next_obs,
            done=np.bool_(done),
        )
        obs = env.reset()[0].astype(np.float32) if done else next_obs

        # --- Gradient update(s) ---
        if len(replay_buffer) >= batch_size:
            for _ in range(update_every):
                batch = replay_buffer.sample(batch_size)
                batch_t = {k: ptu.from_numpy(v) for k, v in batch.items() if isinstance(v, np.ndarray)}
                metrics = agent.update(
                    batch_t["observations"],
                    batch_t["actions"],
                    batch_t["rewards"],
                    batch_t["next_observations"],
                    batch_t["dones"],
                    step,
                )

            # Log episode stats when an episode finishes
            if done and "episode" in info:
                metrics["train/episode_return"] = info["episode"]["return"]
                metrics["train/episode_length"] = info["episode"]["length"]

            if step % args.log_interval == 0:
                train_logger.log(metrics, step=step)

        # --- Periodic evaluation ---
        if step % args.eval_interval == 0:
            eval_metrics = evaluate(agent, env_name, args.num_eval_trajectories, ep_len)
            eval_logger.log(eval_metrics, step=step)

    env.close()
    dump_log(agent, train_logger, eval_logger, config, args.save_dir)


# ---------------------------------------------------------------------------
# Argument parsing + entry point
# ---------------------------------------------------------------------------

def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Walker2d-v4", help="Gymnasium env id, e.g. Walker2d-v4, Humanoid-v5")
    parser.add_argument("--agent_config", type=str, default=None, help="Python import path to a config function")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default="Debug")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", type=int, default=0)
    parser.add_argument("--training_steps", type=int, default=500_000)
    parser.add_argument("--log_interval", type=int, default=1_000)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--num_eval_trajectories", type=int, default=10)
    return parser.parse_args(args=args)


def main(args):
    agent_config = getattr(args, "agent_config", None)
    base_config = getattr(args, "base_config", None)

    if agent_config is not None:
        import importlib
        module_path, fn_name = agent_config.rsplit(".", 1)
        module = importlib.import_module(module_path)
        config_fn = getattr(module, fn_name)
        config = config_fn(args.env_name)
    elif base_config is not None:
        import configs
        config = configs.configs[base_config](args.env_name)
    else:
        from configs.sac_dl_config import sac_dl_config
        config = sac_dl_config(args.env_name)

    config["training_steps"] = args.training_steps
    config["seed"] = args.seed

    exp_name = args.exp_name or f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"
    args.save_dir = os.path.join("exp", args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    setup_wandb(project="cs285_project", name=exp_name, group=args.run_group, config=config)
    train_logger = Logger(os.path.join(args.save_dir, "train.csv"))
    eval_logger = Logger(os.path.join(args.save_dir, "eval.csv"))

    run_online_training_loop(config, train_logger, eval_logger, args)


if __name__ == "__main__":
    args = setup_arguments()
    main(args)
