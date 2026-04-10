"""
explore_env.py — inspect a gymnasium environment before training.

Usage:
    python explore_env.py
    python explore_env.py --env Humanoid-v5
    python explore_env.py --env Walker2d-v4 --steps 200
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import gymnasium


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def explore(env_name: str, steps: int):

    # ------------------------------------------------------------------
    # 1. Build the environment
    # ------------------------------------------------------------------
    section("ENVIRONMENT")
    env = gymnasium.make(env_name)
    print(f"  Name            : {env_name}")
    print(f"  Max episode steps: {getattr(env.spec, 'max_episode_steps', 'N/A')}")
    print(f"  Reward threshold: {getattr(env.spec, 'reward_threshold', 'N/A')}")

    # ------------------------------------------------------------------
    # 2. Observation space
    # ------------------------------------------------------------------
    section("OBSERVATION SPACE")
    obs_space = env.observation_space
    print(f"  Type  : {type(obs_space).__name__}")
    print(f"  Shape : {obs_space.shape}")
    print(f"  Dtype : {obs_space.dtype}")
    print(f"  Low   : min={obs_space.low.min():.3f}, max={obs_space.low.max():.3f}")
    print(f"  High  : min={obs_space.high.min():.3f}, max={obs_space.high.max():.3f}")

    # ------------------------------------------------------------------
    # 3. Action space
    # ------------------------------------------------------------------
    section("ACTION SPACE")
    ac_space = env.action_space
    discrete = isinstance(ac_space, gymnasium.spaces.Discrete)
    print(f"  Type     : {type(ac_space).__name__}")
    if discrete:
        print(f"  N actions: {ac_space.n}")
    else:
        print(f"  Shape    : {ac_space.shape}")
        print(f"  Dtype    : {ac_space.dtype}")
        print(f"  Low      : {ac_space.low}")
        print(f"  High     : {ac_space.high}")

    # ------------------------------------------------------------------
    # 4. Reset — first observation
    # ------------------------------------------------------------------
    section("RESET")
    obs, info = env.reset(seed=0)
    print(f"  obs shape : {obs.shape}")
    print(f"  obs dtype : {obs.dtype}")
    print(f"  obs[:8]   : {obs[:8]}")
    print(f"  obs min   : {obs.min():.4f}  max: {obs.max():.4f}  mean: {obs.mean():.4f}")
    print(f"  info keys : {list(info.keys())}")

    # ------------------------------------------------------------------
    # 5. Step — a few random actions
    # ------------------------------------------------------------------
    section(f"STEPPING ({steps} random actions)")

    rewards = []
    ep_rewards = []
    ep_lengths = []
    ep_reward = 0.0
    ep_len = 0
    obs, _ = env.reset(seed=0)

    for i in range(steps):
        action = ac_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_reward += reward
        ep_len += 1
        rewards.append(reward)

        # Print first 3 transitions in detail
        if i < 3:
            print(f"\n  --- Step {i} ---")
            print(f"  action      : {action if discrete else action[:4]} {'...' if not discrete and len(action) > 4 else ''}")
            print(f"  next_obs[:4]: {next_obs[:4]}")
            print(f"  reward      : {reward:.4f}")
            print(f"  terminated  : {terminated}  truncated: {truncated}")
            print(f"  info keys   : {list(info.keys())}")

        if done:
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)
            ep_reward = 0.0
            ep_len = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

    # ------------------------------------------------------------------
    # 6. Reward statistics
    # ------------------------------------------------------------------
    section("REWARD STATISTICS (random policy)")
    rewards = np.array(rewards)
    print(f"  Per-step reward : min={rewards.min():.4f}  max={rewards.max():.4f}  "
          f"mean={rewards.mean():.4f}  std={rewards.std():.4f}")
    if ep_rewards:
        ep_rewards = np.array(ep_rewards)
        ep_lengths = np.array(ep_lengths)
        print(f"  Episodes seen   : {len(ep_rewards)}")
        print(f"  Episode return  : min={ep_rewards.min():.2f}  max={ep_rewards.max():.2f}  "
              f"mean={ep_rewards.mean():.2f}")
        print(f"  Episode length  : min={ep_lengths.min()}  max={ep_lengths.max()}  "
              f"mean={ep_lengths.mean():.1f}")
    else:
        print(f"  No complete episodes in {steps} steps "
              f"(episode length > {steps})")

    # ------------------------------------------------------------------
    # 7. What an agent needs — summary
    # ------------------------------------------------------------------
    section("AGENT INTERFACE SUMMARY")
    ob_dim = int(np.prod(obs_space.shape))
    ac_dim = ac_space.n if discrete else int(np.prod(ac_space.shape))
    print(f"  ob_dim   : {ob_dim}")
    print(f"  ac_dim   : {ac_dim}")
    print(f"  discrete : {discrete}")
    print(f"  obs dtype fed to network : float32")
    if not discrete:
        print(f"  action bounds : [{ac_space.low.min():.2f}, {ac_space.high.max():.2f}]  "
              f"(use tanh squashing in policy)")
    print()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Humanoid-v5")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    explore(args.env, args.steps)
