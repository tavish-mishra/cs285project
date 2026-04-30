import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import tqdm

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log

WANDB_PROJECT = "cs285_project"


def run_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace):
    """Offline RL training loop (trains on a fixed dataset)."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    env, dataset = config["make_env_and_dataset"]()

    example_batch = dataset.sample(1)
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        example_batch['observations'].shape[1:],
        example_batch['actions'].shape[-1],
        **config["agent_kwargs"],
    )

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    for step in tqdm.trange(config["training_steps"] + 1, dynamic_ncols=True):
        batch = dataset.sample(config["batch_size"])

        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        if step % args.log_interval == 0:
            train_logger.log(metrics, step=step)

        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]

            eval_logger.log(
                {
                    "eval/mean_return": float(np.mean(returns)),
                    "eval/std_return": float(np.std(returns)),
                    "eval/min_return": float(np.min(returns)),
                    "eval/max_return": float(np.max(returns)),
                },
                step=step,
            )

    dump_log(agent, train_logger, eval_logger, config, args.save_dir)


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["online", "offline"], default="offline",
        help="Training mode: 'online' collects data while training, "
             "'offline' trains on a fixed dataset",
    )
    parser.add_argument("--base_config", type=str, default="sacbc")
    parser.add_argument("--env_name", type=str, default="Humanoid-v5")
    parser.add_argument("--exp_name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default="Debug")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--training_steps", type=int, default=500_000)
    parser.add_argument("--minari_dataset", type=str, default=None,
                        help="Minari dataset ID (e.g. mujoco/humanoid/expert-v0). "
                             "Auto-resolved from env_name if omitted.")
    parser.add_argument("--dataset_size", type=int, default=None,
                        help="Number of random transitions if no Minari dataset is available")
    parser.add_argument("--log_interval", type=int, default=1_000)
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--num_eval_trajectories", type=int, default=10)

    parser.add_argument("--expectile", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)
    return args


def main(args):
    if args.mode == "online":
        from scripts.run_online import main as online_main
        online_main(args)
        return

    # Offline mode
    logdir_prefix = "exp"

    config_kwargs = {}
    if getattr(args, "minari_dataset", None) is not None:
        config_kwargs["minari_dataset"] = args.minari_dataset
    if args.dataset_size is not None:
        config_kwargs["dataset_size"] = args.dataset_size

    config = configs.configs[args.base_config](args.env_name, **config_kwargs)

    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['training_steps'] = args.training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories

    print("CONFIG", config)

    exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"

    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"

    setup_wandb(project=WANDB_PROJECT, name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    run_training_loop(config, train_logger, eval_logger, args)


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        main(args)
