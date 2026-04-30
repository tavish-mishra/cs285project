import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path so local Modal execution can resolve imports.
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import modal

APP_NAME = "cs285-project"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/root/exp"
DEFAULT_GPU = "H100"
DEFAULT_CPU = 2.0
DEFAULT_MEMORY = 4096  # MB
volume = modal.Volume.from_name("cs285-final-project", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""
    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0")
    .uv_sync()
    .run_commands(
        "python -c \""
        "import minari; "
        "minari.download_dataset('mujoco/humanoid/medium-v0'); "
        "minari.download_dataset('mujoco/humanoid/expert-v0'); "
        "minari.download_dataset('mujoco/walker2d/medium-v0'); "
        "minari.download_dataset('mujoco/walker2d/expert-v0')"
        "\""
    )
)
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
}


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 12,
    env=env,
    image=image,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY,
)
def modal_remote(*args: str) -> None:
    from scripts.run import main, setup_arguments
    parsed = setup_arguments(args)
    main(parsed)
    volume.commit()


@app.local_entrypoint()
def cli(
    env_name: str = "Humanoid-v5",
    mode: str = "offline",
    base_config: str = "sacbc",
    minari_dataset: str = "",
    training_steps: int = 500_000,
    dataset_size: int = 100_000,
    seed: int = 0,
    run_group: str = "Debug",
    log_interval: int = 1_000,
    eval_interval: int = 10_000,
    num_eval_trajectories: int = 10,
    alpha: float = 0.0,
    no_gpu: bool = False,
):
    """Launch a training run on Modal. All flags map to run.py arguments."""
    args: list[str] = [
        "--mode", mode,
        "--base_config", base_config,
        "--env_name", env_name,
        "--seed", str(seed),
        "--training_steps", str(training_steps),
        "--dataset_size", str(dataset_size),
        "--run_group", run_group,
        "--log_interval", str(log_interval),
        "--eval_interval", str(eval_interval),
        "--num_eval_trajectories", str(num_eval_trajectories),
    ]
    if minari_dataset:
        args += ["--minari_dataset", minari_dataset]
    if alpha != 0.0:
        args += ["--alpha", str(alpha)]
    if no_gpu:
        args.append("--no_gpu")

    modal_remote.remote(*args)
