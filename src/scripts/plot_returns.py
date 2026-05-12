"""Plot eval returns across all experiments in modal_exp/.

Usage:
    uv run python src/scripts/plot_returns.py
    uv run python src/scripts/plot_returns.py --exp-root modal_exp --save plots/returns.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_runs(exp_root: Path):
    """Return list of (run_group, variant, dataframe) tuples."""
    runs = []
    for csv_path in sorted(exp_root.rglob("eval.csv")):
        parts = csv_path.relative_to(exp_root).parts
        if len(parts) < 3:
            continue
        run_group, variant = parts[0], parts[1]
        df = pd.read_csv(csv_path)
        runs.append((run_group, variant, df))
    return runs


def parse_run(run_group: str, variant: str):
    """Return (family, bottleneck, depth, target) — any may be None."""
    if run_group == "Baseline":
        return "Baseline", None, None, None
    family = "FCN" if "FCN" in run_group else "Transformer"
    bottleneck = run_group.split("-")[-1]  # Bottleneck | Same | Expanding
    parts = variant.replace("_target", "").split("_")
    target = parts[-1]                     # linear | log
    depth = parts[0] if "layer" in variant else None  # "2" | "3" | None
    return family, bottleneck, depth, target


def make_label(family, depth, target):
    if family == "Baseline":
        return "Baseline (SAC+BC)"
    if depth is not None:
        return f"{family}-{depth}L · {target}"
    return f"{family} · {target}"


STYLES = {
    ("FCN", "2", "linear"):     dict(color="#1f77b4", linestyle="-"),
    ("FCN", "2", "log"):        dict(color="#1f77b4", linestyle="--"),
    ("FCN", "3", "linear"):     dict(color="#d62728", linestyle="-"),
    ("FCN", "3", "log"):        dict(color="#d62728", linestyle="--"),
    ("Transformer", None, "linear"): dict(color="#2ca02c", linestyle="-"),
    ("Transformer", None, "log"):    dict(color="#2ca02c", linestyle="--"),
    ("Baseline", None, None):   dict(color="black", linestyle=":", linewidth=2),
}


def plot_run(ax, df, label, style):
    step = df["step"]
    mean = df["eval/mean_return"]
    ax.plot(step, mean, label=label, **style)


def plot_single_run(run_group: str, variant: str, df: pd.DataFrame, save_dir: Path):
    fam, bn, depth, target = parse_run(run_group, variant)
    label = make_label(fam, depth, target)

    if fam == "Baseline":
        title = "Mean Return on Humanoid (Baseline SAC+BC)"
    else:
        model = f"FCN-{depth}L" if depth is not None else "Transformer"
        title = f"Mean Return on Humanoid with {model} Encoder and {target.capitalize()} Target"

    fig, ax = plt.subplots(figsize=(8, 5))
    style = STYLES.get((fam, depth, target), {}) or {"color": "#1f77b4"}
    plot_run(ax, df, label, style)
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval mean return")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = save_dir / f"{run_group}__{variant}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=Path("modal_exp"))
    parser.add_argument("--save", type=Path, default=None,
                        help="Path for the combined grid plot. Ignored when --individual is set.")
    parser.add_argument("--individual", action="store_true",
                        help="Save one PNG per run instead of the combined grid.")
    parser.add_argument("--out-dir", type=Path, default=Path("plots/individual"),
                        help="Directory for individual plots when --individual is set.")
    args = parser.parse_args()

    runs = load_runs(args.exp_root)
    if not runs:
        raise SystemExit(f"No eval.csv found under {args.exp_root}")

    if args.individual:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for run_group, variant, df in runs:
            plot_single_run(run_group, variant, df, args.out_dir)
        return

    # Separate baseline so we can overlay it on every panel.
    baseline = [(rg, v, df) for rg, v, df in runs if rg == "Baseline"]
    encoder_runs = [(rg, v, df) for rg, v, df in runs if rg != "Baseline"]

    bottlenecks = ["Bottleneck", "Same", "Expanding"]
    families = ["FCN", "Transformer"]

    fig, axes = plt.subplots(
        len(families), len(bottlenecks),
        figsize=(18, 9), sharex=True, sharey=True,
    )

    for r, family in enumerate(families):
        for c, bottleneck in enumerate(bottlenecks):
            ax = axes[r, c]
            for rg, v, df in encoder_runs:
                fam, bn, depth, target = parse_run(rg, v)
                if fam != family or bn != bottleneck:
                    continue
                style = STYLES.get((fam, depth, target), {})
                plot_run(ax, df, make_label(fam, depth, target), style)

            for rg, v, df in baseline:
                style = STYLES[("Baseline", None, None)]
                plot_run(ax, df, make_label("Baseline", None, None), style)

            if r == 0:
                ax.set_title(bottleneck)
            if c == 0:
                ax.set_ylabel(f"{family}\neval mean return")
            if r == len(families) - 1:
                ax.set_xlabel("Training step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Humanoid-v5 offline — eval return by encoder configuration", fontsize=14)
    fig.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
