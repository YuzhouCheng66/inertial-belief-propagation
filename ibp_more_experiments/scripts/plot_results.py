#!/usr/bin/env python3
"""Create four independent plots and combine them into the repository's 2x2 summary figure."""
from __future__ import annotations
from pathlib import Path
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

TRAJ = pd.read_csv(RESULTS / "trajectories.csv")
PER_CASE = pd.read_csv(RESULTS / "per_case.csv")

TASKS = ["Gaussian/PDE", "Non-Gaussian Ising", "Complex tensor network"]


def save_trajectory(task: str, out: Path) -> None:
    fig = plt.figure(figsize=(6.4, 4.4))
    d = TRAJ[TRAJ.domain == task]
    for method in ("BP", "I-BP"):
        s = d[d.method == method]
        plt.plot(s.full_bp_evaluations, s.residual_over_tolerance, label=method)
    plt.axhline(1.0, linestyle="--", linewidth=1.0, label="Stopping threshold")
    plt.yscale("log")
    plt.xlabel("Full BP update evaluations")
    plt.ylabel("Residual / stopping threshold")
    plt.title(task)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_speedup(out: Path) -> None:
    fig = plt.figure(figsize=(6.4, 4.4))
    for domain_id, task in enumerate(TASKS):
        d = PER_CASE[PER_CASE.domain == domain_id]
        xs = [1, 2, 4]
        values = []
        lows = []
        highs = []
        for t in xs:
            ratio = d[f"BP_{t}"] / d[f"IBP_{t}"]
            med = ratio.median()
            values.append(med); lows.append(med - ratio.min()); highs.append(ratio.max() - med)
        plt.errorbar(xs, values, yerr=[lows, highs], marker="o", capsize=4, label=task)
    plt.xticks([1, 2, 4])
    plt.xlabel("Threads used by both BP and I-BP")
    plt.ylabel("Wall-time speedup: BP / I-BP")
    plt.title("Same-core speedup")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def combine(paths: list[Path], out: Path) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = max(i.width for i in imgs); h = max(i.height for i in imgs)
    canvas = Image.new("RGB", (2*w, 2*h), "white")
    for k, im in enumerate(imgs):
        x = (k % 2) * w + (w - im.width)//2
        y = (k // 2) * h + (h - im.height)//2
        canvas.paste(im, (x, y))
    canvas.save(out, quality=95)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        panels = []
        for i, task in enumerate(TASKS):
            p = td / f"panel_{i}.png"; save_trajectory(task, p); panels.append(p)
        p = td / "panel_speedup.png"; save_speedup(p); panels.append(p)
        out = FIGURES / "bp_vs_ibp_2x2.png"
        combine(panels, out)
    print(out)

if __name__ == "__main__":
    main()
