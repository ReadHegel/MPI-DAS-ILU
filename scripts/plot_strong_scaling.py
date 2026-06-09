#!/home/hegel/venv_glob/bin/python
"""Strong scaling plot (process count vs. runtime) for the MPI ILU benchmark.

Usage:
    ~/venv_glob/bin/python scripts/plot_strong_scaling.py
    ~/venv_glob/bin/python scripts/plot_strong_scaling.py -i scripts/hook1498_scaling.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Hook_1498 on Okeanos (from benchmark runs)
DEFAULT_DATA = [
    (1, 4.960609),
    (2, 3.032587),
    (4, 1.944593),
    (8, 1.401476),
    (12, 1.219064),
    (24, 1.049495),
    (48, 1.279899),
    (96, 0.909666),
    (120, 0.850140),
]


def load_data(path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        rows = DEFAULT_DATA
    else:
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.lower().startswith("p,"):
                    continue
                p_str, t_str = line.split(",")
                rows.append((int(p_str), float(t_str)))
        rows.sort(key=lambda row: row[0])

    procs = np.array([row[0] for row in rows], dtype=float)
    times = np.array([row[1] for row in rows], dtype=float)
    return procs, times


def plot_strong_scaling(
    procs: np.ndarray,
    times: np.ndarray,
    output: Path,
    title: str,
    show_ideal: bool,
) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(procs, times, "o-", color="#1f77b4", linewidth=2, markersize=7, label="Measured")

    if show_ideal and len(times) > 0:
        t1 = times[0]
        ideal = t1 / procs
        ax.plot(procs, ideal, "--", color="#888888", linewidth=1.5, label=r"Ideal ($T_1/p$)")

    ax.set_xlabel("Number of MPI processes ($p$)")
    ax.set_ylabel("Runtime [s]")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_xticks(procs)
    ax.set_xticklabels([str(int(p)) for p in procs])
    ax.minorticks_off()
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.pdf')}")


def plot_speedup(
    procs: np.ndarray,
    times: np.ndarray,
    output: Path,
    title: str,
) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    speedup = times[0] / times
    ideal = procs / procs[0]  # S = p / p_1, equals p when p_1 = 1

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.loglog(
        procs,
        ideal,
        linestyle=(0, (1, 3)),
        color="#4a7ebB",
        linewidth=2,
        label="Ideal",
        zorder=1,
    )
    ax.loglog(
        procs,
        speedup,
        "o-",
        color="#e67e22",
        linewidth=2,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Hook_1498",
        zorder=3,
    )

    ax.set_xlabel("Number of MPI processes")
    ax.set_ylabel("Speedup")
    ax.set_title(title)
    ax.set_xticks(procs)
    ax.set_xticklabels([str(int(p)) for p in procs], rotation=25, ha="right")
    ax.minorticks_on()
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="0.7")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot strong scaling results (p vs. time).")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="CSV file with columns: p,time (header optional)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("figures/hook1498_strong_scaling.png"),
        help="Output image path (.png; .pdf written alongside)",
    )
    parser.add_argument(
        "--title",
        default="Strong scaling — Janna/Hook_1498 (Okeanos)",
        help="Plot title",
    )
    parser.add_argument(
        "--no-ideal",
        action="store_true",
        help="Do not draw the ideal strong-scaling reference line",
    )
    args = parser.parse_args()

    procs, times = load_data(args.input)

    plot_strong_scaling(
        procs,
        times,
        args.output,
        args.title,
        show_ideal=not args.no_ideal,
    )

    speedup_output = args.output.parent / "hook1498_speedup.png"
    plot_speedup(
        procs,
        times,
        speedup_output,
        "Speedup — Janna/Hook_1498 (Okeanos)",
    )


if __name__ == "__main__":
    main()
