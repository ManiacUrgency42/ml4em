#!/usr/bin/env python3
"""
Render the scaling benchmark JSON files as plots.

Reads whatever scaling_gpu.py and scaling_cpu.py have written into
logs/benchmarks/ and draws one panel per available result.  Missing results
are skipped rather than erroring, so this is safe to run after either
benchmark alone.

GPU panel
---------
Throughput (light curves/second) against GPU count.  Each measurement point
is drawn as a violin over its repeat trials, so run-to-run spread is visible
instead of being averaged away — a tight violin means the number is
trustworthy, a wide one means the node was contended or the workload is
imbalanced across shards.  A dashed line through the origin shows what
perfect scaling from the single-GPU rate would look like.

Multiple GPU models can be overlaid by passing several --gpu-json paths;
they are drawn as separate series so different hardware can be compared on
one axis.

CPU panel
---------
Runtime against core count, both axes logarithmic, which turns ideal
scaling into a straight line of slope -1 and makes departures from it
obvious.  Three curves are drawn:

  measured     what actually happened
  Amdahl fit   the model T(n) = T1 (s + (1-s)/n) with s fitted
  ideal        T(1)/n

Parallel efficiency is annotated at each measured point, since that is the
number that decides where adding cores stops being worth it.

Usage
-----
    python benchmarks/plot_scaling.py
    python benchmarks/plot_scaling.py --out logs/benchmarks/scaling.png --dpi 300

    # Overlay several GPU models:
    python benchmarks/plot_scaling.py \\
        --gpu-json logs/benchmarks/scaling_gpu_a100.json \\
                   logs/benchmarks/scaling_gpu_v100.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_RESULTS_DIR = "logs/benchmarks"
_PALETTE = ["#2f6fb2", "#c0472b", "#3f8f5f", "#8a5fb0", "#c88a2e", "#4a4a4a"]


def _load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GPU panel
# ---------------------------------------------------------------------------

def plot_gpu(ax, datasets: list[dict]) -> None:
    import numpy as np

    all_counts: set[int] = set()

    for i, data in enumerate(datasets):
        colour = _PALETTE[i % len(_PALETTE)]
        points = data["points"]
        counts = [p["n_gpus"] for p in points]
        all_counts.update(counts)

        # Violin per point.  widths scale with x so they stay legible when the
        # GPU counts are spread out (1, 2, 4, 8).
        for p in points:
            samples = p["throughputs"]
            if len(samples) < 2:
                continue
            parts = ax.violinplot(
                [samples], positions=[p["n_gpus"]],
                widths=max(0.35, p["n_gpus"] * 0.35),
                showextrema=False, showmedians=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(colour)
                body.set_edgecolor(colour)
                body.set_alpha(0.28)

        medians = [p["median"] for p in points]
        ax.plot(counts, medians, "o-", color=colour, lw=1.8, ms=5,
                label=data.get("label", data.get("gpu_model", "GPU")), zorder=3)

        # Perfect-scaling reference anchored on this series' single-GPU rate.
        if i == 0 and counts:
            per_gpu = medians[0] / counts[0]
            xs = np.array([0, max(counts)], dtype=float)
            ax.plot(xs, per_gpu * xs, "--", color="0.55", lw=1.2,
                    label="linear scaling", zorder=1)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Throughput (light curves s$^{-1}$)")
    ax.set_title("Period-finding GPU scaling")
    ax.set_xticks(sorted(all_counts))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(frameon=False, fontsize=9, loc="upper left")


# ---------------------------------------------------------------------------
# CPU panel
# ---------------------------------------------------------------------------

def plot_cpu(ax, data: dict) -> None:
    points = data["points"]
    cores  = [p["n_cores"]    for p in points]
    meas   = [p["seconds"]    for p in points]
    ideal  = [p["ideal"]      for p in points]
    amdahl = [p["amdahl"]     for p in points]

    ax.plot(cores, meas,   "o-",  color=_PALETTE[0], lw=1.8, ms=5, label="measured")
    ax.plot(cores, amdahl, "s--", color=_PALETTE[1], lw=1.4, ms=4,
            label=f"Amdahl fit (s = {data['serial_fraction']:.3f})")
    ax.plot(cores, ideal,  ":",   color="0.45", lw=1.4, label="ideal")

    # Efficiency label at each measured point.  Offset above the marker; the
    # y axis is log so the offset has to be multiplicative.
    for p in points:
        ax.annotate(f"{p['efficiency']:.2f}",
                    xy=(p["n_cores"], p["seconds"]),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=_PALETTE[0])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of CPU cores")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Period-finding CPU scaling ({data['n_sources']} sources)")
    ax.set_xticks(cores)
    ax.set_xticklabels([str(c) for c in cores])
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.legend(frameon=False, fontsize=9)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the scaling benchmark results")
    parser.add_argument("--gpu-json", nargs="+",
                        default=[os.path.join(_RESULTS_DIR, "scaling_gpu.json")],
                        help="One or more GPU result files to overlay")
    parser.add_argument("--cpu-json",
                        default=os.path.join(_RESULTS_DIR, "scaling_cpu.json"))
    parser.add_argument("--out",
                        default=os.path.join(_RESULTS_DIR, "scaling.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")           # headless — these run on compute nodes
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting:\n"
              "    pip install 'ml4em[plots]'    # or: pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    gpu_data = [d for d in (_load(p) for p in args.gpu_json) if d]
    cpu_data = _load(args.cpu_json)

    if not gpu_data and not cpu_data:
        print(f"No scaling results found in {_RESULTS_DIR}.\n"
              "Run benchmarks/scaling_gpu.py and/or benchmarks/scaling_cpu.py first.",
              file=sys.stderr)
        sys.exit(1)

    n_panels = bool(gpu_data) + bool(cpu_data)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.6))
    axes = [axes] if n_panels == 1 else list(axes)

    i = 0
    if gpu_data:
        plot_gpu(axes[i], gpu_data)
        i += 1
    if cpu_data:
        plot_cpu(axes[i], cpu_data)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
