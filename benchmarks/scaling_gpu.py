#!/usr/bin/env python3
"""
GPU strong-scaling benchmark — light curves/second vs. number of GPUs.

Measures aggregate period-finding throughput as a function of how many GPUs
are working on the problem, with enough repeat trials to show the run-to-run
spread rather than a single lucky number.

How the measurement works
-------------------------
periodfind holds one CUDA context per process, so "use N GPUs" means
"run N processes, each pinned to one GPU".  The parent process:

  1. Fetches the benchmark region once and caches it to a .npz.  Nothing
     after this point touches the network.
  2. For each GPU count N, splits the cached sources into N contiguous
     shards and launches N subprocesses, each with CUDA_VISIBLE_DEVICES
     set to a single distinct device.
  3. Waits for all N to finish.  Aggregate throughput is

         total_sources / max(worker_wall_times)

     — the *slowest* worker, not the mean, because the job is not done
     until the last shard is done.  Using the mean would inflate the
     numbers and hide load imbalance.
  4. Repeats --trials times so the plot can show a distribution.

Every worker runs a GPU warmup batch that is excluded from its timer, so
CUDA JIT compilation does not land in the measured region.

Why throughput and not speedup
------------------------------
Speedup curves hide the absolute rate.  The number that matters is light
curves per second per GPU, because that is what converts a source count
into GPU-hours.  Perfect scaling shows up as a straight line through the
origin; any flattening is the point where adding GPUs stops paying.

Usage
-----
    # On a node with 4 GPUs:
    python benchmarks/scaling_gpu.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --gpu-counts 1 2 4 --trials 5

    # Then plot:
    python benchmarks/plot_scaling.py

Output
------
    logs/benchmarks/scaling_gpu.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _scaling_common as sc

log = logging.getLogger("scaling_gpu")

_DEFAULT_GPU_COUNTS = [1, 2, 4, 8]
_DEFAULT_TRIALS     = 5


# ---------------------------------------------------------------------------
# Worker mode — one process, one GPU, one shard
# ---------------------------------------------------------------------------

def run_worker(args) -> None:
    """Time period finding on one shard and print the result as JSON.

    Runs inside a subprocess with CUDA_VISIBLE_DEVICES already narrowed to
    a single device by the parent, so periodfind sees exactly one GPU.
    Output goes to stdout as a single JSON line; all logging goes to stderr
    so the parent can parse stdout unambiguously.
    """
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    from ml4em.config.loader import load_config

    cfg  = load_config(args.config)
    path = sc.cache_path(args.ra, args.dec, args.radius_arcsec, cfg)
    if not os.path.exists(path):
        print(json.dumps({"error": f"cache missing: {path}"}))
        sys.exit(1)

    sources = sc._load_cache(path)

    # Contiguous shard.  Sources are ordered as Kowalski returned them, which
    # mixes n_obs randomly, so contiguous slices are already load-balanced in
    # expectation — no need to interleave.
    n     = len(sources)
    lo    = (n * args.shard_index) // args.n_shards
    hi    = (n * (args.shard_index + 1)) // args.n_shards
    shard = sources[lo:hi]

    if not shard:
        print(json.dumps({"seconds": 0.0, "n_sources": 0}))
        return

    seconds = sc.time_period_finding(
        shard,
        cfg,
        device     = "gpu",
        batch_size = args.batch_size or cfg.features.feature_batch_size,
        warmup     = True,
    )
    print(json.dumps({"seconds": seconds, "n_sources": len(shard)}))


# ---------------------------------------------------------------------------
# Parent mode — orchestrate the sweep
# ---------------------------------------------------------------------------

def _launch_shards(args, n_gpus: int) -> list[dict]:
    """Run one trial at a given GPU count; return each worker's result dict."""
    procs = []
    for i in range(n_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker",
            "--shard-index",   str(i),
            "--n-shards",      str(n_gpus),
            "--config",        args.config,
            "--ra",            str(args.ra),
            "--dec",           str(args.dec),
            "--radius-arcsec", str(args.radius_arcsec),
        ]
        if args.batch_size:
            cmd += ["--batch-size", str(args.batch_size)]

        procs.append(subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ))

    results = []
    for i, p in enumerate(procs):
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(
                f"GPU worker {i} failed (exit {p.returncode}):\n{err[-2000:]}"
            )
        try:
            results.append(json.loads(out.strip().splitlines()[-1]))
        except (json.JSONDecodeError, IndexError):
            raise RuntimeError(f"GPU worker {i} produced no JSON:\n{out}\n{err[-2000:]}")

    for i, r in enumerate(results):
        if "error" in r:
            raise RuntimeError(f"GPU worker {i}: {r['error']}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU strong-scaling benchmark for period finding"
    )
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--ra",            type=float, default=116.7)
    parser.add_argument("--dec",           type=float, default=36.2)
    parser.add_argument("--radius-arcsec", type=float, default=1800.0)
    parser.add_argument("--gpu-counts",    type=int, nargs="+", default=_DEFAULT_GPU_COUNTS,
                        help="GPU counts to measure (clipped to what is visible)")
    parser.add_argument("--trials",        type=int, default=_DEFAULT_TRIALS,
                        help="Repeat measurements per GPU count (default: 5)")
    parser.add_argument("--batch-size",    type=int, default=None,
                        help="Override features.feature_batch_size")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Re-fetch light curves instead of using the cache")
    parser.add_argument("--label",         default=None,
                        help="Series label for the plot (default: detected GPU model)")

    # Worker-mode flags — not for interactive use.
    parser.add_argument("--worker",      action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0,  help=argparse.SUPPRESS)
    parser.add_argument("--n-shards",    type=int, default=1,  help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    sc.setup_logging()

    from ml4em.config.loader import load_config
    from ml4em.config import get_ztf_token

    cfg   = load_config(args.config)
    token = get_ztf_token()

    n_visible = sc.n_visible_gpus()
    if n_visible == 0:
        log.error("No GPUs visible (nvidia-smi found none). "
                  "Run scaling_cpu.py instead.")
        sys.exit(1)

    counts = sorted({c for c in args.gpu_counts if 1 <= c <= n_visible})
    if not counts:
        log.error("None of --gpu-counts %s fit in the %d visible GPU(s).",
                  args.gpu_counts, n_visible)
        sys.exit(1)
    if max(args.gpu_counts) > n_visible:
        log.warning("Clipping --gpu-counts to %d visible GPU(s): %s",
                    n_visible, counts)

    # ── Fetch once, before anything is timed ─────────────────────────────────
    sources = sc.load_or_fetch_sources(
        args.ra, args.dec, args.radius_arcsec, cfg, token, refresh=args.refresh_cache
    )
    n_sources  = len(sources)
    batch_size = args.batch_size or cfg.features.feature_batch_size
    device     = sc.gpu_name()
    label      = args.label or device

    log.info("GPU scaling: %d sources | %d GPU(s) visible (%s) | batch=%d | %d trials",
             n_sources, n_visible, device, batch_size, args.trials)

    # ── Sweep ────────────────────────────────────────────────────────────────
    points: list[dict] = []
    t_start = time.perf_counter()

    for n_gpus in counts:
        throughputs: list[float] = []

        for trial in range(args.trials):
            workers  = _launch_shards(args, n_gpus)
            makespan = max(w["seconds"] for w in workers)
            thr      = n_sources / makespan if makespan > 0 else 0.0
            throughputs.append(thr)

            log.info("  %d GPU(s)  trial %d/%d  makespan %6.2fs  → %6.1f LC/s",
                     n_gpus, trial + 1, args.trials, makespan, thr)

        points.append({
            "n_gpus":      n_gpus,
            "throughputs": throughputs,
            "median":      float(sorted(throughputs)[len(throughputs) // 2]),
            "min":         min(throughputs),
            "max":         max(throughputs),
        })

    payload = {
        "benchmark":     "scaling_gpu",
        "label":         label,
        "gpu_model":     device,
        "n_gpus_visible": n_visible,
        "n_sources":     n_sources,
        "batch_size":    batch_size,
        "trials":        args.trials,
        "region":        {"ra": args.ra, "dec": args.dec,
                          "radius_arcsec": args.radius_arcsec},
        "elapsed_total": time.perf_counter() - t_start,
        "points":        points,
    }
    out = sc.write_results("scaling_gpu", payload)

    # ── Summary ──────────────────────────────────────────────────────────────
    base = points[0]["median"]
    sep  = "─" * 68
    print(f"\n  GPU scaling  |  {n_sources} sources  |  {device}  |  batch={batch_size}")
    print(f"\n{sep}")
    print(f"  {'GPUs':>5}  {'Median LC/s':>12}  {'Spread':>16}  {'Speedup':>8}  {'Efficiency':>10}")
    print(sep)
    for p in points:
        spread = f"{p['min']:.1f}–{p['max']:.1f}"
        speed  = p["median"] / base if base else 0.0
        eff    = speed / p["n_gpus"]
        print(f"  {p['n_gpus']:>5}  {p['median']:>12.1f}  {spread:>16}"
              f"  {speed:>7.2f}x  {eff:>9.0%}")
    print(sep)
    print(f"\n  Per-GPU rate at {points[-1]['n_gpus']} GPUs: "
          f"{points[-1]['median'] / points[-1]['n_gpus']:.1f} LC/s/GPU")
    print(f"  Wrote {out}")
    print(f"  Plot with: python benchmarks/plot_scaling.py\n")


if __name__ == "__main__":
    main()
