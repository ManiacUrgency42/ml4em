#!/usr/bin/env python3
"""
CPU strong-scaling benchmark — runtime vs. core count, with parallel efficiency.

Measures how period-finding wall time falls as more CPU cores are given to
the same fixed workload, and fits Amdahl's law to the result so the serial
fraction is explicit rather than implied.

Why each core count needs its own process
-----------------------------------------
periodfind's CPU backend parallelises with rayon.  Rayon sizes its global
thread pool once, the first time it is touched, from RAYON_NUM_THREADS.
Changing that variable inside a running process does nothing.  So every
core-count measurement is launched as a fresh subprocess with the variable
set beforehand — otherwise every point after the first would silently reuse
the first point's thread count and the curve would come out flat.

What the numbers mean
---------------------
For a fixed problem size, ideal scaling would give

    T_ideal(n) = T(1) / n

Parallel efficiency is how much of that you actually got:

    efficiency(n) = T_ideal(n) / T_measured(n)

Efficiency of 1.0 is perfect; 0.5 means half your cores are wasted.  It
decays because some of the work cannot be parallelised — array setup,
GIL-bound Python glue, memory bandwidth saturation.

Amdahl's law names that directly.  If a fraction s of the runtime is
serial, then

    T(n) = T(1) · (s + (1 - s)/n)

The script fits s by least squares over the measured points.  A small s
means "buy more cores"; a large s means the runtime is floored at
T(1) · s no matter how many cores you throw at it, and the useful request
is the core count where efficiency drops below what you consider
acceptable.

Usage
-----
    python benchmarks/scaling_cpu.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --core-counts 1 2 4 8 16 32 --trials 3

    # Then plot:
    python benchmarks/plot_scaling.py

Output
------
    logs/benchmarks/scaling_cpu.json
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

log = logging.getLogger("scaling_cpu")

_DEFAULT_CORE_COUNTS = [1, 2, 4, 8, 16, 32]
_DEFAULT_TRIALS      = 3


# ---------------------------------------------------------------------------
# Worker mode — one process, a fixed rayon thread count
# ---------------------------------------------------------------------------

def run_worker(args) -> None:
    """Time period finding on the full source set and print JSON to stdout.

    RAYON_NUM_THREADS is already set in this process's environment by the
    parent, before periodfind is imported, which is the only point at which
    rayon will honour it.
    """
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    from ml4em.config.loader import load_config

    cfg  = load_config(args.config)
    path = sc.cache_path(args.ra, args.dec, args.radius_arcsec, cfg)
    if not os.path.exists(path):
        print(json.dumps({"error": f"cache missing: {path}"}))
        sys.exit(1)

    sources = sc._load_cache(path)
    if args.max_sources and len(sources) > args.max_sources:
        sources = sources[: args.max_sources]

    seconds = sc.time_period_finding(
        sources,
        cfg,
        device     = "cpu",
        batch_size = args.batch_size or cfg.features.feature_batch_size,
        warmup     = True,
    )
    print(json.dumps({"seconds": seconds, "n_sources": len(sources)}))


# ---------------------------------------------------------------------------
# Amdahl fit
# ---------------------------------------------------------------------------

def fit_amdahl(core_counts: list[int], times: list[float]) -> tuple[float, float]:
    """Least-squares fit of T(n) = T1 * (s + (1-s)/n).

    Returns (serial_fraction, t1).  The model is linear in s once T1 is
    fixed, so T1 is taken from the single-core measurement and s is solved
    in closed form rather than by iterative optimisation.
    """
    import numpy as np

    n = np.asarray(core_counts, dtype=np.float64)
    t = np.asarray(times,       dtype=np.float64)
    t1 = float(t[0])

    # t/t1 = s + (1-s)/n  =>  (t/t1 - 1/n) = s * (1 - 1/n)
    y = t / t1 - 1.0 / n
    x = 1.0 - 1.0 / n

    mask = x > 0          # the n=1 point carries no information about s
    if not mask.any():
        return 0.0, t1

    s = float((x[mask] @ y[mask]) / (x[mask] @ x[mask]))
    return max(0.0, min(1.0, s)), t1


# ---------------------------------------------------------------------------
# Parent mode
# ---------------------------------------------------------------------------

def _run_point(args, n_cores: int) -> float:
    """Launch one subprocess pinned to n_cores rayon threads; return seconds."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(n_cores)
    # Keep BLAS/OpenMP from opening a second, unrelated thread pool that would
    # contend for the same cores and pollute the measurement.
    env["OMP_NUM_THREADS"]        = str(n_cores)
    env["OPENBLAS_NUM_THREADS"]   = str(n_cores)
    env["MKL_NUM_THREADS"]        = str(n_cores)

    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--worker",
        "--config",        args.config,
        "--ra",            str(args.ra),
        "--dec",           str(args.dec),
        "--radius-arcsec", str(args.radius_arcsec),
    ]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.max_sources:
        cmd += ["--max-sources", str(args.max_sources)]

    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"CPU worker at {n_cores} core(s) failed (exit {p.returncode}):"
            f"\n{p.stderr[-2000:]}"
        )
    try:
        result = json.loads(p.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise RuntimeError(
            f"CPU worker at {n_cores} core(s) produced no JSON:"
            f"\n{p.stdout}\n{p.stderr[-2000:]}"
        )
    if "error" in result:
        raise RuntimeError(f"CPU worker at {n_cores} core(s): {result['error']}")
    return float(result["seconds"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU strong-scaling benchmark for period finding"
    )
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--ra",            type=float, default=116.7)
    parser.add_argument("--dec",           type=float, default=36.2)
    parser.add_argument("--radius-arcsec", type=float, default=1800.0)
    parser.add_argument("--core-counts",   type=int, nargs="+",
                        default=_DEFAULT_CORE_COUNTS,
                        help="Core counts to measure (clipped to available cores)")
    parser.add_argument("--trials",        type=int, default=_DEFAULT_TRIALS,
                        help="Repeat measurements per core count (default: 3)")
    parser.add_argument("--batch-size",    type=int, default=None,
                        help="Override features.feature_batch_size")
    parser.add_argument("--max-sources",   type=int, default=None,
                        help="Cap the source count — single-core runs are slow")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Re-fetch light curves instead of using the cache")
    parser.add_argument("--label",         default=None,
                        help="Series label for the plot")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    sc.setup_logging()

    from ml4em.config.loader import load_config
    from ml4em.config import get_ztf_token

    cfg   = load_config(args.config)
    token = get_ztf_token()

    # SLURM restricts the process to --cpus-per-task, so sched_getaffinity is
    # the honest ceiling; os.cpu_count() reports the whole node.
    n_avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
              else (os.cpu_count() or 1)

    counts = sorted({c for c in args.core_counts if 1 <= c <= n_avail})
    if not counts:
        log.error("None of --core-counts %s fit in the %d available core(s).",
                  args.core_counts, n_avail)
        sys.exit(1)
    if max(args.core_counts) > n_avail:
        log.warning("Clipping --core-counts to %d available core(s): %s",
                    n_avail, counts)

    sources = sc.load_or_fetch_sources(
        args.ra, args.dec, args.radius_arcsec, cfg, token, refresh=args.refresh_cache
    )
    n_sources = min(len(sources), args.max_sources) if args.max_sources else len(sources)
    batch_size = args.batch_size or cfg.features.feature_batch_size

    log.info("CPU scaling: %d sources | %d core(s) available | batch=%d | %d trials",
             n_sources, n_avail, batch_size, args.trials)
    if 1 in counts:
        log.info("Single-core point runs the whole workload serially — "
                 "use --max-sources if this is too slow.")

    # ── Sweep ────────────────────────────────────────────────────────────────
    points: list[dict] = []
    t_start = time.perf_counter()

    for n_cores in counts:
        times = [_run_point(args, n_cores) for _ in range(args.trials)]
        best  = min(times)   # min, not mean: noise on a shared node only adds time

        log.info("  %3d core(s)  best %8.2fs  (trials: %s)",
                 n_cores, best, ", ".join(f"{t:.1f}" for t in times))

        points.append({
            "n_cores": n_cores,
            "times":   times,
            "seconds": best,
            "min":     min(times),
            "max":     max(times),
        })

    # ── Efficiency and Amdahl fit ────────────────────────────────────────────
    core_counts = [p["n_cores"] for p in points]
    times       = [p["seconds"] for p in points]
    t1          = times[0]

    for p in points:
        ideal            = t1 / p["n_cores"]
        p["ideal"]       = ideal
        p["speedup"]     = t1 / p["seconds"]
        p["efficiency"]  = ideal / p["seconds"]

    serial_frac, _ = fit_amdahl(core_counts, times)
    for p in points:
        p["amdahl"] = t1 * (serial_frac + (1.0 - serial_frac) / p["n_cores"])

    max_speedup = (1.0 / serial_frac) if serial_frac > 0 else float("inf")

    payload = {
        "benchmark":       "scaling_cpu",
        "label":           args.label or f"{n_sources} sources",
        "n_sources":       n_sources,
        "n_cores_avail":   n_avail,
        "batch_size":      batch_size,
        "trials":          args.trials,
        "region":          {"ra": args.ra, "dec": args.dec,
                            "radius_arcsec": args.radius_arcsec},
        "serial_fraction": serial_frac,
        "max_speedup":     None if serial_frac == 0 else max_speedup,
        "t1_seconds":      t1,
        "elapsed_total":   time.perf_counter() - t_start,
        "points":          points,
    }
    out = sc.write_results("scaling_cpu", payload)

    # ── Summary ──────────────────────────────────────────────────────────────
    sep = "─" * 72
    print(f"\n  CPU scaling  |  {n_sources} sources  |  batch={batch_size}")
    print(f"\n{sep}")
    print(f"  {'Cores':>5}  {'Runtime':>10}  {'Ideal':>10}  {'Amdahl':>10}"
          f"  {'Speedup':>8}  {'Efficiency':>10}")
    print(sep)
    for p in points:
        print(f"  {p['n_cores']:>5}  {p['seconds']:>9.1f}s  {p['ideal']:>9.1f}s"
              f"  {p['amdahl']:>9.1f}s  {p['speedup']:>7.2f}x  {p['efficiency']:>9.0%}")
    print(sep)
    print(f"\n  Amdahl serial fraction: {serial_frac:.4f}")
    if serial_frac > 0:
        print(f"  Theoretical max speedup: {max_speedup:.1f}x "
              f"(runtime floor {t1 * serial_frac:.1f}s)")
    knee = [p for p in points if p["efficiency"] >= 0.7]
    if knee:
        print(f"  Highest core count above 70% efficiency: {knee[-1]['n_cores']}")
    print(f"\n  Wrote {out}")
    print(f"  Plot with: python benchmarks/plot_scaling.py\n")


if __name__ == "__main__":
    main()
