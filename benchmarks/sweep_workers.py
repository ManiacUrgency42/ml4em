#!/usr/bin/env python3
"""
n_workers sweep — find the optimal Kowalski parallelism for LC fetch.

Runs the Kowalski find query (round trip 2) repeatedly with increasing
n_workers values on the same set of source IDs.  The near query (round
trip 1) runs once to get IDs, then the find query is re-run for each
n_workers value.

What n_workers controls
-----------------------
fetch_batch() chunks source IDs into slices of limit_per_query (1000),
then sends n_workers chunks simultaneously per sliding-window iteration.
More workers = fewer iterations = less time waiting on network latency.

Three ceilings limit the benefit of more workers:
  1. Kowalski server throttling — shared resource, too many parallel queries
     degrade or get deprioritised.
  2. Network bandwidth — simultaneous large responses can saturate the NIC.
  3. Thread overhead — more OS threads than CPU cores adds context switching.

Use this sweep to find the knee of the throughput curve.  The n_workers
where throughput stops climbing is your ceiling — set that in config.yaml.

Usage
-----
    python benchmarks/sweep_workers.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800

    # Custom worker values:
    python benchmarks/sweep_workers.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --workers 1 2 4 8 16 32

Example output
--------------
    Region: RA=116.7  Dec=36.2  radius=1800 arcsec  |  1247 source IDs

    ─────────────────────────────────────────────────────────────────
      n_workers    Iterations    Find time (s)    Throughput    Speedup
    ─────────────────────────────────────────────────────────────────
              1          1247          87.3s          14/s        1.0x
              2           624          45.1s          28/s        1.9x
              4           312          24.3s          51/s        3.6x
              8           156          12.7s          98/s        6.9x   ← plateau
             16            78          12.1s         103/s        7.3x
             32            39          13.8s          90/s        6.3x   ← degraded
    ─────────────────────────────────────────────────────────────────
    Recommended: n_workers = 8  (highest throughput before plateau/degradation)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_WORKERS = [1, 2, 4, 8, 16]
_DEFAULT_RADIUS  = 1800.0


def main():
    parser = argparse.ArgumentParser(
        description="Sweep n_workers to find the Kowalski LC fetch throughput ceiling"
    )
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--ra",            type=float, default=116.7)
    parser.add_argument("--dec",           type=float, default=36.2)
    parser.add_argument("--radius-arcsec", type=float, default=_DEFAULT_RADIUS)
    parser.add_argument("--workers",       type=int, nargs="+", default=_DEFAULT_WORKERS,
                        help="Worker counts to sweep (default: 1 2 4 8 16)")
    args = parser.parse_args()

    from ml4em.config.loader import load_config
    from ml4em.config import get_ztf_token
    from ml4em.data.ztf import ZTFSource

    cfg   = load_config(args.config)
    token = get_ztf_token()

    # ── Round trip 1: near query — runs once ──────────────────────────────────
    log.info("Near query: RA=%.4f Dec=%.4f radius=%.0f arcsec",
             args.ra, args.dec, args.radius_arcsec)

    cfg.sources.ztf.n_workers = 1
    ztf = ZTFSource(cfg.sources.ztf, token)

    source_ids = ztf.near_ids(args.ra, args.dec, args.radius_arcsec)

    if not source_ids:
        log.error("No sources found. Try a larger --radius-arcsec.")
        sys.exit(1)

    log.info("Got %d source IDs — starting worker sweep", len(source_ids))
    limit = cfg.sources.ztf.limit_per_query

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results: list[tuple[int, int, float]] = []   # (n_workers, n_iters, t_find)

    for n_workers in sorted(args.workers):
        # Modify n_workers on the existing ZTFSource (reuses Kowalski connection)
        ztf._cfg.n_workers = n_workers
        import math
        n_chunks = math.ceil(len(source_ids) / limit)
        n_iters  = math.ceil(n_chunks / n_workers)

        log.info("n_workers=%d  (%d chunks, %d iterations)...", n_workers, n_chunks, n_iters)
        t0 = time.perf_counter()
        ztf.fetch_batch(source_ids)
        t_find = time.perf_counter() - t0

        log.info("  → %.3fs  (%.0f sources/s)", t_find, len(source_ids) / t_find)
        results.append((n_workers, n_iters, t_find))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ids    = len(source_ids)
    t_base   = results[0][2]   # single-worker time as baseline for speedup
    best_idx = min(range(len(results)), key=lambda i: results[i][2])

    sep = "─" * 67
    print(f"\n  Region: RA={args.ra}  Dec={args.dec}  radius={args.radius_arcsec:.0f} arcsec"
          f"  |  {n_ids} source IDs  |  limit_per_query={limit}")
    print(f"\n{sep}")
    print(f"  {'n_workers':>9}  {'Iterations':>10}  {'Find time':>12}  {'Throughput':>11}  {'Speedup':>8}")
    print(sep)

    for i, (n_workers, n_iters, t_find) in enumerate(results):
        marker = "  ← best" if i == best_idx else ""
        print(f"  {n_workers:>9}  {n_iters:>10}  {t_find:>10.1f}s"
              f"  {n_ids / t_find:>8.0f}/s  {t_base / t_find:>7.1f}x{marker}")

    print(sep)
    best_workers = results[best_idx][0]
    print(f"\n  Recommended: n_workers = {best_workers}")
    print(f"  Set in config.yaml:  sources.ztf.n_workers: {best_workers}")
    print(f"                       features.catalog.n_workers: {best_workers}")
    print()


if __name__ == "__main__":
    main()
