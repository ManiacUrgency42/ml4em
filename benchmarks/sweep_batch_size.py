#!/usr/bin/env python3
"""
feature_batch_size sweep — find the optimal GPU batch size for period finding.

Fetches a real ZTF region once, then runs period finding repeatedly with
increasing feature_batch_size values on the same light curves.

What feature_batch_size controls
---------------------------------
The feature pipeline chunks the source list into batches of feature_batch_size
before calling periodfind.  Larger batches:
  - Better GPU lane utilization (more parallelism per CUDA kernel launch)
  - Higher VRAM usage (light curve arrays for all sources in the batch)

The goal is the largest batch size that fits in GPU memory.  If you exceed
VRAM, periodfind raises an OOM error — the last successful size is your ceiling.

Relationship to n_workers
--------------------------
feature_batch_size and n_workers are independent.  n_workers controls how many
Kowalski queries run in parallel (network).  feature_batch_size controls how
many sources go into each periodfind GPU call (compute).  Tune them separately.

Usage
-----
    python benchmarks/sweep_batch_size.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --device cuda

    # Custom batch sizes:
    python benchmarks/sweep_batch_size.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --device cuda \\
        --batch-sizes 100 250 500 1000 2000 4000

Example output
--------------
    Region: RA=116.7  Dec=36.2  radius=1800 arcsec  |  1052 valid sources
    Device: cuda  |  Algorithms: CE, AOV, LS, MHF

    ────────────────────────────────────────────────────────────────────
      batch_size    Period time (s)    Throughput    vs batch=100
    ────────────────────────────────────────────────────────────────────
             100             62.1s          17/s           1.0x
             250             28.4s          37/s           2.2x
             500             16.9s          62/s           3.7x
            1000             14.3s          74/s           4.4x  ← best
            2000             14.1s          75/s           4.4x
            4000          OOM — VRAM exceeded
    ────────────────────────────────────────────────────────────────────
    Recommended: feature_batch_size = 1000  (last stable before plateau/OOM)
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

_DEFAULT_BATCH_SIZES = [100, 250, 500, 1000, 2000]
_DEFAULT_RADIUS      = 1800.0


def main():
    parser = argparse.ArgumentParser(
        description="Sweep feature_batch_size to find GPU throughput ceiling"
    )
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--ra",            type=float, default=116.7)
    parser.add_argument("--dec",           type=float, default=36.2)
    parser.add_argument("--radius-arcsec", type=float, default=_DEFAULT_RADIUS)
    parser.add_argument("--device",        default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch-sizes",   type=int, nargs="+", default=_DEFAULT_BATCH_SIZES,
                        help="Batch sizes to sweep (default: 100 250 500 1000 2000)")
    parser.add_argument("--n-workers",     type=int, default=None,
                        help="n_workers for the initial LC fetch (overrides config)")
    args = parser.parse_args()

    from ml4em.config.loader import load_config
    from ml4em.config import get_ztf_token
    from ml4em.data.ztf import ZTFSource
    from collections import defaultdict
    import periodfind
    from ml4em.features.period import PeriodExtractor

    cfg   = load_config(args.config)
    token = get_ztf_token()
    if args.n_workers is not None:
        cfg.sources.ztf.n_workers = args.n_workers

    # ── Fetch LCs once ────────────────────────────────────────────────────────
    log.info("Fetching LCs: RA=%.4f Dec=%.4f radius=%.0f arcsec",
             args.ra, args.dec, args.radius_arcsec)

    ztf = ZTFSource(cfg.sources.ztf, token)
    source_ids, lcs = ztf.fetch_by_region(args.ra, args.dec, args.radius_arcsec)

    if not lcs:
        log.error("No light curves returned. Try a larger --radius-arcsec.")
        sys.exit(1)

    # Group and filter by min_observations
    groups: dict = defaultdict(list)
    for lc in lcs:
        groups[lc.source_id].append(lc)
    sources = [groups.get(sid, []) for sid in source_ids]
    valid   = [s for s in sources if s and max(lc.n_obs for lc in s) >= cfg.features.min_observations]
    n_valid = len(valid)
    log.info("%d valid sources (>= %d obs)", n_valid, cfg.features.min_observations)

    if n_valid == 0:
        log.error("No valid sources for feature extraction.")
        sys.exit(1)

    # ── GPU warmup ────────────────────────────────────────────────────────────
    log.info("GPU warmup (not timed)...")
    periodfind.set_device(args.device)
    PeriodExtractor(cfg.features.period).extract(valid[:min(50, n_valid)])
    log.info("Warmup complete")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results: list[tuple[int, float | str]] = []   # (batch_size, time_or_"OOM")

    for batch_size in sorted(args.batch_sizes):
        log.info("batch_size=%d ...", batch_size)
        try:
            ext = PeriodExtractor(cfg.features.period)
            t0  = time.perf_counter()
            for i in range(0, n_valid, batch_size):
                ext.extract(valid[i : i + batch_size])
            t_period = time.perf_counter() - t0
            log.info("  → %.3fs  (%.0f sources/s)", t_period, n_valid / t_period)
            results.append((batch_size, t_period))
        except (RuntimeError, MemoryError) as exc:
            log.warning("  → OOM at batch_size=%d: %s", batch_size, exc)
            results.append((batch_size, "OOM"))
            break   # larger batch sizes will also OOM

    # ── Summary ───────────────────────────────────────────────────────────────
    valid_results = [(bs, t) for bs, t in results if isinstance(t, float)]
    best_idx = min(range(len(valid_results)), key=lambda i: valid_results[i][1]) if valid_results else 0
    t_base = valid_results[0][1] if valid_results else 1.0

    sep = "─" * 68
    print(f"\n  Region: RA={args.ra}  Dec={args.dec}  radius={args.radius_arcsec:.0f} arcsec"
          f"  |  {n_valid} valid sources")
    print(f"  Device: {args.device}  |  Algorithms: {', '.join(cfg.features.period.algorithms)}")
    print(f"\n{sep}")
    print(f"  {'batch_size':>10}  {'Period time':>14}  {'Throughput':>11}  {'vs batch=' + str(results[0][0]):>14}")
    print(sep)

    for i, (batch_size, result) in enumerate(results):
        if isinstance(result, str):
            print(f"  {batch_size:>10}  {'OOM — VRAM exceeded':>14}")
        else:
            marker = "  ← best" if i == best_idx else ""
            print(f"  {batch_size:>10}  {result:>12.1f}s"
                  f"  {n_valid / result:>8.0f}/s  {t_base / result:>12.1f}x{marker}")

    print(sep)
    if valid_results:
        best_bs = valid_results[best_idx][0]
        print(f"\n  Recommended: feature_batch_size = {best_bs}")
        print(f"  Set in config.yaml:  features.feature_batch_size: {best_bs}")
    print()


if __name__ == "__main__":
    main()
