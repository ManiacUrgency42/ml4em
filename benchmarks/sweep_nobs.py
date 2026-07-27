#!/usr/bin/env python3
"""
n_obs vs period finding time — understand how LC length drives GPU compute.

Fetches a real ZTF region, groups light curves into n_obs buckets, then
times period finding separately for each bucket.  This tells you how period
finding time scales with the number of observations per light curve.

Why this matters
----------------
Period finding (GPU) scales with n_obs because each trial period requires
evaluating all observations.  A source with 2000 obs takes significantly
longer than one with 200 obs.  The n_obs distribution of your target sky
region determines how long your production MSI runs will take.

After running this benchmark you can:
  - Identify the dominant n_obs range in your target fields
  - Estimate per-source compute from the ms/source column
  - Project total GPU-hours: (n_sources × ms_per_source) / 3600000

Usage
-----
    python benchmarks/sweep_nobs.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --device cuda

    # Custom bucket boundaries:
    python benchmarks/sweep_nobs.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --device cuda \\
        --buckets 50 100 200 500 1000 2000

Example output
--------------
    Region: RA=116.7  Dec=36.2  radius=1800 arcsec  |  1052 valid sources
    Device: cuda  |  Algorithms: CE, AOV, LS, MHF

    ──────────────────────────────────────────────────────────────────────
      n_obs bucket    Sources    Period time (s)    ms / source    % of total
    ──────────────────────────────────────────────────────────────────────
        50 – 100          87             0.41s            4.7         2.9%
       100 – 200         213             1.83s            8.6        12.8%
       200 – 500         418             5.12s           12.2        35.8%
       500 – 1000        287             5.63s           19.6        39.4%
      1000 – 2000         47             1.29s           27.4         9.0%
    ──────────────────────────────────────────────────────────────────────
      Total             1052            14.28s           13.6       100.0%
    ──────────────────────────────────────────────────────────────────────

    Compute projection (period finding only):
      At 1,000,000 sources with this n_obs distribution:
        GPU time = ~3.8 GPU-hours
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_BUCKETS = [50, 100, 200, 500, 1000, 2000, 999_999]
_DEFAULT_RADIUS  = 1800.0


def main():
    parser = argparse.ArgumentParser(
        description="Measure period finding time vs n_obs on real ZTF light curves"
    )
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--ra",            type=float, default=116.7)
    parser.add_argument("--dec",           type=float, default=36.2)
    parser.add_argument("--radius-arcsec", type=float, default=_DEFAULT_RADIUS)
    parser.add_argument("--device",        default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--buckets",       type=int, nargs="+", default=_DEFAULT_BUCKETS,
                        help="n_obs bucket boundaries (default: 50 100 200 500 1000 2000)")
    parser.add_argument("--n-workers",     type=int, default=None)
    args = parser.parse_args()

    from ml4em.config.loader import load_config
    from ml4em.config import get_ztf_token
    from ml4em.data.ztf import ZTFSource
    from ml4em.features.period import PeriodExtractor
    from collections import defaultdict
    import periodfind

    cfg   = load_config(args.config)
    token = get_ztf_token()
    if args.n_workers is not None:
        cfg.sources.ztf.n_workers = args.n_workers

    # ── Fetch LCs ─────────────────────────────────────────────────────────────
    log.info("Fetching LCs: RA=%.4f Dec=%.4f radius=%.0f arcsec",
             args.ra, args.dec, args.radius_arcsec)

    ztf = ZTFSource(cfg.sources.ztf, token)
    source_ids, lcs = ztf.fetch_by_region(args.ra, args.dec, args.radius_arcsec)

    if not lcs:
        log.error("No light curves returned. Try a larger --radius-arcsec.")
        sys.exit(1)

    groups: dict = defaultdict(list)
    for lc in lcs:
        groups[lc.source_id].append(lc)
    sources = [groups.get(sid, []) for sid in source_ids]
    valid   = [s for s in sources if s and max(lc.n_obs for lc in s) >= cfg.features.min_observations]
    n_valid = len(valid)
    log.info("%d valid sources", n_valid)

    if n_valid == 0:
        log.error("No valid sources.")
        sys.exit(1)

    # ── Sort into n_obs buckets ───────────────────────────────────────────────
    # Use the primary band (most observations) to determine n_obs per source
    def primary_n_obs(lcs_list):
        return max(lc.n_obs for lc in lcs_list)

    boundaries = sorted(set(args.buckets))
    if boundaries[0] > cfg.features.min_observations:
        boundaries = [cfg.features.min_observations] + boundaries

    buckets: list[list] = [[] for _ in range(len(boundaries))]
    for src_lcs in valid:
        n = primary_n_obs(src_lcs)
        # Find which bucket this source falls in
        placed = False
        for b_idx in range(len(boundaries) - 1):
            if boundaries[b_idx] <= n < boundaries[b_idx + 1]:
                buckets[b_idx].append(src_lcs)
                placed = True
                break
        if not placed:
            buckets[-1].append(src_lcs)

    # Remove empty buckets
    filled = [(boundaries[i], boundaries[i + 1] if i + 1 < len(boundaries) else None,
               buckets[i])
              for i in range(len(boundaries))
              if buckets[i]]

    log.info("n_obs distribution:")
    for lo, hi, bucket in filled:
        hi_str = str(hi) if hi and hi < 999_999 else "+"
        log.info("  [%d – %s): %d sources", lo, hi_str, len(bucket))

    # ── GPU warmup ────────────────────────────────────────────────────────────
    log.info("GPU warmup (not timed)...")
    periodfind.set_device(args.device)
    PeriodExtractor(cfg.features.period).extract(valid[:min(50, n_valid)])
    log.info("Warmup complete")

    # ── Time period finding per bucket ────────────────────────────────────────
    bucket_results = []   # (label, n_sources, t_period)

    for lo, hi, bucket in filled:
        hi_str = str(hi) if hi and hi < 999_999 else "+"
        label  = f"{lo:>6} – {hi_str}"
        n      = len(bucket)
        log.info("Timing bucket [%s): %d sources...", label.strip(), n)

        ext = PeriodExtractor(cfg.features.period)
        t0  = time.perf_counter()
        ext.extract(bucket)
        t_period = time.perf_counter() - t0

        log.info("  → %.3fs  (%.1f ms/source)", t_period, t_period / n * 1000)
        bucket_results.append((label, n, t_period))

    # ── Summary ───────────────────────────────────────────────────────────────
    t_total_period = sum(t for _, _, t in bucket_results)
    ms_per_src_overall = t_total_period / n_valid * 1000

    sep = "─" * 70
    print(f"\n  Region: RA={args.ra}  Dec={args.dec}  radius={args.radius_arcsec:.0f} arcsec"
          f"  |  {n_valid} valid sources")
    print(f"  Device: {args.device}  |  Algorithms: {', '.join(cfg.features.period.algorithms)}")
    print(f"\n{sep}")
    print(f"  {'n_obs bucket':>14}  {'Sources':>8}  {'Period time':>14}  {'ms / source':>12}  {'% of total':>10}")
    print(sep)

    for label, n, t_period in bucket_results:
        ms_per = t_period / n * 1000
        pct    = t_period / t_total_period * 100 if t_total_period > 0 else 0
        print(f"  {label:>14}  {n:>8}  {t_period:>12.2f}s  {ms_per:>12.1f}  {pct:>9.1f}%")

    print(sep)
    print(f"  {'Total':>14}  {n_valid:>8}  {t_total_period:>12.2f}s"
          f"  {ms_per_src_overall:>12.1f}  {'100.0%':>10}")
    print(sep)

    # ── Compute projection ────────────────────────────────────────────────────
    print(f"\n  Compute projection (period finding only):")
    for scale in [100_000, 1_000_000, 10_000_000]:
        gpu_hours = scale * ms_per_src_overall / 1000 / 3600
        print(f"    {scale:>12,} sources → ~{gpu_hours:.1f} GPU-hours")
    print(f"\n  Note: projection assumes same n_obs distribution as this region.")
    print(f"        Actual time scales with n_obs — check bucket breakdown above.")
    print()


if __name__ == "__main__":
    main()
