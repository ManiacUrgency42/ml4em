#!/usr/bin/env python3
"""
Single-source latency benchmark.

Times each pipeline stage for one ZTF source independently.  Use this to
understand where time goes for a single source and to compare CPU vs GPU
for period finding.

This is NOT a throughput benchmark — single-source latency does not predict
batch throughput because GPU utilization is near zero on one source.  Use
batch_throughput.py for MSI compute estimates.

Usage
-----
    # Default position (known ZTF WDB candidate):
    python benchmarks/single_latency.py --config config.yaml

    # CPU vs GPU comparison:
    python benchmarks/single_latency.py --config config.yaml --device cpu
    python benchmarks/single_latency.py --config config.yaml --device cuda

Example output
--------------
    ──────────────────────────────────────────
      Stage                        Time
    ──────────────────────────────────────────
      Kowalski connect             0.31s
      Fetch (cone search)          2.14s
    ──────────────────────────────────────────
      Statistics                   0.02s
      Period finding               18.43s
      dm/dt histogram              0.08s
      Gaia xmatch                  1.21s
    ──────────────────────────────────────────
      Total feature                19.74s
      Total end-to-end             22.19s
    ──────────────────────────────────────────

      Source:       1234567890
      Bands:        g, r, i
      Max obs:      412
      Device:       cpu
      Algorithms:   CE, AOV, LS, MHF
      Period grid:  8240 points  (freq-spaced, spp=10, baseline=728.3d)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _scaling_common as sc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_RA             = 2.569364   # confirmed 67-pt r-band source in ZTF_sources_20240515
_DEFAULT_DEC            = -22.4367011
_DEFAULT_RADIUS_ARCSEC  = 2.0


def _fmt(seconds: float) -> str:
    return f"{seconds:8.3f}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-source latency benchmark — fetch + feature extraction"
    )
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--ra",       type=float, default=_DEFAULT_RA)
    parser.add_argument("--dec",      type=float, default=_DEFAULT_DEC)
    parser.add_argument("--radius",   type=float, default=_DEFAULT_RADIUS_ARCSEC,
                        help="Cone search radius in arcsec")
    parser.add_argument("--device",   default=None, choices=["cpu", "cuda", "auto"])
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated list, e.g. CE,AOV,LS")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    from ml4em.config.loader import load_config, get_ztf_token
    cfg = load_config(args.config)
    if args.device:
        cfg.features.device = args.device
    if args.algorithms:
        cfg.features.period.algorithms = args.algorithms.split(",")
    device = cfg.features.device

    log.info("device=%s  ra=%.4f  dec=%.4f  radius=%.1f\"", device, args.ra, args.dec, args.radius)

    # ── Kowalski connection ───────────────────────────────────────────────────
    from ml4em.data.ztf import ZTFSource
    token = get_ztf_token()

    t0 = time.perf_counter()
    ztf = ZTFSource(cfg.sources.ztf, token)
    t_connect = time.perf_counter() - t0
    log.info("Kowalski connected (%.3fs)", t_connect)

    # ── Fetch ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    lcs = ztf.fetch_by_position(args.ra, args.dec, radius_arcsec=args.radius)
    t_fetch = time.perf_counter() - t0

    if not lcs:
        log.error("No ZTF sources found at (%.4f, %.4f) within %.1f arcsec.",
                  args.ra, args.dec, args.radius)
        sys.exit(1)

    grouped: dict[str, list] = defaultdict(list)
    for lc in lcs:
        grouped[lc.source_id].append(lc)

    target_id  = max(grouped, key=lambda sid: max(lc.n_obs for lc in grouped[sid]))
    target_lcs = grouped[target_id]
    n_obs_max  = max(lc.n_obs for lc in target_lcs)
    bands      = [lc.band for lc in target_lcs]

    log.info("Fetch complete (%.3fs) | %d sources in cone | using %s (%d obs, bands=%s)",
             t_fetch, len(grouped), target_id, n_obs_max, bands)

    # ── Period grid info ──────────────────────────────────────────────────────
    primary  = max(target_lcs, key=lambda lc: lc.n_obs)
    baseline = float(primary.time.max() - primary.time.min())
    spp      = cfg.features.period.samples_per_peak
    f_min    = max(2.0 / baseline, 1.0 / cfg.features.period.max_period_days)
    f_max    = 1.0 / cfg.features.period.min_period_days
    df       = 1.0 / (spp * baseline)
    n_grid   = max(1, int((f_max - f_min) / df))
    grid_desc = f"freq-spaced, spp={spp}, baseline={baseline:.1f}d"
    log.info("Period grid: %d points — %s", n_grid, grid_desc)

    # ── Feature extraction ────────────────────────────────────────────────────
    import periodfind
    from ml4em.features.statistics import StatisticsExtractor
    from ml4em.features.period     import PeriodExtractor
    from ml4em.features.dmdt       import DmdtExtractor
    from ml4em.features.catalog    import CatalogExtractor

    device = sc.normalize_device(device)
    periodfind.set_device(device)

    # One entry per band, as FeaturePipeline.run_batch does.  Passing all bands
    # as a single entry would time only the longest one, so the reported
    # latency would be for a fraction of the work the pipeline actually does
    # for this source.
    source_batch = [[lc] for lc in target_lcs]

    # CUDA warmup: first GPU call pays JIT compile cost — run throwaway first
    if device == "gpu":
        log.info("CUDA warmup (not timed)...")
        t0 = time.perf_counter()
        _warm = PeriodExtractor(cfg.features.period)
        _warm.prepare(source_batch)
        _warm.extract(source_batch)
        log.info("Warmup complete (%.3fs)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    StatisticsExtractor().extract(source_batch)
    t_stats = time.perf_counter() - t0

    _period_ext = PeriodExtractor(cfg.features.period)
    _period_ext.prepare(source_batch)
    t0 = time.perf_counter()
    _period_ext.extract(source_batch)
    t_period = time.perf_counter() - t0

    t0 = time.perf_counter()
    DmdtExtractor(cfg.features.dmdt).extract(source_batch)
    t_dmdt = time.perf_counter() - t0

    t0 = time.perf_counter()
    CatalogExtractor(cfg.features.catalog, kowalski_client=ztf.client).extract(source_batch)
    t_catalog = time.perf_counter() - t0

    t_features = t_stats + t_period + t_dmdt + t_catalog

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "─" * 42
    print(f"\n{sep}")
    print(f"  {'Stage':<28}{'Time':>8}")
    print(sep)
    print(f"  {'Kowalski connect':<28}{_fmt(t_connect)}")
    print(f"  {'Fetch (cone search)':<28}{_fmt(t_fetch)}")
    print(sep)
    print(f"  {'Statistics':<28}{_fmt(t_stats)}")
    print(f"  {'Period finding':<28}{_fmt(t_period)}")
    print(f"  {'dm/dt histogram':<28}{_fmt(t_dmdt)}")
    print(f"  {'Gaia xmatch':<28}{_fmt(t_catalog)}")
    print(sep)
    print(f"  {'Total feature':<28}{_fmt(t_features)}")
    print(f"  {'Total end-to-end':<28}{_fmt(t_connect + t_fetch + t_features)}")
    print(sep)
    print(f"\n  Source:       {target_id}")
    print(f"  Bands:        {', '.join(bands)}")
    print(f"  Max obs:      {n_obs_max}")
    print(f"  Device:       {device}")
    print(f"  Algorithms:   {', '.join(cfg.features.period.algorithms)}")
    print(f"  Period grid:  {n_grid} points  ({grid_desc})")
    print()


if __name__ == "__main__":
    main()
