#!/usr/bin/env python3
"""
Single-source throughput benchmark: fetch → feature extraction.

Times each stage independently so you can see exactly where the wall time goes
and compare CPU vs GPU for the period-finding step.

Usage
-----
    # Default position (known ZTF source):
    python scripts/benchmark_single.py

    # Specific sky coordinate:
    python scripts/benchmark_single.py --ra 116.7 --dec 36.2

    # With explicit config and device override:
    python scripts/benchmark_single.py \
        --config /data/config_msi.yaml \
        --device cuda

Output
------
    Stage              Time (s)
    ─────────────────────────────
    Kowalski connect     0.31
    Fetch (cone search)  2.14
    Statistics           0.02
    Period finding      18.43
    dm/dt histogram      0.08
    Catalog (Gaia)       0.00
    ─────────────────────────────
    Total feature        18.53
    ─────────────────────────────
    Light curves found:  3  (g, r, i)
    Observations (max):  412
    Device:              cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Default: a ZTF-covered WDB candidate position
_DEFAULT_RA  = 116.7354
_DEFAULT_DEC =  36.1980
_DEFAULT_RADIUS_ARCSEC = 2.0


def _fmt(seconds: float) -> str:
    return f"{seconds:8.3f}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fetch + feature extraction for a single ZTF source"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--ra",  type=float, default=_DEFAULT_RA,
                        help=f"Right ascension in degrees (default: {_DEFAULT_RA})")
    parser.add_argument("--dec", type=float, default=_DEFAULT_DEC,
                        help=f"Declination in degrees (default: {_DEFAULT_DEC})")
    parser.add_argument("--radius", type=float, default=_DEFAULT_RADIUS_ARCSEC,
                        help=f"Cone search radius in arcsec (default: {_DEFAULT_RADIUS_ARCSEC})")
    parser.add_argument("--device", default=None,
                        choices=["cpu", "cuda", "auto"],
                        help="Override features.device from config")
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated algorithm list, e.g. CE,AOV,LS (overrides config)")
    parser.add_argument("--samples-per-peak", type=float, default=None,
                        help="Build frequency-spaced grid with this oversampling factor (overrides n_freq_grid)")
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────────
    from ml4em.config.loader import load_config, get_ztf_token
    cfg = load_config(args.config)
    if args.device:
        cfg.features.device = args.device
    if args.algorithms:
        cfg.features.period.algorithms = args.algorithms.split(",")
    if args.samples_per_peak:
        cfg.features.period.samples_per_peak = args.samples_per_peak
    device = cfg.features.device

    log.info("Config loaded  |  device=%s  |  ra=%.4f  dec=%.4f  radius=%.1f\"",
             device, args.ra, args.dec, args.radius)

    # ── Kowalski connection ──────────────────────────────────────────────────────
    from ml4em.data.ztf import ZTFSource
    token = get_ztf_token()

    t0 = time.perf_counter()
    ztf = ZTFSource(cfg.sources.ztf, token)
    t_connect = time.perf_counter() - t0
    log.info("Kowalski connected  (%.3fs)", t_connect)

    # ── Fetch ───────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    lcs = ztf.fetch_by_position(args.ra, args.dec, radius_arcsec=args.radius)
    t_fetch = time.perf_counter() - t0

    if not lcs:
        log.error("No ZTF sources found at (%.4f, %.4f) within %.1f arcsec.",
                  args.ra, args.dec, args.radius)
        log.error("Try a larger --radius or a different --ra/--dec.")
        sys.exit(1)

    # Group by source_id; pick the source with the most observations
    grouped: dict[str, list] = defaultdict(list)
    for lc in lcs:
        grouped[lc.source_id].append(lc)

    target_id = max(grouped, key=lambda sid: max(lc.n_obs for lc in grouped[sid]))
    target_lcs = grouped[target_id]
    n_obs_max = max(lc.n_obs for lc in target_lcs)
    bands = [lc.band for lc in target_lcs]

    log.info("Fetch complete  (%.3fs)  |  %d sources in cone  |  using source %s  "
             "(%d obs, bands=%s)",
             t_fetch, len(grouped), target_id, n_obs_max, bands)

    # ── Per-extractor timing ─────────────────────────────────────────────────────
    from ml4em.features.statistics import StatisticsExtractor
    from ml4em.features.period     import PeriodExtractor
    from ml4em.features.dmdt       import DmdtExtractor
    from ml4em.features.catalog    import CatalogExtractor

    import periodfind
    periodfind.set_device(device)

    source_batch = [target_lcs]  # single source wrapped in list

    # ── Period grid info ─────────────────────────────────────────────────────
    if cfg.features.period.samples_per_peak is not None:
        primary = max(target_lcs, key=lambda lc: lc.n_obs)
        baseline = float(primary.time.max() - primary.time.min())
        f_min = 2.0 / baseline                                   # scope-ml: 2 cycles minimum
        f_max = 1.0 / cfg.features.period.min_period_days
        df = 1.0 / (cfg.features.period.samples_per_peak * baseline)
        n_grid = max(1, int((f_max - f_min) / df))
        grid_desc = (f"freq-spaced  spp={cfg.features.period.samples_per_peak}"
                     f"  baseline={baseline:.1f}d  fmin=2/baseline")
    else:
        n_grid = cfg.features.period.n_freq_grid
        grid_desc = "period-spaced  (linspace)"
    log.info("Period grid: %d points — %s", n_grid, grid_desc)

    # ── CUDA warmup ──────────────────────────────────────────────────────────
    # First GPU call on a fresh process pays CUDA JIT compilation overhead.
    # Run a throwaway call so timing reflects steady-state GPU throughput.
    if device == "cuda":
        log.info("CUDA warmup (not timed)...")
        t0 = time.perf_counter()
        PeriodExtractor(cfg.features.period).extract(source_batch)
        log.info("Warmup complete (%.3fs)", time.perf_counter() - t0)

    t0 = time.perf_counter()
    StatisticsExtractor().extract(source_batch)
    t_stats = time.perf_counter() - t0

    t0 = time.perf_counter()
    PeriodExtractor(cfg.features.period).extract(source_batch)
    t_period = time.perf_counter() - t0

    t0 = time.perf_counter()
    DmdtExtractor(cfg.features.dmdt).extract(source_batch)
    t_dmdt = time.perf_counter() - t0

    t0 = time.perf_counter()
    CatalogExtractor(cfg.features.catalog).extract(source_batch)
    t_catalog = time.perf_counter() - t0

    t_features_total = t_stats + t_period + t_dmdt + t_catalog

    # ── Summary ─────────────────────────────────────────────────────────────────
    sep = "─" * 38
    print(f"\n{sep}")
    print(f"  {'Stage':<26}{'Time':>8}")
    print(sep)
    print(f"  {'Kowalski connect':<26}{_fmt(t_connect)}")
    print(f"  {'Fetch (cone search)':<26}{_fmt(t_fetch)}")
    print(sep)
    print(f"  {'Statistics':<26}{_fmt(t_stats)}")
    print(f"  {'Period finding':<26}{_fmt(t_period)}")
    print(f"  {'dm/dt histogram':<26}{_fmt(t_dmdt)}")
    print(f"  {'Catalog (Gaia xmatch)':<26}{_fmt(t_catalog)}")
    print(sep)
    print(f"  {'Total feature':<26}{_fmt(t_features_total)}")
    print(f"  {'Total end-to-end':<26}{_fmt(t_connect + t_fetch + t_features_total)}")
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
