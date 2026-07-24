#!/usr/bin/env python3
"""
Batch throughput benchmark: real ZTF light curves through the full pipeline.

Mirrors the production workflow exactly:
  Round trip 1 (near)  — spatial index query to discover source IDs in a sky region
  Round trip 2 (find)  — fetch full light curve data for those IDs
  Feature extraction   — Statistics → Period finding → dm/dt (GPU batched)
  Round trip 3 (Gaia)  — Kowalski cone_search against Gaia_EDR3

This matches scope-ml's two-hop LC fetch pattern (get_lightcurves_via_coords)
plus the external_xmatch Gaia step.

Usage (MSI — real data)
-----------------------
    # Benchmark a ~quad-sized sky region centred on a known ZTF field:
    python scripts/benchmark_batch.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --device cuda

    # Smaller region for a quick test:
    python scripts/benchmark_batch.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 600

Fallback (no Kowalski — local dev only)
----------------------------------------
    python scripts/benchmark_batch.py --synthetic --n-sources 200 --n-obs 50

Output
------
    Region: RA=116.7  Dec=36.2  radius=1800 arcsec    Device: cuda    Batch: 1000
    ─────────────────────────────────────────────────────────────────────────────
      Stage                 Sources    Total (s)    Per src (ms)    Throughput
    ─────────────────────────────────────────────────────────────────────────────
      Near (ID discovery)     1 247       1.203            0.97       1 036/s
      Find (LC fetch)         1 247       7.841            6.29         159/s
      Statistics              1 052       0.043            0.04      24 465/s
      Period finding          1 052      14.312           13.60          74/s
      dm/dt histogram         1 052       0.219            0.21       4 804/s
      Gaia xmatch             1 052       2.088            1.98         504/s
    ─────────────────────────────────────────────────────────────────────────────
      Total                   1 052      25.706           24.43          41/s
    ─────────────────────────────────────────────────────────────────────────────
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

_DEFAULT_BATCH_SIZE  = 1_000
_DEFAULT_DEVICE      = "cpu"
_DEFAULT_RADIUS      = 1800.0   # arcsec — roughly a ZTF quad footprint


# ---------------------------------------------------------------------------
# Synthetic fallback (local dev only)
# ---------------------------------------------------------------------------

def _make_synthetic_sources(n_sources: int, n_obs: int, rng: np.random.Generator):
    from ml4em.types import LightCurve
    sources = []
    t0_hjd  = 2_458_800.0
    for i in range(n_sources):
        period    = float(rng.uniform(0.05, 5.0))
        amplitude = float(rng.uniform(0.05, 0.5))
        phase     = float(rng.uniform(0, 2 * np.pi))
        mean_mag  = float(rng.uniform(17.0, 20.0))
        noise     = float(rng.uniform(0.01, 0.05))
        t = np.sort(rng.uniform(0, 730.0, n_obs)) + t0_hjd
        m = mean_mag + amplitude * np.sin(2 * np.pi * t / period + phase) + rng.normal(0, noise, n_obs)
        e = np.full(n_obs, noise)
        sources.append([LightCurve(f"synth_{i:06d}", t, m, e, "r", "simulated",
                                   float(rng.uniform(0, 360)), float(rng.uniform(-30, 90)))])
    return sources


# ---------------------------------------------------------------------------
# Per-extractor timing
# ---------------------------------------------------------------------------

def _time_extractors(sources, cfg, device, batch_size, kowalski_client=None):
    import periodfind
    from ml4em.features.statistics import StatisticsExtractor
    from ml4em.features.period     import PeriodExtractor
    from ml4em.features.dmdt       import DmdtExtractor
    from ml4em.features.catalog    import CatalogExtractor

    periodfind.set_device(device)

    valid = [s for s in sources if s]
    n = len(valid)
    if n == 0:
        log.error("No valid sources after filtering.")
        sys.exit(1)

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    ext = StatisticsExtractor()
    for i in range(0, n, batch_size):
        ext.extract(valid[i : i + batch_size])
    timings["Statistics"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ext = PeriodExtractor(cfg.features.period)
    period_results = []
    for i in range(0, n, batch_size):
        period_results.extend(ext.extract(valid[i : i + batch_size]))
    timings["Period finding"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ext = DmdtExtractor(cfg.features.dmdt)
    for i in range(0, n, batch_size):
        ext.extract(valid[i : i + batch_size])
    timings["dm/dt histogram"] = time.perf_counter() - t0

    if kowalski_client is not None:
        t0 = time.perf_counter()
        ext = CatalogExtractor(cfg.features.catalog, kowalski_client=kowalski_client)
        for i in range(0, n, batch_size):
            ext.extract(valid[i : i + batch_size])
        timings["Gaia xmatch"] = time.perf_counter() - t0
    else:
        log.info("Gaia xmatch skipped (no Kowalski client)")

    return timings, period_results, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch throughput benchmark — real ZTF light curves, two-hop Kowalski fetch"
    )

    # Real data (primary)
    parser.add_argument("--config",        default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--ra",            type=float, default=116.7,
                        help="Region centre RA in degrees (default: 116.7)")
    parser.add_argument("--dec",           type=float, default=36.2,
                        help="Region centre Dec in degrees (default: 36.2)")
    parser.add_argument("--radius-arcsec", type=float, default=_DEFAULT_RADIUS,
                        help=f"Search radius in arcsec (default: {_DEFAULT_RADIUS} ≈ ZTF quad)")

    # Synthetic fallback
    parser.add_argument("--synthetic",     action="store_true",
                        help="Use synthetic data instead of Kowalski (local dev only)")
    parser.add_argument("--n-sources",     type=int, default=500,
                        help="Synthetic mode: number of sources (default: 500)")
    parser.add_argument("--n-obs",         type=int, default=300,
                        help="Synthetic mode: observations per source (default: 300)")
    parser.add_argument("--seed",          type=int, default=42)

    # Common
    parser.add_argument("--batch-size",    type=int, default=_DEFAULT_BATCH_SIZE,
                        help=f"Sources per GPU batch (default: {_DEFAULT_BATCH_SIZE})")
    parser.add_argument("--device",        default=_DEFAULT_DEVICE,
                        choices=["cpu", "cuda", "auto"],
                        help=f"periodfind device (default: {_DEFAULT_DEVICE})")
    parser.add_argument("--algorithms",    default=None,
                        help="Comma-separated algorithm list, e.g. CE,AOV,LS")
    parser.add_argument("--warmup",        action="store_true",
                        help="Run one throwaway batch before timing (GPU warmup)")

    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    if args.synthetic:
        log.warning("Running in synthetic mode — Kowalski fetch and Gaia are NOT timed.")
        log.warning("Use real-data mode on MSI for meaningful throughput numbers.")
        from ml4em.config.schema import FeatureConfig
        from ml4em.config.schema import PeriodConfig
        class _Cfg:
            class features:
                period  = PeriodConfig()
                dmdt    = __import__('ml4em.config.schema', fromlist=['DmdtConfig']).DmdtConfig()
                catalog = __import__('ml4em.config.schema', fromlist=['CatalogConfig']).CatalogConfig()
        cfg = _Cfg()
        if args.algorithms:
            cfg.features.period = PeriodConfig(algorithms=args.algorithms.split(","))
    else:
        from ml4em.config.loader import load_config
        from ml4em.config import get_ztf_token
        cfg   = load_config(args.config)
        token = get_ztf_token()
        if args.algorithms:
            cfg.features.period.algorithms = args.algorithms.split(",")

    # ── Data acquisition ─────────────────────────────────────────────────────
    kowalski_client = None
    t_near = t_find = None
    sources = []

    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        sources = _make_synthetic_sources(args.n_sources, args.n_obs, rng)
        log.info("Generated %d synthetic sources", len(sources))
        region_label = f"synthetic ({args.n_sources} sources)"

    else:
        from ml4em.data.ztf import ZTFSource
        ztf = ZTFSource(cfg.sources.ztf, token)
        kowalski_client = ztf.client

        log.info("Round trip 1 — near query: RA=%.4f Dec=%.4f radius=%.0f arcsec",
                 args.ra, args.dec, args.radius_arcsec)
        t0 = time.perf_counter()

        # near query (ID discovery)
        near_query = {
            "query_type": "near",
            "query": {
                "max_distance"  : args.radius_arcsec,
                "distance_units": "arcsec",
                "radec"         : {"query_coords": [args.ra, args.dec]},
                "catalogs"      : {
                    cfg.sources.ztf.collection_sources: {
                        "filter": {}, "projection": {"_id": 1},
                    }
                },
            },
            "kwargs": {"max_time_ms": 30000, "limit": 100_000},
        }
        near_resp = ztf.client.query(queries=[near_query], use_batch_query=True, max_n_threads=1)
        t_near = time.perf_counter() - t0

        source_ids: list[str] = []
        for _inst, resp_list in near_resp.items():
            for resp in resp_list:
                if resp.get("status") != "success":
                    continue
                hits = (resp.get("data", {})
                            .get(cfg.sources.ztf.collection_sources, {})
                            .get("query_coords", []))
                source_ids.extend(str(doc["_id"]) for doc in hits)

        log.info("Near query found %d source IDs (%.3fs)", len(source_ids), t_near)

        if not source_ids:
            log.error("No sources found in region. Try a larger --radius-arcsec.")
            sys.exit(1)

        log.info("Round trip 2 — find query: fetching light curves for %d sources (n_workers=%d)",
                 len(source_ids), cfg.sources.ztf.n_workers)
        t0 = time.perf_counter()
        lcs = ztf.fetch_batch(source_ids)
        t_find = time.perf_counter() - t0
        log.info("Find query returned %d light curves (%.3fs)", len(lcs), t_find)

        # Group by source_id
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for lc in lcs:
            groups[lc.source_id].append(lc)
        sources = [groups.get(sid, []) for sid in source_ids]

        region_label = f"RA={args.ra}  Dec={args.dec}  radius={args.radius_arcsec:.0f} arcsec"

    # ── GPU warmup ───────────────────────────────────────────────────────────
    if args.warmup or args.device == "cuda":
        log.info("GPU warmup (not timed)...")
        import periodfind
        from ml4em.features.period import PeriodExtractor
        periodfind.set_device(args.device)
        warmup = [s for s in sources if s][:min(50, len(sources))]
        if warmup:
            PeriodExtractor(cfg.features.period).extract(warmup)
        log.info("Warmup complete")

    # ── Per-extractor timing ─────────────────────────────────────────────────
    timings, period_results, n_valid = _time_extractors(
        sources, cfg, args.device, args.batch_size,
        kowalski_client=kowalski_client,
    )

    # ── Algorithm breakdown ──────────────────────────────────────────────────
    algo_counts: dict[str, int] = {}
    nan_count = 0
    for r in period_results:
        algo = r.get("period_algorithm", "")
        if not algo or (isinstance(r.get("period"), float) and np.isnan(r["period"])):
            nan_count += 1
        else:
            algo_counts[algo] = algo_counts.get(algo, 0) + 1

    # ── Summary ──────────────────────────────────────────────────────────────
    t_total = sum(timings.values())
    if t_near is not None:
        t_total += t_near + t_find

    sep = "─" * 76

    print(f"\n  Region: {region_label}    Device: {args.device}    Batch: {args.batch_size}")
    print(f"\n{sep}")
    print(f"  {'Stage':<28}{'Sources':>9}{'Total (s)':>11}{'Per src (ms)':>14}{'Throughput':>12}")
    print(sep)

    if t_near is not None:
        n_ids = len(source_ids)
        print(f"  {'Near (ID discovery)':<28}{n_ids:>9}{t_near:>10.3f}s"
              f"{t_near / n_ids * 1000:>12.2f}  {n_ids / t_near:>8.0f}/s")
        print(f"  {'Find (LC fetch)':<28}{n_ids:>9}{t_find:>10.3f}s"
              f"{t_find / n_ids * 1000:>12.2f}  {n_ids / t_find:>8.0f}/s")
        print(sep)

    for stage, t in timings.items():
        print(f"  {stage:<28}{n_valid:>9}{t:>10.3f}s"
              f"{t / n_valid * 1000:>12.2f}  {n_valid / t:>8.0f}/s")

    print(sep)
    print(f"  {'Total':<28}{n_valid:>9}{t_total:>10.3f}s"
          f"{t_total / n_valid * 1000:>12.2f}  {n_valid / t_total:>8.0f}/s")
    print(sep)

    if not args.synthetic:
        skipped = len(source_ids) - n_valid
        print(f"\n  {skipped} sources skipped (< {cfg.features.min_observations} obs or no clean data)")

    print(f"\n  Period algorithm breakdown ({n_valid} sources):")
    for algo, count in sorted(algo_counts.items(), key=lambda x: -x[1]):
        print(f"    {algo:<6} {count:>6}  ({count / n_valid * 100:.1f}%)")
    if nan_count:
        print(f"    {'NaN':<6} {nan_count:>6}  ({nan_count / n_valid * 100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
