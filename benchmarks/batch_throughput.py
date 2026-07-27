#!/usr/bin/env python3
"""
End-to-end batch throughput benchmark.

Runs the full production pipeline on a real ZTF sky region and reports
per-stage timing.  This is the primary benchmark for estimating MSI
compute hours and for comparing performance across node configurations.

Pipeline stages timed
---------------------
  Round trip 1  near query    Kowalski spatial index → source IDs in region
  Round trip 2  find query    fetch_batch() sliding window → full light curves
                Statistics    periodfind BasicStats extractor
                Period        CE / AOV / LS / MHF / FPW (GPU batched via periodfind)
                dm/dt         periodfind DmDt histogram
  Round trip 3  Gaia xmatch  CatalogExtractor cone_search → Gaia EDR3 features

The near + find two-hop pattern matches scope-ml's get_lightcurves_via_coords
exactly.  The find query uses a sliding window: IDs are chunked into slices of
limit_per_query (default 1000), sent n_workers chunks at a time.

Usage
-----
    python benchmarks/batch_throughput.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --device cuda --n-workers 8

    # GPU warmup run (recommended — avoids CUDA JIT in timing)
    python benchmarks/batch_throughput.py \\
        --config config.yaml \\
        --ra 116.7 --dec 36.2 --radius-arcsec 1800 \\
        --device cuda --n-workers 8 --warmup

Example output
--------------
    Region: RA=116.7  Dec=36.2  radius=1800 arcsec    Device: cuda    Batch: 1000
    Workers: 8 (LC) / 8 (Gaia)    limit_per_query: 1000

    ────────────────────────────────────────────────────────────────────────────
      Stage                    Sources    Total (s)    Per src (ms)    Throughput
    ────────────────────────────────────────────────────────────────────────────
      Near (ID discovery)        1 247       1.203            0.97       1 036/s
      Find (LC fetch)            1 247       7.841            6.29         159/s
    ────────────────────────────────────────────────────────────────────────────
      Statistics                 1 052       0.043            0.04      24 465/s
      Period finding             1 052      14.312           13.60          74/s
      dm/dt histogram            1 052       0.219            0.21       4 804/s
      Gaia xmatch                1 052       2.088            1.98         504/s
    ────────────────────────────────────────────────────────────────────────────
      Total                      1 052      25.706           24.43          41/s
    ────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 1_000
_DEFAULT_DEVICE     = "cpu"
_DEFAULT_RADIUS     = 1800.0   # arcsec — roughly a ZTF quad footprint


# ---------------------------------------------------------------------------
# Per-extractor timing
# ---------------------------------------------------------------------------

def _time_extractors(sources, cfg, device, batch_size, kowalski_client):
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

    t0 = time.perf_counter()
    ext = CatalogExtractor(cfg.features.catalog, kowalski_client=kowalski_client)
    for i in range(0, n, batch_size):
        ext.extract(valid[i : i + batch_size])
    timings["Gaia xmatch"] = time.perf_counter() - t0

    return timings, period_results, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end batch throughput benchmark — real ZTF light curves"
    )
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--ra",            type=float, default=116.7)
    parser.add_argument("--dec",           type=float, default=36.2)
    parser.add_argument("--radius-arcsec", type=float, default=_DEFAULT_RADIUS,
                        help=f"Search radius in arcsec (default: {_DEFAULT_RADIUS} ≈ ZTF quad)")
    parser.add_argument("--batch-size",    type=int, default=_DEFAULT_BATCH_SIZE,
                        help=f"Sources per GPU batch (default: {_DEFAULT_BATCH_SIZE})")
    parser.add_argument("--device",        default=_DEFAULT_DEVICE,
                        choices=["cpu", "cuda", "auto"])
    parser.add_argument("--algorithms",    default=None,
                        help="Comma-separated list, e.g. CE,AOV,LS,MHF,FPW")
    parser.add_argument("--warmup",        action="store_true",
                        help="Run one throwaway GPU batch before timing (recommended with --device cuda)")
    parser.add_argument("--n-workers",     type=int, default=None,
                        help="Parallel Kowalski threads (overrides config; sweep with sweep_workers.py)")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    from ml4em.config.loader import load_config
    from ml4em.config import get_ztf_token

    cfg   = load_config(args.config)
    token = get_ztf_token()
    if args.algorithms:
        cfg.features.period.algorithms = args.algorithms.split(",")
    if args.n_workers is not None:
        cfg.sources.ztf.n_workers      = args.n_workers
        cfg.features.catalog.n_workers = args.n_workers

    # ── Data acquisition ─────────────────────────────────────────────────────
    from ml4em.data.ztf import ZTFSource

    ztf = ZTFSource(cfg.sources.ztf, token)

    # Round trip 1: near query — spatial index, returns IDs only
    log.info("Round trip 1 — near: RA=%.4f Dec=%.4f radius=%.0f arcsec",
             args.ra, args.dec, args.radius_arcsec)
    t0 = time.perf_counter()
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

    log.info("Near query: %d source IDs (%.3fs)", len(source_ids), t_near)
    if not source_ids:
        log.error("No sources found. Try a larger --radius-arcsec.")
        sys.exit(1)

    # Round trip 2: find query — sliding window, full LC data
    log.info("Round trip 2 — find: %d sources, n_workers=%d",
             len(source_ids), cfg.sources.ztf.n_workers)
    t0 = time.perf_counter()
    lcs = ztf.fetch_batch(source_ids)
    t_find = time.perf_counter() - t0
    log.info("Find: %d light curves (%.3fs)", len(lcs), t_find)

    groups: dict = defaultdict(list)
    for lc in lcs:
        groups[lc.source_id].append(lc)
    sources = [groups.get(sid, []) for sid in source_ids]

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

    # ── Feature extraction timing ─────────────────────────────────────────────
    timings, period_results, n_valid = _time_extractors(
        sources, cfg, args.device, args.batch_size,
        kowalski_client=ztf.client,
    )

    # ── Algorithm breakdown ───────────────────────────────────────────────────
    algo_counts: dict[str, int] = {}
    nan_count = 0
    for r in period_results:
        algo = r.get("period_algorithm", "")
        if not algo or (isinstance(r.get("period"), float) and np.isnan(r["period"])):
            nan_count += 1
        else:
            algo_counts[algo] = algo_counts.get(algo, 0) + 1

    # ── Summary ──────────────────────────────────────────────────────────────
    t_total = sum(timings.values()) + t_near + t_find
    n_ids   = len(source_ids)
    skipped = n_ids - n_valid

    worker_info = (f"    Workers: {cfg.sources.ztf.n_workers} (LC) / "
                   f"{cfg.features.catalog.n_workers} (Gaia)"
                   f"    limit_per_query: {cfg.sources.ztf.limit_per_query}")

    sep = "─" * 76
    print(f"\n  Region: RA={args.ra}  Dec={args.dec}  radius={args.radius_arcsec:.0f} arcsec"
          f"    Device: {args.device}    Batch: {args.batch_size}{worker_info}")
    print(f"\n{sep}")
    print(f"  {'Stage':<28}{'Sources':>9}{'Total (s)':>11}{'Per src (ms)':>14}{'Throughput':>12}")
    print(sep)
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
    print(f"\n  {skipped} sources skipped (< {cfg.features.min_observations} obs or no clean data)")
    print(f"\n  Period algorithm breakdown ({n_valid} sources):")
    for algo, count in sorted(algo_counts.items(), key=lambda x: -x[1]):
        print(f"    {algo:<6} {count:>6}  ({count / n_valid * 100:.1f}%)")
    if nan_count:
        print(f"    {'NaN':<6} {nan_count:>6}  ({nan_count / n_valid * 100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
