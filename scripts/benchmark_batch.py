#!/usr/bin/env python3
"""
Batch throughput benchmark: real ZTF light curves through the full pipeline.

Fetches a batch of real sources from Kowalski and times every stage of the
production pipeline — Kowalski fetch, statistics, period finding, dm/dt,
and Gaia xmatch — exactly as a production MSI run would execute them.

This is the benchmark that matters.  Single-source latency (benchmark_single.py)
tells you nothing about batch GPU throughput or Kowalski I/O amortisation.

Primary mode (real data — use this on MSI)
------------------------------------------
Provide a plain-text file of ZTF source IDs (one integer per line).
The script fetches all of them in one batched Kowalski query, then runs
the full feature pipeline:

    # Create a source ID file (e.g. from a ZTF quad list):
    python scripts/benchmark_batch.py \\
        --source-ids /path/to/source_ids.txt \\
        --config config.yaml \\
        --device cuda

Fallback mode (synthetic data — for local dev without Kowalski)
---------------------------------------------------------------
Omit --source-ids to generate synthetic ZTF-like sinusoidal light curves.
Only GPU compute stages are timed; Kowalski fetch and Gaia are excluded.
Use this to verify the script runs before deploying to MSI.

    python scripts/benchmark_batch.py --n-sources 100 --n-obs 50

Output (real data mode)
-----------------------
    N sources: 1000    N obs: ~310 (median)    Device: cuda    Batch: 1000
    ─────────────────────────────────────────────────────────────────────
      Stage                 Total (s)    Per source (ms)    Throughput
    ─────────────────────────────────────────────────────────────────────
      Kowalski fetch           8.241             8.24         121/s
      Statistics               0.041             0.04      24 390/s
      Period finding          11.832            11.83          85/s
      dm/dt histogram          0.203             0.20       4 926/s
      Gaia xmatch              2.114             2.11         473/s
    ─────────────────────────────────────────────────────────────────────
      Total pipeline          22.431            22.43          45/s
    ─────────────────────────────────────────────────────────────────────
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

_DEFAULT_N_SOURCES    = 1_000
_DEFAULT_N_OBS        = 300
_DEFAULT_BATCH_SIZE   = 1_000
_DEFAULT_DEVICE       = "cpu"


# ---------------------------------------------------------------------------
# Real data: fetch from Kowalski
# ---------------------------------------------------------------------------

def _fetch_from_kowalski(
    source_ids: list[str],
    cfg,
    token: str,
) -> tuple:
    """Fetch real ZTF light curves for the given source IDs.

    Returns (ztf_source, grouped_lcs, fetch_time_seconds).
    grouped_lcs is a list[list[LightCurve]] — one inner list per source_id,
    containing one LightCurve per band (g/r/i).  Sources with no clean
    observations after cadence filtering are dropped.
    """
    from ml4em.data.ztf import ZTFSource

    log.info("Connecting to Kowalski at %s ...", cfg.sources.ztf.host)
    ztf = ZTFSource(cfg.sources.ztf, token)

    log.info("Fetching %d sources (n_workers=%d)...", len(source_ids), cfg.sources.ztf.n_workers)
    t0 = time.perf_counter()
    lcs = ztf.fetch_batch(source_ids)
    t_fetch = time.perf_counter() - t0
    log.info("Fetched %d light curves in %.3fs", len(lcs), t_fetch)

    # Group by source_id (one inner list per source, all bands together)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for lc in lcs:
        groups[lc.source_id].append(lc)

    # Preserve original source_id order; sources with no data get empty list
    grouped = [groups.get(sid, []) for sid in source_ids]
    n_empty = sum(1 for g in grouped if not g)
    if n_empty:
        log.warning("%d/%d sources returned no clean light curves", n_empty, len(source_ids))

    return ztf, grouped, t_fetch


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def _make_synthetic_sources(
    n_sources: int,
    n_obs: int,
    rng: np.random.Generator,
) -> list[list]:
    """Generate realistic ZTF-like sinusoidal light curves as a local fallback.

    Only use this when no Kowalski connection is available (local dev / CI).
    For production benchmarks always use real data via --source-ids.
    """
    from ml4em.types import LightCurve

    sources = []
    baseline_days = 730.0
    t0_hjd = 2_458_800.0

    for i in range(n_sources):
        period    = float(rng.uniform(0.05, 5.0))
        amplitude = float(rng.uniform(0.05, 0.5))
        phase     = float(rng.uniform(0, 2 * np.pi))
        mean_mag  = float(rng.uniform(17.0, 20.0))
        noise     = float(rng.uniform(0.01, 0.05))

        t = np.sort(rng.uniform(0, baseline_days, n_obs).astype(np.float64)) + t0_hjd
        m = mean_mag + amplitude * np.sin(2 * np.pi * t / period + phase) + rng.normal(0, noise, n_obs)
        e = np.full(n_obs, noise, dtype=np.float64)

        lc = LightCurve(
            source_id = f"synth_{i:06d}",
            time=t, mag=m, mag_err=e,
            band="r", survey="simulated",
            ra=float(rng.uniform(0, 360)),
            dec=float(rng.uniform(-30, 90)),
        )
        sources.append([lc])

    return sources


# ---------------------------------------------------------------------------
# Per-extractor timing
# ---------------------------------------------------------------------------

def _time_extractors(
    sources: list[list],
    cfg,
    device: str,
    batch_size: int,
    kowalski_client=None,
) -> tuple[dict[str, float], list]:
    """Time each extractor independently on the same grouped source list.

    When kowalski_client is provided, Gaia xmatch is included.
    Returns (timings, period_results).
    """
    import periodfind
    from ml4em.features.statistics import StatisticsExtractor
    from ml4em.features.period     import PeriodExtractor
    from ml4em.features.dmdt       import DmdtExtractor
    from ml4em.features.catalog    import CatalogExtractor

    periodfind.set_device(device)
    timings: dict[str, float] = {}

    # Filter to sources that have at least one light curve
    valid_sources = [s for s in sources if s]
    n = len(valid_sources)
    if n == 0:
        log.error("No valid sources to benchmark after filtering.")
        sys.exit(1)
    log.info("Timing extractors on %d sources with ≥1 light curve", n)

    # -- Statistics -----------------------------------------------------------
    ext_stats = StatisticsExtractor()
    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        ext_stats.extract(valid_sources[i : i + batch_size])
    timings["Statistics"] = time.perf_counter() - t0

    # -- Period finding -------------------------------------------------------
    ext_period = PeriodExtractor(cfg.features.period)
    t0 = time.perf_counter()
    period_results = []
    for i in range(0, n, batch_size):
        period_results.extend(ext_period.extract(valid_sources[i : i + batch_size]))
    timings["Period finding"] = time.perf_counter() - t0

    # -- dm/dt histogram ------------------------------------------------------
    ext_dmdt = DmdtExtractor(cfg.features.dmdt)
    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        ext_dmdt.extract(valid_sources[i : i + batch_size])
    timings["dm/dt histogram"] = time.perf_counter() - t0

    # -- Gaia xmatch ----------------------------------------------------------
    # Only timed when a live Kowalski client is available.
    # In synthetic mode this is skipped — use benchmark_single.py for Gaia latency.
    if kowalski_client is not None:
        ext_catalog = CatalogExtractor(cfg.features.catalog, kowalski_client=kowalski_client)
        t0 = time.perf_counter()
        for i in range(0, n, batch_size):
            ext_catalog.extract(valid_sources[i : i + batch_size])
        timings["Gaia xmatch"] = time.perf_counter() - t0
    else:
        log.info("Gaia xmatch skipped (no Kowalski client — synthetic mode)")

    return timings, period_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch throughput benchmark — real ZTF light curves through the full pipeline"
    )

    # Real data mode
    parser.add_argument("--source-ids", metavar="FILE",
                        help="Plain-text file of ZTF source IDs (one integer per line). "
                             "Required for real-data mode.")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")

    # Synthetic fallback
    parser.add_argument("--n-sources", type=int, default=_DEFAULT_N_SOURCES,
                        help=f"Synthetic mode: number of sources (default: {_DEFAULT_N_SOURCES})")
    parser.add_argument("--n-obs", type=int, default=_DEFAULT_N_OBS,
                        help=f"Synthetic mode: observations per source (default: {_DEFAULT_N_OBS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Synthetic mode: RNG seed (default: 42)")

    # Common
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE,
                        help=f"Sources per GPU batch (default: {_DEFAULT_BATCH_SIZE})")
    parser.add_argument("--device", default=_DEFAULT_DEVICE,
                        choices=["cpu", "cuda", "auto"],
                        help=f"periodfind device (default: {_DEFAULT_DEVICE})")
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated algorithm list, e.g. CE,AOV,LS (overrides config)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run one throwaway batch before timing (GPU warmup)")

    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    from ml4em.config.schema import PeriodConfig

    if args.source_ids:
        # Real data mode — load full config
        from ml4em.config.loader import load_config
        from ml4em.config import get_ztf_token
        cfg = load_config(args.config)
        token = get_ztf_token()
        if args.algorithms:
            cfg.features.period.algorithms = args.algorithms.split(",")
    else:
        # Synthetic fallback mode
        log.warning("No --source-ids provided. Running in synthetic mode.")
        log.warning("This only benchmarks GPU compute, NOT Kowalski I/O or Gaia xmatch.")
        log.warning("For a realistic production benchmark, provide --source-ids on MSI.")
        from ml4em.config.schema import FeatureConfig, WDBConfig
        class _FakeCfg:
            features = FeatureConfig(
                period=PeriodConfig(
                    **({"algorithms": args.algorithms.split(",")} if args.algorithms else {})
                )
            )
        cfg = _FakeCfg()
        token = None

    log.info(
        "Benchmark | mode=%s | device=%s | batch_size=%d | algorithms=%s",
        "real" if args.source_ids else "synthetic",
        args.device, args.batch_size,
        getattr(cfg.features.period, 'algorithms', 'default'),
    )

    # ── Data acquisition ─────────────────────────────────────────────────────
    kowalski_client = None
    t_fetch = None

    if args.source_ids:
        with open(args.source_ids) as f:
            source_ids = [line.strip() for line in f if line.strip()]
        log.info("Loaded %d source IDs from %s", len(source_ids), args.source_ids)

        ztf, sources, t_fetch = _fetch_from_kowalski(source_ids, cfg, token)
        kowalski_client = ztf.client
        n = len([s for s in sources if s])
    else:
        rng = np.random.default_rng(args.seed)
        t0 = time.perf_counter()
        sources = _make_synthetic_sources(args.n_sources, args.n_obs, rng)
        t_fetch_synth = time.perf_counter() - t0
        log.info("Generated %d synthetic sources (%.3fs)", args.n_sources, t_fetch_synth)
        n = len(sources)

    # ── GPU warmup ───────────────────────────────────────────────────────────
    if args.warmup or args.device == "cuda":
        log.info("GPU warmup (not timed)...")
        import periodfind
        from ml4em.features.period import PeriodExtractor
        periodfind.set_device(args.device)
        warmup = [s for s in sources if s][:min(50, n)]
        PeriodExtractor(cfg.features.period).extract(warmup)
        log.info("Warmup complete")

    # ── Per-extractor timing ─────────────────────────────────────────────────
    timings, period_results = _time_extractors(
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
    if t_fetch is not None:
        t_total += t_fetch

    mode_label = f"real (Kowalski)" if args.source_ids else "synthetic (compute only)"
    sep = "─" * 68

    print(f"\n  Mode: {mode_label}    N valid sources: {n}"
          f"    Device: {args.device}    Batch: {args.batch_size}")
    print(f"\n{sep}")
    print(f"  {'Stage':<26}{'Total (s)':>12}{'Per src (ms)':>15}{'Throughput':>13}")
    print(sep)

    if t_fetch is not None:
        print(f"  {'Kowalski fetch':<26}{t_fetch:>11.3f}s"
              f"{t_fetch / n * 1000:>13.2f}  {n / t_fetch:>9.0f}/s")
        print(sep)

    for stage, t in timings.items():
        print(f"  {stage:<26}{t:>11.3f}s"
              f"{t / n * 1000:>13.2f}  {n / t:>9.0f}/s")

    print(sep)
    print(f"  {'Total':<26}{t_total:>11.3f}s"
          f"{t_total / n * 1000:>13.2f}  {n / t_total:>9.0f}/s")
    print(sep)

    print(f"\n  Period algorithm breakdown ({n} sources):")
    for algo, count in sorted(algo_counts.items(), key=lambda x: -x[1]):
        print(f"    {algo:<6} {count:>6}  ({count / n * 100:.1f}%)")
    if nan_count:
        print(f"    {'NaN':<6} {nan_count:>6}  ({nan_count / n * 100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
