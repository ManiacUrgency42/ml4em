#!/usr/bin/env python3
"""
Batch throughput benchmark: synthetic light curves through the full pipeline.

Tests production-scale performance without requiring a live Kowalski connection.
Generates realistic ZTF-like light curves (sinusoidal + Gaussian noise) and
runs the complete feature pipeline, measuring throughput at each stage and overall.

This is the benchmark that actually matters for MSI runs — single-source
latency (benchmark_single.py) tells you nothing about batch GPU throughput.

Usage
-----
    # Default: 1 000 sources, 300 obs each, CPU period-finding
    python scripts/benchmark_batch.py

    # MSI production scale:
    python scripts/benchmark_batch.py --n-sources 10000 --device cuda

    # Sweep over batch sizes to find GPU memory sweet spot:
    python scripts/benchmark_batch.py --n-sources 5000 --batch-size 500

    # Quick smoke test (fast even on CPU):
    python scripts/benchmark_batch.py --n-sources 100 --n-obs 50

Output
------
    N sources: 1000    N obs: 300    Device: cuda    Batch size: 1000
    ──────────────────────────────────────────────────────────────────
      Stage                 Total (s)    Per source (ms)    Throughput
    ──────────────────────────────────────────────────────────────────
      Synthetic data gen       0.412             0.41       2 427/s
      Statistics               0.038             0.04      26 316/s
      Period finding          12.341            12.34          81/s
      dm/dt histogram          0.187             0.19       5 348/s
      ──────────────────────────────────────────────────────────────
      Total pipeline          12.566            12.57          80/s
    ──────────────────────────────────────────────────────────────────
    Period algorithm breakdown:
      CE   found    412 (41.2%)
      AOV  found    198 (19.8%)
      LS   found    389 (38.9%)
      NaN           1 (0.1%)
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
_DEFAULT_ALGORITHMS   = ["CE", "AOV", "LS", "MHF"]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _make_synthetic_sources(
    n_sources: int,
    n_obs: int,
    rng: np.random.Generator,
) -> list[list]:
    """Generate realistic ZTF-like sinusoidal light curves.

    Each source has a random period in [0.05, 5.0] days, amplitude [0.05, 0.5] mag,
    mean magnitude in [17, 20], and Gaussian noise matching ZTF's typical ~0.02 mag
    photometric precision at r < 20.

    Returns a list of single-element lists (one band per source) matching the
    FeaturePipeline.run_batch() input format.
    """
    from ml4em.types import LightCurve

    sources = []
    baseline_days  = 730.0          # 2 years of ZTF coverage
    t0_hjd         = 2_458_800.0    # approximate ZTF start HJD

    for i in range(n_sources):
        period    = float(rng.uniform(0.05, 5.0))
        amplitude = float(rng.uniform(0.05, 0.5))
        phase     = float(rng.uniform(0, 2 * np.pi))
        mean_mag  = float(rng.uniform(17.0, 20.0))
        noise     = float(rng.uniform(0.01, 0.05))   # photometric error

        t = np.sort(rng.uniform(0, baseline_days, n_obs).astype(np.float64)) + t0_hjd
        signal = amplitude * np.sin(2 * np.pi * t / period + phase)
        m = mean_mag + signal + rng.normal(0, noise, n_obs)
        e = np.full(n_obs, noise, dtype=np.float64)

        lc = LightCurve(
            source_id = f"synth_{i:06d}",
            time      = t,
            mag       = m,
            mag_err   = e,
            band      = "r",
            survey    = "simulated",
            ra        = float(rng.uniform(0, 360)),
            dec       = float(rng.uniform(-30, 90)),
        )
        sources.append([lc])

    return sources


# ---------------------------------------------------------------------------
# Per-extractor timing (without FeaturePipeline overhead)
# ---------------------------------------------------------------------------

def _time_extractors(
    sources: list[list],
    cfg,
    device: str,
    batch_size: int,
) -> dict:
    """Time each extractor independently on the same source list.

    Returns dict of {stage_name: seconds}.
    """
    import periodfind
    from ml4em.features.statistics import StatisticsExtractor
    from ml4em.features.period     import PeriodExtractor
    from ml4em.features.dmdt       import DmdtExtractor

    periodfind.set_device(device)
    timings: dict[str, float] = {}

    # -- Statistics -----------------------------------------------------------
    ext_stats = StatisticsExtractor()
    t0 = time.perf_counter()
    for i in range(0, len(sources), batch_size):
        ext_stats.extract(sources[i : i + batch_size])
    timings["Statistics"] = time.perf_counter() - t0

    # -- Period finding -------------------------------------------------------
    ext_period = PeriodExtractor(cfg.period)
    t0 = time.perf_counter()
    period_results = []
    for i in range(0, len(sources), batch_size):
        period_results.extend(ext_period.extract(sources[i : i + batch_size]))
    timings["Period finding"] = time.perf_counter() - t0

    # -- dm/dt histogram ------------------------------------------------------
    ext_dmdt = DmdtExtractor(cfg.dmdt)
    t0 = time.perf_counter()
    for i in range(0, len(sources), batch_size):
        ext_dmdt.extract(sources[i : i + batch_size])
    timings["dm/dt histogram"] = time.perf_counter() - t0

    return timings, period_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch throughput benchmark — synthetic sources through the full pipeline"
    )
    parser.add_argument("--n-sources",  type=int,   default=_DEFAULT_N_SOURCES,
                        help=f"Number of synthetic sources (default: {_DEFAULT_N_SOURCES})")
    parser.add_argument("--n-obs",      type=int,   default=_DEFAULT_N_OBS,
                        help=f"Observations per source (default: {_DEFAULT_N_OBS})")
    parser.add_argument("--batch-size", type=int,   default=_DEFAULT_BATCH_SIZE,
                        help=f"Sources per GPU batch (default: {_DEFAULT_BATCH_SIZE})")
    parser.add_argument("--device",     default=_DEFAULT_DEVICE,
                        choices=["cpu", "cuda", "auto"],
                        help=f"periodfind device (default: {_DEFAULT_DEVICE})")
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated period algorithm list (default: CE,AOV,LS,MHF)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="RNG seed for reproducible synthetic data (default: 42)")
    parser.add_argument("--warmup",     action="store_true",
                        help="Run one throwaway batch before timing (GPU warmup)")
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    from ml4em.config.schema import FeatureConfig, PeriodConfig

    period_cfg_kwargs: dict = {}
    if args.algorithms:
        period_cfg_kwargs["algorithms"] = args.algorithms.split(",")

    cfg = FeatureConfig(period=PeriodConfig(**period_cfg_kwargs))

    log.info(
        "Benchmark config | n_sources=%d | n_obs=%d | device=%s | "
        "batch_size=%d | algorithms=%s",
        args.n_sources, args.n_obs, args.device,
        args.batch_size, cfg.period.algorithms,
    )

    # ── Synthetic data ───────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    sources = _make_synthetic_sources(args.n_sources, args.n_obs, rng)
    t_gen = time.perf_counter() - t0
    log.info("Generated %d synthetic sources (%.3fs)", args.n_sources, t_gen)

    # ── GPU warmup ───────────────────────────────────────────────────────────
    if args.warmup or args.device == "cuda":
        log.info("GPU warmup (not timed)...")
        import periodfind
        from ml4em.features.period import PeriodExtractor
        periodfind.set_device(args.device)
        warmup_batch = sources[:min(50, args.n_sources)]
        PeriodExtractor(cfg.period).extract(warmup_batch)
        log.info("Warmup complete")

    # ── Per-extractor timing ─────────────────────────────────────────────────
    timings, period_results = _time_extractors(
        sources, cfg, args.device, args.batch_size
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

    # ── Summary ─────────────────────────────────────────────────────────────
    n = args.n_sources
    t_total = sum(timings.values())

    sep  = "─" * 68
    sep2 = "─" * 68
    print(f"\n  N sources: {n}    N obs: {args.n_obs}    "
          f"Device: {args.device}    Batch size: {args.batch_size}")
    print(f"\n{sep}")
    print(f"  {'Stage':<26}{'Total (s)':>12}{'Per src (ms)':>15}{'Throughput':>13}")
    print(sep)
    print(f"  {'Synthetic data gen':<26}{t_gen:>11.3f}s"
          f"{t_gen / n * 1000:>13.2f}  {n / t_gen:>9.0f}/s")
    print(sep)
    for stage, t in timings.items():
        print(f"  {stage:<26}{t:>11.3f}s"
              f"{t / n * 1000:>13.2f}  {n / t:>9.0f}/s")
    print(sep)
    print(f"  {'Total pipeline':<26}{t_total:>11.3f}s"
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
