"""
Shared helpers for the scaling benchmarks.

Both scaling_gpu.py and scaling_cpu.py need the same three things:

  1. A fixed, reproducible set of light curves.  Scaling curves are only
     meaningful if every measurement point processes identical data, so the
     light curves are fetched from Kowalski once and cached to a local .npz.
     Every subsequent trial and every worker subprocess loads that cache —
     no network traffic enters the timed region.

  2. A way to run period finding on a slice of those light curves with an
     explicit device, returning wall time.

  3. A common JSON result format so plot_scaling.py can render either
     scaling curve without knowing which script produced it.

The cache is keyed by every config field that changes which epochs come back
(region, collection, cadence and epoch-count cuts, bands, program IDs) so
changing any of those produces a new file rather than silently reusing stale
data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_CACHE_DIR   = "logs/benchmarks/cache"
_RESULTS_DIR = "logs/benchmarks"


# ---------------------------------------------------------------------------
# Light curve cache
# ---------------------------------------------------------------------------

def cache_key(ra: float, dec: float, radius: float, cfg) -> str:
    """Stable hash of every parameter that affects the fetched light curves."""
    payload = "|".join(str(x) for x in (
        round(ra, 6),
        round(dec, 6),
        round(radius, 3),
        cfg.sources.ztf.collection_sources,
        cfg.sources.ztf.min_cadence_days,
        cfg.sources.ztf.max_timestamp_hjd,
        cfg.sources.ztf.min_nobs,
        sorted(cfg.sources.ztf.bands),
        sorted(cfg.sources.ztf.program_ids),
        cfg.features.min_observations,
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def cache_path(ra: float, dec: float, radius: float, cfg) -> str:
    return os.path.join(_CACHE_DIR, f"lcs_{cache_key(ra, dec, radius, cfg)}.npz")


def _save_cache(path: str, sources: list[list]) -> None:
    """Flatten list-of-lists of LightCurve into a single .npz.

    Ragged arrays are stored concatenated with an offsets array so the
    structure can be rebuilt exactly on load.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    times, mags, errs        = [], [], []
    offsets                  = [0]
    source_index             = []   # which source each LC belongs to
    band, sid, ra_arr, dec_arr = [], [], [], []

    for s_idx, lcs in enumerate(sources):
        for lc in lcs:
            times.append(lc.time)
            mags.append(lc.mag)
            errs.append(lc.mag_err)
            offsets.append(offsets[-1] + len(lc.time))
            source_index.append(s_idx)
            band.append(lc.band)
            sid.append(lc.source_id)
            ra_arr.append(lc.ra)
            dec_arr.append(lc.dec)

    tmp = path[:-4] + ".tmp.npz"  # keep .npz suffix so numpy doesn't append another
    np.savez_compressed(
        tmp,
        time         = np.concatenate(times) if times else np.zeros(0),
        mag          = np.concatenate(mags)  if mags  else np.zeros(0),
        mag_err      = np.concatenate(errs)  if errs  else np.zeros(0),
        offsets      = np.array(offsets, dtype=np.int64),
        source_index = np.array(source_index, dtype=np.int64),
        band         = np.array(band),
        source_id    = np.array(sid),
        ra           = np.array(ra_arr, dtype=np.float64),
        dec          = np.array(dec_arr, dtype=np.float64),
        n_sources    = np.array([len(sources)], dtype=np.int64),
    )
    os.replace(tmp, path)


def _load_cache(path: str) -> list[list]:
    from ml4em.types import LightCurve

    # np.load on a compressed .npz returns a lazy NpzFile, and its __getitem__
    # re-inflates the whole member on EVERY access.  The loop below touches
    # seven members per light curve, so indexing the NpzFile directly turns a
    # seconds-long load into hours of redundant zlib work.  Materialize each
    # member exactly once and index plain ndarrays instead.
    with np.load(path, allow_pickle=False) as npz:
        d = {k: npz[k] for k in npz.files}
    offsets   = d["offsets"]
    src_index = d["source_index"]
    n_sources = int(d["n_sources"][0])

    sources: list[list] = [[] for _ in range(n_sources)]
    for i in range(len(src_index)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        sources[int(src_index[i])].append(
            LightCurve(
                source_id = str(d["source_id"][i]),
                time      = d["time"][lo:hi].astype(np.float64),
                mag       = d["mag"][lo:hi].astype(np.float64),
                mag_err   = d["mag_err"][lo:hi].astype(np.float64),
                band      = str(d["band"][i]),
                survey    = "ztf",
                ra        = float(d["ra"][i]),
                dec       = float(d["dec"][i]),
            )
        )
    return sources


def load_or_fetch_sources(
    ra: float,
    dec: float,
    radius_arcsec: float,
    cfg,
    token: str,
    refresh: bool = False,
) -> list[list]:
    """Return the cached valid sources for a region, fetching them if needed.

    A 'source' is a list of LightCurve (one per band).  Only sources meeting
    cfg.features.min_observations are returned, so every scaling measurement
    operates on exactly the workload the production pipeline would see.
    """
    path = cache_path(ra, dec, radius_arcsec, cfg)

    if os.path.exists(path) and not refresh:
        sources = _load_cache(path)
        log.info("Loaded %d cached sources from %s", len(sources), path)
        return sources

    from ml4em.data.ztf import ZTFSource

    log.info("Cache miss — fetching from Kowalski (RA=%.4f Dec=%.4f r=%.0f\")",
             ra, dec, radius_arcsec)
    ztf = ZTFSource(cfg.sources.ztf, token)
    source_ids, lcs = ztf.fetch_by_region(ra, dec, radius_arcsec)
    if not lcs:
        raise RuntimeError("No light curves returned — try a larger radius.")

    groups: dict[str, list] = defaultdict(list)
    for lc in lcs:
        groups[lc.source_id].append(lc)

    min_obs = cfg.features.min_observations
    sources = [
        groups[sid] for sid in source_ids
        if groups.get(sid) and max(lc.n_obs for lc in groups[sid]) >= min_obs
    ]

    log.info("Fetched %d light curves → %d valid sources (>= %d obs)",
             len(lcs), len(sources), min_obs)
    _save_cache(path, sources)
    log.info("Cached to %s", path)
    return sources


# ---------------------------------------------------------------------------
# Workload shaping
# ---------------------------------------------------------------------------

def split_by_band(sources: list[list], min_obs: int) -> list[list]:
    """Expand per-source groups into one single-band group per light curve.

    FeaturePipeline.run_batch does exactly this before handing work to the
    extractors, so a source with g and r light curves is two units of work,
    not one.  Benchmarks that time the per-source groups directly measure only
    the longest band — the extractors pick it internally — and therefore
    understate real pipeline cost by roughly the mean band count per source
    (2–3x for ZTF).  Every timing path here splits first so the numbers can be
    multiplied by a source count without a hidden correction factor.

    Bands below `min_obs` epochs are dropped, matching the pipeline's own
    per-unit validity cut.
    """
    return [[lc] for lcs in sources for lc in lcs if lc.n_obs >= min_obs]


# ---------------------------------------------------------------------------
# Timed period finding
# ---------------------------------------------------------------------------

def time_period_finding(
    sources: list[list],
    cfg,
    device: str,
    batch_size: int,
    warmup: bool = True,
) -> float:
    """Run period finding over `sources` and return wall seconds.

    `device` is periodfind's vocabulary: 'cpu' or 'gpu'.
    A warmup batch is run first (and excluded) so CUDA JIT compilation and
    rayon thread-pool spin-up do not land inside the measured region.

    `sources` is a list of per-source groups; it is split into one unit per
    band before timing, matching the pipeline.  The returned time is therefore
    the cost of the whole source list including every band, and dividing by
    len(sources) gives a true sources/second figure.

    prepare() is called on the full work list before timing, exactly as the
    pipeline does.  This is not optional for a benchmark: the frequency grid is
    sized from the longest baseline in whatever list it is given, so letting
    each batch build its own grid would make the per-batch work depend on which
    sources landed in that batch, and the measured time would then vary with
    batch composition rather than with the quantity under test.
    """
    import periodfind
    from ml4em.features.period import PeriodExtractor

    periodfind.set_device(normalize_device(device))

    units = split_by_band(sources, cfg.features.min_observations)

    if warmup:
        ext = PeriodExtractor(cfg.features.period)
        ext.prepare(units)
        ext.extract(units[: min(32, len(units))])

    ext = PeriodExtractor(cfg.features.period)
    ext.prepare(units)
    t0 = time.perf_counter()
    for i in range(0, len(units), batch_size):
        ext.extract(units[i : i + batch_size])
    return time.perf_counter() - t0


def normalize_device(device: str) -> str:
    """Map user-facing device names onto periodfind's 'cpu' | 'gpu'."""
    d = device.lower()
    if d in ("cuda", "gpu"):
        return "gpu"
    if d == "cpu":
        return "cpu"
    if d == "auto":
        import periodfind
        return periodfind.get_device()
    raise ValueError(f"Unknown device '{device}'. Use cpu, gpu/cuda, or auto.")


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def write_results(name: str, payload: dict[str, Any]) -> str:
    """Write a benchmark result JSON and return its path."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    path = os.path.join(_RESULTS_DIR, f"{name}.json")
    tmp  = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def gpu_name() -> str:
    """Best-effort GPU model string, e.g. 'A100-SXM4-40GB'.  'unknown' if none."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            raw = out.stdout.strip().splitlines()[0].strip()
            return raw.replace("NVIDIA ", "").replace("Tesla ", "")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def n_visible_gpus() -> int:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return len([l for l in out.stdout.strip().splitlines() if l.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
