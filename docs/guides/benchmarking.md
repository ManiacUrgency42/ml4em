# Benchmarking

ml4em has a dedicated `benchmarks/` directory with scripts for measuring
end-to-end throughput, tuning parallelism, and projecting MSI compute hours.
Run these on MSI before production runs to find the optimal configuration
for your node allocation.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `benchmarks/batch_throughput.py` | End-to-end pipeline on a real ZTF region — primary benchmark for compute estimation |
| `benchmarks/single_latency.py` | Single-source latency — useful for CPU vs GPU comparison, not throughput estimation |
| `benchmarks/sweep_workers.py` | Sweep `n_workers` to find the Kowalski LC fetch throughput ceiling |
| `benchmarks/sweep_batch_size.py` | Sweep `feature_batch_size` to find the GPU throughput ceiling and VRAM limit |
| `benchmarks/sweep_nobs.py` | Bucket real ZTF LCs by observation count, time period finding per bucket |

**Use `batch_throughput.py` for all MSI compute estimates.**
The single-source script is for local development only — single-source latency
does not predict batch throughput because GPU utilization is near zero on one source.

---

## Canonical benchmark field

Always use the same sky coordinates so timing is comparable across runs and machines:

| Parameter | Value |
|-----------|-------|
| RA | 116.7° |
| Dec | 36.2° |
| Radius | 1800 arcsec (≈ ZTF quad) |

---

## Step 1 — End-to-end baseline

Get a single-run baseline with default config before tuning anything:

```bash
python benchmarks/batch_throughput.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda --warmup
```

The output header shows exactly what configuration ran:

```
Region: RA=116.7  Dec=36.2  radius=1800 arcsec    Device: cuda    Batch: 1000
Workers: 1 (LC) / 1 (Gaia)    limit_per_query: 1000
```

---

## Step 2 — Find the n_workers ceiling

`n_workers` controls how many Kowalski queries run in parallel per sliding-window
iteration.  More workers reduces time waiting on network latency — up to the point
where Kowalski server throttling or bandwidth becomes the bottleneck.

Run the worker sweep to find the knee of the throughput curve:

```bash
python benchmarks/sweep_workers.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800
```

The script runs the find query once per worker count and prints:

```
  n_workers    Iterations    Find time (s)    Throughput    Speedup
  ─────────────────────────────────────────────────────────────────
          1          1247          87.3s          14/s        1.0x
          4           312          24.3s          51/s        3.6x
          8           156          12.7s          98/s        6.9x   ← best
         16            78          12.1s         103/s        7.3x
         32            39          13.8s          90/s        6.3x   ← degraded
```

The recommended value is printed at the bottom.  Set it permanently in `config.yaml`:

```yaml
sources:
  ztf:
    n_workers: 8
    limit_per_query: 1000   # do not change — MongoDB constraint
features:
  catalog:
    n_workers: 8
```

!!! note "limit_per_query is not a tuning knob"
    Keep `limit_per_query` at 1000.  It caps the size of MongoDB `$in` filters
    to keep individual queries fast.  Matches scope-ml's default.

---

## Step 3 — Find the GPU batch size ceiling

`feature_batch_size` controls how many sources are sent to periodfind per GPU call.
Larger batches improve GPU lane utilization but use more VRAM.  The ceiling is the
largest batch that fits in your GPU's memory without OOM.

```bash
python benchmarks/sweep_batch_size.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda
```

Output stops at OOM:

```
  batch_size    Period time (s)    Throughput    vs batch=100
  ──────────────────────────────────────────────────────────
         100             62.1s          17/s           1.0x
         500             16.9s          62/s           3.7x
        1000             14.3s          74/s           4.4x  ← best
        2000             14.1s          75/s           4.4x
        4000          OOM — VRAM exceeded
```

Set the recommended value in `config.yaml`:

```yaml
features:
  feature_batch_size: 1000
```

---

## Step 4 — Understand how n_obs drives compute time

Period finding time scales with observations per light curve (n_obs).  This sweep
groups real ZTF LCs by n_obs and times period finding per bucket — giving you the
ms/source breakdown by LC length.

```bash
python benchmarks/sweep_nobs.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda
```

Output includes a compute projection for your science scope:

```
  n_obs bucket    Sources    Period time (s)    ms / source    % of total
  ──────────────────────────────────────────────────────────────────────
      50 – 100         87             0.41s            4.7         2.9%
     100 – 200        213             1.83s            8.6        12.8%
     200 – 500        418             5.12s           12.2        35.8%
     500 – 1000       287             5.63s           19.6        39.4%
    1000 – 2000        47             1.29s           27.4         9.0%

  Compute projection (period finding only):
      100,000 sources →  ~0.4 GPU-hours
    1,000,000 sources →  ~3.8 GPU-hours
   10,000,000 sources → ~37.8 GPU-hours
```

!!! note "Projection assumes your target fields have a similar n_obs distribution"
    Run this sweep on a region representative of your actual science target.
    A dense galactic field with short LCs is much faster per source than a
    sparse high-latitude field with long baselines.

---

## Step 5 — Final compute estimate

After Steps 1–4, compute total GPU-hours as:

```
GPU-hours = (total_sources × ms_per_source_from_sweep_nobs) / 3_600_000
```

Add the Kowalski fetch time from `batch_throughput.py` for total wall time.
Kowalski stages consume wall time (node reserved) but minimal GPU-hours.

---

## GPU warmup

Pass `--warmup` to run one throwaway batch before timing starts.
Always use this for GPU period finding benchmarks — without it the first
batch includes CUDA JIT compile time, which inflates the timing.

```bash
python benchmarks/batch_throughput.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda --n-workers 8 --warmup
```

