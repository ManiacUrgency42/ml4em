# Benchmarking

ml4em ships two benchmark scripts in `scripts/` for measuring end-to-end throughput
on real ZTF data.  Run these on MSI before committing to a production config to find
the right `n_workers` for your node allocation.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_single.py` | Single-source latency — one ZTF source ID through the full feature pipeline |
| `scripts/benchmark_batch.py`  | Batch throughput — real ZTF region, two-hop Kowalski fetch + full feature extraction |

**Use `benchmark_batch.py` for all MSI performance work.**  The single-source script
is for local development only; single-source latency is not meaningful at production
scale.

---

## What benchmark_batch.py does

The script mirrors the exact production workflow, staged and timed separately:

```
Round trip 1  near query    Kowalski spatial index → source IDs in sky region
Round trip 2  find query    fetch_batch() sliding window → full light curves
              Statistics    periodfind BasicStats
              Period        CE / AOV / LS / MHF (GPU batched)
              dm/dt         periodfind DmDt histogram
Round trip 3  Gaia xmatch   CatalogExtractor cone_search → Gaia EDR3 features
```

The near+find two-hop pattern matches scope-ml's `get_lightcurves_via_coords` exactly.
`find` uses the sliding window: IDs are chunked into slices of `limit_per_query` (1000),
sent `n_workers` chunks at a time so Kowalski threads stay saturated.

---

## Canonical benchmark field

Always use the same sky coordinates so timing is comparable across runs and machines.
The default field is a mid-latitude ZTF region with ~1000–1500 sources per quad:

| Parameter | Value |
|-----------|-------|
| RA | 116.7° |
| Dec | 36.2° |
| Radius | 1800 arcsec (≈ ZTF quad) |

Avoid the galactic plane for benchmarking — source density is extreme there and
unrepresentative of production WDB science runs.

---

## Finding the right n_workers

`n_workers` controls how many Kowalski queries run in parallel per sliding-window
iteration.  The right value depends on your MSI node's network allocation and
Kowalski's server-side thread limits — you must find it empirically.

Run the worker sweep below and watch the **Find (LC fetch)** throughput row.
The value where throughput stops climbing is your ceiling.

```bash
# Baseline — single-threaded
python scripts/benchmark_batch.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda --n-workers 1

# Sweep upward
python scripts/benchmark_batch.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda --n-workers 4

python scripts/benchmark_batch.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda --n-workers 8

python scripts/benchmark_batch.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda --n-workers 16
```

The output header shows the active settings so you always know what you ran:

```
Region: RA=116.7  Dec=36.2  radius=1800 arcsec    Device: cuda    Batch: 1000
Workers: 8 (LC) / 8 (Gaia)    limit_per_query: 1000
```

Once you find the optimal value, set it permanently in `config.yaml`:

```yaml
sources:
  ztf:
    n_workers: 8        # tune this to your MSI node
    limit_per_query: 1000   # do not change — MongoDB constraint
features:
  catalog:
    n_workers: 8        # match ztf n_workers
```

!!! note "limit_per_query is not a tuning knob"
    Keep `limit_per_query` at 1000.  It caps the size of MongoDB `$in` filters to
    avoid slow server-side scans.  This matches scope-ml's default and should not
    be changed regardless of your node size.

---

## GPU warmup

Period finding via periodfind/CUDA has a one-time JIT compile cost on first use.
Pass `--warmup` to run one throwaway batch before timing starts:

```bash
python scripts/benchmark_batch.py \
    --config config.yaml \
    --ra 116.7 --dec 36.2 --radius-arcsec 1800 \
    --device cuda --n-workers 8 --warmup
```

Always use `--warmup` when benchmarking GPU period finding — without it the first
batch includes compile time, which inflates the period finding row.

---

## Synthetic fallback (local dev only)

If you do not have a Kowalski token, you can run against synthetic light curves
to test the feature extraction stages.  Kowalski fetch and Gaia xmatch are not
timed in this mode.

```bash
python scripts/benchmark_batch.py --synthetic --n-sources 500 --n-obs 300
```

!!! warning
    Synthetic mode produces meaningless throughput numbers for the Kowalski stages.
    Use real-data mode on MSI for any numbers you intend to act on.

---

## Example output

```
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
```

Period finding dominates at ~14s because it is GPU-bound.  The Kowalski stages
(near + find + Gaia) together take ~11s and are network-bound — this is where
`n_workers` has the most impact.
