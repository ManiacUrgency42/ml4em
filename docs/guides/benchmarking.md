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
| `benchmarks/scaling_cpu.py` | CPU core scaling and Amdahl fit for period finding |
| `benchmarks/scaling_gpu.py` | Multi-GPU strong scaling for period finding |
| `benchmarks/plot_scaling.py` | Renders the figures from the two scaling runs |

**Use `batch_throughput.py` for all MSI compute estimates.**
The single-source script is for local development only — single-source latency
does not predict batch throughput because GPU utilization is near zero on one source.

---

## Running on MSI

Each Python benchmark has a matching SLURM wrapper in `benchmarks/slurm/`.
Submit them from the repository root:

```bash
cd ~/ml4em
sbatch benchmarks/slurm/batch_throughput.sh
sbatch benchmarks/slurm/sweep_workers.sh
sbatch benchmarks/slurm/sweep_batch_size.sh
sbatch benchmarks/slurm/sweep_nobs.sh
sbatch benchmarks/slurm/scaling_cpu.sh
sbatch benchmarks/slurm/scaling_gpu.sh
sbatch benchmarks/slurm/single_latency.sh
```

Extra flags are forwarded to the Python script, so you can override a parameter
without editing anything:

```bash
sbatch benchmarks/slurm/batch_throughput.sh --n-workers 8 --device cuda
sbatch benchmarks/slurm/scaling_cpu.sh --max-sources 2000 --trials 3
```

| Script | Partition | Resources |
|--------|-----------|-----------|
| `batch_throughput.sh` | `msigpu` | 1×A100, 16 CPU, 32 GB, 30 min |
| `single_latency.sh` | `msigpu` | 1×A100, 4 CPU, 16 GB, 15 min |
| `sweep_batch_size.sh` | `msigpu` | 1×A100, 8 CPU, 64 GB, 1 h |
| `sweep_nobs.sh` | `msigpu` | 1×A100, 8 CPU, 32 GB, 1 h |
| `scaling_gpu.sh` | `msigpu` | 4×A100, 32 CPU, 128 GB, 4 h |
| `sweep_workers.sh` | `msismall` | 32 CPU, 32 GB, 1 h |
| `scaling_cpu.sh` | `msismall` | 64 CPU, 128 GB, 12 h |

The two Kowalski-bound benchmarks run on `msismall` because nothing in them
touches a GPU — `sweep_workers` measures network throughput and `scaling_cpu`
deliberately measures the CPU backend.

Check that your account is authorized for a partition before queuing a long job:

```bash
sbatch --test-only benchmarks/slurm/scaling_gpu.sh
```

### Preconditions

Every wrapper sources `/scratch.global/$USER/ml4em_data/.env` and then refuses to
start unless all of the following hold:

- `/scratch.global/$USER/ml4em_gpu.sif` exists (`sbatch slurm/pull_image.sh`)
- `/scratch.global/$USER/ml4em_data/config_msi.yaml` exists
- `ML4EM_ZTF_TOKEN` is set

Neither is checked by the Python scripts themselves, and both would otherwise
appear several minutes in as a pydantic validation error or a Kowalski
authentication failure that does not name the missing file. Run
`python scripts/get_credentials.py` to populate the token.

The benchmarks run inside the same Apptainer image as the demo, with the same
`apptainer run --bind REPO:/app/ml4em --bind DATA:/data` invocation, so there is
one environment to keep working rather than two. GPU jobs add `--nv`; the two
CPU jobs (`scaling_cpu.sh`, `sweep_workers.sh`) omit it.

The rest of the SLURM conventions these scripts follow — submitting from the
repo root, the tracked `logs/` directory — are described in
[Deployment → SLURM conventions](../deployment.md#slurm-conventions).

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

## Step 6 — Parallel scaling

Steps 1–5 size a single run. The two scaling benchmarks answer a different
question: how much faster does period finding get when you add hardware.

Both fetch the region once, cache the light curves, and then re-run period
finding on that same cached set at every parallelism level, so the measurement
isolates compute from network variance.

```bash
sbatch benchmarks/slurm/scaling_cpu.sh
sbatch benchmarks/slurm/scaling_gpu.sh
```

`scaling_cpu.py` runs each core count in its own subprocess. That is not an
implementation detail worth hiding: periodfind's Rayon thread pool is sized once
from `RAYON_NUM_THREADS` when the extension loads, so a single process cannot
change its core count between measurements. It fits Amdahl's law to the results
and reports the serial fraction:

```
  Cores     Runtime       Ideal      Amdahl  ...
```

`scaling_gpu.py` shards the workload across local devices via
`CUDA_VISIBLE_DEVICES` and reports median light curves per second with the
run-to-run spread. It uses no MPI, so it cannot span nodes — the SLURM wrapper
requests 4 A100s on one node for that reason.

Run `sweep_batch_size.sh` (Step 3) first. Scaling measured at a batch size below
the per-GPU optimum understates what the hardware can do, because each device is
underutilised at every point on the curve.

Both write JSON to `logs/benchmarks/` and the SLURM wrappers finish by calling:

```bash
python benchmarks/plot_scaling.py
```

which renders `logs/benchmarks/scaling.png` from whichever of
`scaling_cpu.json` and `scaling_gpu.json` are present. The plot call is
tolerant of a missing matplotlib — the benchmark result is the JSON, and the
figure is a convenience on top of it.

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

