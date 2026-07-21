# Deployment — Apptainer

The pipeline is packaged as a Docker image, stored on GHCR, and converted to an
Apptainer `.sif` file on MSI. Your live git checkout is bind-mounted into the container
at runtime, so any Python code you change is picked up immediately with `git pull` —
no rebuild needed.

!!! tip "How Docker and GHCR fit into this"
    If you want to understand how the image is built, how GHCR caching works, or when
    CI triggers a rebuild, see [Background → Docker & GHCR](background/docker-ghcr.md).

---

## 1. Clone the repo on MSI

SSH into MSI and clone the repo with submodules. `periodfind` lives at
`external/periodfind` as a git submodule — without `--recurse-submodules` the
build will fail.

```bash
git clone --recurse-submodules https://github.com/ManiacUrgency42/ml4em.git ~/ml4em
cd ~/ml4em
```

---

## 2. First-time data setup

Do this once. These directories and files persist across jobs on scratch.

### 2a. Create scratch directories

MSI's home quota is small (~10 GB). Keep all large files on scratch:

```bash
DATA=/scratch.global/$USER/ml4em_data

mkdir -p $DATA/features $DATA/models $DATA/predictions
mkdir -p /scratch.global/$USER/apptainer_cache
mkdir -p /scratch.global/$USER/tmp
```

### 2b. Copy your catalog

`wdb_sources.csv` is the WDB catalog (ra, dec positions of target sources). Copy it
to scratch from your local machine:

```bash
# From your local machine:
scp data/wdb_sources.csv jin00404@login.msi.umn.edu:/scratch.global/jin00404/ml4em_data/
```

### 2c. Write a config for MSI

The MSI config uses absolute scratch paths instead of relative ones. Create it at
`/scratch.global/$USER/ml4em_data/config_msi.yaml`:

```yaml
sources:
  ztf:
    host: melman.caltech.edu
    port: 443
    timeout: 300
    collection_sources: ZTF_sources_84525009
    max_timestamp_hjd: 2459951.5
    bands: [g, r, i]
    min_cadence_days: 0.020833

features:
  device: cuda          # use GPU for periodfind
  min_observations: 50

storage:
  catalog_path: /data/wdb_sources.csv
  labels_path:  /data/labels.csv
  features_dir:    /data/features
  models_dir:      /data/models
  predictions_dir: /data/predictions

training:
  val_fraction:  0.1
  test_fraction: 0.1
  seed: 42

inference:
  confidence_thresholds:
    high:   0.9
    medium: 0.7
```

The bind mount in the run command maps `/scratch.global/$USER/ml4em_data` → `/data`
inside the container, so all paths above resolve correctly.

### 2d. Store your Kowalski token

Never put tokens in `config_msi.yaml`. Store them in a `.env` file:

```bash
echo "ML4EM_ZTF_TOKEN=your_token_here" > /scratch.global/$USER/ml4em_data/.env
chmod 600 /scratch.global/$USER/ml4em_data/.env
```

The `--env-file` flag in the run scripts injects this into the container at runtime.

---

## 3. Pull the image on MSI

Submit the SLURM pull job from your repo root. This downloads the Docker image
from GHCR, decompresses every layer, merges them into a single filesystem, and
converts it to a `.sif` (SquashFS Image Format) file on scratch. This is the
one-time cost — you never repeat it unless a compiled dependency changes.

```bash
cd ~/ml4em
mkdir -p logs
sbatch slurm/pull_image.sh
```

Monitor progress:

```bash
squeue -u $USER
tail -f logs/pull_ml4em_<JOBID>.out
```

The pull takes 20–40 minutes (5–8 GB download + squashfs conversion). Output is
written to `/scratch.global/$USER/ml4em_gpu.sif`.

!!! warning "Do not pull on the login node"
    The login node enforces a 15-minute CPU limit — not enough time to convert a
    5–8 GB image. `slurm/pull_image.sh` submits the job to a compute node
    automatically via SLURM.

---

## 4. Run the demo

### Batch job (recommended)

The repo includes `slurm/run_demo.sh`, a complete SLURM script that sets up the
environment and runs the end-to-end demo pipeline. Submit it from your repo root:

```bash
mkdir -p logs
sbatch slurm/run_demo.sh
```

Monitor:

```bash
squeue -u $USER
tail -f logs/ml4em_demo_<JOBID>.out
```

The script requests one A100 GPU, bind-mounts your live git checkout into the
container, and runs `scripts/run_demo.py` with the MSI config. Outputs:

- `/scratch.global/$USER/ml4em_data/features/demo.parquet` — extracted feature vectors
- `/scratch.global/$USER/ml4em_data/models/logistic_demo/` — saved model

### Interactive run (for debugging only)

!!! warning "Must be on a GPU compute node"
    `apptainer run --nv` requires GPU drivers. **Never run it on the login node** —
    request an interactive GPU node first:

    ```bash
    srun --account=cough052 --partition=a100 --gres=gpu:a100:1 \
         --mem=16g --time=1:00:00 --pty bash
    ```

Once on the compute node:

```bash
DATA=/scratch.global/$USER/ml4em_data

module load apptainer

apptainer run --nv \
    --bind $HOME/ml4em:/app/ml4em \
    --bind $DATA:/data \
    --env-file $DATA/.env \
    /scratch.global/$USER/ml4em_gpu.sif \
    python scripts/run_demo.py --config /data/config_msi.yaml
```

| Flag | Purpose |
|------|---------|
| `--nv` | Pass NVIDIA GPU drivers through to the container |
| `--bind .../ml4em:/app/ml4em` | Mount your live code — `git pull` picks up changes without rebuilding |
| `--bind .../ml4em_data:/data` | Mount data directory — catalog, config, output features/models |
| `--env-file .env` | Inject `ML4EM_ZTF_TOKEN` into the container |

GPU device is controlled by `features.device` in `config_msi.yaml` (`"cpu"` / `"cuda"` / `"auto"`).

---

## Updating the pipeline after the initial pull

**The `.sif` is a one-time cost.** Once it exists on scratch, you never pull it
again unless a compiled dependency changes. All Python code — models, scripts,
pipeline logic — lives in your git checkout and is bind-mounted live into the
container at runtime.

The day-to-day workflow is just:

```bash
# On MSI login node — pick up any code changes from GitHub
git pull

# Submit your job — the container immediately sees the updated code
sbatch slurm/run_demo.sh
```

That's it. No rebuild, no re-pull, no reinstall.

## When to re-pull the .sif

The image only needs to be re-pulled when compiled dependencies change — things
that are baked into the image at build time and cannot be bind-mounted.

| Change | Action needed on MSI |
|--------|----------------------|
| `src/ml4em/` Python code | `git pull` only — no re-pull |
| `scripts/`, `slurm/`, `docs/` | `git pull` only — no re-pull |
| `pyproject.toml` — new pip dependency | Re-pull `.sif` after CI rebuilds image |
| `external/periodfind` submodule update | Re-pull `.sif` after CI rebuilds image |
| CUDA driver compatibility change | Re-pull `.sif` after CI rebuilds image |

CI rebuilds the image automatically when you merge to `main` — you just wait
for the build to finish, then run `sbatch slurm/pull_image.sh` again on MSI.
