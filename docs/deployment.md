# Deployment

ml4em runs on MSI via Apptainer. `periodfind` — the GPU-accelerated feature extraction
library — requires a Rust + CUDA build that cannot be done with `pip`, so the full
toolchain is baked into a Docker image published on GHCR (GitHub Container Registry).

**Most researchers never need to build the image.** The pipeline is installed in
editable mode inside the container, and your live git checkout is bind-mounted over
it at runtime — so any Python code you add or change is picked up immediately without
touching the image. Just pull and run.

| You are… | What you need to do |
|----------|---------------------|
| Running the pipeline or adding a model | `apptainer pull` once, then `git pull` for updates |
| Adding a new Python dependency (`pyproject.toml`) | Open a PR → merge to `main` → CI rebuilds automatically → re-pull the `.sif` |
| Maintaining the image (periodfind / CUDA / Dockerfile) | Same as above — CI rebuilds on merge |

The image is rebuilt automatically by GitHub Actions whenever `Dockerfile`,
`pyproject.toml`, or `external/periodfind/` changes on `main`. The section below
documents how to trigger a manual build if needed.

---

## 1. Build the image (maintainers only)

**Normally you don't need to do this** — merging to `main` triggers the GitHub Actions
workflow (`.github/workflows/docker.yml`) which builds and pushes the GPU image on
native x86_64 runners automatically. Layer caching means rebuilds after a
`pyproject.toml`-only change skip the Rust/CUDA compilation entirely.

Only run a local build if you need to test a Dockerfile change before merging, or if
CI is unavailable. On Apple Silicon, the `--platform linux/amd64` flag forces QEMU
emulation — the GPU build will take 3+ hours. Prefer pushing to a branch and letting
CI build it.

```bash
# GPU image — for MSI production runs
docker build --platform linux/amd64 --target gpu \
    -t ghcr.io/maniacurgency42/ml4em:gpu .

# CPU image — for local testing only
docker build --platform linux/amd64 --target cpu \
    -t ghcr.io/maniacurgency42/ml4em:cpu .
```

!!! note "Submodule required"
    `periodfind` lives at `external/periodfind` as a git submodule. Make sure it is
    initialized before building:
    ```bash
    git clone --recurse-submodules <repo-url>
    # or after a plain clone:
    git submodule update --init
    ```

---

## 2. Push to GHCR (local)

GHCR (GitHub Container Registry) hosts the image so MSI can pull it without needing
Docker installed. Authenticate with a GitHub Personal Access Token (PAT):

**GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → Generate new token**

Select scopes: `write:packages`, `read:packages`

```bash
echo YOUR_PAT | docker login ghcr.io -u ManiacUrgency42 --password-stdin
```

Then push:

```bash
docker push ghcr.io/maniacurgency42/ml4em:gpu
docker push ghcr.io/maniacurgency42/ml4em:cpu
```

The GPU image is 5–8 GB. If the push times out mid-upload, rerun — Docker skips
already-uploaded layers.

!!! note "First push only"
    After the first push, make the package public so MSI can pull without credentials:
    **GitHub → your profile → Packages → ml4em → Package Settings → Change visibility → Public**

---

## 3. First-time MSI setup

Do this once after SSH-ing into MSI. These directories persist across jobs.

### 3a. Create scratch directories

MSI's home quota is small. Keep all large files on scratch:

```bash
DATA=/scratch.global/$USER/ml4em_data

mkdir -p $DATA/features $DATA/models $DATA/predictions
mkdir -p /scratch.global/$USER/apptainer_cache
mkdir -p /scratch.global/$USER/tmp
```

### 3b. Copy your catalog

`wdb_sources.csv` is the WDB catalog (ra, dec positions of target sources). Copy it
to scratch from your local machine:

```bash
# From your local machine:
scp data/wdb_sources.csv jin00404@login.msi.umn.edu:/scratch.global/jin00404/ml4em_data/
```

### 3c. Write a config for MSI

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

### 3d. Store your Kowalski token

Never put tokens in `config_msi.yaml`. Store them in a `.env` file:

```bash
echo "ML4EM_ZTF_TOKEN=your_token_here" > /scratch.global/$USER/ml4em_data/.env
chmod 600 /scratch.global/$USER/ml4em_data/.env
```

The `--env-file` flag in the run scripts injects this into the container at runtime.

---

## 4. Pull the image on MSI

The repo includes a ready-made SLURM script that handles the pull correctly —
no interactive node needed:

```bash
# From your ml4em repo root on MSI:
mkdir -p logs
sbatch slurm/pull_image.sh
```

Monitor progress:

```bash
squeue -u $USER
tail -f logs/pull_ml4em_<JOBID>.out
```

The pull takes 20–40 minutes (5–8 GB download + squashfs conversion). Output is
written to `/scratch.global/$USER/ml4em_gpu.sif`. You only need to re-run this
when a new image is pushed to GHCR.

!!! warning "Do not pull on the login node"
    The login node enforces a 15-minute CPU limit — not enough time to convert a
    5–8 GB image. `slurm/pull_image.sh` submits the job to a compute node
    automatically via SLURM.

---

## 5. Run the demo on MSI

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

## When to rebuild

The image only needs to be rebuilt (and re-pulled on MSI) when compiled dependencies change.

| Change | Rebuild needed? |
|--------|----------------|
| `src/ml4em/` code changes | **No** — `git pull` on MSI is enough |
| `docs/` or `config/` changes | **No** |
| `pyproject.toml` dependency changes | **Yes** |
| `external/periodfind` submodule update | **Yes** |
| CUDA driver compatibility change | **Yes** |
