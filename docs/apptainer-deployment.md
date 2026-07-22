# Deployment — Apptainer

The pipeline ships as a pre-built container image. You download it once, then
run jobs against your live git checkout — no recompilation needed when you
change Python code.

!!! tip "How the image is built"
    See [Background → Docker & GHCR](background/docker-ghcr.md) for details on
    how the image is built and when CI triggers a rebuild.

---

## 1. Clone the repo

MSI login nodes cannot authenticate to GitHub via SSH. Use a Personal Access
Token (PAT) embedded in the HTTPS URL. Replace `<PAT>` with your token.

**Run on MSI:**
```bash
git clone --recurse-submodules https://<PAT>@github.com/ManiacUrgency42/ml4em.git ~/ml4em
cd ~/ml4em
```

!!! tip "Creating a classic PAT"
    GitHub → Settings → Developer settings → Personal access tokens → Tokens
    (classic). Grant `repo` scope.
    [Step-by-step guide →](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic)

---

## 2. First-time data setup

Do this once. Everything here persists across SLURM jobs on scratch.

### 2a. Create scratch directories

MSI home directories have a small quota (~10 GB). All large files go on scratch.

**Run on MSI:**
```bash
mkdir -p /scratch.global/$USER/ml4em_data/{features,models,predictions}
mkdir -p /scratch.global/$USER/{apptainer_cache,tmp}
```

### 2b. Copy your catalog

`wdb_sources.csv` is the source catalog (RA/Dec positions of WDB targets).

**Run on your laptop:**
```bash
scp data/wdb_sources.csv jin00404@login.msi.umn.edu:/scratch.global/jin00404/ml4em_data/
```

### 2c. Write the MSI config

The storage paths use `/data` — the run command maps your scratch directory to
`/data` inside the container at runtime.

**Run on MSI:**
```bash
cat > /scratch.global/$USER/ml4em_data/config_msi.yaml << 'EOF'
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
  device: cuda
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
EOF
```

### 2d. Get your Kowalski token

Run this script on the MSI login node. It prompts for your Kowalski username
and password, fetches a token, and saves it to your scratch directory.

**Run on MSI:**
```bash
python3 ~/ml4em/scripts/get_credentials.py --show-password
```

Your token is stored at `/scratch.global/$USER/ml4em_data/.env`. To update
your credentials at any time — for example if your token expires — just re-run
the script.

---

## 3. Pull the container image

Downloads the pre-built image from GHCR and converts it to a `.sif` file on
scratch. Runs as a SLURM job because the conversion (20–40 min, ~6 GB download)
exceeds the login node's CPU time limit.

**Run on MSI:**
```bash
cd ~/ml4em
mkdir -p logs
sbatch slurm/pull_image.sh
```

Monitor progress:

**Run on MSI:**
```bash
squeue -u $USER
tail -f logs/pull_ml4em_<JOBID>.out
```

Output: `/scratch.global/$USER/ml4em_gpu.sif`

---

## 4. Run the demo

### Batch job (recommended)

**Run on MSI:**
```bash
cd ~/ml4em
mkdir -p logs
sbatch slurm/run_demo.sh
```

Monitor:

**Run on MSI:**
```bash
squeue -u $USER
tail -f logs/ml4em_demo_<JOBID>.out
```

Outputs:

- `/scratch.global/$USER/ml4em_data/features/demo.parquet` — extracted feature vectors
- `/scratch.global/$USER/ml4em_data/models/logistic_demo/` — saved model

### Interactive run (debugging only)

First, request a GPU compute node:

**Run on MSI:**
```bash
srun --account=cough052 --partition=a100 --gres=gpu:a100:1 \
     --mem=16g --time=1:00:00 --pty bash
```

Once on the compute node:

**Run on MSI (compute node):**
```bash
module load apptainer

apptainer run --nv \
    --bind $HOME/ml4em:/app/ml4em \
    --bind /scratch.global/$USER/ml4em_data:/data \
    --env-file /scratch.global/$USER/ml4em_data/.env \
    /scratch.global/$USER/ml4em_gpu.sif \
    python scripts/run_demo.py --config /data/config_msi.yaml
```

| Flag | Purpose |
|------|---------|
| `--nv` | Pass NVIDIA GPU drivers through to the container |
| `--bind .../ml4em:/app/ml4em` | Mount your live code — `git pull` picks up changes without rebuilding |
| `--bind .../ml4em_data:/data` | Mount scratch — catalog, config, outputs |
| `--env-file .env` | Inject `ML4EM_ZTF_TOKEN` into the container |

---

## Day-to-day workflow

The `.sif` is a one-time download. All Python code lives in your git checkout
and is mounted live into the container. Updating the pipeline is:

**Run on MSI:**
```bash
git pull
sbatch slurm/run_demo.sh
```

## When to re-pull the `.sif`

The image only needs to be re-pulled when compiled dependencies change.

| Change | Action |
|--------|--------|
| `src/ml4em/` Python code | `git pull` only |
| `scripts/`, `slurm/`, `docs/` | `git pull` only |
| `pyproject.toml` — new pip dependency | Re-pull after CI rebuilds image |
| `external/periodfind` submodule update | Re-pull after CI rebuilds image |
| CUDA driver compatibility change | Re-pull after CI rebuilds image |

CI rebuilds automatically on merge to `main`. Once the build finishes, run
`sbatch slurm/pull_image.sh` on MSI.
