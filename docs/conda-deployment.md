# Deployment — Conda

`periodfind` — the period-finding library ml4em depends on — cannot be installed
via `conda` or plain `pip`. It requires a compiled build from Rust and Cython
source. A one-shot SLURM job handles this automatically.

!!! tip "Jupyter notebooks"
    Once the environment is set up, you can run ml4em interactively in a Jupyter
    notebook via [MSI Open OnDemand](https://ondemand.msi.umn.edu). Launch a
    JupyterLab session, select the `ml4em-gpu` kernel, and import away.

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

## 2. Create the conda environment

**Run on MSI:**
```bash
module load conda
module load cuda/11.8.0
conda env create -f ~/ml4em/environment-gpu.yml
```

!!! note "CUDA is not inside the conda env"
    `environment-gpu.yml` does not install a CUDA toolkit — that would conflict
    with MSI's system drivers. `module load cuda/11.8.0` puts `nvcc` on your
    PATH so periodfind's CUDA extensions can be compiled in the next step.

---

## 3. Build periodfind and install ml4em

The Rust compilation can exceed the login node's 15-minute CPU limit. Submit it
as a SLURM job — no GPU needed, it runs on a CPU node.

**Run on MSI:**
```bash
cd ~/ml4em
mkdir -p logs
sbatch slurm/setup_conda.sh
```

Monitor progress:

**Run on MSI:**
```bash
squeue -u $USER
tail -f logs/ml4em_conda_setup_<JOBID>.out
```

Verify the install once the job completes:

**Run on MSI:**
```bash
module load conda
conda activate ml4em-gpu
python -c 'import ml4em; import periodfind; print("OK")'
```

---

## 4. First-time data setup

Do this once. Everything here persists across SLURM jobs on scratch.

### 4a. Create scratch directories

MSI home directories have a small quota (~10 GB). All large files go on scratch.

**Run on MSI:**
```bash
mkdir -p /scratch.global/$USER/ml4em_data/{features,models,predictions}
mkdir -p /scratch.global/$USER/tmp
```

### 4b. Copy your catalog

`wdb_sources.csv` is the source catalog (RA/Dec positions of WDB targets).

**Run on your laptop:**
```bash
scp data/wdb_sources.csv jin00404@login.msi.umn.edu:/scratch.global/jin00404/ml4em_data/
```

### 4c. Write the MSI config

**Run on MSI:**
```bash
cat > /scratch.global/$USER/ml4em_data/config_msi.yaml << EOF
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
  catalog_path: /scratch.global/$USER/ml4em_data/wdb_sources.csv
  labels_path:  /scratch.global/$USER/ml4em_data/labels.csv
  features_dir:    /scratch.global/$USER/ml4em_data/features
  models_dir:      /scratch.global/$USER/ml4em_data/models
  predictions_dir: /scratch.global/$USER/ml4em_data/predictions

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

### 4d. Get your Kowalski token

Run this script on the MSI login node. It prompts for your Kowalski username
and password, fetches a token, and saves it to your scratch directory.

**Run on MSI:**
```bash
python3 ~/ml4em/scripts/get_credentials.py --show-password
```

Your token is stored at `/scratch.global/$USER/ml4em_data/.env`. To update
your credentials at any time, re-run the script.

---

## 5. Run the demo

### Batch job (recommended)

**Run on MSI:**
```bash
cd ~/ml4em
mkdir -p logs
sbatch slurm/run_demo_conda.sh
```

Monitor:

**Run on MSI:**
```bash
squeue -u $USER
tail -f logs/ml4em_demo_conda_<JOBID>.out
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
module load conda cuda/11.8.0
conda activate ml4em-gpu

set -a; source /scratch.global/$USER/ml4em_data/.env; set +a

python ~/ml4em/scripts/run_demo.py --config /scratch.global/$USER/ml4em_data/config_msi.yaml
```

---

## Day-to-day workflow

The conda environment is a one-time build. All Python code lives in your git
checkout as an editable install. Updating the pipeline is:

**Run on MSI:**
```bash
git pull
sbatch slurm/run_demo_conda.sh
```

## When to rebuild the environment

| Change | Action |
|--------|--------|
| `src/ml4em/` Python code | `git pull` only |
| `scripts/`, `slurm/`, `docs/` | `git pull` only |
| `pyproject.toml` — new pure-Python dep | `pip install -e ".[ztf,catalog,training,dev]"` inside the active env |
| `external/periodfind` submodule update | Full rebuild (see below) |
| CUDA version change | Full rebuild (see below) |

Full rebuild from scratch:

**Run on MSI:**
```bash
conda env remove -n ml4em-gpu --yes
cd ~/ml4em && mkdir -p logs
sbatch slurm/setup_conda.sh
```
