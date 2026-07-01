# Deployment — Conda

## Why a setup script is required

`periodfind` — the period-finding library ml4em depends on — cannot be installed via
`conda` or plain `pip`. It requires a two-step compiled build:

1. **Rust → Python wheel** via `maturin`
2. **Cython extensions** (optionally CUDA-accelerated)

A one-shot script (`scripts/setup_conda.sh`) handles both steps after the conda
environment is created. This is the only step that differs from a standard conda
workflow.

---

## 1. Clone the repo (with submodules)

`periodfind` lives at `external/periodfind` as a git submodule:

```bash
git clone --recurse-submodules https://github.com/ManiacUrgency42/ml4em.git
cd ml4em
# or after a plain clone:
git submodule update --init
```

---

## 2. Create the conda environment

=== "GPU (MSI production)"

    The CUDA module provides `nvcc`, which is required to build periodfind's CUDA
    extensions. Load it **before** creating the environment:

    ```bash
    module load conda
    module load cuda/11.8.0
    conda env create -f environment-gpu.yml
    ```

    !!! note "CUDA is not inside the conda env"
        `environment-gpu.yml` does not install a CUDA toolkit — that would conflict
        with MSI's system drivers. Instead, the `module load cuda/11.8.0` command
        puts `nvcc` on your PATH at build time.

=== "CPU (local dev / CPU nodes)"

    ```bash
    conda env create -f environment-cpu.yml
    ```

    Or with the Makefile (creates the environment and runs the setup script in one
    step — see § 3 below):

    ```bash
    make conda-cpu
    ```

---

## 3. Build periodfind and install ml4em

`scripts/setup_conda.sh` does four things in order:

1. Installs torch with the correct wheel (CPU or CUDA 11.8)
2. Compiles the periodfind Rust extension with `maturin`
3. Compiles the periodfind Cython extensions (with CUDA if in `gpu` mode)
4. Installs ml4em in editable mode: `pip install -e ".[ztf,catalog,training,dev]"`

=== "GPU (MSI production)"

    On MSI, submit this as a SLURM job. The Rust compilation can exceed the login
    node's 15-minute CPU limit. No GPU is needed to compile — the job runs on a
    CPU node (`amdsmall`).

    ```bash
    mkdir -p logs
    sbatch slurm/setup_conda.sh
    ```

    Monitor:

    ```bash
    squeue -u $USER
    tail -f logs/ml4em_conda_setup_<JOBID>.out
    ```

=== "CPU (local dev / CPU nodes)"

    ```bash
    conda run -n ml4em-cpu bash scripts/setup_conda.sh cpu
    ```

Verify the install once complete:

```bash
module load conda          # MSI only
conda activate ml4em-gpu   # or ml4em-cpu
python -c 'import ml4em; import periodfind; print("OK")'
```

---

## 4. First-time MSI data setup

Do this once after SSH-ing into MSI. These directories persist across jobs.

### 4a. Create scratch directories

MSI's home quota is small. Keep all large files on scratch:

```bash
DATA=/scratch.global/$USER/ml4em_data

mkdir -p $DATA/features $DATA/models $DATA/predictions
mkdir -p /scratch.global/$USER/tmp
```

### 4b. Copy your catalog

`wdb_sources.csv` is the WDB catalog (ra, dec positions of target sources):

```bash
# From your local machine:
scp data/wdb_sources.csv jin00404@login.msi.umn.edu:/scratch.global/jin00404/ml4em_data/
```

### 4c. Write a config for MSI

Create `/scratch.global/$USER/ml4em_data/config_msi.yaml`:

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
  catalog_path: /scratch.global/<USER>/ml4em_data/wdb_sources.csv
  labels_path:  /scratch.global/<USER>/ml4em_data/labels.csv
  features_dir: /scratch.global/<USER>/ml4em_data/features
  models_dir:   /scratch.global/<USER>/ml4em_data/models
  predictions_dir: /scratch.global/<USER>/ml4em_data/predictions

training:
  val_fraction:  0.1
  test_fraction: 0.1
  seed: 42

inference:
  confidence_thresholds:
    high:   0.9
    medium: 0.7
```

Replace `<USER>` with your MSI username (e.g. `jin00404`).

### 4d. Store your Kowalski token

Never put tokens in `config_msi.yaml`. Store them in a `.env` file:

```bash
echo "ML4EM_ZTF_TOKEN=your_token_here" > /scratch.global/$USER/ml4em_data/.env
chmod 600 /scratch.global/$USER/ml4em_data/.env
```

---

## 5. Run the demo

### Batch job (recommended)

```bash
mkdir -p logs
sbatch slurm/run_demo_conda.sh
```

Monitor:

```bash
squeue -u $USER
tail -f logs/ml4em_demo_conda_<JOBID>.out
```

Output files:

- `/scratch.global/$USER/ml4em_data/features/demo.parquet` — extracted feature vectors
- `/scratch.global/$USER/ml4em_data/models/logistic_demo/` — saved model

### Interactive run (for debugging)

!!! warning "Must be on a GPU compute node"
    Never run GPU workloads on the login node. Request an interactive GPU node first:

    ```bash
    srun --account=cough052 --partition=a100 --gres=gpu:a100:1 \
         --mem=16g --time=1:00:00 --pty bash
    ```

Once on the compute node:

```bash
DATA=/scratch.global/$USER/ml4em_data

module load conda cuda/11.8.0
conda activate ml4em-gpu

set -a; source "${DATA}/.env"; set +a

python scripts/run_demo.py --config "${DATA}/config_msi.yaml"
```

---

## When to rebuild the environment

| Change | Action |
|--------|--------|
| `src/ml4em/` code changes | **None** — `git pull` is enough (editable install) |
| `pyproject.toml` pure-Python dep changes | `pip install -e ".[ztf,catalog,training,dev]"` inside the active env |
| `external/periodfind` submodule update | Full rebuild (see below) |
| CUDA version change | Full rebuild (see below) |

Full rebuild from scratch:

```bash
conda env remove -n ml4em-gpu --yes
sbatch slurm/setup_conda.sh
```
