# Deployment

ml4em runs on MSI via Apptainer. `periodfind` — the GPU-accelerated feature extraction
library — requires a Rust + CUDA build that cannot be done with `pip`, so the full
toolchain is baked into a Docker image. You build it once, push it to GHCR (GitHub's
container registry), and pull it on MSI. Code changes never require a rebuild.

---

## 1. Build the image

Run these commands from your local machine, in the repo root where the `Dockerfile` lives.

```bash
# GPU image — for MSI production runs
docker build --platform linux/amd64 --target gpu \
    -t ghcr.io/maniacurgency42/ml4em:gpu .

# CPU image — for local testing only
docker build --platform linux/amd64 --target cpu \
    -t ghcr.io/maniacurgency42/ml4em:cpu .
```

`--platform linux/amd64` is required on Apple Silicon. Without it Docker produces an
arm64 image that won't run on MSI's x86_64 nodes.

The GPU build takes ~45 minutes — it compiles the Rust toolchain and CUDA extensions
from scratch. You won't need to do this often. See [When to rebuild](#when-to-rebuild).

!!! note "Submodule required"
    `periodfind` lives at `external/periodfind` as a git submodule. Make sure it is
    initialized before building:
    ```bash
    git clone --recurse-submodules <repo-url>
    # or after a plain clone:
    git submodule update --init
    ```

---

## 2. Push to GHCR

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

## 3. Pull on MSI

MSI doesn't allow Docker (it requires root access). Instead, Apptainer converts the
Docker image into a `.sif` file that runs unprivileged on HPC nodes.

**Run this on a compute node, not the login node.** The login node enforces a 15-minute
CPU limit — not enough time to convert a 5–8 GB image. First, request an interactive
compute node:

```bash
srun --time=1:00:00 --mem=8g --pty bash
```

Then pull the image. Redirect temp files to scratch — the default `/tmp` is too small
for a GPU image:

```bash
export APPTAINER_TMPDIR=/scratch.global/$USER/tmp
mkdir -p $APPTAINER_TMPDIR

apptainer pull \
    /scratch.global/$USER/ml4em_gpu.sif \
    docker://ghcr.io/maniacurgency42/ml4em:gpu
```

Store the `.sif` in `/scratch.global`, not `$HOME` — home quotas are small. You only
need to pull again when a new image is pushed to GHCR.

---

## 4. Run on MSI

ml4em is installed inside the image in **editable mode** — Python imports directly from
`/app/ml4em/src` rather than a frozen copy. By bind-mounting your live git checkout over
that path, any `git pull` on MSI is immediately picked up without touching the image.

```bash
apptainer run --nv \
    --bind /users/7/jin00404/ml4em:/app/ml4em \
    --bind /scratch.global/$USER/data:/data \
    /scratch.global/$USER/ml4em_gpu.sif \
    python -m ml4em.run --config /data/config.yaml
```

| Flag | Purpose |
|------|---------|
| `--nv` | Pass NVIDIA GPU through to the container |
| `--bind .../ml4em:/app/ml4em` | Mount your live code — `git pull` picks up changes instantly |
| `--bind .../data:/data` | Mount your data directory inside the container |

GPU device is controlled by `features.device` in `config.yaml` (`"cpu"` / `"gpu"` / `"auto"`).

### SLURM job script

For non-interactive batch jobs, submit via SLURM. Adapt this template to your run:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00

apptainer run --nv \
    --bind /users/7/jin00404/ml4em:/app/ml4em \
    --bind /scratch.global/$USER/data:/data \
    /scratch.global/$USER/ml4em_gpu.sif \
    python -m ml4em.run --config /data/config.yaml
```

---

## When to rebuild

The image only needs to be rebuilt (and re-pulled on MSI) when the compiled dependencies
change — not for routine code or doc updates.

| Change | Rebuild needed? |
|--------|----------------|
| `src/ml4em/` code changes | **No** — `git pull` on MSI is enough |
| `docs/` changes | **No** |
| `pyproject.toml` dependency changes | **Yes** |
| `external/periodfind` submodule update | **Yes** |
| CUDA driver compatibility change | **Yes** |
