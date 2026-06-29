# Deployment

ml4em runs on MSI via Apptainer. The workflow is linear: build a Docker image locally,
push it to GHCR, pull it on MSI as a `.sif` file, and run it.

`periodfind` — the GPU-accelerated feature extraction library — requires a Rust + CUDA
build that cannot be done with `pip`. Docker handles this once; the resulting image is
reused indefinitely.

---

## 1. Build the image

From the repo root:

```bash
# GPU image — MSI production
docker build --platform linux/amd64 --target gpu \
    -t ghcr.io/maniacurgency42/ml4em:gpu .

# CPU image — local testing only
docker build --platform linux/amd64 --target cpu \
    -t ghcr.io/maniacurgency42/ml4em:cpu .
```

`--platform linux/amd64` is required on Apple Silicon. Without it Docker produces an
arm64 image that won't run on MSI's x86_64 nodes.

The GPU build takes ~45 minutes (Rust toolchain + CUDA extensions). See
[When to rebuild](#when-to-rebuild) — you won't need to do this often.

!!! note "Submodule required"
    `periodfind` lives at `external/periodfind` as a git submodule. Initialize it before
    building:
    ```bash
    git clone --recurse-submodules <repo-url>
    # or after a plain clone:
    git submodule update --init
    ```

---

## 2. Push to GHCR

Authenticate with a GitHub PAT. Generate one at:
**GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → Generate new token**

Select scopes: `write:packages`, `read:packages`

```bash
echo YOUR_PAT | docker login ghcr.io -u ManiacUrgency42 --password-stdin
```

Push both images:

```bash
docker push ghcr.io/maniacurgency42/ml4em:gpu
docker push ghcr.io/maniacurgency42/ml4em:cpu
```

The GPU image is 5–8 GB. If the push times out, rerun — Docker skips already-uploaded
layers.

!!! note "First push only"
    Make the package public so MSI can pull without credentials:
    **GitHub → your profile → Packages → ml4em → Package Settings → Change visibility → Public**

---

## 3. Pull on MSI

**Run on a compute node, not the login node.** The login node enforces a 15-minute CPU
limit — not enough to convert a 5–8 GB image.

```bash
# Request an interactive compute node
srun --time=1:00:00 --mem=8g --pty bash

# Redirect tmp to scratch — default /tmp is too small for a GPU image
export APPTAINER_TMPDIR=/scratch.global/$USER/tmp
mkdir -p $APPTAINER_TMPDIR

apptainer pull \
    /scratch.global/$USER/ml4em_gpu.sif \
    docker://ghcr.io/maniacurgency42/ml4em:gpu
```

Store the `.sif` in `/scratch.global`, not `$HOME` — it's 5–8 GB and home quotas are
small. The `.sif` only needs to be pulled again when a new image is pushed to GHCR.

---

## 4. Run on MSI

ml4em is installed in **editable mode** inside the image. This means Python imports
directly from `/app/ml4em/src` — not from a frozen copy in site-packages. Bind-mounting
your live git checkout over that path makes `git pull` on MSI equivalent to a code
update, with no image rebuild required.

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
| `--bind .../ml4em:/app/ml4em` | Mount your live checkout — code changes via `git pull`, no rebuild |
| `--bind .../data:/data` | Mount your data directory inside the container |

GPU device is controlled by `features.device` in `config.yaml` (`"cpu"` / `"gpu"` / `"auto"`).

### SLURM job script

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

| Change | Rebuild needed? |
|--------|----------------|
| `src/ml4em/` code changes | **No** — `git pull` on MSI is enough |
| `docs/` changes | **No** |
| `pyproject.toml` dependency changes | **Yes** |
| `external/periodfind` submodule update | **Yes** |
| CUDA driver compatibility change | **Yes** |
