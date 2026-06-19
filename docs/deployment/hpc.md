# MSI / HPC (Apptainer)

MSI (Minnesota Supercomputing Institute) GPU nodes cannot run Docker directly — no
root access. Apptainer (formerly Singularity) converts Docker images into `.sif` files
that run unprivileged.

## One-time pull

**Run on a compute node, not a login node.** Login nodes enforce a 15-minute CPU time
limit which is not enough to convert the GPU image.

Request an interactive compute node first:

```bash
srun --time=1:00:00 --mem=8g --pty bash
```

Then pull, redirecting temp files to scratch (the default `/tmp` is too small):

```bash
export APPTAINER_TMPDIR=/scratch.global/$USER/tmp
mkdir -p /scratch.global/$USER/tmp

apptainer pull \
    /scratch.global/$USER/ml4em_gpu.sif \
    docker://ghcr.io/<org>/ml4em:gpu
```

The `.sif` file only needs to be pulled once and can be reused across all jobs.

## Running the pipeline

```bash
apptainer run --nv \
    --bind /scratch.global/$USER/data:/data \
    /scratch.global/$USER/ml4em_gpu.sif \
    python -m ml4em.run --config /data/config.yaml
```

**Key flags:**

| Flag | Purpose |
|------|---------|
| `--nv` | Pass NVIDIA GPU through to the container (required for GPU period-finding) |
| `--bind /scratch.global/$USER/data:/data` | Mount your scratch data directory inside the container at `/data` |

The GPU device is controlled by `features.device` in `config.yaml`
(`"cpu"` / `"gpu"` / `"auto"`). The `.sif` file supports both.

## SLURM job script

A template SLURM script is in `slurm/pull_image.sh`. Copy and adapt it for your runs:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00

apptainer run --nv \
    --bind /scratch.global/$USER/data:/data \
    /scratch.global/$USER/ml4em_gpu.sif \
    python -m ml4em.run --config /data/config.yaml
```

## CPU-only runs

For feature extraction on a CPU node:

```bash
apptainer pull \
    /scratch.global/$USER/ml4em_cpu.sif \
    docker://ghcr.io/<org>/ml4em:cpu

apptainer run \
    --bind /scratch.global/$USER/data:/data \
    /scratch.global/$USER/ml4em_cpu.sif \
    python -m ml4em.run --config /data/config.yaml
```

Omit `--nv` for CPU-only jobs. Set `features.device: cpu` in `config.yaml`.
