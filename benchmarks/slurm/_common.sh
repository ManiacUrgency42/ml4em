#!/bin/bash
# ==============================================================================
# _common.sh — shared setup for every benchmark SLURM wrapper
#
# Not submitted directly.  Each benchmark wrapper sources this file, which
# resolves paths, loads credentials, checks preconditions, and defines
# ml4em_run() — the one place that knows how to start a Python process.
#
# Environment selection
# ---------------------
# The benchmarks themselves are environment-agnostic: they are plain Python
# scripts.  How Python gets started is decided here and nowhere else.
#
#   ML4EM_LAUNCHER=apptainer   run inside /scratch.global/$USER/ml4em_gpu.sif
#   ML4EM_LAUNCHER=conda       run inside the ml4em-gpu conda environment
#   ML4EM_LAUNCHER=python      run against whatever python is on PATH
#
# If unset, whichever of the first two is actually present on the machine is
# used, preferring Apptainer.  Override per job without editing anything:
#
#   ML4EM_LAUNCHER=conda sbatch benchmarks/slurm/scaling_gpu.sh
#
# Contract for the wrapper that sources this
# ------------------------------------------
#   ML4EM_GPU=1   set before sourcing if the job needs a GPU (default 0)
#   ML4EM_CONFIG  set here; a path valid in whichever environment was chosen
#   ml4em_run     call with a repo-relative script path and its arguments
# ==============================================================================

set -uo pipefail

# sbatch copies the submitted script to /var/spool on the compute node, so
# BASH_SOURCE points there and not at the checkout.  SLURM_SUBMIT_DIR is the
# directory sbatch was called from; the BASH_SOURCE form is the fallback for
# running a wrapper directly with bash.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="${ML4EM_DATA_DIR:-/scratch.global/$USER/ml4em_data}"
SIF="${ML4EM_SIF:-/scratch.global/$USER/ml4em_gpu.sif}"
ML4EM_GPU="${ML4EM_GPU:-0}"

cd "${REPO_DIR}" || exit 1

if [[ -f "${DATA_DIR}/.env" ]]; then
    set -a; source "${DATA_DIR}/.env"; set +a
fi

# Both are prerequisites, and failing here names the missing one instead of
# surfacing as a pydantic or Kowalski error several minutes in.
if [[ ! -f "${DATA_DIR}/config_msi.yaml" ]]; then
    echo "ERROR: ${DATA_DIR}/config_msi.yaml not found." >&2
    echo "       cp config.example.yaml ${DATA_DIR}/config_msi.yaml and edit storage.*" >&2
    exit 1
fi
if [[ -z "${ML4EM_ZTF_TOKEN:-}" ]]; then
    echo "ERROR: ML4EM_ZTF_TOKEN is not set (expected in ${DATA_DIR}/.env)." >&2
    echo "       Run: python scripts/get_credentials.py" >&2
    exit 1
fi

# ------------------------------------------------------------------ launcher

if [[ -z "${ML4EM_LAUNCHER:-}" ]]; then
    if [[ -f "${SIF}" ]]; then
        ML4EM_LAUNCHER=apptainer
    elif command -v conda >/dev/null 2>&1 || module load conda >/dev/null 2>&1; then
        ML4EM_LAUNCHER=conda
    else
        echo "ERROR: no runnable environment found." >&2
        echo "       Apptainer: sbatch slurm/pull_image.sh   (creates ${SIF})" >&2
        echo "       Conda:     sbatch slurm/setup_conda.sh  (creates ml4em-gpu)" >&2
        exit 1
    fi
fi

# ML4EM_CONFIG is set per launcher rather than bind-mounting DATA_DIR at its
# real path inside the container: the container already sees it at /data, and
# one variable is cheaper than a second bind that has to exist in the image.
case "${ML4EM_LAUNCHER}" in
apptainer)
    module purge
    module load apptainer
    if [[ ! -f "${SIF}" ]]; then
        echo "ERROR: ${SIF} not found." >&2
        echo "       Run: sbatch slurm/pull_image.sh" >&2
        exit 1
    fi
    ML4EM_CONFIG=/data/config_msi.yaml
    ;;
conda)
    module purge
    module load conda
    [[ "${ML4EM_GPU}" == "1" ]] && module load cuda/11.8.0
    if ! conda env list | grep -qE '(^|/)ml4em-gpu[[:space:]]'; then
        echo "ERROR: conda environment ml4em-gpu not found." >&2
        echo "       Run: sbatch slurm/setup_conda.sh" >&2
        exit 1
    fi
    ML4EM_CONFIG="${DATA_DIR}/config_msi.yaml"
    ;;
python)
    ML4EM_CONFIG="${DATA_DIR}/config_msi.yaml"
    ;;
*)
    echo "ERROR: unknown ML4EM_LAUNCHER='${ML4EM_LAUNCHER}'" >&2
    echo "       Expected one of: apptainer, conda, python" >&2
    exit 1
    ;;
esac

echo "ml4em benchmark: launcher=${ML4EM_LAUNCHER} gpu=${ML4EM_GPU} repo=${REPO_DIR}"

# Run a repo-relative Python script with the chosen launcher.
#   ml4em_run benchmarks/scaling_gpu.py --config "${ML4EM_CONFIG}" --trials 5
ml4em_run() {
    case "${ML4EM_LAUNCHER}" in
    apptainer)
        local nv=()
        [[ "${ML4EM_GPU}" == "1" ]] && nv+=(--nv)
        apptainer run ${nv[@]+"${nv[@]}"} \
            --bind "${REPO_DIR}:/app/ml4em" \
            --bind "${DATA_DIR}:/data" \
            --env-file "${DATA_DIR}/.env" \
            --env PYTHONPATH=/data/pyshim \
            "${SIF}" \
            python "$@"
        ;;
    conda)
        conda run --no-capture-output -n ml4em-gpu python "$@"
        ;;
    python)
        python "$@"
        ;;
    esac
}
