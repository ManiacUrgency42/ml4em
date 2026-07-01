.PHONY: build test test-unit conda-cpu conda-gpu

IMAGE         ?= ml4em:cpu
PLATFORM       = linux/amd64
CONDA_ENV_CPU  = ml4em-cpu
CONDA_ENV_GPU  = ml4em-gpu

# ── Docker ────────────────────────────────────────────────────────────────────
build:
	docker build --platform $(PLATFORM) --target cpu -t $(IMAGE) .

# ── Test (all — includes integration tests that hit real ZTF) ─────────────────
# Credentials are injected at runtime via --env-file; .env is never baked into
# the image (.dockerignore excludes it).  If .env is missing, this fails loudly
# rather than silently skipping the integration tests.
test:
	@test -f .env || { \
	    echo ""; \
	    echo "ERROR: .env not found."; \
	    echo "Create it with your Kowalski token:"; \
	    echo "  echo 'ML4EM_ZTF_TOKEN=<your-token>' > .env"; \
	    echo ""; \
	    exit 1; \
	}
	docker run --rm --platform $(PLATFORM) --env-file .env $(IMAGE) \
	    python -m pytest tests/ -v

# ── Test (unit only — no credentials or network required) ─────────────────────
test-unit:
	docker run --rm --platform $(PLATFORM) $(IMAGE) \
	    python -m pytest tests/ -v -m "not integration"

# ── Conda (local — no Docker required) ───────────────────────────────────────
# Creates the conda environment and builds periodfind in one step.
# If the environment already exists, remove it first:
#   conda env remove -n ml4em-cpu --yes
conda-cpu:
	conda env create -f environment-cpu.yml
	conda run -n $(CONDA_ENV_CPU) bash scripts/setup_conda.sh cpu

# GPU build requires nvcc on PATH. On MSI use slurm/setup_conda.sh instead.
# Locally: ensure the CUDA toolkit is installed and nvcc is on PATH, then:
#   conda env remove -n ml4em-gpu --yes  (if env already exists)
conda-gpu:
	conda env create -f environment-gpu.yml
	conda run -n $(CONDA_ENV_GPU) bash scripts/setup_conda.sh gpu
