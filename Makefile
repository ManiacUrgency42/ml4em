.PHONY: build test test-unit

IMAGE    ?= ml4em:cpu
PLATFORM  = linux/amd64

# ── Build ─────────────────────────────────────────────────────────────────────
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
