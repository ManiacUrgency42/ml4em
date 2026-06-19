# GitHub Container Registry (GHCR)

Pre-built Docker images are hosted on GHCR so that collaborators and HPC nodes can
pull them without rebuilding from source.

## Step 1 — Authenticate

Generate a GitHub Personal Access Token (PAT) with `write:packages` scope:

**GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
→ Generate new token**

Select scopes: `write:packages`, `read:packages`, `delete:packages`

```bash
echo YOUR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

## Step 2 — Build

See [Docker Builds](docker.md) for build commands. After building:

```bash
docker build --platform linux/amd64 --target cpu -t ghcr.io/<org>/ml4em:cpu .
docker build --platform linux/amd64 --target gpu -t ghcr.io/<org>/ml4em:gpu .
```

## Step 3 — Push

```bash
docker push ghcr.io/<org>/ml4em:cpu
docker push ghcr.io/<org>/ml4em:gpu
```

The GPU image is 5–8 GB. If the push drops mid-upload (network timeout), simply
rerun — Docker skips already-uploaded layers.

## Step 4 — Make the package public

After the first push, make the package public so MSI (and others) can pull without
credentials:

**GitHub → your profile → Packages → ml4em → Package Settings →
Change visibility → Public**

## Pulling

```bash
docker pull ghcr.io/<org>/ml4em:cpu
docker pull ghcr.io/<org>/ml4em:gpu
```
