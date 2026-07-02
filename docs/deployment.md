# Deployment

<div class="grid cards" markdown>

-   **Apptainer** — recommended for most users

    ---

    Download a pre-built software package and run it directly on MSI. No compilation
    or environment setup required — the package is rebuilt automatically whenever
    the codebase changes. Setup takes about 30 minutes, mostly waiting for a ~6 GB
    download to complete.

    **MSI only** — Apptainer is available on MSI via `module load` but is not
    a standard tool on personal laptops.

    [Apptainer deployment →](apptainer-deployment.md)

-   **Conda**

    ---

    Build a Python environment using the standard conda and pip tools you may already
    know. The right choice if you want to run ml4em interactively in a Jupyter
    notebook, or if you are developing locally on your own machine. Setup compiles
    the period-finding library from source.

    **MSI and local** — the same Conda setup works on MSI GPU nodes and on a
    personal laptop (CPU-only mode).

    [Conda deployment →](conda-deployment.md)

</div>

---

## Not sure which to choose?

```mermaid
flowchart TD
    A([Where am I running ml4em?]) --> B{MSI or local laptop?}
    B -- Local laptop --> C[Conda — CPU mode]
    B -- MSI --> D{Do I want Jupyter\nor interactive work?}
    D -- Yes --> E[Conda — GPU mode]
    D -- No --> F[Apptainer ★ recommended]
```

If you are on MSI and have no strong preference, go with **Apptainer**. It requires
less setup and is the path used and tested by the core team.

---

## Side-by-side comparison

!!! note "Why does Conda setup take 30–45 minutes?"
    The setup time is not spent installing Python packages — it is spent compiling
    the period-finding library (`periodfind`) from Rust and CUDA C++ source code.
    See [Architecture → periodfind](architecture/periodfind.md#why-setup-takes-so-long)
    for a full explanation of what is being compiled and why it only needs to happen once.

| | Apptainer ★ | Conda |
|---|---|---|
| **Recommended** | Yes, for most MSI users | For Jupyter / interactive work, or local development |
| **Where it runs** | MSI only | MSI + local laptop |
| **Setup time** | ~30 min | ~30–45 min |
| **What setup involves** | Downloading a pre-built ~6 GB software package to MSI | Compiling the period-finding library from source, then installing Python packages |
| **Tools required** | Apptainer (already available on MSI via `module load`) | Conda (already available on MSI via `module load`, or via Miniforge locally) |
| **After a code change** | `git pull` — nothing else needed | `git pull` — nothing else needed |
| **Jupyter notebooks** | Not supported | Supported |
| **When you need to redo setup** | Only if the compiled dependencies change (rare) | Only if the compiled dependencies change (rare) |

Both paths install ml4em in **editable mode**: changes to Python source files are
picked up immediately with `git pull` — no rebuild or reinstall needed.
