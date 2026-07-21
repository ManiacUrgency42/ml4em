# Background

This section covers everything you need to understand ml4em before diving into code —
both the astrophysics concepts and the technical infrastructure that makes the pipeline
run. No prior astronomy or systems knowledge is assumed.

!!! info "Who this section is for"
    **If you come from an EECS/CS background:** the astrophysics terms in this codebase
    will be unfamiliar. Variable names like `chi2red`, `stetson_k`, `gaia_ruwe`, and
    `f1_relphi2` are completely opaque without context. This section gives you that
    context — and also explains the compiled library (`periodfind`) and deployment
    infrastructure (Docker, GHCR, Apptainer) that power the pipeline.

    **If you come from an astrophysics background:** the software design patterns used
    here (Protocols, dataclasses, batch-first APIs) are explained in the
    [Architecture](../architecture/overview.md) section. This section focuses on the
    science and on the technical toolchain (Rust, CUDA, Docker) you will encounter
    during setup.

    **You don't need to read these pages in order.** When you hit an unfamiliar term in
    the code or setup docs, come here.

---

## Astrophysics

What the data is, where it comes from, and what the features mean.

<div class="grid cards" markdown>

-   **Light Curves**

    ---

    What a light curve is; magnitude; MJD; photometric bands.

    [Light Curves →](light-curves.md)

-   **Surveys (ZTF & Rubin)**

    ---

    What ZTF and Rubin are; Kowalski; TAP; source IDs; table schemas.

    [Surveys →](surveys.md)

-   **Period Finding**

    ---

    What a period is; all six algorithms; agreement scoring; Fourier features.

    [Period Finding →](period-finding.md)

-   **Variability Statistics**

    ---

    All 22 scalar statistics — `chi2red`, `stetson_k`, `iqr`, and more — explained in plain English.

    [Variability Statistics →](variability-statistics.md)

-   **The dm/dt Histogram**

    ---

    The 26×26 image feature: what it encodes and why it captures variability structure.

    [dm/dt →](dmdt.md)

-   **Gaia & Stellar Catalogs**

    ---

    Gaia EDR3; parallax; BP–RP colour index; RUWE as a binarity indicator.

    [Gaia →](gaia.md)

</div>

---

## Infrastructure

How the compiled library and deployment pipeline work.

<div class="grid cards" markdown>

-   **periodfind**

    ---

    Why Python is too slow for period finding; the Rust + Rayon CPU backend; the CUDA GPU backend; Cython and maturin; why setup takes 20–45 minutes and why it only happens once.

    [periodfind →](periodfind.md)

-   **Docker & GHCR**

    ---

    What Docker images are; GHCR's two roles (distribution + build cache); the GitHub Actions CI workflow; Dockerfile layer structure; how to build and push images.

    [Docker & GHCR →](docker-ghcr.md)

</div>
