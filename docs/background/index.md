# Background

This section explains every astrophysics concept used in ml4em from first principles.
No prior astronomy knowledge is assumed.

!!! info "Who this section is for"
    **If you come from an EECS/CS background:** the astrophysics terms in this codebase
    will be unfamiliar. Variable names like `chi2red`, `stetson_k`, `gaia_ruwe`, and
    `f1_relphi2` are completely opaque without context. This section gives you that
    context.

    **If you come from an astrophysics background:** the software design patterns used
    here (Protocols, dataclasses, batch-first APIs, Protocol-based dependency injection)
    are explained in the [Architecture](../architecture/overview.md) section.

    **You don't need to read these pages in order.** When you hit an unfamiliar term in
    the code, come here.

---

## Pages in this section

| Page | What it explains |
|------|-----------------|
| [Light Curves](light-curves.md) | What a light curve is; magnitude; MJD; photometric bands |
| [Surveys (ZTF & Rubin)](surveys.md) | What ZTF and Rubin are; Kowalski; TAP; source IDs; table schemas |
| [Period Finding](period-finding.md) | What a period is; all six algorithms; agreement scoring; Fourier features |
| [Variability Statistics](variability-statistics.md) | All 22 scalar statistics explained in plain English |
| [The dm/dt Histogram](dmdt.md) | The 26×26 image feature — what it is and why it works |
| [Gaia & Stellar Catalogs](gaia.md) | Gaia EDR3; parallax; BP–RP colour; RUWE |
