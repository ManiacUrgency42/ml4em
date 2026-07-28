# Guide: Add a New Feature Extractor

Adding a new feature extractor requires **one new file** and passing it to
`FeaturePipeline`. No Protocol registration or base class is needed.

---

## Step 1 — Create the file

```
src/ml4em/features/my_extractor.py
```

## Step 2 — Implement `extract()`

The interface is batch-first: input is a list of sources (each source is a list of
bands), output is one dict per source.

```python
from typing import Any
from ml4em.types import LightCurve

class MyExtractor:
    def extract(self, sources: list[list[LightCurve]]) -> list[dict[str, Any]]:
        results = []
        for lcs in sources:
            try:
                value = self._compute(lcs)
                results.append({"my_feature": value})
            except Exception:
                results.append({})   # NEVER raise — return empty dict on failure
        return results

    def _compute(self, lcs: list[LightCurve]) -> float:
        # your computation here
        ...
```

**Rules:**

- `extract()` is the **only member the Protocol requires**.
- The method must **never raise**. On any failure (missing data, computation error),
  return `{}` for that source. The pipeline fills in NaN for missing fields.
- The returned list must have **exactly one entry per input source**, in the same order.
  The pipeline checks this and discards your extractor's output for the whole chunk if it
  does not match, because a short list would misalign features onto the wrong sources.
- Dict keys must match field names in `FeatureVector` exactly. If you want to add a
  new field, you must first add it to `FeatureVector` in `types.py` and
  `SCALAR_FIELDS` in `models/base.py`.
- Times arrive as float64. If you hand them to a float32 consumer, use
  `ml4em.features.base.to_float32_time()` rather than casting directly — see
  [Feature layer → Time precision](../layers/features.md#time-precision) for why.
- Processing is done in Python-level loops in this example, but for performance you
  should batch across sources (like `StatisticsExtractor` does with periodfind).

## Step 2b — (optional) Implement `prepare()`

If your extractor needs a quantity that must be **identical for every source in the run**
— a shared grid, a global normalisation, a fitted scaler — compute it in `prepare()`:

```python
    def prepare(self, sources: list[list[LightCurve]]) -> None:
        """Called once with the complete source list, before chunking."""
        self._global_baseline = max(
            float(lc.time.max() - lc.time.min())
            for lcs in sources for lc in lcs if lc.n_obs >= 2
        )
```

The pipeline calls it once with the full per-band source list before splitting into
chunks. `PeriodExtractor` uses it to build its frequency grid; without it the grid would
be rebuilt per chunk and the same star would get a different period depending on which
sources happened to share its batch.

`prepare()` is **not declared on the `FeatureExtractor` Protocol**, and you should not
add it there. `FeatureExtractor` is `@runtime_checkable`, so `isinstance()` against it
tests for every declared member. Declaring an optional method would make every extractor
that does not need one — including `StatisticsExtractor` and `DmdtExtractor` — fail an
`isinstance()` check against the protocol they satisfy. The pipeline probes for the
method with `getattr(extractor, "prepare", None)` and calls it only if present, so simply
defining it is enough.

## Step 3 — Add it to FeaturePipeline

=== "Pass at construction"

    ```python
    from ml4em.features import FeaturePipeline, StatisticsExtractor, PeriodExtractor
    from ml4em.features.my_extractor import MyExtractor

    pipeline = FeaturePipeline(
        extractors=[
            StatisticsExtractor(),
            PeriodExtractor(cfg.features.period),
            MyExtractor(),
        ],
        min_observations=50,
        device="auto",
    )
    ```

=== "Add to FeaturePipeline.default()"

    Edit `src/ml4em/features/pipeline.py` to include your extractor in the default
    ordering inside `FeaturePipeline.default()`. This makes it part of the standard
    pipeline for all users.

## Step 4 — (if needed) Add fields to FeatureVector

If your extractor produces a new feature, add it to `FeatureVector` in `types.py`:

```python
# src/ml4em/types.py
@dataclass
class FeatureVector:
    ...
    my_feature: float = field(default=np.nan)  # add here
```

And to `SCALAR_FIELDS` in `models/base.py` (append at the end to preserve existing
model compatibility):

```python
SCALAR_FIELDS: list[str] = [
    ...,
    "my_feature",   # append at the end
]
N_SCALAR_FEATURES = len(SCALAR_FIELDS)
```

A name in `SCALAR_FIELDS` that `FeatureVector` does not declare is checked at import and
raises immediately, so a typo fails loudly instead of leaving every model training on a
column that is silently always NaN.

!!! warning "Adding to SCALAR_FIELDS invalidates existing saved models"
    If you append a new field to `SCALAR_FIELDS`, any model trained on the old
    45-field ordering will produce wrong predictions on the new 46-field input.
    Retrain the model after adding new fields.

Adding a field to `FeatureVector` also changes the schema fingerprint the feature
pipeline stores in its checkpoints, so any in-flight checkpointed run will restart from
chunk zero rather than mixing old and new field sets. That is intentional.

---

## Example: median g−r colour extractor

This extractor computes the median colour index (g magnitude minus r magnitude) for
sources observed in both bands.

```python
import numpy as np
from typing import Any
from ml4em.types import LightCurve

class ColourExtractor:
    """Median g-r colour from multi-band light curves."""

    def extract(self, sources: list[list[LightCurve]]) -> list[dict[str, Any]]:
        results = []
        for lcs in sources:
            try:
                results.append(self._colour(lcs))
            except Exception:
                results.append({})
        return results

    def _colour(self, lcs: list[LightCurve]) -> dict[str, float]:
        g = next((lc for lc in lcs if lc.band == "g"), None)
        r = next((lc for lc in lcs if lc.band == "r"), None)
        if g is None or r is None:
            return {}
        return {"colour_g_r": float(np.median(g.mag) - np.median(r.mag))}
```

This extractor requires adding `colour_g_r: float = field(default=np.nan)` to
`FeatureVector` and `"colour_g_r"` to `SCALAR_FIELDS`.

!!! warning "This example needs multi-band groups"
    `FeaturePipeline.run_batch()` expands each source into **one single-band group per
    `LightCurve`** before calling extractors, so `lcs` normally holds exactly one band and
    a cross-band extractor written this way will always return `{}`. A colour feature
    needs either its own pass over the un-expanded `list[list[LightCurve]]`, or a join
    across the per-band `FeatureVector` rows after the pipeline has run. The rest of the
    example — the never-raise contract, the one-dict-per-source shape, the field
    registration — applies unchanged.
