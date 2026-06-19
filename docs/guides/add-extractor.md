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
- The method must **never raise**. On any failure (missing data, computation error),
  return `{}` for that source. The pipeline fills in NaN for missing fields.
- Dict keys must match field names in `FeatureVector` exactly. If you want to add a
  new field, you must first add it to `FeatureVector` in `types.py` and
  `SCALAR_FIELDS` in `models/base.py`.
- Processing is done in Python-level loops in this example, but for performance you
  should batch across sources (like `StatisticsExtractor` does with periodfind).

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

!!! warning "Adding to SCALAR_FIELDS invalidates existing saved models"
    If you append a new field to `SCALAR_FIELDS`, any model trained on the old
    43-field ordering will produce wrong predictions on the new 44-field input.
    Retrain the model after adding new fields.

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
