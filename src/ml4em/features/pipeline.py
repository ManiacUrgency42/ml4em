"""
FeaturePipeline — compose extractors into a batch of FeatureVectors.

The pipeline:
1. Sets the periodfind device (CPU/GPU) once before processing.
2. Expands each source into one entry per photometric band.
3. Runs each extractor's optional prepare() over the complete list.
4. Chunks that list into batches of feature_batch_size.
5. Within each batch, partitions entries by min_observations.
6. Calls each extractor once for the valid entries in the batch.
7. Merges extractor outputs and constructs FeatureVectors.
8. Returns a default (all-NaN) FeatureVector for skipped entries.
9. Optionally checkpoints after every chunk for HPC fault tolerance.

Output granularity
------------------
One FeatureVector per (source, band), not per source.  Bands are processed
independently, so a source observed in g and r produces two rows sharing a
source_id and distinguished by the `band` field.  Whether the two bands agree
on a period is itself evidence, and collapsing to a single band would both
discard epochs and hide that check.

Checkpoint / resume
-------------------
Set FeatureConfig.checkpoint_dir to a unique per-run path on scratch storage.
After every chunk the pipeline atomically writes a checkpoint file.  On the
next invocation with the same checkpoint_dir, it detects the file, restores
the already-completed FeatureVectors, and resumes from the next chunk.
On successful completion the checkpoint file is deleted automatically.

Matches scope-ml's checkpoint pattern in generate_features.py
(_save_period_checkpoint / _load_period_checkpoint) extended to cover the
full pipeline output rather than just period features.

Usage
-----
    from ml4em.features import FeaturePipeline
    from ml4em.config import load_config

    cfg = load_config()
    pipeline = FeaturePipeline.default(cfg.features)

    fvs = pipeline.run_batch(all_lcs_grouped_by_source)

Single-source use (e.g. tests):
    fv = pipeline.run_batch([lcs])[0]
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pickle
from typing import Any

from ml4em.config.schema import FeatureConfig
from ml4em.types import FeatureVector, LightCurve

from .base import FeatureExtractor
from .catalog import CatalogExtractor
from .dmdt import DmdtExtractor
from .period import PeriodExtractor
from .statistics import StatisticsExtractor

log = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "feature_checkpoint.pkl"


def _schema_fingerprint() -> tuple[str, ...]:
    """Field names of FeatureVector, used to detect stale checkpoints.

    Unpickling a FeatureVector saved before a field was added does not fail:
    the missing attribute resolves to the dataclass default, so the restored
    rows silently carry None where later rows carry real values.  Comparing
    the field list makes that mismatch visible instead.
    """
    return tuple(f.name for f in dataclasses.fields(FeatureVector))


def _chunks(lst: list, size: int):
    """Yield successive chunks of `size` from `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _resolve_device(device: str) -> str:
    """Map an ml4em device string onto the 'cpu'/'gpu' literals periodfind accepts.

    periodfind.set_device() raises on anything other than 'cpu' or 'gpu', so
    'auto' must be resolved here.  periodfind's own _resolve_device(None)
    already performs the CUDA-extension import plus nvidia-smi probe, so 'auto'
    delegates to it rather than duplicating the detection logic.
    """
    import periodfind

    key = device.strip().lower()
    if key == "cuda":
        key = "gpu"
    if key in ("cpu", "gpu"):
        return key
    if key == "auto":
        return periodfind._resolve_device()
    raise ValueError(
        f"Unknown device {device!r}. Valid: 'auto', 'cpu', 'gpu' (alias 'cuda')."
    )


def _split_by_band(grouped_lcs: list[list[LightCurve]]) -> list[list[LightCurve]]:
    """Expand per-source groups into one single-band group per LightCurve.

    Extractors receive ``list[list[LightCurve]]`` and internally select the band
    with the most epochs.  Feeding them one band at a time is therefore all it
    takes to get per-band features: a source with g and r light curves becomes
    two entries and yields two FeatureVectors sharing a source_id.

    This matters because the alternative — searching only the longest band —
    discards roughly 40% of a typical ZTF source's epochs and, more importantly,
    throws away the cross-band consistency check.  A period that reproduces in
    both g and r is far more likely to be astrophysical than one that appears in
    a single band, where it may just be that band's cadence pattern.
    """
    return [[lc] for lcs in grouped_lcs for lc in lcs]


class FeaturePipeline:
    """Compose a list of FeatureExtractors into FeatureVectors.

    Parameters
    ----------
    extractors:
        Ordered list of extractors to run.  Results are merged left-to-right.
    min_observations:
        Bands with fewer observations than this are skipped — a default
        (all-NaN) FeatureVector is returned for them.
    compute_dmdt:
        If False, the dmdt key is dropped from the final FeatureVector even
        if a DmdtExtractor is present.
    device:
        periodfind device — 'auto', 'cpu' or 'gpu' ('cuda' is accepted as an
        alias for 'gpu').  Resolved to a concrete device and set once before
        the first batch, then reused for all subsequent calls.
    batch_size:
        Number of sources per periodfind batch call.
        Lower this if GPU runs out of memory on long light curves.
    checkpoint_dir:
        Directory for checkpoint files.  None disables checkpointing.
        Use a unique per-run path on scratch storage on MSI.
    """

    def __init__(
        self,
        extractors: list[FeatureExtractor],
        min_observations: int = 50,
        compute_dmdt: bool = True,
        device: str = "auto",
        batch_size: int = 1000,
        checkpoint_dir: str | None = None,
    ) -> None:
        self._extractors      = extractors
        self._min_obs         = min_observations
        self._compute_dmdt    = compute_dmdt
        self._device          = device
        self._batch_size      = batch_size
        self._checkpoint_dir  = checkpoint_dir

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def default(
        cls,
        config: FeatureConfig,
        kowalski_client=None,
    ) -> "FeaturePipeline":
        """Build the standard pipeline from a FeatureConfig.

        Extractor order: statistics → period → dmdt → catalog.

        Parameters
        ----------
        config:
            FeatureConfig section from the root PipelineConfig.
        kowalski_client:
            Optional authenticated penquins Kowalski instance.
            Pass ``ztf_source.client`` here to enable live Gaia EDR3
            cross-matching.  If None, Gaia features are skipped and all
            gaia_* fields in every FeatureVector remain None.
        """
        extractors: list[FeatureExtractor] = [
            StatisticsExtractor(),
            PeriodExtractor(config.period),
        ]
        if config.compute_dmdt:
            extractors.append(DmdtExtractor(config.dmdt))
        extractors.append(CatalogExtractor(config.catalog, kowalski_client=kowalski_client))

        return cls(
            extractors      = extractors,
            min_observations = config.min_observations,
            compute_dmdt    = config.compute_dmdt,
            device          = config.device,
            batch_size      = config.feature_batch_size,
            checkpoint_dir  = config.checkpoint_dir,
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _ckpt_path(self) -> str:
        return os.path.join(self._checkpoint_dir, _CHECKPOINT_FILENAME)  # type: ignore[arg-type]

    def _save_checkpoint(
        self,
        completed_chunk: int,
        n_total_sources: int,
        results: list[FeatureVector],
    ) -> None:
        """Atomically write checkpoint to disk.

        Uses write-to-tmp-then-rename so a crash during the write cannot
        leave a corrupt checkpoint file — matches scope-ml's pattern.
        """
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        state = {
            "completed_chunk"  : completed_chunk,   # 0-indexed last completed chunk
            "n_total_sources"  : n_total_sources,
            "schema"           : _schema_fingerprint(),
            "results"          : results,
        }
        ckpt_path = self._ckpt_path()
        tmp_path  = ckpt_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(state, f, protocol=4)
        os.replace(tmp_path, ckpt_path)   # atomic on POSIX (MSI Lustre / scratch)
        log.info(
            "[checkpoint] Saved after chunk %d  (%d/%d sources done)",
            completed_chunk, len(results), n_total_sources,
        )

    def _load_checkpoint(self, n_total_sources: int) -> tuple[int, list[FeatureVector]] | None:
        """Load checkpoint if one exists and is consistent with this run.

        Returns (start_chunk, partial_results) or None if no valid checkpoint.
        """
        if self._checkpoint_dir is None:
            return None
        ckpt_path = self._ckpt_path()
        if not os.path.exists(ckpt_path):
            return None
        try:
            with open(ckpt_path, "rb") as f:
                state = pickle.load(f)
        except Exception as exc:
            log.warning("[checkpoint] Failed to load checkpoint (%s) — starting fresh", exc)
            return None

        saved_n = state.get("n_total_sources")
        if saved_n != n_total_sources:
            log.warning(
                "[checkpoint] Source count mismatch (saved %s, current %d) — starting fresh",
                saved_n, n_total_sources,
            )
            return None

        saved_schema = state.get("schema")
        if saved_schema != _schema_fingerprint():
            log.warning(
                "[checkpoint] FeatureVector schema changed since this "
                "checkpoint was written — starting fresh so the output does "
                "not mix old and new field sets"
            )
            return None

        completed_chunk = state["completed_chunk"]
        results         = state["results"]
        log.info(
            "[checkpoint] Resuming from chunk %d  (%d/%d sources already done)",
            completed_chunk + 1, len(results), n_total_sources,
        )
        return completed_chunk + 1, results   # next chunk to process

    def _delete_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion."""
        try:
            os.remove(self._ckpt_path())
            log.info("[checkpoint] Deleted after successful completion")
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _primary_n_obs(self, lcs: list[LightCurve]) -> int:
        return max(lc.n_obs for lc in lcs) if lcs else 0

    def _build_feature_vector(
        self,
        lcs: list[LightCurve],
        features: dict[str, Any],
    ) -> FeatureVector:
        primary = max(lcs, key=lambda lc: lc.n_obs)
        valid_fields = {f.name for f in dataclasses.fields(FeatureVector)}

        kwargs: dict[str, Any] = {
            "source_id": primary.source_id,
            "survey"   : primary.survey,
            "band"     : primary.band,
            "ra"       : primary.ra,
            "dec"      : primary.dec,
        }
        for key, val in features.items():
            if key in valid_fields:
                kwargs[key] = val

        if not self._compute_dmdt:
            kwargs.pop("dmdt", None)

        return FeatureVector(**kwargs)

    def _default_fv(self, lcs: list[LightCurve]) -> FeatureVector:
        primary = max(lcs, key=lambda lc: lc.n_obs)
        return FeatureVector(
            source_id=primary.source_id,
            survey=primary.survey,
            band=primary.band,
            ra=primary.ra,
            dec=primary.dec,
            n_obs=primary.n_obs,
        )

    def _process_chunk(
        self, chunk: list[list[LightCurve]]
    ) -> list[FeatureVector]:
        """Process one chunk of sources through all extractors."""
        valid_mask      = [self._primary_n_obs(lcs) >= self._min_obs for lcs in chunk]
        valid_sources   = [lcs for lcs, ok in zip(chunk, valid_mask) if ok]
        valid_positions = [i   for i, ok   in enumerate(valid_mask)   if ok]

        merged: list[dict[str, Any]] = [{} for _ in valid_sources]
        if valid_sources:
            for extractor in self._extractors:
                # base.py asks extractors not to raise, but that is a contract
                # rather than a guarantee — an extractor calling into numpy,
                # periodfind or a network catalog can still fail.  Losing one
                # extractor should cost its own features, not the whole chunk
                # and not a multi-hour job.
                try:
                    partial = extractor.extract(valid_sources)
                except Exception:
                    log.exception(
                        "[pipeline] %s.extract failed on a chunk of %d entries; "
                        "its features stay unset",
                        type(extractor).__name__, len(valid_sources),
                    )
                    continue
                # Extractors are a documented extension point, so the
                # one-dict-per-entry contract is checked rather than assumed;
                # a short list would otherwise misalign features onto the
                # wrong sources.
                if len(partial) != len(valid_sources):
                    log.error(
                        "[pipeline] %s.extract returned %d dicts for %d entries; "
                        "discarding its output for this chunk",
                        type(extractor).__name__, len(partial), len(valid_sources),
                    )
                    continue
                for i, d in enumerate(partial):
                    merged[i].update(d)

        fvs: list[FeatureVector | None] = [None] * len(chunk)
        for pos, lcs, features in zip(valid_positions, valid_sources, merged):
            fvs[pos] = self._build_feature_vector(lcs, features)

        for i, (lcs, ok) in enumerate(zip(chunk, valid_mask)):
            if not ok:
                fvs[i] = self._default_fv(lcs)

        return fvs  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_batch(
        self,
        grouped_lcs: list[list[LightCurve]],
        batch_size: int | None = None,
    ) -> list[FeatureVector]:
        """Extract features for multiple sources, resuming from checkpoint if available.

        Parameters
        ----------
        grouped_lcs:
            Each element is all LightCurves for one source (one per band).
        batch_size:
            Override the instance batch_size for this call.

        Returns
        -------
        list[FeatureVector]
            One FeatureVector per (source, band), ordered by source and then by
            the band order within that source's group.  A source with g and r
            light curves therefore contributes two consecutive entries.
        """
        if not grouped_lcs:
            return []

        import periodfind
        device = _resolve_device(self._device)
        periodfind.set_device(device)
        log.info("[pipeline] periodfind device: %s", device)

        # One entry per band.  Done before prepare() and before chunking so the
        # frequency grid and the checkpoint source count both refer to the same
        # expanded list.
        per_band = _split_by_band(grouped_lcs)

        # Field-wide setup.  Extractors that derive a quantity which must be
        # identical for every source — PeriodExtractor's frequency grid, built
        # from the longest baseline in the field — compute it here.  Doing it
        # per chunk would make a star's period depend on which other stars
        # happened to share its batch.
        for extractor in self._extractors:
            prepare = getattr(extractor, "prepare", None)
            if prepare is not None:
                prepare(per_band)

        size    = batch_size or self._batch_size
        chunks  = list(_chunks(per_band, size))
        n_total = len(per_band)

        # ── Resume from checkpoint if available ──────────────────────────
        start_chunk = 0
        results: list[FeatureVector] = []

        if self._checkpoint_dir is not None:
            loaded = self._load_checkpoint(n_total)
            if loaded is not None:
                start_chunk, results = loaded

        # ── Process remaining chunks ─────────────────────────────────────
        for chunk_idx, chunk in enumerate(chunks):
            if chunk_idx < start_chunk:
                continue

            chunk_fvs = self._process_chunk(chunk)
            results.extend(chunk_fvs)

            if self._checkpoint_dir is not None:
                self._save_checkpoint(chunk_idx, n_total, results)

        # ── Clean up on success ──────────────────────────────────────────
        if self._checkpoint_dir is not None:
            self._delete_checkpoint()

        return results
