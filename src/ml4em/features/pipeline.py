"""
FeaturePipeline — compose extractors into a batch of FeatureVectors.

The pipeline:
1. Sets the periodfind device (CPU/GPU) once before processing.
2. Chunks the source list into batches of feature_batch_size.
3. Within each batch, partitions sources by min_observations.
4. Calls each extractor once for the valid sources in the batch.
5. Merges extractor outputs and constructs FeatureVectors.
6. Returns a default (all-NaN) FeatureVector for skipped sources.
7. Optionally checkpoints after every chunk for HPC fault tolerance.

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


def _chunks(lst: list, size: int):
    """Yield successive chunks of `size` from `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


class FeaturePipeline:
    """Compose a list of FeatureExtractors into FeatureVectors.

    Parameters
    ----------
    extractors:
        Ordered list of extractors to run.  Results are merged left-to-right.
    min_observations:
        Sources with fewer observations than this in the primary band are
        skipped — a default (all-NaN) FeatureVector is returned.
    compute_dmdt:
        If False, the dmdt key is dropped from the final FeatureVector even
        if a DmdtExtractor is present.
    device:
        periodfind device — 'cpu', 'cuda', or 'auto'.
        Set once before the first batch and reused for all subsequent calls.
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
                partial = extractor.extract(valid_sources)
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
            One FeatureVector per source, in the same order as grouped_lcs.
        """
        if not grouped_lcs:
            return []

        import periodfind
        periodfind.set_device(self._device)

        size   = batch_size or self._batch_size
        chunks = list(_chunks(grouped_lcs, size))
        n_total = len(grouped_lcs)

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
