#!/usr/bin/env python3
"""
End-to-end pipeline demo: WDB catalog → ZTF light curves → features →
LogisticExampleClassifier → inference results.

Positives: sources in wdb_sources.csv, matched within 2 arcsec in ZTF.
Negatives: other ZTF sources returned by the same 30-arcsec cone search,
           sampled from the same sky region with no extra selection.

Usage
-----
    # From ml4em/ directory:
    python scripts/run_demo.py

    # With a specific config:
    python scripts/run_demo.py --config /path/to/config.yaml

    # Save the trained model to a custom location:
    python scripts/run_demo.py --save-model models/logistic_demo
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

from ml4em.config.loader import get_ztf_token, load_config
from ml4em.data.ztf import ZTFSource
from ml4em.features.pipeline import FeaturePipeline
from ml4em.inference.loader import load_model
from ml4em.inference.predictor import StandardPredictor
from ml4em.models.logistic_example import LogisticExampleClassifier, LogisticExampleConfig
from ml4em.training.dataset import FeatureDataset
from ml4em.types import LightCurve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Cone search radii
_POSITIVE_RADIUS_ARCSEC = 2.0    # within this → positive (WDB)
_NEGATIVE_RADIUS_ARCSEC = 30.0   # outer cone → negative candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angular_sep_arcsec(
    ra1: float, dec1: float,
    ra2: float, dec2: float,
) -> float:
    """Great-circle separation in arcseconds (haversine formula)."""
    dra  = math.radians(ra2 - ra1)
    ddec = math.radians(dec2 - dec1)
    a = (
        math.sin(ddec / 2) ** 2
        + math.cos(math.radians(dec1))
        * math.cos(math.radians(dec2))
        * math.sin(dra / 2) ** 2
    )
    return 2.0 * math.degrees(math.asin(math.sqrt(min(a, 1.0)))) * 3600.0


def _load_catalog(path: str) -> list[dict]:
    """Read wdb_sources.csv → list of rows with ra, dec keys."""
    p = Path(path)
    if not p.exists():
        log.error("Catalog not found: %s", path)
        log.error("Place wdb_sources.csv at the catalog_path in config.yaml.")
        sys.exit(1)
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %d sources from %s", len(rows), path)
    return rows


def _fetch_and_label(
    catalog: list[dict],
    ztf: ZTFSource,
) -> tuple[dict[str, list[LightCurve]], dict[str, int]]:
    """Cone-search each catalog position and assign labels.

    Returns
    -------
    grouped : {source_id: [LightCurve, ...]}
        All bands for each unique ZTF source found.
    labels  : {source_id: 0 or 1}
        1 = within _POSITIVE_RADIUS_ARCSEC of a WDB position.
        0 = returned by the same cone search but further away.
    """
    grouped: dict[str, list[LightCurve]] = defaultdict(list)
    labels:  dict[str, int]              = {}

    for i, row in enumerate(catalog):
        ra  = float(row["ra"])
        dec = float(row["dec"])
        name = row.get("obj_id") or row.get("name") or f"src_{i}"

        lcs = ztf.fetch_by_position(ra, dec, radius_arcsec=_NEGATIVE_RADIUS_ARCSEC)
        if not lcs:
            log.warning("  %s (%.4f, %.4f): no ZTF sources found", name, ra, dec)
            continue

        n_pos = n_neg = 0
        for lc in lcs:
            sid = lc.source_id
            grouped[sid].append(lc)
            if sid not in labels:
                sep = _angular_sep_arcsec(ra, dec, lc.ra, lc.dec)
                if sep <= _POSITIVE_RADIUS_ARCSEC:
                    labels[sid] = 1
                    n_pos += 1
                else:
                    labels[sid] = 0
                    n_neg += 1

        log.info("  %s: %d positive, %d negative ZTF sources", name, n_pos, n_neg)

    return dict(grouped), labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ml4em end-to-end demo with LogisticExampleClassifier"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in cwd)",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Directory to save the trained model (default: storage.models_dir/logistic_demo)",
    )
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    model_save_path = args.save_model or str(
        Path(cfg.storage.models_dir) / "logistic_demo"
    )

    # ── ZTF connection ──────────────────────────────────────────────────────────
    log.info("Connecting to Kowalski...")
    token = get_ztf_token()     # reads ML4EM_ZTF_TOKEN from env / .env
    ztf   = ZTFSource(cfg.sources.ztf, token)

    # ── Catalog ─────────────────────────────────────────────────────────────────
    catalog = _load_catalog(cfg.storage.catalog_path)

    # ── Fetch light curves ──────────────────────────────────────────────────────
    log.info("Fetching ZTF light curves (positives + negatives from same cone)...")
    grouped, labels = _fetch_and_label(catalog, ztf)

    n_pos = sum(1 for v in labels.values() if v == 1)
    n_neg = sum(1 for v in labels.values() if v == 0)
    log.info("Total: %d positive, %d negative ZTF sources", n_pos, n_neg)

    if n_pos == 0:
        log.error("No positive sources found. Check catalog_path and ZTF credentials.")
        sys.exit(1)
    if n_neg == 0:
        log.warning(
            "No negative sources found — positives may be isolated on the sky. "
            "The model cannot train without negatives."
        )
        sys.exit(1)

    # ── Feature extraction ──────────────────────────────────────────────────────
    log.info("Extracting features (device=%s)...", cfg.features.device)
    pipeline      = FeaturePipeline.default(cfg.features)
    sources_list  = list(grouped.values())
    feature_vectors = pipeline.run_batch(sources_list)
    log.info("Features extracted for %d sources", len(feature_vectors))

    # Persist to parquet so features don't need to be recomputed
    parquet_path = Path(cfg.storage.features_dir) / "demo.parquet"
    FeatureDataset.save_feature_vectors(feature_vectors, str(parquet_path))
    log.info("Feature vectors saved to %s", parquet_path)

    # ── Build labeled dataset ───────────────────────────────────────────────────
    dataset = FeatureDataset.from_feature_vectors(feature_vectors, labels)
    log.info("Labeled samples: %d  |  class counts: %s",
             len(dataset), dataset.class_counts())

    if len(dataset) < 4:
        log.error("Not enough labeled samples to split into train/test.")
        sys.exit(1)

    train_ds, _, test_ds = dataset.split(
        val_fraction=cfg.training.val_fraction,
        test_fraction=cfg.training.test_fraction,
        seed=cfg.training.seed,
    )
    log.info("Train: %d  |  Test: %d", len(train_ds), len(test_ds))

    # ── Train ───────────────────────────────────────────────────────────────────
    log.info("Training LogisticExampleClassifier (n_epochs=300)...")
    model = LogisticExampleClassifier(
        LogisticExampleConfig(n_epochs=300, learning_rate=1e-2)
    )
    model.fit(
        [s.feature for s in train_ds],
        [s.label   for s in train_ds],
    )
    log.info("Training complete.")

    # ── Feature weights ─────────────────────────────────────────────────────────
    log.info("Top 10 features by learned weight magnitude:")
    for name, w in list(model.weights().items())[:10]:
        log.info("  %-35s  %+.4f", name, w)

    # ── Save + reload through inference layer ───────────────────────────────────
    model.save(model_save_path)
    log.info("Model saved to %s", model_save_path)

    loaded    = load_model(model_save_path)
    predictor = StandardPredictor(loaded, cfg.inference)
    candidates = predictor.predict([s.feature for s in test_ds])

    # ── Results ─────────────────────────────────────────────────────────────────
    by_conf: dict[str, list] = defaultdict(list)
    for c in candidates:
        by_conf[c.confidence].append(c)

    log.info("─── Inference results on test set (%d sources) ───", len(candidates))
    for tier in ("high", "medium", "low"):
        srcs = by_conf.get(tier, [])
        log.info("  %s confidence: %d sources", tier, len(srcs))
        for c in srcs[:3]:
            log.info(
                "    %-20s  P=%.3f  period=%.4f d  [%s]",
                c.source_id, c.probability, c.period, c.period_algorithm or "—",
            )


if __name__ == "__main__":
    main()
