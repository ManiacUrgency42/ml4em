/* ════════════════════════════════════════════════════════════════
   ml4em  —  Interactive Architecture Diagram
   ════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  /* ── Component data ────────────────────────────────────────── */
  var COMPONENTS = {

    /* Foundation */
    types: {
      layer: "foundation",
      icon:  "📋",
      name:  "types.py",
      file:  "src/ml4em/types.py",
      status: "complete",
      desc: "Defines the three data contracts that cross every layer boundary: <code>LightCurve</code>, <code>FeatureVector</code>, and <code>Candidate</code>. All are frozen dataclasses — immutable by design. No raw dicts, tuples, or unnamed arrays are passed between layers. <code>FeatureVector</code> holds 48 fields; every float field defaults to <code>np.nan</code> so partial execution is always explicit and detectable.",
      link: "data-contracts/"
    },
    constants: {
      layer: "foundation",
      icon:  "🔢",
      name:  "constants.py",
      file:  "src/ml4em/constants.py",
      status: "complete",
      desc: "Physical constants (G, c, M☉) and survey-specific parameters: ZTF band IDs (1→g, 2→r, 3→i), Rubin LSST bands, dm/dt histogram binning (26×26 log-spaced time axis), and cross-match radii. Nothing here is tunable at runtime — all adjustable parameters live in <code>config.yaml</code> instead.",
      link: "layers/foundation/"
    },
    config: {
      layer: "foundation",
      icon:  "⚙️",
      name:  "config/",
      file:  "src/ml4em/config/",
      status: "complete",
      desc: "Pydantic v2 models that define the exact shape of <code>config.yaml</code>. The root is <code>PipelineConfig</code>, which nests <code>ZTFConfig</code>, <code>PeriodConfig</code>, <code>DmdtConfig</code>, <code>TrainingConfig</code>, <code>InferenceConfig</code>, and more. Environment variables resolve automatically at load time. One call to <code>loader.py</code> gives you a fully validated config object.",
      link: "layers/foundation/"
    },

    /* Data */
    ztf: {
      layer: "data",
      icon:  "🔭",
      name:  "ZTFSource",
      file:  "src/ml4em/data/ztf.py",
      status: "complete",
      desc: "Queries the Zwicky Transient Facility via the Kowalski API (penquins client). Automatically applies two quality filters: removes bad observations where <code>catflags ≠ 0</code>, and drops intra-night duplicates where <code>Δt &lt; 30 min</code>. Returns one <code>LightCurve</code> per band (g, r, i) for each requested source.",
      link: "layers/data/"
    },
    rubin: {
      layer: "data",
      icon:  "🌌",
      name:  "RubinSource",
      file:  "src/ml4em/data/rubin.py",
      status: "stub",
      desc: "TAP/SQL interface to Vera C. Rubin Observatory LSST data products. The query structure and <code>LightCurveSource</code> protocol implementation are in place. Pending connection to a live Rubin TAP endpoint once survey operations begin. A drop-in replacement for <code>ZTFSource</code> — the rest of the pipeline is unchanged.",
      link: "layers/data/"
    },
    simulated: {
      layer: "data",
      icon:  "💻",
      name:  "SimulatedSource",
      file:  "src/ml4em/data/simulation.py",
      status: "stub",
      desc: "Generates synthetic light curves via the Lcurve library for testing and pipeline validation before real survey data is available. Wraps Lcurve as a <code>LightCurveSource</code> — the rest of the pipeline cannot tell the data is synthetic.",
      link: "layers/data/"
    },

    /* Features */
    statistics: {
      layer: "features",
      icon:  "📊",
      name:  "StatisticsExtractor",
      file:  "src/ml4em/features/statistics.py",
      status: "complete",
      desc: "Computes 22 scalar variability metrics on the primary band (whichever has most observations). Includes Stetson I/J/K indices, Anderson-Darling and Shapiro-Wilk normality tests, inverse von Neumann ratio, robust RMS (RoMS), normalized excess variance, IQR, skewness, and 7 quantile ratios (i60r–i90r). Backed by periodfind's <code>BasicStats</code> GPU/CPU kernel.",
      link: "layers/features/"
    },
    period: {
      layer: "features",
      icon:  "〰️",
      name:  "PeriodExtractor",
      file:  "src/ml4em/features/period.py",
      status: "complete",
      desc: "Runs 6 period-finding algorithms in parallel — Conditional Entropy (CE), Analysis of Variance (AOV), Lomb-Scargle (LS), Multi-Harmonic Fit (MHF), FPW, and Box-Least-Squares (BLS). Finds consensus where ≥ 2 algorithms agree within 2% tolerance. Then decomposes the phased light curve into 14 Fourier coefficients (4 relative amplitudes, 4 relative phases, f1 power, BIC, a, b, amplitude, phi0). All via periodfind.",
      link: "layers/features/"
    },
    dmdt: {
      layer: "features",
      icon:  "🗺️",
      name:  "DmdtExtractor",
      file:  "src/ml4em/features/dmdt.py",
      status: "complete",
      desc: "Builds a 26×26 pairwise (Δt, Δmag) histogram over all N(N−1)/2 observation pairs. Time axis is log-spaced to capture variability across many timescales; magnitude axis is linear. Output is L2-normalized before storage. Encodes variability timescale structure as a 2D image — useful as CNN input or mixed alongside scalar features.",
      link: "layers/features/"
    },
    catalog: {
      layer: "features",
      icon:  "⭐",
      name:  "CatalogExtractor",
      file:  "src/ml4em/features/catalog.py",
      status: "stub",
      desc: "Cross-matches each source position (RA/Dec from <code>LightCurve</code>) against Gaia EDR3 via a 2″ cone search. Returns 4 astrometric features: <code>gaia_parallax</code>, <code>gaia_parallax_error</code>, <code>gaia_bp_rp</code> (colour index), and <code>gaia_ruwe</code> (a binarity quality indicator). TAP query drafted; Gaia endpoint integration is pending.",
      link: "layers/features/"
    },
    pipeline: {
      layer: "features",
      icon:  "🔄",
      name:  "FeaturePipeline",
      file:  "src/ml4em/features/pipeline.py",
      status: "complete",
      desc: "Orchestrates all four extractors in fixed order: Statistics → Period → Dmdt → Catalog. Chunks sources into configurable batches, filters out sources below <code>min_observations</code> (default 50), and merges each extractor's output dict into a single <code>FeatureVector</code>. Never raises — sources that fail any extractor receive an all-NaN <code>FeatureVector</code> and processing continues. Sets the periodfind device (CPU or GPU) once at startup.",
      link: "layers/features/"
    },

    /* Training */
    dataset: {
      layer: "training",
      icon:  "📁",
      name:  "FeatureDataset",
      file:  "src/ml4em/training/dataset.py",
      status: "partial",
      desc: "Loads pre-computed <code>FeatureVector</code> objects from parquet files and joins them with a CSV label file (<code>source_id → 0/1</code>). Produces <code>LabeledSample</code> objects and supports configurable train/val/test splits. Label join logic is complete; the parquet loading path is a stub pending storage schema finalization.",
      link: "layers/training/"
    },
    trainer: {
      layer: "training",
      icon:  "🏋️",
      name:  "StandardTrainer",
      file:  "src/ml4em/training/trainer.py",
      status: "partial",
      desc: "Wraps any object implementing the <code>MLModel</code> protocol and drives training using <code>TrainingConfig</code> (learning rate, batch size, max epochs, early-stopping patience). Interface and shell are defined; the inner training loop is pending a complete <code>MLModel</code> implementation.",
      link: "layers/training/"
    },

    /* Models */
    xgboost: {
      layer: "models",
      icon:  "🌲",
      name:  "XGBoostClassifier",
      file:  "src/ml4em/models/xgboost.py",
      status: "partial",
      desc: "Reference <code>MLModel</code> implementation using XGBoost gradient-boosted trees. Operates on the 43 scalar fields from <code>FeatureVector</code> — excludes metadata and the dm/dt image. Handles NaN values natively with no imputation required. Serializes to <code>weights.json</code> + <code>manifest.json</code> for versioned loading via the inference registry. Implements <code>predict_proba()</code>, <code>save()</code>, and <code>load()</code>.",
      link: "layers/models/"
    },

    /* Inference */
    loader: {
      layer: "inference",
      icon:  "📦",
      name:  "load_model()",
      file:  "src/ml4em/inference/loader.py",
      status: "complete",
      desc: "Reads <code>manifest.json</code> from a saved model directory to determine the class name, then dispatches to the correct <code>load()</code> classmethod via <code>_MODEL_REGISTRY</code>. Adding support for a new model type requires just one line: register it in the registry dict. Returns any object implementing <code>MLModel</code>.",
      link: "layers/inference/"
    },
    predictor: {
      layer: "inference",
      icon:  "🎯",
      name:  "StandardPredictor",
      file:  "src/ml4em/inference/predictor.py",
      status: "partial",
      desc: "Implements the <code>Predictor</code> protocol: calls <code>model.predict_proba(features)</code> to get a probability array of shape (N, 2), then passes it along with the original <code>FeatureVector</code> list to <code>postprocess()</code>. Shell is fully defined; execution depends on a complete <code>MLModel</code> implementation.",
      link: "layers/inference/"
    },
    postprocess: {
      layer: "inference",
      icon:  "✅",
      name:  "postprocess()",
      file:  "src/ml4em/inference/postprocess.py",
      status: "complete",
      desc: "Converts a raw probability array into a list of <code>Candidate</code> dataclass instances. Applies three confidence tiers: High (probability ≥ 0.9), Medium (≥ 0.7), Low (below 0.7). Populates each Candidate's sky position (RA, Dec), period, and period algorithm from the source's <code>FeatureVector</code>. This is where raw model output becomes science-ready results.",
      link: "layers/inference/"
    }
  };

  var LAYER_COLORS = {
    foundation : "#F59E0B",
    data       : "#60A5FA",
    features   : "#A78BFA",
    models     : "#FB923C",
    training   : "#FB923C",
    inference  : "#34D399"
  };

  var STATUS_LABELS = {
    complete : "✓ complete",
    stub     : "stub",
    partial  : "partial"
  };

  var activeCard = null;

  /* ── Helpers ────────────────────────────────────────────────── */
  function hexToRgba(hex, a) {
    var r = parseInt(hex.slice(1,3), 16);
    var g = parseInt(hex.slice(3,5), 16);
    var b = parseInt(hex.slice(5,7), 16);
    return "rgba(" + r + "," + g + "," + b + "," + a + ")";
  }

  function $(id) { return document.getElementById(id); }

  /* ── Open panel ─────────────────────────────────────────────── */
  function openPanel(id) {
    var d = COMPONENTS[id];
    if (!d) return;

    var panel    = $("detail-panel");
    var backdrop = $("panel-backdrop");
    var color    = LAYER_COLORS[d.layer] || "#60A5FA";

    /* badge */
    var badge = panel.querySelector(".panel-layer-badge");
    badge.textContent        = d.layer.toUpperCase();
    badge.style.color        = color;
    badge.style.borderColor  = color;
    badge.style.backgroundColor = hexToRgba(color, 0.1);

    /* content */
    panel.querySelector(".panel-icon").textContent  = d.icon;
    panel.querySelector(".panel-name").textContent  = d.name;
    panel.querySelector(".panel-file").textContent  = d.file;
    panel.querySelector(".panel-desc").innerHTML    = d.desc;

    /* status */
    var statusEl = panel.querySelector(".panel-status-badge");
    statusEl.className   = "panel-status-badge card-status status-" + d.status;
    statusEl.textContent = STATUS_LABELS[d.status] || d.status;

    /* docs link */
    var link = panel.querySelector(".panel-link");
    link.href                  = d.link;
    link.style.color           = color;
    link.style.borderColor     = hexToRgba(color, 0.5);
    link.style.backgroundColor = hexToRgba(color, 0.07);

    /* open */
    panel.classList.add("open");
    backdrop.classList.add("open");

    /* highlight card */
    if (activeCard) activeCard.classList.remove("active");
    var card = document.querySelector('[data-id="' + id + '"]');
    if (card) { card.classList.add("active"); activeCard = card; }
  }

  /* ── Close panel ────────────────────────────────────────────── */
  function closePanel() {
    $("detail-panel").classList.remove("open");
    $("panel-backdrop").classList.remove("open");
    if (activeCard) { activeCard.classList.remove("active"); activeCard = null; }
  }

  /* ── Init ───────────────────────────────────────────────────── */
  function init() {
    /* wire cards */
    document.querySelectorAll("[data-id]").forEach(function (card) {
      card.addEventListener("click", function () { openPanel(card.dataset.id); });
    });

    /* close triggers */
    var backdrop = $("panel-backdrop");
    if (backdrop) backdrop.addEventListener("click", closePanel);

    var closeBtn = document.querySelector(".panel-close");
    if (closeBtn) closeBtn.addEventListener("click", closePanel);

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closePanel();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  /* expose globally for MkDocs Material SPA navigation re-init */
  window.archDiagramInit  = init;
  window.archClosePanel   = closePanel;

  /* re-init after MkDocs instant navigation */
  document.addEventListener("DOMContentSwitch", init);

}());
