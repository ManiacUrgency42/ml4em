# Inference

Model loading and prediction.

::: ml4em.inference.base
    options:
      members:
        - Predictor
      show_root_heading: false
      show_source: true

::: ml4em.inference.loader
    options:
      members:
        - load_model
      show_root_heading: false
      show_source: true

::: ml4em.inference.predictor
    options:
      members:
        - StandardPredictor
      show_root_heading: false
      show_source: true

::: ml4em.inference.postprocess
    options:
      members:
        - probabilities_to_candidates
      show_root_heading: false
      show_source: true
