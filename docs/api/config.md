# Config

Pydantic configuration models and YAML loader.

::: ml4em.config.schema
    options:
      members:
        - PipelineConfig
        - SourcesConfig
        - ZTFConfig
        - RubinConfig
        - FeatureConfig
        - PeriodConfig
        - DmdtConfig
        - CatalogConfig
        - StorageConfig
        - TrainingConfig
        - InferenceConfig
      show_root_heading: false
      show_source: true

::: ml4em.config.loader
    options:
      members:
        - load_config
        - load_default_config
        - get_ztf_token
        - get_rubin_token
      show_root_heading: false
      show_source: true
