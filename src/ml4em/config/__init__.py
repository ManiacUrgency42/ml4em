"""Configuration package — schema and loader."""

from .loader import get_ztf_token, load_config, load_default_config
from .schema import (
    CatalogConfig,
    DmdtConfig,
    FeatureConfig,
    InferenceConfig,
    PeriodConfig,
    SourcesConfig,
    StorageConfig,
    TrainingConfig,
    PipelineConfig,
    ZTFConfig,
)

__all__ = [
    # Root
    "PipelineConfig",
    # Sources
    "SourcesConfig", "ZTFConfig",
    # Features
    "FeatureConfig", "PeriodConfig", "DmdtConfig", "CatalogConfig",
    # Other layers
    "StorageConfig", "TrainingConfig", "InferenceConfig",
    # Loader
    "load_config", "load_default_config", "get_ztf_token",
]
