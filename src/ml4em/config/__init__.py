"""Configuration package — schema and loader."""

from .loader import get_rubin_token, get_ztf_token, load_config, load_default_config
from .schema import (
    CatalogConfig,
    ClassificationConfig,
    DmdtConfig,
    FeatureConfig,
    InferenceConfig,
    PeriodConfig,
    RubinConfig,
    SourcesConfig,
    StellarClassConfig,
    StorageConfig,
    TrainingConfig,
    WDBConfig,
    ZTFConfig,
)

__all__ = [
    # Root
    "WDBConfig",
    # Classification
    "ClassificationConfig", "StellarClassConfig",
    # Sources
    "SourcesConfig", "ZTFConfig", "RubinConfig",
    # Features
    "FeatureConfig", "PeriodConfig", "DmdtConfig", "CatalogConfig",
    # Other layers
    "StorageConfig", "TrainingConfig", "InferenceConfig",
    # Loader
    "load_config", "load_default_config", "get_ztf_token", "get_rubin_token",
]
