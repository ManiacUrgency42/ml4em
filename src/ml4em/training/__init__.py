"""Training pipeline for ml4em.

Components
----------
Trainer          Protocol — any object with fit() and save()
FeatureDataset   Labeled feature vectors loaded from storage
StandardTrainer  Concrete trainer for any MLModel

Usage
-----
    from ml4em.training import FeatureDataset, StandardTrainer, Trainer
"""

from .base import Trainer
from .dataset import FeatureDataset
from .trainer import StandardTrainer

__all__ = [
    "Trainer",
    "FeatureDataset",
    "StandardTrainer",
]
