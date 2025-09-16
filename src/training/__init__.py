"""
Training modules for adversarial synthetic data generation.

This module contains training loops, fairness constraints, privacy mechanisms,
and evaluation metrics for GAN-based synthetic data generation.
"""

from .trainer import GANTrainer, WGANGPTrainer, CGANTrainer
from .fairness import FairnessConstraints, DemographicParityLoss, EqualizedOddsLoss
from .privacy import PrivacyEngine, DifferentialPrivacyGAN
from .evaluation import SyntheticDataEvaluator, QualityMetrics
from .utils import EarlyStopping, ModelCheckpoint, TrainingLogger

__all__ = [
    "GANTrainer",
    "WGANGPTrainer", 
    "CGANTrainer",
    "FairnessConstraints",
    "DemographicParityLoss",
    "EqualizedOddsLoss",
    "PrivacyEngine",
    "DifferentialPrivacyGAN",
    "SyntheticDataEvaluator",
    "QualityMetrics",
    "EarlyStopping",
    "ModelCheckpoint",
    "TrainingLogger"
]
