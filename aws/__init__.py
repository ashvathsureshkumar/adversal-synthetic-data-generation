"""
AWS infrastructure components for scalable synthetic data generation.

This module provides AWS integrations for cloud-based training, storage,
and deployment of synthetic data generation models.
"""

from .s3_manager import S3Manager, DatasetManager
from .sagemaker_trainer import SageMakerTrainer, ModelDeployment
# from .ec2_manager import EC2Manager, TrainingCluster  # Optional advanced feature

__all__ = [
    "S3Manager",
    "DatasetManager",
    "SageMakerTrainer", 
    "ModelDeployment"
    # "EC2Manager",      # Optional advanced feature
    # "TrainingCluster"  # Optional advanced feature
]
