"""
AWS SageMaker integration for scalable model training and deployment.

This module provides SageMaker integration for training synthetic data generation
models at scale and deploying them as endpoints.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.processing import ProcessingInput, ProcessingOutput
import pandas as pd
import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from datetime import datetime
import tarfile
import tempfile
from dataclasses import dataclass


@dataclass
class TrainingJob:
    """Information about a SageMaker training job."""
    job_name: str
    status: str
    creation_time: datetime
    training_start_time: Optional[datetime]
    training_end_time: Optional[datetime]
    model_artifacts: Optional[str]
    training_image: str
    instance_type: str
    instance_count: int
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]


class SageMakerTrainer:
    """
    SageMaker integration for training synthetic data generation models.
    
    Provides scalable training with automatic hyperparameter tuning,
    distributed training, and model artifact management.
    """
    
    def __init__(
        self,
        role: str,
        bucket_name: str,
        region_name: str = "us-west-2",
        base_job_name: str = "synthetic-data-training"
    ):
        self.role = role
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.base_job_name = base_job_name
        
        # Initialize SageMaker session
        self.session = sagemaker.Session(boto3.Session(region_name=region_name))
        self.logger = logging.getLogger(__name__)
        
        # Training configuration
        self.default_instance_type = "ml.g4dn.xlarge"  # GPU instance for deep learning
        self.default_volume_size = 30  # GB
        self.default_max_run = 24 * 60 * 60  # 24 hours in seconds
        
        self.logger.info(f"SageMaker trainer initialized for region: {region_name}")
    
    def create_training_script(
        self,
        script_dir: str,
        model_type: str = "wgan_gp",
        fairness_enabled: bool = True,
        privacy_enabled: bool = True
    ) -> str:
        """
        Create training script for SageMaker.
        
        Args:
            script_dir: Directory to save the script
            model_type: Type of model to train
            fairness_enabled: Whether to enable fairness constraints
            privacy_enabled: Whether to enable privacy mechanisms
            
        Returns:
            Path to the created training script
        """
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, "train.py")
        
        script_content = f'''
import argparse
import json
import logging
import os
import sys
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add the source directory to Python path
sys.path.append("/opt/ml/code")

# Import our modules (these would be packaged with the training script)
from src.models.wgan_gp import WGAN_GP
from src.models.cgan import ConditionalGAN
from src.training.trainer import WGANGPTrainer, CGANTrainer
from src.training.fairness import FairnessConstraints
from src.training.privacy import PrivacyEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument("--model_type", type=str, default="{model_type}")
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--generator_dims", type=str, default="128,256,512")
    parser.add_argument("--discriminator_dims", type=str, default="512,256,128")
    parser.add_argument("--embedding_dim", type=int, default=64)
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--n_critic", type=int, default=5)
    
    # WGAN-GP specific
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    
    # Fairness and privacy
    parser.add_argument("--fairness_enabled", type=bool, default={str(fairness_enabled).lower()})
    parser.add_argument("--fairness_weight", type=float, default=0.1)
    parser.add_argument("--privacy_enabled", type=bool, default={str(privacy_enabled).lower()})
    parser.add_argument("--privacy_epsilon", type=float, default=1.0)
    parser.add_argument("--privacy_delta", type=float, default=1e-5)
    
    # SageMaker specific arguments
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current_host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--num_gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))
    
    return parser.parse_args()


def load_data(data_dir):
    """Load training data from the specified directory."""
    data_file = os.path.join(data_dir, "data.parquet")
    if os.path.exists(data_file):
        df = pd.read_parquet(data_file)
        logger.info(f"Loaded data with shape: {{df.shape}}")
        return df
    else:
        raise FileNotFoundError(f"Data file not found: {{data_file}}")


def create_model(args, input_dim):
    """Create the appropriate model based on arguments."""
    device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")
    
    generator_dims = [int(x) for x in args.generator_dims.split(",")]
    discriminator_dims = [int(x) for x in args.discriminator_dims.split(",")]
    
    if args.model_type == "wgan_gp":
        model = WGAN_GP(
            input_dim=input_dim,
            noise_dim=args.noise_dim,
            generator_hidden_dims=generator_dims,
            discriminator_hidden_dims=discriminator_dims,
            lambda_gp=args.lambda_gp,
            fairness_lambda=args.fairness_weight if args.fairness_enabled else 0.0,
            device=device
        )
    elif args.model_type == "cgan":
        # For cGAN, we'd need condition_dim - this is simplified
        model = ConditionalGAN(
            input_dim=input_dim,
            condition_dim=args.embedding_dim,  # Simplified
            noise_dim=args.noise_dim,
            generator_hidden_dims=generator_dims,
            discriminator_hidden_dims=discriminator_dims,
            fairness_lambda=args.fairness_weight if args.fairness_enabled else 0.0,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {{args.model_type}}")
    
    return model, device


def train_model(args, model, data, device):
    """Train the model with the given data."""
    # Convert data to tensors
    data_tensor = torch.FloatTensor(data.values).to(device)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create trainer
    if args.model_type == "wgan_gp":
        trainer = WGANGPTrainer(model, device=device)
    else:
        trainer = CGANTrainer(model, device=device)
    
    # Setup optimizers
    trainer.setup_optimizers(
        g_lr=args.learning_rate,
        d_lr=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2
    )
    
    # Setup privacy if enabled
    if args.privacy_enabled:
        trainer.setup_privacy_engine(
            epsilon=args.privacy_epsilon,
            delta=args.privacy_delta
        )
    
    # Setup fairness if enabled
    if args.fairness_enabled:
        trainer.setup_fairness_constraints(
            protected_attributes=["attr_0"],  # Simplified
            constraint_weight=args.fairness_weight
        )
    
    # Train the model
    logger.info(f"Starting training for {{args.epochs}} epochs")
    training_history = trainer.train(
        dataloader=dataloader,
        num_epochs=args.epochs
    )
    
    return training_history


def save_model(args, model, training_history):
    """Save the trained model and artifacts."""
    # Save model
    model_path = os.path.join(args.model_dir, "model.pt")
    model.save_model(model_path)
    
    # Save training history
    history_path = os.path.join(args.model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2, default=str)
    
    # Save model configuration
    config = {{
        "model_type": args.model_type,
        "noise_dim": args.noise_dim,
        "fairness_enabled": args.fairness_enabled,
        "privacy_enabled": args.privacy_enabled,
        "hyperparameters": vars(args)
    }}
    
    config_path = os.path.join(args.model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model saved to {{args.model_dir}}")


def main():
    args = parse_args()
    
    # Load data
    data = load_data(args.train)
    
    # Create model
    model, device = create_model(args, data.shape[1])
    
    # Train model
    training_history = train_model(args, model, data, device)
    
    # Save model
    save_model(args, model, training_history)
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        self.logger.info(f"Training script created: {script_path}")
        return script_path
    
    def prepare_training_package(
        self,
        source_dir: str,
        output_dir: str,
        include_dependencies: bool = True
    ) -> str:
        """
        Prepare a training package for SageMaker.
        
        Args:
            source_dir: Source directory containing code
            output_dir: Output directory for the package
            include_dependencies: Whether to include dependencies
            
        Returns:
            Path to the created package
        """
        os.makedirs(output_dir, exist_ok=True)
        package_path = os.path.join(output_dir, "training_package.tar.gz")
        
        with tarfile.open(package_path, "w:gz") as tar:
            # Add source code
            tar.add(source_dir, arcname="src")
            
            # Add requirements if requested
            if include_dependencies:
                requirements_path = os.path.join(source_dir, "..", "requirements.txt")
                if os.path.exists(requirements_path):
                    tar.add(requirements_path, arcname="requirements.txt")
        
        self.logger.info(f"Training package created: {package_path}")
        return package_path
    
    def launch_training_job(
        self,
        training_data_s3_uri: str,
        script_path: str,
        hyperparameters: Dict[str, Any],
        instance_type: str = None,
        instance_count: int = 1,
        volume_size: int = None,
        max_run: int = None,
        job_name: Optional[str] = None,
        validation_data_s3_uri: Optional[str] = None,
        checkpoint_s3_uri: Optional[str] = None,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Launch a SageMaker training job.
        
        Args:
            training_data_s3_uri: S3 URI for training data
            script_path: Path to training script
            hyperparameters: Training hyperparameters
            instance_type: EC2 instance type
            instance_count: Number of instances
            volume_size: EBS volume size in GB
            max_run: Maximum runtime in seconds
            job_name: Custom job name
            validation_data_s3_uri: S3 URI for validation data
            checkpoint_s3_uri: S3 URI for checkpoints
            tags: Resource tags
            
        Returns:
            Training job name
        """
        # Set defaults
        instance_type = instance_type or self.default_instance_type
        volume_size = volume_size or self.default_volume_size
        max_run = max_run or self.default_max_run
        
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            job_name = f"{self.base_job_name}-{timestamp}"
        
        # Create PyTorch estimator
        pytorch_estimator = PyTorch(
            entry_point=os.path.basename(script_path),
            source_dir=os.path.dirname(script_path),
            role=self.role,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            max_run=max_run,
            framework_version="1.12",
            py_version="py38",
            hyperparameters=hyperparameters,
            sagemaker_session=self.session,
            tags=tags,
            checkpoint_s3_uri=checkpoint_s3_uri,
            use_spot_instances=False,  # Can be enabled for cost savings
            metric_definitions=[
                {"Name": "train:generator_loss", "Regex": "Generator Loss: ([0-9\\.]+)"},
                {"Name": "train:discriminator_loss", "Regex": "Discriminator Loss: ([0-9\\.]+)"},
                {"Name": "train:wasserstein_distance", "Regex": "Wasserstein Distance: ([0-9\\.]+)"},
                {"Name": "train:fairness_loss", "Regex": "Fairness Loss: ([0-9\\.]+)"}
            ]
        )
        
        # Prepare input data configuration
        inputs = {"train": training_data_s3_uri}
        if validation_data_s3_uri:
            inputs["test"] = validation_data_s3_uri
        
        # Launch training
        pytorch_estimator.fit(inputs, job_name=job_name)
        
        self.logger.info(f"Training job launched: {job_name}")
        return job_name
    
    def monitor_training_job(
        self,
        job_name: str,
        poll_interval: int = 60
    ) -> TrainingJob:
        """
        Monitor a training job and return its status.
        
        Args:
            job_name: Name of the training job
            poll_interval: Polling interval in seconds
            
        Returns:
            Training job information
        """
        sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
        
        while True:
            try:
                response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
                
                status = response['TrainingJobStatus']
                self.logger.info(f"Training job {job_name} status: {status}")
                
                if status in ['Completed', 'Failed', 'Stopped']:
                    # Job finished, return final status
                    training_job = TrainingJob(
                        job_name=job_name,
                        status=status,
                        creation_time=response['CreationTime'],
                        training_start_time=response.get('TrainingStartTime'),
                        training_end_time=response.get('TrainingEndTime'),
                        model_artifacts=response.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
                        training_image=response['AlgorithmSpecification']['TrainingImage'],
                        instance_type=response['ResourceConfig']['InstanceType'],
                        instance_count=response['ResourceConfig']['InstanceCount'],
                        hyperparameters=response.get('HyperParameters', {}),
                        metrics=self._get_training_metrics(job_name)
                    )
                    return training_job
                
                # Wait before next poll
                time.sleep(poll_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring training job: {e}")
                time.sleep(poll_interval)
    
    def _get_training_metrics(self, job_name: str) -> Dict[str, Any]:
        """Get training metrics for a completed job."""
        cloudwatch = boto3.client('cloudwatch', region_name=self.region_name)
        
        try:
            # Get metrics from CloudWatch
            # This is a simplified version - in practice, you'd want to get actual metrics
            metrics = {
                'final_generator_loss': 0.0,
                'final_discriminator_loss': 0.0,
                'training_duration_minutes': 0
            }
            return metrics
        except Exception as e:
            self.logger.warning(f"Failed to get training metrics: {e}")
            return {}
    
    def list_training_jobs(
        self,
        status_filter: Optional[str] = None,
        max_results: int = 100
    ) -> List[TrainingJob]:
        """
        List training jobs.
        
        Args:
            status_filter: Filter by job status
            max_results: Maximum number of results
            
        Returns:
            List of training jobs
        """
        sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
        
        list_args = {
            'MaxResults': max_results,
            'SortBy': 'CreationTime',
            'SortOrder': 'Descending'
        }
        
        if status_filter:
            list_args['StatusEquals'] = status_filter
        
        response = sagemaker_client.list_training_jobs(**list_args)
        
        training_jobs = []
        for job_summary in response['TrainingJobSummaries']:
            job_name = job_summary['TrainingJobName']
            
            # Get detailed information
            detail_response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            
            training_job = TrainingJob(
                job_name=job_name,
                status=job_summary['TrainingJobStatus'],
                creation_time=job_summary['CreationTime'],
                training_start_time=job_summary.get('TrainingStartTime'),
                training_end_time=job_summary.get('TrainingEndTime'),
                model_artifacts=detail_response.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
                training_image=detail_response['AlgorithmSpecification']['TrainingImage'],
                instance_type=detail_response['ResourceConfig']['InstanceType'],
                instance_count=detail_response['ResourceConfig']['InstanceCount'],
                hyperparameters=detail_response.get('HyperParameters', {}),
                metrics={}
            )
            training_jobs.append(training_job)
        
        return training_jobs
    
    def download_model_artifacts(
        self,
        job_name: str,
        local_path: str
    ) -> bool:
        """
        Download model artifacts from a completed training job.
        
        Args:
            job_name: Name of the training job
            local_path: Local path to save artifacts
            
        Returns:
            True if download successful
        """
        sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
        
        try:
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            model_artifacts_uri = response.get('ModelArtifacts', {}).get('S3ModelArtifacts')
            
            if not model_artifacts_uri:
                self.logger.error(f"No model artifacts found for job: {job_name}")
                return False
            
            # Parse S3 URI
            if model_artifacts_uri.startswith('s3://'):
                bucket_name = model_artifacts_uri.split('/')[2]
                key = '/'.join(model_artifacts_uri.split('/')[3:])
                
                # Download from S3
                s3_client = boto3.client('s3', region_name=self.region_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3_client.download_file(bucket_name, key, local_path)
                
                self.logger.info(f"Model artifacts downloaded to: {local_path}")
                return True
            else:
                self.logger.error(f"Invalid S3 URI: {model_artifacts_uri}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to download model artifacts: {e}")
            return False


class ModelDeployment:
    """
    SageMaker model deployment and endpoint management.
    """
    
    def __init__(
        self,
        role: str,
        region_name: str = "us-west-2"
    ):
        self.role = role
        self.region_name = region_name
        self.session = sagemaker.Session(boto3.Session(region_name=region_name))
        self.logger = logging.getLogger(__name__)
    
    def create_model(
        self,
        model_artifacts_s3_uri: str,
        model_name: str,
        inference_image_uri: Optional[str] = None
    ) -> str:
        """
        Create a SageMaker model from training artifacts.
        
        Args:
            model_artifacts_s3_uri: S3 URI of model artifacts
            model_name: Name for the model
            inference_image_uri: Custom inference image URI
            
        Returns:
            Model name
        """
        if inference_image_uri is None:
            # Use default PyTorch inference image
            inference_image_uri = sagemaker.image_uris.retrieve(
                "pytorch",
                self.region_name,
                version="1.12",
                py_version="py38",
                instance_type="ml.m5.large",
                image_scope="inference"
            )
        
        pytorch_model = PyTorchModel(
            model_data=model_artifacts_s3_uri,
            role=self.role,
            image_uri=inference_image_uri,
            framework_version="1.12",
            py_version="py38",
            sagemaker_session=self.session,
            name=model_name
        )
        
        # Create the model
        pytorch_model.create(instance_type="ml.m5.large")
        
        self.logger.info(f"Model created: {model_name}")
        return model_name
    
    def deploy_endpoint(
        self,
        model_name: str,
        endpoint_name: str,
        instance_type: str = "ml.m5.large",
        instance_count: int = 1,
        auto_scaling_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Deploy a model to a SageMaker endpoint.
        
        Args:
            model_name: Name of the model to deploy
            endpoint_name: Name for the endpoint
            instance_type: EC2 instance type
            instance_count: Number of instances
            auto_scaling_config: Auto-scaling configuration
            
        Returns:
            Endpoint name
        """
        sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
        
        # Create endpoint configuration
        config_name = f"{endpoint_name}-config"
        
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': instance_count,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        # Create endpoint
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        # Wait for endpoint to be in service
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        self.logger.info(f"Endpoint deployed: {endpoint_name}")
        return endpoint_name
    
    def invoke_endpoint(
        self,
        endpoint_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke a SageMaker endpoint for inference.
        
        Args:
            endpoint_name: Name of the endpoint
            input_data: Input data for inference
            
        Returns:
            Inference results
        """
        runtime = boto3.client('sagemaker-runtime', region_name=self.region_name)
        
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(input_data)
            )
            
            result = json.loads(response['Body'].read().decode())
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to invoke endpoint: {e}")
            raise
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """
        Delete a SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to delete
            
        Returns:
            True if deletion successful
        """
        sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
        
        try:
            # Delete endpoint
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            # Delete endpoint configuration
            config_name = f"{endpoint_name}-config"
            try:
                sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
            except:
                pass  # Config might not exist or already deleted
            
            self.logger.info(f"Endpoint deleted: {endpoint_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete endpoint: {e}")
            return False
