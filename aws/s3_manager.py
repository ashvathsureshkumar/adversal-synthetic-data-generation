"""
AWS S3 integration for data storage and management.

This module provides comprehensive S3 integration for storing datasets,
model checkpoints, synthetic data, and audit logs in the cloud.
"""

import boto3
import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError
import io
from dataclasses import dataclass


@dataclass
class S3Object:
    """Metadata for S3 objects."""
    key: str
    bucket: str
    size: int
    last_modified: datetime
    etag: str
    metadata: Dict[str, str]


class S3Manager:
    """
    Manager for AWS S3 operations.
    
    Handles uploading, downloading, and managing objects in S3 buckets.
    """
    
    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-west-2"
    ):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize S3 client
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # Use default credentials (environment variables, IAM role, etc.)
                self.s3_client = boto3.client('s3', region_name=region_name)
            
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            self.logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.logger.warning(f"Bucket {bucket_name} not found, will create if needed")
            else:
                self.logger.error(f"Failed to connect to S3: {e}")
                raise
    
    def create_bucket_if_not_exists(self) -> bool:
        """
        Create the S3 bucket if it doesn't exist.
        
        Returns:
            True if bucket exists or was created successfully
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    if self.region_name == 'us-east-1':
                        # us-east-1 doesn't need LocationConstraint
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region_name}
                        )
                    
                    self.logger.info(f"Created S3 bucket: {self.bucket_name}")
                    return True
                except ClientError as create_error:
                    self.logger.error(f"Failed to create bucket: {create_error}")
                    return False
            else:
                self.logger.error(f"Error checking bucket: {e}")
                return False
    
    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_path: Path to local file
            s3_key: S3 object key
            metadata: Object metadata
            tags: Object tags
            
        Returns:
            True if upload successful
        """
        try:
            extra_args = {}
            
            if metadata:
                extra_args['Metadata'] = metadata
            
            if tags:
                tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
                extra_args['Tagging'] = '&'.join([f"{k}={v}" for k, v in tags.items()])
            
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            self.logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload {local_path}: {e}")
            return False
    
    def download_file(
        self,
        s3_key: str,
        local_path: str
    ) -> bool:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local path to save file
            
        Returns:
            True if download successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_path
            )
            
            self.logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {s3_key}: {e}")
            return False
    
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        s3_key: str,
        format: str = "parquet",
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload a pandas DataFrame to S3.
        
        Args:
            df: DataFrame to upload
            s3_key: S3 object key
            format: File format ("parquet", "csv", "json")
            metadata: Object metadata
            
        Returns:
            True if upload successful
        """
        try:
            buffer = io.BytesIO()
            
            if format == "parquet":
                df.to_parquet(buffer, index=False)
                content_type = "application/octet-stream"
            elif format == "csv":
                df.to_csv(buffer, index=False)
                content_type = "text/csv"
            elif format == "json":
                df.to_json(buffer, orient="records", indent=2)
                content_type = "application/json"
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            buffer.seek(0)
            
            extra_args = {"ContentType": content_type}
            if metadata:
                extra_args["Metadata"] = metadata
            
            self.s3_client.upload_fileobj(
                buffer,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            self.logger.info(f"Uploaded DataFrame to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload DataFrame: {e}")
            return False
    
    def download_dataframe(
        self,
        s3_key: str,
        format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """
        Download a DataFrame from S3.
        
        Args:
            s3_key: S3 object key
            format: File format ("parquet", "csv", "json")
            
        Returns:
            DataFrame or None if failed
        """
        try:
            buffer = io.BytesIO()
            
            self.s3_client.download_fileobj(
                self.bucket_name,
                s3_key,
                buffer
            )
            
            buffer.seek(0)
            
            if format == "parquet":
                df = pd.read_parquet(buffer)
            elif format == "csv":
                df = pd.read_csv(buffer)
            elif format == "json":
                df = pd.read_json(buffer, orient="records")
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Downloaded DataFrame from s3://{self.bucket_name}/{s3_key}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download DataFrame: {e}")
            return None
    
    def upload_model(
        self,
        model,
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload a PyTorch model to S3.
        
        Args:
            model: PyTorch model or state dict
            s3_key: S3 object key
            metadata: Object metadata
            
        Returns:
            True if upload successful
        """
        try:
            buffer = io.BytesIO()
            
            # Save model state
            import torch
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), buffer)
            else:
                torch.save(model, buffer)
            
            buffer.seek(0)
            
            extra_args = {"ContentType": "application/octet-stream"}
            if metadata:
                extra_args["Metadata"] = metadata
            
            self.s3_client.upload_fileobj(
                buffer,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            self.logger.info(f"Uploaded model to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload model: {e}")
            return False
    
    def download_model(self, s3_key: str) -> Optional[Any]:
        """
        Download a PyTorch model from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Model state dict or None if failed
        """
        try:
            buffer = io.BytesIO()
            
            self.s3_client.download_fileobj(
                self.bucket_name,
                s3_key,
                buffer
            )
            
            buffer.seek(0)
            
            import torch
            model_state = torch.load(buffer, map_location='cpu')
            
            self.logger.info(f"Downloaded model from s3://{self.bucket_name}/{s3_key}")
            return model_state
            
        except Exception as e:
            self.logger.error(f"Failed to download model: {e}")
            return None
    
    def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000
    ) -> List[S3Object]:
        """
        List objects in the S3 bucket.
        
        Args:
            prefix: Object key prefix filter
            max_keys: Maximum number of objects to return
            
        Returns:
            List of S3Object metadata
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            for obj in response.get('Contents', []):
                # Get object metadata
                try:
                    head_response = self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    metadata = head_response.get('Metadata', {})
                except:
                    metadata = {}
                
                s3_obj = S3Object(
                    key=obj['Key'],
                    bucket=self.bucket_name,
                    size=obj['Size'],
                    last_modified=obj['LastModified'],
                    etag=obj['ETag'].strip('"'),
                    metadata=metadata
                )
                objects.append(s3_obj)
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to list objects: {e}")
            return []
    
    def delete_object(self, s3_key: str) -> bool:
        """
        Delete an object from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if deletion successful
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            self.logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete {s3_key}: {e}")
            return False
    
    def get_object_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an S3 object.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Object metadata or None if failed
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'etag': response['ETag'].strip('"'),
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for {s3_key}: {e}")
            return None


class DatasetManager:
    """
    High-level manager for dataset operations in S3.
    
    Provides organized storage and retrieval of datasets, synthetic data,
    and related metadata.
    """
    
    def __init__(self, s3_manager: S3Manager):
        self.s3_manager = s3_manager
        self.logger = logging.getLogger(__name__)
        
        # Standard folder structure
        self.folders = {
            'datasets': 'datasets/',
            'synthetic': 'synthetic-data/',
            'models': 'models/',
            'experiments': 'experiments/',
            'audits': 'audits/',
            'configs': 'configs/'
        }
    
    def upload_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        version: str = "v1",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload a dataset with versioning and metadata.
        
        Args:
            df: Dataset DataFrame
            dataset_name: Name of the dataset
            version: Version identifier
            description: Dataset description
            tags: Additional tags
            
        Returns:
            S3 key of uploaded dataset
        """
        # Create S3 key
        s3_key = f"{self.folders['datasets']}{dataset_name}/{version}/data.parquet"
        
        # Prepare metadata
        metadata = {
            'dataset_name': dataset_name,
            'version': version,
            'upload_timestamp': datetime.now().isoformat(),
            'rows': str(len(df)),
            'columns': str(len(df.columns)),
            'column_names': ','.join(df.columns)
        }
        
        if description:
            metadata['description'] = description
        
        # Upload dataset
        success = self.s3_manager.upload_dataframe(
            df, s3_key, format="parquet", metadata=metadata
        )
        
        if success:
            # Upload metadata file
            metadata_key = f"{self.folders['datasets']}{dataset_name}/{version}/metadata.json"
            metadata_buffer = io.StringIO(json.dumps(metadata, indent=2))
            
            try:
                self.s3_manager.s3_client.upload_fileobj(
                    io.BytesIO(metadata_buffer.getvalue().encode()),
                    self.s3_manager.bucket_name,
                    metadata_key,
                    ExtraArgs={"ContentType": "application/json"}
                )
            except Exception as e:
                self.logger.warning(f"Failed to upload metadata file: {e}")
            
            self.logger.info(f"Dataset uploaded: {dataset_name} {version}")
            return s3_key
        else:
            raise RuntimeError(f"Failed to upload dataset: {dataset_name}")
    
    def download_dataset(
        self,
        dataset_name: str,
        version: str = "latest"
    ) -> Optional[pd.DataFrame]:
        """
        Download a dataset.
        
        Args:
            dataset_name: Name of the dataset
            version: Version identifier ("latest" for most recent)
            
        Returns:
            Dataset DataFrame or None if failed
        """
        if version == "latest":
            # Find the latest version
            version = self.get_latest_version(dataset_name)
            if not version:
                self.logger.error(f"No versions found for dataset: {dataset_name}")
                return None
        
        s3_key = f"{self.folders['datasets']}{dataset_name}/{version}/data.parquet"
        return self.s3_manager.download_dataframe(s3_key, format="parquet")
    
    def get_latest_version(self, dataset_name: str) -> Optional[str]:
        """
        Get the latest version of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Latest version identifier or None
        """
        prefix = f"{self.folders['datasets']}{dataset_name}/"
        objects = self.s3_manager.list_objects(prefix=prefix)
        
        versions = set()
        for obj in objects:
            # Extract version from key like "datasets/name/v1/data.parquet"
            parts = obj.key.split('/')
            if len(parts) >= 4:
                versions.add(parts[2])
        
        if not versions:
            return None
        
        # Simple version sorting (assumes v1, v2, etc.)
        sorted_versions = sorted(versions, key=lambda x: int(x[1:]) if x.startswith('v') and x[1:].isdigit() else 0)
        return sorted_versions[-1] if sorted_versions else None
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets.
        
        Returns:
            List of dataset information
        """
        objects = self.s3_manager.list_objects(prefix=self.folders['datasets'])
        
        datasets = {}
        for obj in objects:
            parts = obj.key.split('/')
            if len(parts) >= 4 and parts[3] == 'data.parquet':
                dataset_name = parts[1]
                version = parts[2]
                
                if dataset_name not in datasets:
                    datasets[dataset_name] = {
                        'name': dataset_name,
                        'versions': [],
                        'latest_version': None,
                        'total_size': 0
                    }
                
                datasets[dataset_name]['versions'].append({
                    'version': version,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'metadata': obj.metadata
                })
                datasets[dataset_name]['total_size'] += obj.size
        
        # Determine latest version for each dataset
        for dataset_info in datasets.values():
            if dataset_info['versions']:
                latest = max(dataset_info['versions'], key=lambda x: x['last_modified'])
                dataset_info['latest_version'] = latest['version']
        
        return list(datasets.values())
    
    def upload_synthetic_data(
        self,
        df: pd.DataFrame,
        experiment_name: str,
        run_id: str,
        model_info: Dict[str, Any],
        quality_metrics: Optional[Dict[str, Any]] = None,
        fairness_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload synthetic data with comprehensive metadata.
        
        Args:
            df: Synthetic data DataFrame
            experiment_name: Name of the experiment
            run_id: Unique run identifier
            model_info: Information about the generating model
            quality_metrics: Data quality metrics
            fairness_metrics: Fairness evaluation metrics
            
        Returns:
            S3 key of uploaded data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{self.folders['synthetic']}{experiment_name}/{run_id}/synthetic_data_{timestamp}.parquet"
        
        # Prepare metadata
        metadata = {
            'experiment_name': experiment_name,
            'run_id': run_id,
            'generation_timestamp': datetime.now().isoformat(),
            'rows': str(len(df)),
            'columns': str(len(df.columns)),
            'model_type': model_info.get('type', 'unknown'),
            'model_version': model_info.get('version', 'unknown')
        }
        
        # Upload synthetic data
        success = self.s3_manager.upload_dataframe(
            df, s3_key, format="parquet", metadata=metadata
        )
        
        if success:
            # Upload comprehensive metadata
            full_metadata = {
                **metadata,
                'model_info': model_info,
                'quality_metrics': quality_metrics or {},
                'fairness_metrics': fairness_metrics or {}
            }
            
            metadata_key = f"{self.folders['synthetic']}{experiment_name}/{run_id}/metadata_{timestamp}.json"
            try:
                metadata_buffer = io.BytesIO(json.dumps(full_metadata, indent=2, default=str).encode())
                self.s3_manager.s3_client.upload_fileobj(
                    metadata_buffer,
                    self.s3_manager.bucket_name,
                    metadata_key,
                    ExtraArgs={"ContentType": "application/json"}
                )
            except Exception as e:
                self.logger.warning(f"Failed to upload synthetic data metadata: {e}")
            
            self.logger.info(f"Synthetic data uploaded: {experiment_name}/{run_id}")
            return s3_key
        else:
            raise RuntimeError(f"Failed to upload synthetic data: {experiment_name}/{run_id}")
    
    def upload_model_checkpoint(
        self,
        model,
        experiment_name: str,
        epoch: int,
        metrics: Dict[str, Any],
        config: Dict[str, Any]
    ) -> str:
        """
        Upload a model checkpoint with training information.
        
        Args:
            model: Model or model state dict
            experiment_name: Name of the experiment
            epoch: Training epoch
            metrics: Training metrics
            config: Model configuration
            
        Returns:
            S3 key of uploaded checkpoint
        """
        s3_key = f"{self.folders['models']}{experiment_name}/checkpoint_epoch_{epoch}.pt"
        
        metadata = {
            'experiment_name': experiment_name,
            'epoch': str(epoch),
            'save_timestamp': datetime.now().isoformat(),
            'model_type': config.get('model_type', 'unknown')
        }
        
        # Add key metrics to metadata
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                metadata[f"metric_{key}"] = str(value)
        
        success = self.s3_manager.upload_model(model, s3_key, metadata=metadata)
        
        if success:
            # Upload training info
            training_info = {
                'epoch': epoch,
                'metrics': metrics,
                'config': config,
                'save_timestamp': datetime.now().isoformat()
            }
            
            info_key = f"{self.folders['models']}{experiment_name}/training_info_epoch_{epoch}.json"
            try:
                info_buffer = io.BytesIO(json.dumps(training_info, indent=2, default=str).encode())
                self.s3_manager.s3_client.upload_fileobj(
                    info_buffer,
                    self.s3_manager.bucket_name,
                    info_key,
                    ExtraArgs={"ContentType": "application/json"}
                )
            except Exception as e:
                self.logger.warning(f"Failed to upload training info: {e}")
            
            return s3_key
        else:
            raise RuntimeError(f"Failed to upload model checkpoint: {experiment_name}")
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get storage usage summary.
        
        Returns:
            Storage summary information
        """
        summary = {
            'total_objects': 0,
            'total_size_bytes': 0,
            'folders': {}
        }
        
        all_objects = self.s3_manager.list_objects()
        
        for obj in all_objects:
            summary['total_objects'] += 1
            summary['total_size_bytes'] += obj.size
            
            # Categorize by folder
            folder = obj.key.split('/')[0] + '/'
            if folder not in summary['folders']:
                summary['folders'][folder] = {
                    'objects': 0,
                    'size_bytes': 0
                }
            
            summary['folders'][folder]['objects'] += 1
            summary['folders'][folder]['size_bytes'] += obj.size
        
        # Convert to human-readable sizes
        summary['total_size_mb'] = summary['total_size_bytes'] / (1024 * 1024)
        for folder_info in summary['folders'].values():
            folder_info['size_mb'] = folder_info['size_bytes'] / (1024 * 1024)
        
        return summary
