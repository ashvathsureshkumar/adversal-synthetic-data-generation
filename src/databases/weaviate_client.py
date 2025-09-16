"""
Weaviate integration for vector storage and similarity search of synthetic data.

This module provides comprehensive Weaviate integration for storing data embeddings,
metadata, and performing similarity searches on synthetic and real data.
"""

import weaviate
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
from datetime import datetime
import uuid
from dataclasses import dataclass
import warnings

from .embedding_generator import EmbeddingGenerator


@dataclass
class DataRecord:
    """Data record for Weaviate storage."""
    id: str
    data_vector: List[float]
    metadata: Dict[str, Any]
    data_type: str  # "real" or "synthetic"
    generation_model: Optional[str] = None
    fairness_metrics: Optional[Dict[str, float]] = None
    privacy_metrics: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None


class WeaviateManager:
    """
    Manager for Weaviate vector database operations.
    
    Handles connection, schema management, and basic operations.
    """
    
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout_config: Tuple[int, int] = (10, 60),
        startup_period: int = 5
    ):
        self.weaviate_url = weaviate_url
        self.api_key = api_key
        self.timeout_config = timeout_config
        self.startup_period = startup_period
        
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        # Default schema configuration
        self.default_class_config = {
            "class": "SyntheticDataRecord",
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {
                    "name": "dataType",
                    "dataType": ["string"],
                    "description": "Type of data: real or synthetic"
                },
                {
                    "name": "generationModel",
                    "dataType": ["string"],
                    "description": "Model used for generation (if synthetic)"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When the data was created/generated"
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata about the data"
                },
                {
                    "name": "fairnessMetrics",
                    "dataType": ["object"],
                    "description": "Fairness evaluation metrics"
                },
                {
                    "name": "privacyMetrics",
                    "dataType": ["object"],
                    "description": "Privacy evaluation metrics"
                },
                {
                    "name": "originalDatasetId",
                    "dataType": ["string"],
                    "description": "ID of the original dataset this was derived from"
                },
                {
                    "name": "qualityScore",
                    "dataType": ["number"],
                    "description": "Overall quality score of the synthetic data"
                }
            ]
        }
    
    def connect(self) -> bool:
        """
        Establish connection to Weaviate.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)
                self.client = weaviate.Client(
                    url=self.weaviate_url,
                    auth_client_secret=auth_config,
                    timeout_config=self.timeout_config,
                    startup_period=self.startup_period
                )
            else:
                self.client = weaviate.Client(
                    url=self.weaviate_url,
                    timeout_config=self.timeout_config,
                    startup_period=self.startup_period
                )
            
            # Test connection
            if self.client.is_ready():
                self.logger.info("Successfully connected to Weaviate")
                return True
            else:
                self.logger.error("Weaviate is not ready")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Weaviate: {e}")
            return False
    
    def create_schema(self, class_config: Optional[Dict] = None) -> bool:
        """
        Create schema in Weaviate.
        
        Args:
            class_config: Custom class configuration (uses default if None)
            
        Returns:
            True if schema created successfully
        """
        if not self.client:
            self.logger.error("Not connected to Weaviate")
            return False
        
        config = class_config or self.default_class_config
        
        try:
            # Check if class already exists
            existing_schema = self.client.schema.get()
            existing_classes = [cls["class"] for cls in existing_schema.get("classes", [])]
            
            if config["class"] in existing_classes:
                self.logger.info(f"Class {config['class']} already exists")
                return True
            
            # Create the class
            self.client.schema.create_class(config)
            self.logger.info(f"Created class {config['class']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create schema: {e}")
            return False
    
    def delete_schema(self, class_name: str = "SyntheticDataRecord") -> bool:
        """
        Delete a class from Weaviate schema.
        
        Args:
            class_name: Name of the class to delete
            
        Returns:
            True if deletion successful
        """
        if not self.client:
            self.logger.error("Not connected to Weaviate")
            return False
        
        try:
            self.client.schema.delete_class(class_name)
            self.logger.info(f"Deleted class {class_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete class {class_name}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Weaviate connection.
        
        Returns:
            Health status information
        """
        health_info = {
            "connected": False,
            "ready": False,
            "live": False,
            "version": None,
            "error": None
        }
        
        try:
            if self.client is None:
                health_info["error"] = "No client connection established"
                return health_info
            
            health_info["connected"] = True
            health_info["ready"] = self.client.is_ready()
            health_info["live"] = self.client.is_live()
            
            # Get version info
            meta = self.client.get_meta()
            health_info["version"] = meta.get("version", "Unknown")
            
        except Exception as e:
            health_info["error"] = str(e)
        
        return health_info


class SyntheticDataVectorStore:
    """
    High-level interface for storing and querying synthetic data in Weaviate.
    """
    
    def __init__(
        self,
        weaviate_manager: WeaviateManager,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        class_name: str = "SyntheticDataRecord"
    ):
        self.weaviate_manager = weaviate_manager
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.class_name = class_name
        self.logger = logging.getLogger(__name__)
        
        # Ensure connection and schema
        if not self.weaviate_manager.client:
            self.weaviate_manager.connect()
        
        self.weaviate_manager.create_schema()
    
    def store_data_batch(
        self,
        data: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        data_type: str = "synthetic",
        generation_model: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Store a batch of data records in Weaviate.
        
        Args:
            data: Data array to store
            metadata_list: List of metadata dictionaries for each data point
            data_type: Type of data ("real" or "synthetic")
            generation_model: Model used for generation (if synthetic)
            batch_size: Batch size for insertion
            
        Returns:
            List of generated IDs for the stored records
        """
        if not self.weaviate_manager.client:
            raise RuntimeError("Not connected to Weaviate")
        
        if len(data) != len(metadata_list):
            raise ValueError("Data and metadata list must have same length")
        
        # Generate embeddings for the data
        embeddings = self.embedding_generator.generate_embeddings(data)
        
        stored_ids = []
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch_end = min(i + batch_size, len(data))
            batch_data = data[i:batch_end]
            batch_metadata = metadata_list[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            
            batch_ids = self._store_batch(
                batch_data,
                batch_embeddings,
                batch_metadata,
                data_type,
                generation_model
            )
            stored_ids.extend(batch_ids)
        
        self.logger.info(f"Stored {len(stored_ids)} records in Weaviate")
        return stored_ids
    
    def _store_batch(
        self,
        data_batch: np.ndarray,
        embeddings_batch: np.ndarray,
        metadata_batch: List[Dict[str, Any]],
        data_type: str,
        generation_model: Optional[str]
    ) -> List[str]:
        """Store a single batch of data."""
        client = self.weaviate_manager.client
        stored_ids = []
        
        try:
            with client.batch as batch:
                batch.batch_size = len(data_batch)
                
                for i, (data_point, embedding, metadata) in enumerate(
                    zip(data_batch, embeddings_batch, metadata_batch)
                ):
                    # Generate unique ID
                    record_id = str(uuid.uuid4())
                    
                    # Prepare properties
                    properties = {
                        "dataType": data_type,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": metadata,
                        "originalDatasetId": metadata.get("dataset_id", "unknown")
                    }
                    
                    if generation_model:
                        properties["generationModel"] = generation_model
                    
                    if "fairness_metrics" in metadata:
                        properties["fairnessMetrics"] = metadata["fairness_metrics"]
                    
                    if "privacy_metrics" in metadata:
                        properties["privacyMetrics"] = metadata["privacy_metrics"]
                    
                    if "quality_score" in metadata:
                        properties["qualityScore"] = metadata["quality_score"]
                    
                    # Add to batch
                    batch.add_data_object(
                        properties,
                        self.class_name,
                        uuid=record_id,
                        vector=embedding.tolist()
                    )
                    
                    stored_ids.append(record_id)
            
        except Exception as e:
            self.logger.error(f"Failed to store batch: {e}")
            # Return partial results if some succeeded
        
        return stored_ids
    
    def similarity_search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_certainty: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in Weaviate.
        
        Args:
            query_vector: Vector to search for
            limit: Maximum number of results
            filters: Optional filters for the search
            min_certainty: Minimum certainty score for results
            
        Returns:
            List of similar records with metadata
        """
        if not self.weaviate_manager.client:
            raise RuntimeError("Not connected to Weaviate")
        
        client = self.weaviate_manager.client
        
        # Convert query vector to list if needed
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        try:
            # Build the query
            query = (
                client.query
                .get(self.class_name, [
                    "dataType", "generationModel", "timestamp", "metadata",
                    "fairnessMetrics", "privacyMetrics", "qualityScore"
                ])
                .with_near_vector({
                    "vector": query_vector,
                    "certainty": min_certainty
                })
                .with_limit(limit)
                .with_additional(["certainty", "distance"])
            )
            
            # Add filters if provided
            if filters:
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            result = query.do()
            
            # Extract results
            if "data" in result and "Get" in result["data"]:
                return result["data"]["Get"][self.class_name]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build where filter for Weaviate query."""
        if not filters:
            return None
        
        conditions = []
        
        for key, value in filters.items():
            if key == "data_type":
                conditions.append({
                    "path": ["dataType"],
                    "operator": "Equal",
                    "valueString": value
                })
            elif key == "generation_model":
                conditions.append({
                    "path": ["generationModel"],
                    "operator": "Equal",
                    "valueString": value
                })
            elif key == "min_quality_score":
                conditions.append({
                    "path": ["qualityScore"],
                    "operator": "GreaterThanEqual",
                    "valueNumber": value
                })
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {
                "operator": "And",
                "operands": conditions
            }
        else:
            return None
    
    def find_similar_synthetic_data(
        self,
        real_data: np.ndarray,
        limit: int = 10,
        generation_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find synthetic data similar to given real data.
        
        Args:
            real_data: Real data to find matches for
            limit: Maximum number of results per query
            generation_model: Filter by specific generation model
            
        Returns:
            List of similar synthetic data records
        """
        # Generate embedding for the real data
        if real_data.ndim == 1:
            real_data = real_data.reshape(1, -1)
        
        query_embedding = self.embedding_generator.generate_embeddings(real_data)[0]
        
        # Set up filters
        filters = {"data_type": "synthetic"}
        if generation_model:
            filters["generation_model"] = generation_model
        
        return self.similarity_search(
            query_vector=query_embedding,
            limit=limit,
            filters=filters
        )
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored data.
        
        Returns:
            Dictionary with data statistics
        """
        if not self.weaviate_manager.client:
            raise RuntimeError("Not connected to Weaviate")
        
        client = self.weaviate_manager.client
        stats = {
            "total_records": 0,
            "real_data_count": 0,
            "synthetic_data_count": 0,
            "generation_models": {},
            "average_quality_score": 0.0
        }
        
        try:
            # Get total count
            result = client.query.aggregate(self.class_name).with_meta_count().do()
            if "data" in result and "Aggregate" in result["data"]:
                stats["total_records"] = result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
            
            # Get counts by data type
            for data_type in ["real", "synthetic"]:
                result = (
                    client.query.aggregate(self.class_name)
                    .with_meta_count()
                    .with_where({
                        "path": ["dataType"],
                        "operator": "Equal",
                        "valueString": data_type
                    })
                    .do()
                )
                
                if "data" in result and "Aggregate" in result["data"]:
                    count = result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
                    stats[f"{data_type}_data_count"] = count
            
            # Get generation model distribution (for synthetic data)
            result = (
                client.query.aggregate(self.class_name)
                .with_group_by_filter(["generationModel"])
                .with_meta_count()
                .with_where({
                    "path": ["dataType"],
                    "operator": "Equal",
                    "valueString": "synthetic"
                })
                .do()
            )
            
            if "data" in result and "Aggregate" in result["data"]:
                for group in result["data"]["Aggregate"][self.class_name]:
                    model = group["groupedBy"]["value"]
                    count = group["meta"]["count"]
                    stats["generation_models"][model] = count
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
        
        return stats
    
    def delete_records(
        self,
        filters: Dict[str, Any],
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Delete records matching the given filters.
        
        Args:
            filters: Filters to select records for deletion
            dry_run: If True, only count records that would be deleted
            
        Returns:
            Dictionary with deletion results
        """
        if not self.weaviate_manager.client:
            raise RuntimeError("Not connected to Weaviate")
        
        client = self.weaviate_manager.client
        
        try:
            # Build where filter
            where_filter = self._build_where_filter(filters)
            
            if dry_run:
                # Just count the records
                result = (
                    client.query.aggregate(self.class_name)
                    .with_meta_count()
                    .with_where(where_filter)
                    .do()
                )
                
                count = 0
                if "data" in result and "Aggregate" in result["data"]:
                    count = result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
                
                return {
                    "dry_run": True,
                    "records_to_delete": count,
                    "deleted": 0
                }
            else:
                # Actually delete the records
                result = client.batch.delete_objects(
                    class_name=self.class_name,
                    where=where_filter
                )
                
                return {
                    "dry_run": False,
                    "records_to_delete": result.get("matches", 0),
                    "deleted": result.get("successful", 0),
                    "failed": result.get("failed", 0)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to delete records: {e}")
            return {
                "dry_run": dry_run,
                "error": str(e),
                "deleted": 0
            }
    
    def export_data(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Export data from Weaviate to pandas DataFrame.
        
        Args:
            filters: Optional filters for data selection
            limit: Maximum number of records to export
            
        Returns:
            DataFrame with exported data
        """
        if not self.weaviate_manager.client:
            raise RuntimeError("Not connected to Weaviate")
        
        client = self.weaviate_manager.client
        
        try:
            # Build query
            query = client.query.get(self.class_name, [
                "dataType", "generationModel", "timestamp", "metadata",
                "fairnessMetrics", "privacyMetrics", "qualityScore"
            ])
            
            if filters:
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            if limit:
                query = query.with_limit(limit)
            
            result = query.do()
            
            # Convert to DataFrame
            if "data" in result and "Get" in result["data"]:
                records = result["data"]["Get"][self.class_name]
                return pd.DataFrame(records)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return pd.DataFrame()
