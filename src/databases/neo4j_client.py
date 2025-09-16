"""
Neo4j integration for data lineage and audit tracking.

This module provides comprehensive Neo4j integration for tracking data lineage,
model provenance, fairness audits, and privacy compliance in synthetic data generation.
"""

from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from enum import Enum


class NodeType(Enum):
    """Enumeration of node types in the lineage graph."""
    DATASET = "Dataset"
    MODEL = "Model"
    GENERATION_RUN = "GenerationRun"
    SYNTHETIC_DATA = "SyntheticData"
    FAIRNESS_AUDIT = "FairnessAudit"
    PRIVACY_AUDIT = "PrivacyAudit"
    USER = "User"
    EXPERIMENT = "Experiment"


class RelationshipType(Enum):
    """Enumeration of relationship types in the lineage graph."""
    TRAINED_ON = "TRAINED_ON"
    GENERATED_BY = "GENERATED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    AUDITED_BY = "AUDITED_BY"
    CREATED_BY = "CREATED_BY"
    USED_IN = "USED_IN"
    EVALUATED_WITH = "EVALUATED_WITH"
    INFLUENCED_BY = "INFLUENCED_BY"


@dataclass
class LineageNode:
    """Base class for lineage graph nodes."""
    id: str
    node_type: NodeType
    name: str
    created_at: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "created_at": self.created_at,
            **self.properties
        }


@dataclass
class LineageRelationship:
    """Base class for lineage graph relationships."""
    from_node_id: str
    to_node_id: str
    relationship_type: RelationshipType
    created_at: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "relationship_type": self.relationship_type.value,
            "created_at": self.created_at,
            **self.properties
        }


class Neo4jManager:
    """
    Manager for Neo4j database operations.
    
    Handles connection, schema management, and basic graph operations.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "synthetic_data"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        
        self.driver = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        Establish connection to Neo4j.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.logger.info("Successfully connected to Neo4j")
                    return True
                else:
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")
    
    def create_constraints(self) -> bool:
        """
        Create database constraints and indexes.
        
        Returns:
            True if constraints created successfully
        """
        if not self.driver:
            self.logger.error("Not connected to Neo4j")
            return False
        
        constraints = [
            # Unique constraints for node IDs
            "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Dataset) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT model_id_unique IF NOT EXISTS FOR (n:Model) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT run_id_unique IF NOT EXISTS FOR (n:GenerationRun) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT data_id_unique IF NOT EXISTS FOR (n:SyntheticData) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT audit_id_unique IF NOT EXISTS FOR (n:FairnessAudit) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT privacy_audit_id_unique IF NOT EXISTS FOR (n:PrivacyAudit) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (n:User) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT experiment_id_unique IF NOT EXISTS FOR (n:Experiment) REQUIRE n.id IS UNIQUE",
            
            # Indexes for common queries
            "CREATE INDEX node_created_at IF NOT EXISTS FOR (n:Dataset) ON (n.created_at)",
            "CREATE INDEX model_type IF NOT EXISTS FOR (n:Model) ON (n.model_type)",
            "CREATE INDEX run_status IF NOT EXISTS FOR (n:GenerationRun) ON (n.status)"
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # Constraint might already exist
                        self.logger.debug(f"Constraint creation result: {e}")
            
            self.logger.info("Database constraints and indexes created")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create constraints: {e}")
            return False
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j")
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Neo4j connection.
        
        Returns:
            Health status information
        """
        health_info = {
            "connected": False,
            "database_exists": False,
            "version": None,
            "node_count": 0,
            "relationship_count": 0,
            "error": None
        }
        
        try:
            if self.driver is None:
                health_info["error"] = "No driver connection established"
                return health_info
            
            with self.driver.session(database=self.database) as session:
                # Test basic connectivity
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    health_info["connected"] = True
                    health_info["database_exists"] = True
                
                # Get version
                version_result = session.run("CALL dbms.components() YIELD versions")
                versions = version_result.single()["versions"]
                health_info["version"] = versions[0] if versions else "Unknown"
                
                # Get node count
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                health_info["node_count"] = node_result.single()["count"]
                
                # Get relationship count
                rel_result = session.run("MATCH ()-[r]-() RETURN count(r) as count")
                health_info["relationship_count"] = rel_result.single()["count"]
                
        except Exception as e:
            health_info["error"] = str(e)
        
        return health_info


class DataLineageTracker:
    """
    High-level interface for tracking data lineage and audit trails.
    """
    
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager
        self.logger = logging.getLogger(__name__)
        
        # Ensure connection and constraints
        if not self.neo4j_manager.driver:
            self.neo4j_manager.connect()
        
        self.neo4j_manager.create_constraints()
    
    def create_dataset_node(
        self,
        dataset_id: str,
        name: str,
        file_path: Optional[str] = None,
        size: Optional[int] = None,
        columns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a dataset node in the lineage graph.
        
        Args:
            dataset_id: Unique identifier for the dataset
            name: Human-readable name
            file_path: Path to the dataset file
            size: Number of records in the dataset
            columns: List of column names
            metadata: Additional metadata
            
        Returns:
            Created node ID
        """
        node = LineageNode(
            id=dataset_id,
            node_type=NodeType.DATASET,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            properties={
                "file_path": file_path,
                "size": size,
                "columns": columns or [],
                "metadata": json.dumps(metadata or {})
            }
        )
        
        query = """
        CREATE (d:Dataset $properties)
        RETURN d.id as id
        """
        
        result = self.neo4j_manager.execute_query(query, {"properties": node.to_dict()})
        self.logger.info(f"Created dataset node: {dataset_id}")
        return result[0]["id"]
    
    def create_model_node(
        self,
        model_id: str,
        name: str,
        model_type: str,
        architecture: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a model node in the lineage graph.
        
        Args:
            model_id: Unique identifier for the model
            name: Human-readable name
            model_type: Type of model (e.g., "WGAN-GP", "cGAN")
            architecture: Model architecture details
            hyperparameters: Training hyperparameters
            training_config: Training configuration
            metadata: Additional metadata
            
        Returns:
            Created node ID
        """
        node = LineageNode(
            id=model_id,
            node_type=NodeType.MODEL,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            properties={
                "model_type": model_type,
                "architecture": json.dumps(architecture or {}),
                "hyperparameters": json.dumps(hyperparameters or {}),
                "training_config": json.dumps(training_config or {}),
                "metadata": json.dumps(metadata or {})
            }
        )
        
        query = """
        CREATE (m:Model $properties)
        RETURN m.id as id
        """
        
        result = self.neo4j_manager.execute_query(query, {"properties": node.to_dict()})
        self.logger.info(f"Created model node: {model_id}")
        return result[0]["id"]
    
    def create_generation_run_node(
        self,
        run_id: str,
        name: str,
        model_id: str,
        dataset_id: str,
        num_samples: int,
        status: str = "running",
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a generation run node in the lineage graph.
        
        Args:
            run_id: Unique identifier for the run
            name: Human-readable name
            model_id: ID of the model used
            dataset_id: ID of the training dataset
            num_samples: Number of synthetic samples generated
            status: Status of the run ("running", "completed", "failed")
            parameters: Generation parameters
            metrics: Performance metrics
            
        Returns:
            Created node ID
        """
        node = LineageNode(
            id=run_id,
            node_type=NodeType.GENERATION_RUN,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            properties={
                "num_samples": num_samples,
                "status": status,
                "parameters": json.dumps(parameters or {}),
                "metrics": json.dumps(metrics or {})
            }
        )
        
        # Create the node and relationships
        query = """
        CREATE (r:GenerationRun $properties)
        WITH r
        MATCH (m:Model {id: $model_id})
        MATCH (d:Dataset {id: $dataset_id})
        CREATE (r)-[:USED_MODEL]->(m)
        CREATE (r)-[:USED_DATASET]->(d)
        RETURN r.id as id
        """
        
        result = self.neo4j_manager.execute_query(
            query,
            {
                "properties": node.to_dict(),
                "model_id": model_id,
                "dataset_id": dataset_id
            }
        )
        
        self.logger.info(f"Created generation run node: {run_id}")
        return result[0]["id"]
    
    def create_synthetic_data_node(
        self,
        data_id: str,
        name: str,
        run_id: str,
        size: int,
        file_path: Optional[str] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a synthetic data node in the lineage graph.
        
        Args:
            data_id: Unique identifier for the synthetic data
            name: Human-readable name
            run_id: ID of the generation run that created this data
            size: Number of synthetic records
            file_path: Path where the data is stored
            quality_metrics: Data quality metrics
            metadata: Additional metadata
            
        Returns:
            Created node ID
        """
        node = LineageNode(
            id=data_id,
            node_type=NodeType.SYNTHETIC_DATA,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            properties={
                "size": size,
                "file_path": file_path,
                "quality_metrics": json.dumps(quality_metrics or {}),
                "metadata": json.dumps(metadata or {})
            }
        )
        
        # Create the node and relationship to generation run
        query = """
        CREATE (s:SyntheticData $properties)
        WITH s
        MATCH (r:GenerationRun {id: $run_id})
        CREATE (s)-[:GENERATED_BY]->(r)
        RETURN s.id as id
        """
        
        result = self.neo4j_manager.execute_query(
            query,
            {"properties": node.to_dict(), "run_id": run_id}
        )
        
        self.logger.info(f"Created synthetic data node: {data_id}")
        return result[0]["id"]
    
    def create_fairness_audit_node(
        self,
        audit_id: str,
        data_id: str,
        auditor: str,
        fairness_metrics: Dict[str, Any],
        protected_attributes: List[str],
        audit_results: Dict[str, Any],
        passed: bool
    ) -> str:
        """
        Create a fairness audit node in the lineage graph.
        
        Args:
            audit_id: Unique identifier for the audit
            data_id: ID of the data being audited
            auditor: Name/ID of the auditor
            fairness_metrics: Computed fairness metrics
            protected_attributes: List of protected attributes evaluated
            audit_results: Detailed audit results
            passed: Whether the audit passed
            
        Returns:
            Created node ID
        """
        node = LineageNode(
            id=audit_id,
            node_type=NodeType.FAIRNESS_AUDIT,
            name=f"Fairness Audit - {audit_id}",
            created_at=datetime.now(timezone.utc).isoformat(),
            properties={
                "auditor": auditor,
                "fairness_metrics": json.dumps(fairness_metrics),
                "protected_attributes": protected_attributes,
                "audit_results": json.dumps(audit_results),
                "passed": passed
            }
        )
        
        # Create the node and relationship to data
        query = """
        CREATE (a:FairnessAudit $properties)
        WITH a
        MATCH (d:SyntheticData {id: $data_id})
        CREATE (a)-[:AUDITED]->(d)
        RETURN a.id as id
        """
        
        result = self.neo4j_manager.execute_query(
            query,
            {"properties": node.to_dict(), "data_id": data_id}
        )
        
        self.logger.info(f"Created fairness audit node: {audit_id}")
        return result[0]["id"]
    
    def create_privacy_audit_node(
        self,
        audit_id: str,
        data_id: str,
        auditor: str,
        privacy_metrics: Dict[str, Any],
        privacy_parameters: Dict[str, Any],
        audit_results: Dict[str, Any],
        risk_score: float
    ) -> str:
        """
        Create a privacy audit node in the lineage graph.
        
        Args:
            audit_id: Unique identifier for the audit
            data_id: ID of the data being audited
            auditor: Name/ID of the auditor
            privacy_metrics: Computed privacy metrics
            privacy_parameters: Privacy parameters used
            audit_results: Detailed audit results
            risk_score: Overall privacy risk score
            
        Returns:
            Created node ID
        """
        node = LineageNode(
            id=audit_id,
            node_type=NodeType.PRIVACY_AUDIT,
            name=f"Privacy Audit - {audit_id}",
            created_at=datetime.now(timezone.utc).isoformat(),
            properties={
                "auditor": auditor,
                "privacy_metrics": json.dumps(privacy_metrics),
                "privacy_parameters": json.dumps(privacy_parameters),
                "audit_results": json.dumps(audit_results),
                "risk_score": risk_score
            }
        )
        
        # Create the node and relationship to data
        query = """
        CREATE (a:PrivacyAudit $properties)
        WITH a
        MATCH (d:SyntheticData {id: $data_id})
        CREATE (a)-[:AUDITED]->(d)
        RETURN a.id as id
        """
        
        result = self.neo4j_manager.execute_query(
            query,
            {"properties": node.to_dict(), "data_id": data_id}
        )
        
        self.logger.info(f"Created privacy audit node: {audit_id}")
        return result[0]["id"]
    
    def get_data_lineage(
        self,
        node_id: str,
        depth: int = 3,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        Get lineage information for a specific node.
        
        Args:
            node_id: ID of the node to trace lineage for
            depth: Maximum depth to traverse
            direction: Direction to traverse ("incoming", "outgoing", "both")
            
        Returns:
            Lineage graph structure
        """
        if direction == "incoming":
            relationship_pattern = "<-[r]-(n)"
        elif direction == "outgoing":
            relationship_pattern = "-[r]->(n)"
        else:  # both
            relationship_pattern = "-[r]-(n)"
        
        query = f"""
        MATCH path = (start {{id: $node_id}}){relationship_pattern}
        WITH path, relationships(path) as rels, nodes(path) as nodes
        WHERE length(path) <= $depth
        RETURN 
            [node in nodes | {{
                id: node.id, 
                node_type: node.node_type, 
                name: node.name,
                created_at: node.created_at,
                properties: properties(node)
            }}] as nodes,
            [rel in rels | {{
                type: type(rel),
                properties: properties(rel)
            }}] as relationships
        """
        
        result = self.neo4j_manager.execute_query(
            query,
            {"node_id": node_id, "depth": depth}
        )
        
        # Organize results into a graph structure
        lineage_graph = {
            "center_node_id": node_id,
            "nodes": {},
            "relationships": []
        }
        
        for record in result:
            nodes = record["nodes"]
            relationships = record["relationships"]
            
            # Add nodes
            for node in nodes:
                node_id_key = node["id"]
                if node_id_key not in lineage_graph["nodes"]:
                    lineage_graph["nodes"][node_id_key] = node
            
            # Add relationships
            for i, rel in enumerate(relationships):
                if i < len(nodes) - 1:
                    lineage_graph["relationships"].append({
                        "from": nodes[i]["id"],
                        "to": nodes[i + 1]["id"],
                        "type": rel["type"],
                        "properties": rel["properties"]
                    })
        
        return lineage_graph
    
    def get_audit_history(
        self,
        data_id: str,
        audit_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit history for a specific data node.
        
        Args:
            data_id: ID of the data node
            audit_type: Type of audit ("fairness", "privacy", or None for all)
            
        Returns:
            List of audit records
        """
        if audit_type == "fairness":
            audit_label = "FairnessAudit"
        elif audit_type == "privacy":
            audit_label = "PrivacyAudit"
        else:
            audit_label = "FairnessAudit|PrivacyAudit"
        
        query = f"""
        MATCH (a:{audit_label})-[:AUDITED]->(d:SyntheticData {{id: $data_id}})
        RETURN a.id as audit_id, a.node_type as audit_type, a.created_at as created_at,
               properties(a) as properties
        ORDER BY a.created_at DESC
        """
        
        return self.neo4j_manager.execute_query(query, {"data_id": data_id})
    
    def find_similar_datasets(
        self,
        dataset_id: str,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find datasets similar to the given dataset.
        
        Args:
            dataset_id: ID of the reference dataset
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar datasets
        """
        # This is a simplified similarity based on column overlap
        # In practice, you might want more sophisticated similarity metrics
        
        query = """
        MATCH (d1:Dataset {id: $dataset_id})
        MATCH (d2:Dataset)
        WHERE d1 <> d2 AND d1.columns IS NOT NULL AND d2.columns IS NOT NULL
        WITH d1, d2, 
             size([col IN d1.columns WHERE col IN d2.columns]) as common_cols,
             size(d1.columns + [col IN d2.columns WHERE NOT col IN d1.columns]) as total_cols
        WITH d1, d2, common_cols, total_cols,
             toFloat(common_cols) / toFloat(total_cols) as similarity
        WHERE similarity >= $threshold
        RETURN d2.id as dataset_id, d2.name as name, similarity
        ORDER BY similarity DESC
        """
        
        return self.neo4j_manager.execute_query(
            query,
            {"dataset_id": dataset_id, "threshold": similarity_threshold}
        )
    
    def get_model_performance_comparison(self) -> List[Dict[str, Any]]:
        """
        Get performance comparison across different models.
        
        Returns:
            List of model performance data
        """
        query = """
        MATCH (m:Model)<-[:USED_MODEL]-(r:GenerationRun)-[:GENERATED_BY]-(s:SyntheticData)
        OPTIONAL MATCH (fa:FairnessAudit)-[:AUDITED]->(s)
        OPTIONAL MATCH (pa:PrivacyAudit)-[:AUDITED]->(s)
        RETURN 
            m.id as model_id,
            m.name as model_name,
            m.model_type as model_type,
            count(s) as synthetic_datasets_generated,
            avg(toFloat(s.size)) as avg_dataset_size,
            avg(CASE WHEN fa.passed THEN 1.0 ELSE 0.0 END) as fairness_pass_rate,
            avg(pa.risk_score) as avg_privacy_risk
        ORDER BY fairness_pass_rate DESC, avg_privacy_risk ASC
        """
        
        return self.neo4j_manager.execute_query(query)
    
    def update_generation_run_status(
        self,
        run_id: str,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update the status of a generation run.
        
        Args:
            run_id: ID of the generation run
            status: New status ("completed", "failed", "running")
            metrics: Updated metrics (if any)
            error_message: Error message (if failed)
            
        Returns:
            True if update successful
        """
        properties_to_set = ["status = $status"]
        parameters = {"run_id": run_id, "status": status}
        
        if metrics:
            properties_to_set.append("metrics = $metrics")
            parameters["metrics"] = json.dumps(metrics)
        
        if error_message:
            properties_to_set.append("error_message = $error_message")
            parameters["error_message"] = error_message
        
        query = f"""
        MATCH (r:GenerationRun {{id: $run_id}})
        SET {', '.join(properties_to_set)}
        RETURN r.id as id
        """
        
        try:
            result = self.neo4j_manager.execute_query(query, parameters)
            return len(result) > 0
        except Exception as e:
            self.logger.error(f"Failed to update generation run status: {e}")
            return False
    
    def get_compliance_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a compliance report for audits within a date range.
        
        Args:
            start_date: Start date in ISO format (optional)
            end_date: End date in ISO format (optional)
            
        Returns:
            Compliance report
        """
        date_filter = ""
        parameters = {}
        
        if start_date:
            date_filter += " AND a.created_at >= $start_date"
            parameters["start_date"] = start_date
        
        if end_date:
            date_filter += " AND a.created_at <= $end_date"
            parameters["end_date"] = end_date
        
        query = f"""
        MATCH (a:FairnessAudit)-[:AUDITED]->(s:SyntheticData)
        WHERE true {date_filter}
        WITH count(a) as total_fairness_audits, 
             sum(CASE WHEN a.passed THEN 1 ELSE 0 END) as passed_fairness_audits
        
        MATCH (pa:PrivacyAudit)-[:AUDITED]->(s:SyntheticData)
        WHERE true {date_filter}
        WITH total_fairness_audits, passed_fairness_audits,
             count(pa) as total_privacy_audits,
             avg(pa.risk_score) as avg_privacy_risk,
             sum(CASE WHEN pa.risk_score < 0.3 THEN 1 ELSE 0 END) as low_risk_privacy_audits
        
        RETURN 
            total_fairness_audits,
            passed_fairness_audits,
            toFloat(passed_fairness_audits) / toFloat(total_fairness_audits) as fairness_pass_rate,
            total_privacy_audits,
            avg_privacy_risk,
            low_risk_privacy_audits,
            toFloat(low_risk_privacy_audits) / toFloat(total_privacy_audits) as low_risk_rate
        """
        
        result = self.neo4j_manager.execute_query(query, parameters)
        
        if result:
            return result[0]
        else:
            return {
                "total_fairness_audits": 0,
                "passed_fairness_audits": 0,
                "fairness_pass_rate": 0.0,
                "total_privacy_audits": 0,
                "avg_privacy_risk": 0.0,
                "low_risk_privacy_audits": 0,
                "low_risk_rate": 0.0
            }
    
    def export_lineage_to_json(
        self,
        output_file: str,
        node_types: Optional[List[str]] = None
    ) -> bool:
        """
        Export the entire lineage graph to JSON format.
        
        Args:
            output_file: Path to output JSON file
            node_types: List of node types to include (None for all)
            
        Returns:
            True if export successful
        """
        try:
            # Build node type filter
            if node_types:
                node_filter = f"WHERE n.node_type IN {node_types}"
            else:
                node_filter = ""
            
            # Get all nodes
            nodes_query = f"""
            MATCH (n)
            {node_filter}
            RETURN collect({{
                id: n.id,
                node_type: n.node_type,
                name: n.name,
                created_at: n.created_at,
                properties: properties(n)
            }}) as nodes
            """
            
            # Get all relationships
            relationships_query = """
            MATCH (n1)-[r]->(n2)
            RETURN collect({
                from: n1.id,
                to: n2.id,
                type: type(r),
                properties: properties(r)
            }) as relationships
            """
            
            nodes_result = self.neo4j_manager.execute_query(nodes_query)
            relationships_result = self.neo4j_manager.execute_query(relationships_query)
            
            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "nodes": nodes_result[0]["nodes"] if nodes_result else [],
                "relationships": relationships_result[0]["relationships"] if relationships_result else []
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Lineage graph exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export lineage graph: {e}")
            return False
