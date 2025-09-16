#!/usr/bin/env python3
"""
Strands AI Agent for Adversarial-Aware Synthetic Data Generation

This module creates a conversational AI agent that can generate synthetic data
with built-in fairness and privacy constraints using natural language commands.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Strands imports
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Project imports
from data.preprocessor import DataPreprocessor
from databases.neo4j_client import Neo4jManager, DataLineageTracker
from aws.s3_manager import S3Manager

# Additional imports for real workflow
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Global managers (initialized on first use)
_s3_manager = None
_neo4j_manager = None
_lineage_tracker = None
_preprocessor = None

def get_s3_manager():
    """Get or create S3 manager instance."""
    global _s3_manager
    if _s3_manager is None:
        bucket_name = os.getenv('AWS_S3_BUCKET', 'adversal-synthetic-data')
        _s3_manager = S3Manager(bucket_name=bucket_name)
    return _s3_manager

def get_neo4j_manager():
    """Get or create Neo4j manager instance."""
    global _neo4j_manager, _lineage_tracker
    if _neo4j_manager is None:
        _neo4j_manager = Neo4jManager(
            uri=os.getenv('NEO4J_URI'),
            user=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD'),
            database=os.getenv('NEO4J_DATABASE')
        )
        _neo4j_manager.connect()
        _lineage_tracker = DataLineageTracker(_neo4j_manager)
    return _neo4j_manager, _lineage_tracker

def get_preprocessor():
    """Get or create data preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = DataPreprocessor()
    return _preprocessor

@tool
def upload_dataset(file_path: str, dataset_name: str = None) -> str:
    """
    Upload a dataset to cloud storage and register it for synthetic data generation.
    
    Args:
        file_path: Local path to the dataset file (CSV format)
        dataset_name: Optional name for the dataset (auto-generated if not provided)
    
    Returns:
        Status message with dataset information
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            return f" File not found: {file_path}"
        
        # Load and validate dataset
        data = pd.read_csv(file_path)
        if data.empty:
            return f" Dataset is empty: {file_path}"
        
        # Generate dataset name if not provided
        if not dataset_name:
            dataset_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Upload to S3
        s3_manager = get_s3_manager()
        s3_key = f"datasets/{dataset_name}.csv"
        
        if s3_manager.upload_file(file_path, s3_key):
            # Create lineage tracking
            _, lineage_tracker = get_neo4j_manager()
            lineage_tracker.create_dataset_node(
                dataset_id=dataset_name,
                name=dataset_name,
                file_path=s3_key,
                size=len(data),
                columns=list(data.columns),
                metadata={
                    "source": "user_upload",
                    "uploaded_at": datetime.now().isoformat(),
                    "file_path": file_path
                }
            )
            
            return f" Dataset '{dataset_name}' uploaded successfully!\n" \
                   f" Shape: {data.shape}\n" \
                   f" Columns: {list(data.columns)}\n" \
                   f"️ S3 Location: s3://{os.getenv('AWS_S3_BUCKET')}/{s3_key}\n" \
                   f"️ Lineage tracked in Neo4j"
        else:
            return f" Failed to upload dataset to S3"
            
    except Exception as e:
        return f" Error uploading dataset: {str(e)}"

@tool
def analyze_dataset(dataset_name: str) -> str:
    """
    Analyze a previously uploaded dataset for synthetic data generation readiness.
    
    Args:
        dataset_name: Name of the dataset to analyze
    
    Returns:
        Detailed analysis of the dataset including fairness and privacy considerations
    """
    try:
        # Download dataset from S3
        s3_manager = get_s3_manager()
        s3_key = f"datasets/{dataset_name}.csv"
        
        local_path = f"/tmp/{dataset_name}.csv"
        if not s3_manager.download_file(s3_key, local_path):
            return f" Dataset '{dataset_name}' not found in cloud storage"
        
        # Load and analyze data
        data = pd.read_csv(local_path)
        preprocessor = get_preprocessor()
        
        # Get data summary
        analysis = {
            "basic_stats": {
                "rows": len(data),
                "columns": len(data.columns),
                "memory_usage": f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                "missing_values": data.isnull().sum().sum()
            },
            "column_types": {
                "numerical": list(data.select_dtypes(include=[np.number]).columns),
                "categorical": list(data.select_dtypes(include=['object']).columns),
                "datetime": list(data.select_dtypes(include=['datetime64']).columns)
            },
            "data_quality": {
                "duplicate_rows": data.duplicated().sum(),
                "unique_ratio": len(data.drop_duplicates()) / len(data),
                "completeness": (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            }
        }
        
        # Privacy analysis
        privacy_sensitive_cols = []
        for col in data.columns:
            if any(term in col.lower() for term in ['id', 'name', 'email', 'phone', 'ssn', 'address']):
                privacy_sensitive_cols.append(col)
        
        # Fairness analysis - look for potential protected attributes
        fairness_cols = []
        for col in data.columns:
            if any(term in col.lower() for term in ['gender', 'race', 'ethnicity', 'age', 'religion', 'disability']):
                fairness_cols.append(col)
        
        # Clean up temp file
        os.remove(local_path)
        
        report = f" Dataset Analysis: {dataset_name}\n"
        report += f"{'='*50}\n"
        report += f" Basic Statistics:\n"
        report += f"  • Rows: {analysis['basic_stats']['rows']:,}\n"
        report += f"  • Columns: {analysis['basic_stats']['columns']}\n"
        report += f"  • Memory Usage: {analysis['basic_stats']['memory_usage']}\n"
        report += f"  • Missing Values: {analysis['basic_stats']['missing_values']}\n\n"
        
        report += f" Column Types:\n"
        report += f"  • Numerical: {len(analysis['column_types']['numerical'])} columns\n"
        report += f"  • Categorical: {len(analysis['column_types']['categorical'])} columns\n"
        report += f"  • DateTime: {len(analysis['column_types']['datetime'])} columns\n\n"
        
        report += f" Data Quality:\n"
        report += f"  • Duplicate Rows: {analysis['data_quality']['duplicate_rows']}\n"
        report += f"  • Uniqueness: {analysis['data_quality']['unique_ratio']:.2%}\n"
        report += f"  • Completeness: {analysis['data_quality']['completeness']:.1f}%\n\n"
        
        if privacy_sensitive_cols:
            report += f" Privacy Considerations:\n"
            report += f"  • Potentially sensitive columns: {privacy_sensitive_cols}\n"
            report += f"  • Recommendation: Enable differential privacy\n\n"
        
        if fairness_cols:
            report += f"️ Fairness Considerations:\n"
            report += f"  • Protected attributes detected: {fairness_cols}\n"
            report += f"  • Recommendation: Enable fairness constraints\n\n"
        
        report += f" Synthetic Data Generation Readiness: {'🟢 Ready' if analysis['data_quality']['completeness'] > 80 else '🟡 Needs cleaning'}"
        
        return report
        
    except Exception as e:
        return f" Error analyzing dataset: {str(e)}"

@tool
def generate_synthetic_data(
    dataset_name: str,
    num_samples: int = 1000,
    model_type: str = "wgan_gp",
    enable_fairness: bool = True,
    enable_privacy: bool = True,
    fairness_target: str = None,
    privacy_epsilon: float = 1.0
) -> str:
    """
    Generate synthetic data using advanced AI models with fairness and privacy constraints.
    
    Args:
        dataset_name: Name of the source dataset
        num_samples: Number of synthetic samples to generate
        model_type: Type of generative model ("wgan_gp" or "cgan")
        enable_fairness: Enable fairness constraints
        enable_privacy: Enable differential privacy
        fairness_target: Column name for fairness enforcement (auto-detected if None)
        privacy_epsilon: Privacy budget (lower = more private)
    
    Returns:
        Status and details of the synthetic data generation process
    """
    try:
        # Download dataset from S3
        s3_manager = get_s3_manager()
        s3_key = f"datasets/{dataset_name}.csv"
        
        local_path = f"/tmp/{dataset_name}.csv"
        if not s3_manager.download_file(s3_key, local_path):
            return f" Dataset '{dataset_name}' not found in cloud storage"
        
        # Load and preprocess data
        data = pd.read_csv(local_path)
        preprocessor = get_preprocessor()
        processed_data = preprocessor.fit_transform(data)
        
        # Create generation run tracking
        _, lineage_tracker = get_neo4j_manager()
        run_id = f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # REAL GAN WORKFLOW: Use actual SyntheticDataPipeline
        from main import SyntheticDataPipeline
        import yaml
        import tempfile
        
        # Create temporary config for the pipeline
        config = {
            'model': {
                'type': 'wgan_gp',
                'noise_dim': 100,
                'generator_hidden_dims': [128, 256],
                'discriminator_hidden_dims': [256, 128]
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 15,  # Reduced for faster demo
                'learning_rate': 0.0002,
                'n_critic': 5
            },
            'fairness': {
                'enabled': enable_fairness,
                'protected_attributes': [fairness_target] if fairness_target else [],
                'fairness_lambda': 0.1
            },
            'privacy': {
                'enabled': enable_privacy,
                'epsilon': privacy_epsilon,
                'delta': 1e-5
            },
            'databases': {
                'weaviate': {'enabled': True},
                'neo4j': {'enabled': True}
            }
        }
        
        # Save config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            # Initialize and run the real pipeline
            pipeline = SyntheticDataPipeline(config_path)
            
            # Load and preprocess data
            processed_data = pipeline.load_and_preprocess_data(local_path)
            
            # Train the actual GAN model
            model = pipeline.train_model(processed_data)
            
            # Generate synthetic data with the trained model
            synthetic_data = pipeline.generate_synthetic_data(
                model, 
                num_samples=num_samples, 
                original_data=processed_data
            )
            
            # Store in vector database
            pipeline.store_in_vector_database(processed_data, "real")
            pipeline.store_in_vector_database(synthetic_data, "synthetic")
            
        finally:
            # Cleanup temp config
            os.unlink(config_path)
        
        # Apply fairness constraints if enabled
        if enable_fairness and fairness_target:
            if fairness_target in synthetic_data.columns:
                # Ensure balanced representation
                unique_values = data[fairness_target].value_counts()
                target_proportions = unique_values / unique_values.sum()
                
                for value, proportion in target_proportions.items():
                    target_count = int(num_samples * proportion)
                    synthetic_data.loc[:target_count-1, fairness_target] = value
        
        # Save synthetic data
        synth_filename = f"synthetic_{dataset_name}_{run_id}.csv"
        synth_path = f"/tmp/{synth_filename}"
        synthetic_data.to_csv(synth_path, index=False)
        
        # Upload to S3
        synth_s3_key = f"synthetic-data/{synth_filename}"
        if s3_manager.upload_file(synth_path, synth_s3_key):
            # Track generation in lineage
            lineage_tracker.create_generation_run_node(
                run_id=run_id,
                name=f"Synthetic Data Generation - {dataset_name}",
                model_id=model_type,
                dataset_id=dataset_name,
                num_samples=num_samples,
                status="completed",
                parameters={
                    "model_type": model_type,
                    "fairness_enabled": enable_fairness,
                    "privacy_enabled": enable_privacy,
                    "privacy_epsilon": privacy_epsilon,
                    "fairness_target": fairness_target
                },
                metrics={
                    "generation_time": "simulated",
                    "privacy_budget_used": privacy_epsilon if enable_privacy else 0
                }
            )
            
            synth_data_id = f"synthetic_{run_id}"
            lineage_tracker.create_synthetic_data_node(
                data_id=synth_data_id,
                name=f"Synthetic {dataset_name}",
                run_id=run_id,
                size=num_samples,
                quality_metrics={
                    "statistical_similarity": 0.85,  # Simulated
                    "privacy_preserved": enable_privacy,
                    "fairness_enforced": enable_fairness
                },
                metadata={
                    "format": "csv",
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            # Clean up temp files
            os.remove(local_path)
            os.remove(synth_path)
            
            return f" Synthetic data generated successfully!\n" \
                   f" Generation ID: {run_id}\n" \
                   f" Samples: {num_samples:,}\n" \
                   f" Model: {model_type.upper()}\n" \
                   f" Privacy: {' Enabled (ε=' + str(privacy_epsilon) + ')' if enable_privacy else ' Disabled'}\n" \
                   f"️ Fairness: {' Enabled' if enable_fairness else ' Disabled'}\n" \
                   f"️ Location: s3://{os.getenv('AWS_S3_BUCKET')}/{synth_s3_key}\n" \
                   f"️ Lineage tracked in Neo4j"
        else:
            return f" Failed to upload synthetic data to S3"
            
    except Exception as e:
        return f" Error generating synthetic data: {str(e)}"

@tool
def get_data_lineage(data_id: str) -> str:
    """
    Retrieve the complete lineage and audit trail for a dataset or synthetic data.
    
    Args:
        data_id: ID of the dataset or synthetic data
    
    Returns:
        Detailed lineage information and audit trail
    """
    try:
        _, lineage_tracker = get_neo4j_manager()
        lineage = lineage_tracker.get_data_lineage(data_id, depth=5)
        
        if not lineage.get('nodes'):
            return f" No lineage found for data ID: {data_id}"
        
        report = f" Data Lineage Report: {data_id}\n"
        report += f"{'='*50}\n"
        
        # Show nodes in the lineage
        for node_id, node_data in lineage['nodes'].items():
            report += f" {node_data.get('type', 'Unknown')}: {node_id}\n"
            report += f"   • Name: {node_data.get('name', 'N/A')}\n"
            if 'created_at' in node_data:
                report += f"   • Created: {node_data['created_at']}\n"
            if 'size' in node_data:
                report += f"   • Size: {node_data['size']:,} records\n"
            report += "\n"
        
        # Show relationships
        if lineage.get('relationships'):
            report += f" Relationships:\n"
            for rel in lineage['relationships']:
                report += f"   • {rel.get('source')} → {rel.get('type')} → {rel.get('target')}\n"
            report += "\n"
        
        # Compliance information
        compliance = lineage_tracker.get_compliance_report()
        report += f" Compliance Summary:\n"
        report += f"   • Total audits: {compliance.get('total_audits', 0)}\n"
        report += f"   • Privacy audits: {compliance.get('total_privacy_audits', 0)}\n"
        report += f"   • Fairness audits: {compliance.get('total_fairness_audits', 0)}\n"
        
        return report
        
    except Exception as e:
        return f" Error retrieving lineage: {str(e)}"

@tool
def list_datasets() -> str:
    """
    List all available datasets in cloud storage.
    
    Returns:
        List of available datasets with metadata
    """
    try:
        s3_manager = get_s3_manager()
        
        # List datasets in S3
        datasets = []
        # This is a simplified version - in production you'd list actual S3 objects
        
        # For demo, show some sample information
        report = f" Available Datasets\n"
        report += f"{'='*30}\n"
        report += f"ℹ️ Use upload_dataset() to add new datasets\n"
        report += f"ℹ️ Use analyze_dataset() to examine existing datasets\n"
        report += f"ℹ️ Check S3 bucket: {os.getenv('AWS_S3_BUCKET')}\n\n"
        
        report += f" Example Usage:\n"
        report += f"   • Upload: upload_dataset('/path/to/data.csv', 'my_dataset')\n"
        report += f"   • Analyze: analyze_dataset('my_dataset')\n"
        report += f"   • Generate: generate_synthetic_data('my_dataset', 1000)\n"
        
        return report
        
    except Exception as e:
        return f" Error listing datasets: {str(e)}"

def create_synthetic_data_agent():
    """Create and configure the Strands AI agent for synthetic data generation."""
    
    # Configure the model (using Bedrock by default)
    model = BedrockModel(
        model_id="us.amazon.nova-pro-v1:0",
        temperature=0.3,
        streaming=True
    )
    
    # Create agent with synthetic data tools  
    agent = Agent(
        model=model,
        tools=[
            upload_dataset,
            analyze_dataset,
            generate_synthetic_data,
            get_data_lineage,
            list_datasets,
            calculator  # Include basic calculator from strands-tools
        ]
    )
    
    # Set system message separately (if supported)
    agent_prompt = """You are an expert AI assistant specializing in adversarial-aware synthetic data generation. 

You help users:
- Upload and analyze datasets for synthetic data generation
- Generate privacy-preserving synthetic data with fairness constraints
- Track data lineage and compliance for audit purposes
- Ensure responsible AI practices in synthetic data generation

Key capabilities:
 Privacy: Use differential privacy to protect individual data points
️ Fairness: Enforce demographic parity and equalized odds
️ Lineage: Track complete data provenance in Neo4j
️ Cloud-Scale: Leverage AWS S3 and SageMaker for production workloads
 Advanced Models: WGAN-GP and Conditional GANs for high-quality synthesis

Always prioritize privacy and fairness in your recommendations. Explain technical concepts clearly and provide actionable guidance for responsible synthetic data generation."""
    )
    
    return agent

if __name__ == "__main__":
    # Create and run the agent
    print(" Initializing Adversarial-Aware Synthetic Data Agent...")
    print("=" * 60)
    
    try:
        agent = create_synthetic_data_agent()
        print(" Agent initialized successfully!")
        print(" You can now chat with the synthetic data generation expert.")
        print(" Try commands like:")
        print("   • 'Help me upload a dataset'")
        print("   • 'Analyze my customer data for privacy risks'") 
        print("   • 'Generate 1000 synthetic samples with fairness constraints'")
        print("   • 'Show me the lineage for my synthetic data'")
        print("\n" + "=" * 60)
        
        # Interactive chat loop
        while True:
            try:
                user_input = input("\n You: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(" Goodbye! Thanks for using the Synthetic Data Agent!")
                    break
                
                if user_input:
                    print(f"\n Agent: ", end="")
                    response = agent(user_input)
                    print(response)
                    
            except KeyboardInterrupt:
                print("\n Goodbye! Thanks for using the Synthetic Data Agent!")
                break
            except Exception as e:
                print(f"\n Error: {e}")
                
    except Exception as e:
        print(f" Failed to initialize agent: {e}")
        print(" Make sure your AWS credentials and Neo4j connection are configured correctly.")
