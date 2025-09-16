#!/usr/bin/env python3
"""
Test Weaviate cluster connection and setup schemas for synthetic data
"""

import sys
import os
import weaviate
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

def test_weaviate_connection():
    """Test connection to your new Weaviate cluster."""
    
    print("️ Testing Weaviate Connection")
    print("=" * 40)
    
    # Get credentials from environment
    weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    weaviate_key = os.getenv('WEAVIATE_API_KEY')
    
    print(f" Connecting to: {weaviate_url}")
    
    try:
        # Create client
        if weaviate_key:
            client = weaviate.Client(
                url=weaviate_url,
                auth_client_secret=weaviate.auth.AuthApiKey(api_key=weaviate_key),
                additional_headers={
                    "X-OpenAI-Api-Key": "dummy"  # Not needed for basic operations
                }
            )
        else:
            client = weaviate.Client(url=weaviate_url)
        
        # Test connection
        if client.is_ready():
            print(" Weaviate cluster is ready!")
            
            # Get cluster info
            meta = client.get_meta()
            print(f" Cluster info:")
            print(f"   • Version: {meta.get('version', 'Unknown')}")
            print(f"   • Modules: {list(meta.get('modules', {}).keys())}")
            
            # Create schemas for synthetic data
            print("\n️ Setting up schemas for synthetic data...")
            
            # Schema for synthetic datasets
            dataset_schema = {
                "class": "SyntheticDataset",
                "description": "Synthetic datasets with fairness and privacy metadata",
                "properties": [
                    {
                        "name": "name",
                        "dataType": ["text"],
                        "description": "Dataset name"
                    },
                    {
                        "name": "description", 
                        "dataType": ["text"],
                        "description": "Dataset description"
                    },
                    {
                        "name": "size",
                        "dataType": ["int"],
                        "description": "Number of records"
                    },
                    {
                        "name": "columns",
                        "dataType": ["text[]"],
                        "description": "Column names"
                    },
                    {
                        "name": "fairness_score",
                        "dataType": ["number"],
                        "description": "Fairness compliance score"
                    },
                    {
                        "name": "privacy_epsilon",
                        "dataType": ["number"], 
                        "description": "Differential privacy parameter"
                    },
                    {
                        "name": "quality_score",
                        "dataType": ["number"],
                        "description": "Statistical utility score"
                    },
                    {
                        "name": "bias_metrics",
                        "dataType": ["text"],
                        "description": "JSON of bias analysis results"
                    },
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "Creation timestamp"
                    }
                ],
                "vectorizer": "none"  # We'll add vectors manually
            }
            
            # Create schema if it doesn't exist
            existing_classes = [cls['class'] for cls in client.schema.get()['classes']]
            
            if "SyntheticDataset" not in existing_classes:
                client.schema.create_class(dataset_schema)
                print(" SyntheticDataset schema created")
            else:
                print("ℹ️ SyntheticDataset schema already exists")
            
            # Test adding a sample record
            print("\n Testing data insertion...")
            
            sample_data = {
                "name": f"test_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "description": "Test synthetic dataset for Weaviate integration",
                "size": 1000,
                "columns": ["age", "income", "credit_score", "outcome"],
                "fairness_score": 0.92,
                "privacy_epsilon": 1.0,
                "quality_score": 0.87,
                "bias_metrics": "{\"gender_disparity\": 0.02, \"racial_disparity\": 0.01}",
                "created_at": datetime.now().isoformat()
            }
            
            result = client.data_object.create(
                data_object=sample_data,
                class_name="SyntheticDataset"
            )
            
            print(f" Sample record inserted: {result}")
            
            # Test querying
            print("\n Testing query functionality...")
            
            query_result = client.query.get("SyntheticDataset", ["name", "fairness_score", "quality_score"]).with_limit(5).do()
            
            datasets = query_result.get("data", {}).get("Get", {}).get("SyntheticDataset", [])
            print(f" Found {len(datasets)} datasets in Weaviate")
            
            for dataset in datasets:
                print(f"   • {dataset['name']}: Fairness={dataset['fairness_score']}, Quality={dataset['quality_score']}")
            
            print("\n Weaviate integration successful!")
            print(" Ready for synthetic data embedding and search!")
            
        else:
            print(" Weaviate cluster not ready")
            return False
            
    except Exception as e:
        print(f" Connection failed: {e}")
        print(" Make sure WEAVIATE_URL and WEAVIATE_API_KEY are set correctly")
        return False
    
    return True

if __name__ == "__main__":
    success = test_weaviate_connection()
    
    if success:
        print("\n" + "="*50)
        print(" Next steps:")
        print("   • Your Weaviate cluster is ready!")
        print("   • Schemas created for synthetic data")
        print("   • Integration with your AI agent is complete")
        print("   • Ready for hackathon demo!")
    else:
        print("\n" + "="*50)
        print(" Troubleshooting:")
        print("   • Check your WEAVIATE_URL in .env")
        print("   • Verify WEAVIATE_API_KEY if using authentication")
        print("   • Ensure cluster is running and accessible")
