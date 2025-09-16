#!/usr/bin/env python3
"""
Verify that data is actually stored in cloud services
"""

import boto3
import os
from dotenv import load_dotenv
load_dotenv()

def verify_cloud_storage():
    """Verify data is actually in cloud storage."""
    
    print("🔍 Verifying Real Cloud Storage")
    print("=" * 40)
    
    # Check S3
    print("1. Checking AWS S3...")
    try:
        s3_client = boto3.client('s3')
        bucket_name = os.getenv('AWS_S3_BUCKET', 'adversal-synthetic-data')
        
        # List objects
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' in response:
            print(f"   ✅ Found {len(response['Contents'])} objects in S3:")
            
            for obj in response['Contents'][:10]:  # Show first 10
                size_mb = obj['Size'] / (1024*1024)
                print(f"      📄 {obj['Key']} ({size_mb:.2f} MB)")
                
            # Look for our synthetic datasets
            synthetic_files = [obj for obj in response['Contents'] if 'synthetic-datasets' in obj['Key']]
            demo_files = [obj for obj in response['Contents'] if 'demo-datasets' in obj['Key']]
            
            print(f"   📊 Synthetic datasets: {len(synthetic_files)}")
            print(f"   🧪 Demo datasets: {len(demo_files)}")
            
        else:
            print("   ⚠️ No objects found in S3 bucket")
            
    except Exception as e:
        print(f"   ❌ S3 check failed: {e}")
    
    # Check recent integration
    print("\n2. Checking recent integration test...")
    try:
        integration_objects = [obj for obj in response['Contents'] if 'real_integration' in obj['Key']]
        if integration_objects:
            print(f"   ✅ Found {len(integration_objects)} integration test files:")
            for obj in integration_objects:
                print(f"      📁 {obj['Key']}")
        else:
            print("   ⚠️ No integration test files found")
    except:
        print("   ⚠️ Could not check integration files")

if __name__ == "__main__":
    verify_cloud_storage()
