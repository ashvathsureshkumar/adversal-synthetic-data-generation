#!/usr/bin/env python3
"""
Helper script to gather AWS configuration information for the project.
This script will help you identify your AWS settings and test connectivity.
"""

import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound

def get_aws_account_id():
    """Get the AWS account ID."""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        return identity.get('Account')
    except Exception as e:
        return f"Error: {e}"

def get_current_region():
    """Get the current AWS region."""
    try:
        session = boto3.Session()
        return session.region_name or "us-east-1"
    except Exception as e:
        return f"Error: {e}"

def check_s3_bucket_region(bucket_name):
    """Check the region of the S3 bucket."""
    try:
        s3 = boto3.client('s3')
        response = s3.get_bucket_location(Bucket=bucket_name)
        region = response.get('LocationConstraint')
        return region if region else 'us-east-1'  # us-east-1 returns None
    except Exception as e:
        return f"Error: {e}"

def check_sagemaker_role():
    """Check if SageMaker execution role exists."""
    try:
        iam = boto3.client('iam')
        roles = iam.list_roles()
        sagemaker_roles = [role for role in roles['Roles'] 
                          if 'sagemaker' in role['RoleName'].lower()]
        return sagemaker_roles
    except Exception as e:
        return f"Error: {e}"

def test_aws_connectivity():
    """Test basic AWS connectivity."""
    print(" AWS Configuration Discovery")
    print("=" * 50)
    
    try:
        # Test basic connectivity
        print("1. Testing AWS credentials...")
        account_id = get_aws_account_id()
        if "Error" not in str(account_id):
            print(f"    AWS Account ID: {account_id}")
        else:
            print(f"    {account_id}")
            print("    You need to configure AWS credentials first!")
            print("   Run: aws configure")
            return False
            
        # Get region info
        print("\n2. Checking AWS region...")
        current_region = get_current_region()
        print(f"    Current region: {current_region}")
        
        # Check S3 bucket
        print("\n3. Checking S3 bucket 'adversal-synthetic-data'...")
        bucket_region = check_s3_bucket_region("adversal-synthetic-data")
        if "Error" not in str(bucket_region):
            print(f"    Bucket region: {bucket_region}")
            print(f"    Bucket URL: https://s3.console.aws.amazon.com/s3/buckets/adversal-synthetic-data")
        else:
            print(f"    {bucket_region}")
            
        # Check SageMaker roles
        print("\n4. Checking SageMaker execution roles...")
        sagemaker_roles = check_sagemaker_role()
        if isinstance(sagemaker_roles, list) and sagemaker_roles:
            print("    Found SageMaker roles:")
            for role in sagemaker_roles[:3]:  # Show first 3
                print(f"      - {role['RoleName']}: {role['Arn']}")
        else:
            print("   ️  No SageMaker roles found. You'll need to create one.")
            print("    Go to: https://console.aws.amazon.com/iam/home#/roles")
            
        # Generate configuration
        print(f"\n{'='*50}")
        print(" Your AWS Configuration:")
        print(f"{'='*50}")
        print(f"AWS_DEFAULT_REGION={bucket_region if 'Error' not in str(bucket_region) else current_region}")
        print(f"AWS_S3_BUCKET=adversal-synthetic-data")
        print(f"AWS_S3_PREFIX=synthetic-data/")
        
        if isinstance(sagemaker_roles, list) and sagemaker_roles:
            print(f"SAGEMAKER_EXECUTION_ROLE={sagemaker_roles[0]['Arn']}")
        else:
            print(f"SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::{account_id}:role/SageMakerExecutionRole")
            
        print(f"\n Copy these values to your .env file!")
        print(f" Don't forget to add your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        
        return True
        
    except NoCredentialsError:
        print(" No AWS credentials found!")
        print(" Please configure AWS credentials:")
        print("   1. Install AWS CLI: pip install awscli")
        print("   2. Configure: aws configure")
        print("   3. Enter your Access Key ID and Secret Access Key")
        return False
        
    except Exception as e:
        print(f" Error checking AWS configuration: {e}")
        return False

def show_credential_instructions():
    """Show instructions for getting AWS credentials."""
    print(f"\n{'='*50}")
    print(" How to Get AWS Access Keys:")
    print(f"{'='*50}")
    print("1. Go to AWS IAM Console:")
    print("   https://console.aws.amazon.com/iam/home#/users")
    print("\n2. Create a new user (or select existing):")
    print("   - Username: adversarial-data-user")
    print("   - Access type:  Programmatic access")
    print("\n3. Attach permissions:")
    print("   - AmazonS3FullAccess")
    print("   - AmazonSageMakerFullAccess")
    print("\n4. Save your credentials:")
    print("   - Access Key ID (starts with AKIA...)")
    print("   - Secret Access Key (long string)")
    print("\n5. Configure locally:")
    print("   aws configure")
    print("   (Enter your Access Key ID and Secret)")

if __name__ == "__main__":
    success = test_aws_connectivity()
    if not success:
        show_credential_instructions()
