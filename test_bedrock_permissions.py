#!/usr/bin/env python3
"""
Test AWS Bedrock permissions and model access
"""

import boto3
import json
from botocore.exceptions import ClientError

def test_bedrock_permissions():
    """Test if Bedrock permissions are configured correctly."""
    
    print(" Testing AWS Bedrock Permissions")
    print("=" * 40)
    
    try:
        # Create Bedrock client
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        print("1. Testing Bedrock service access...")
        
        # Test listing foundation models
        try:
            models = bedrock.list_foundation_models()
            print(f"    Can list foundation models: {len(models.get('modelSummaries', []))} models found")
            
            # Check for specific models
            model_ids = [model['modelId'] for model in models.get('modelSummaries', [])]
            
            target_models = [
                'us.amazon.nova-pro-v1:0',
                'anthropic.claude-3-sonnet-20240229-v1:0'
            ]
            
            for model_id in target_models:
                if model_id in model_ids:
                    print(f"    Model available: {model_id}")
                else:
                    print(f"   ️  Model not available: {model_id}")
                    
        except ClientError as e:
            print(f"    Cannot list models: {e}")
            return False
        
        print("\n2. Testing model invocation permissions...")
        
        # Test simple model invocation
        try:
            test_message = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "Hello! Just testing permissions."}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 100,
                    "temperature": 0.3
                }
            }
            
            # Try Nova Pro first
            try:
                response = bedrock_runtime.converse(
                    modelId="us.amazon.nova-pro-v1:0",
                    **test_message
                )
                print(f"    Nova Pro working! Response: {response['output']['message']['content'][0]['text'][:50]}...")
                return True
                
            except ClientError as nova_error:
                print(f"   ️  Nova Pro error: {nova_error}")
                
                # Try Claude as backup
                try:
                    response = bedrock_runtime.converse(
                        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                        **test_message
                    )
                    print(f"    Claude 3 Sonnet working! Response: {response['output']['message']['content'][0]['text'][:50]}...")
                    return True
                    
                except ClientError as claude_error:
                    print(f"    Claude 3 Sonnet error: {claude_error}")
                    return False
                    
        except Exception as e:
            print(f"    Model invocation failed: {e}")
            return False
            
    except Exception as e:
        print(f" Bedrock service error: {e}")
        return False

def show_permission_instructions():
    """Show instructions for adding Bedrock permissions."""
    
    print(f"\n{'='*60}")
    print(" How to Add Bedrock Permissions:")
    print(f"{'='*60}")
    print("1. Go to AWS IAM Console:")
    print("   https://console.aws.amazon.com/iam/home#/groups")
    print()
    print("2. Find your group: 'AdversarialDataGenerators'")
    print("   Click on the group name")
    print()
    print("3. Click 'Attach policies'")
    print("   Search for: AmazonBedrockFullAccess")
    print("   Select it and click 'Attach policy'")
    print()
    print("4. Enable model access:")
    print("   Go to: https://console.aws.amazon.com/bedrock/home#/modelaccess")
    print("   Click 'Manage model access'")
    print("   Enable: Amazon Nova Pro")
    print("   Enable: Claude 3 Sonnet (backup)")
    print()
    print("5. Wait 2-3 minutes for permissions to propagate")
    print("   Then run this test again!")

if __name__ == "__main__":
    success = test_bedrock_permissions()
    
    if success:
        print(f"\n{'='*60}")
        print(" SUCCESS! Bedrock permissions are working!")
        print(" Your AI agent is ready for full conversation!")
        print(" Run: python strands_bedrock_agent.py")
        print(f"{'='*60}")
    else:
        show_permission_instructions()
        print(f"\n After adding permissions, run this test again:")
        print(f"   python test_bedrock_permissions.py")
