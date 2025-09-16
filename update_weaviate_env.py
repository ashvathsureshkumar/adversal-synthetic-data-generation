#!/usr/bin/env python3
"""
Update environment with Weaviate credentials
"""

import os

def update_env_file():
    """Update .env file with Weaviate credentials."""
    
    # Your Weaviate configuration
    weaviate_url = "https://xfyxantiswcea7wo2qeufw.c0.us-west3.gcp.weaviate.cloud"
    weaviate_api_key = "Uk5wM0k1SXMvVkwwdnlFOV9JcjN2TWVIOGFmUWtqSWhrb2NqbUwzeVdnNXlHZFRzTDh4OFgvV3kxdWVRPV92MjAw"
    
    print("🔧 Updating environment with Weaviate credentials...")
    
    # Read current .env file
    env_content = ""
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            env_content = f.read()
    
    # Update Weaviate settings
    lines = env_content.split('\n')
    updated_lines = []
    weaviate_url_updated = False
    weaviate_key_updated = False
    
    for line in lines:
        if line.startswith('WEAVIATE_URL='):
            updated_lines.append(f'WEAVIATE_URL={weaviate_url}')
            weaviate_url_updated = True
        elif line.startswith('WEAVIATE_API_KEY='):
            updated_lines.append(f'WEAVIATE_API_KEY={weaviate_api_key}')
            weaviate_key_updated = True
        else:
            updated_lines.append(line)
    
    # Add if not found
    if not weaviate_url_updated:
        updated_lines.append(f'WEAVIATE_URL={weaviate_url}')
    if not weaviate_key_updated:
        updated_lines.append(f'WEAVIATE_API_KEY={weaviate_api_key}')
    
    # Write back to .env
    with open('.env', 'w') as f:
        f.write('\n'.join(updated_lines))
    
    print("✅ Environment updated successfully!")
    print(f"🔗 Weaviate URL: {weaviate_url}")
    print(f"🔑 API Key configured")
    print(f"🧪 Ready to test connection!")

if __name__ == "__main__":
    update_env_file()
    
    print("\n" + "="*50)
    print("🎯 Next steps:")
    print("1. Test connection: python test_weaviate_connection.py")
    print("2. If successful, your vector database is ready!")
    print("3. Your AI agent can now store and search embeddings!")
    print("="*50)
