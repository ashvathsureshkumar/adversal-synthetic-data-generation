#!/usr/bin/env python3
"""
Launch script for the Adversarial-Aware Synthetic Data Generator dashboard.

This script starts the Streamlit dashboard with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    print("🧬 Starting Adversarial-Aware Synthetic Data Generator Dashboard...")
    print("=" * 60)
    
    # Get the directory containing this script
    project_root = Path(__file__).parent
    streamlit_app = project_root / "streamlit_app.py"
    
    # Check if streamlit app exists
    if not streamlit_app.exists():
        print(f"❌ Error: Streamlit app not found at {streamlit_app}")
        return 1
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(project_root / "src")
    
    # Streamlit configuration
    config_args = [
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light"
    ]
    
    # Build the command
    cmd = ["streamlit", "run", str(streamlit_app)] + config_args
    
    print("🚀 Launching dashboard...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   App location: {streamlit_app}")
    print(f"   URL: http://localhost:8501")
    print("\n" + "=" * 60)
    print("📱 Dashboard should open automatically in your browser.")
    print("   If not, navigate to: http://localhost:8501")
    print("=" * 60)
    
    try:
        # Run streamlit
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
        return 0
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install it with:")
        print("   pip install streamlit")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
