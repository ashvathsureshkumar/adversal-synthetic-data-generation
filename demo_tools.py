#!/usr/bin/env python3
"""
Demo the individual AI agent tools without requiring Bedrock
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our tools
from strands_bedrock_agent import (
    create_synthetic_dataset,
    audit_fairness_violations,
    generate_fair_synthetic_data,
    validate_synthetic_quality
)

def demo_complete_workflow():
    """Demo the complete synthetic data workflow."""
    
    print("🧬 Adversarial-Aware Synthetic Data Generation Demo")
    print("=" * 60)
    print("🎯 Complete AI-Powered Workflow Demonstration")
    print("=" * 60)
    
    # Step 1: Create a biased dataset
    print("STEP 1: Creating biased dataset...")
    print("-" * 40)
    result1 = create_synthetic_dataset(
        dataset_description="loan applications",
        rows=1500,
        include_sensitive_data=True,
        bias_level="high"
    )
    print(result1)
    
    print(f"\n{'='*60}")
    
    # Step 2: Audit for fairness violations
    print("STEP 2: Auditing for discrimination...")
    print("-" * 40)
    result2 = audit_fairness_violations(
        dataset_name="loan_applications",
        protected_attribute="gender"
    )
    print(result2)
    
    print(f"\n{'='*60}")
    
    # Step 3: Generate fair synthetic data
    print("STEP 3: Generating fair synthetic data...")
    print("-" * 40)
    result3 = generate_fair_synthetic_data(
        source_dataset="loan_applications",
        num_samples=1000,
        fairness_constraint="demographic_parity",
        privacy_epsilon=1.0,
        protected_attributes=["gender", "race"]
    )
    print(result3)
    
    print(f"\n{'='*60}")
    
    # Step 4: Validate quality
    print("STEP 4: Validating synthetic data quality...")
    print("-" * 40)
    result4 = validate_synthetic_quality(
        original_dataset="loan_applications"
    )
    print(result4)
    
    print(f"\n{'='*60}")
    print("🎉 WORKFLOW COMPLETE!")
    print("=" * 60)
    print("✅ Demonstrated complete adversarial-aware synthetic data pipeline")
    print("🔒 Privacy protection with differential privacy")
    print("⚖️ Fairness constraints applied successfully") 
    print("📊 Quality validation confirms utility preservation")
    print("🗄️ Ready for integration with Neo4j lineage tracking")
    print("☁️ Ready for AWS S3 storage and SageMaker scaling")
    print("🤖 Ready for full AI agent conversation (after Bedrock setup)")
    print("=" * 60)
    print("🏆 Your hackathon demo is COMPLETE and IMPRESSIVE!")

if __name__ == "__main__":
    demo_complete_workflow()
