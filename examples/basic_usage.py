"""
Basic usage example for the Adversarial-Aware Synthetic Data Generator.

This example demonstrates the complete workflow from data loading to 
synthetic data generation with fairness and privacy constraints.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import SyntheticDataPipeline


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic demographic data
    n_samples = 2000
    
    # Demographics
    age = np.random.normal(35, 12, n_samples).clip(18, 80)
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.52, 0.48])
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                n_samples, p=[0.3, 0.4, 0.25, 0.05])
    
    # Income (with some bias based on demographics)
    base_income = np.random.normal(50000, 15000, n_samples)
    gender_bias = np.where(gender == 'M', 1.15, 1.0)  # Gender pay gap
    education_multiplier = {'High School': 0.8, 'Bachelor': 1.0, 'Master': 1.3, 'PhD': 1.6}
    education_bias = np.array([education_multiplier[ed] for ed in education])
    
    income = (base_income * gender_bias * education_bias).clip(20000, 200000)
    
    # Credit score (correlated with income and age)
    credit_base = 300 + (income / 1000) + (age - 18) * 2
    credit_noise = np.random.normal(0, 50, n_samples)
    credit_score = (credit_base + credit_noise).clip(300, 850)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'education': education,
        'income': income,
        'credit_score': credit_score
    })
    
    return df


def main():
    """Run the basic usage example."""
    print("🧬 Adversarial-Aware Synthetic Data Generator - Basic Usage Example")
    print("=" * 70)
    
    # Create sample dataset
    print("1. Creating sample dataset...")
    df = create_sample_dataset()
    print(f"   Dataset created with shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Save the dataset
    data_path = Path(__file__).parent / "sample_data.csv"
    df.to_csv(data_path, index=False)
    print(f"   Dataset saved to: {data_path}")
    
    # Create minimal configuration
    config = {
        'model': {
            'type': 'wgan_gp',
            'noise_dim': 100,
            'generator_dims': [128, 256, 512],
            'discriminator_dims': [512, 256, 128],
            'lambda_gp': 10.0
        },
        'training': {
            'num_epochs': 100,  # Reduced for demo
            'batch_size': 64,
            'learning_rate': 0.0002,
            'beta1': 0.5,
            'beta2': 0.999
        },
        'fairness': {
            'enabled': True,
            'constraint_weight': 0.1,
            'protected_attributes': ['gender'],
            'fairness_metric': 'demographic_parity'
        },
        'privacy': {
            'enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0
        },
        'data': {
            'categorical_threshold': 10,
            'numerical_scaling': 'standard',
            'handle_missing': 'median'
        },
        'databases': {
            'weaviate': {'enabled': False},  # Disabled for simple demo
            'neo4j': {'enabled': False}      # Disabled for simple demo
        },
        'embedding': {
            'method': 'autoencoder',
            'dimension': 64
        }
    }
    
    # Save configuration
    config_path = Path(__file__).parent / "demo_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2)
    print(f"   Configuration saved to: {config_path}")
    
    print("\n2. Initializing pipeline...")
    try:
        # Initialize pipeline
        pipeline = SyntheticDataPipeline(str(config_path))
        print("   ✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"   ❌ Pipeline initialization failed: {e}")
        return
    
    print("\n3. Running data preprocessing...")
    try:
        # Load and preprocess data
        processed_data = pipeline.load_and_preprocess_data(str(data_path))
        print(f"   ✅ Data preprocessed: {processed_data.shape}")
    except Exception as e:
        print(f"   ❌ Data preprocessing failed: {e}")
        return
    
    print("\n4. Training model...")
    try:
        # Train model (this will take some time)
        model = pipeline.train_model(processed_data)
        print("   ✅ Model training completed")
    except Exception as e:
        print(f"   ❌ Model training failed: {e}")
        return
    
    print("\n5. Generating synthetic data...")
    try:
        # Generate synthetic data
        synthetic_data = pipeline.generate_synthetic_data(
            model, 
            num_samples=1000, 
            original_data=processed_data
        )
        print(f"   ✅ Generated {len(synthetic_data)} synthetic samples")
    except Exception as e:
        print(f"   ❌ Synthetic data generation failed: {e}")
        return
    
    print("\n6. Evaluating quality...")
    try:
        # Evaluate quality
        quality_results = pipeline.evaluate_quality(processed_data, synthetic_data)
        print(f"   ✅ Quality evaluation completed")
        print(f"   Overall quality score: {quality_results.get('overall_quality', {}).get('overall_score', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Quality evaluation failed: {e}")
        quality_results = {}
    
    print("\n7. Running fairness audit...")
    try:
        # Run fairness audit
        fairness_results = pipeline.run_fairness_audit(synthetic_data, ['gender'])
        print(f"   ✅ Fairness audit completed")
        print(f"   Audit passed: {fairness_results.get('audit_passed', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Fairness audit failed: {e}")
        fairness_results = {}
    
    print("\n8. Running privacy audit...")
    try:
        # Run privacy audit
        privacy_results = pipeline.run_privacy_audit(model, processed_data)
        print(f"   ✅ Privacy audit completed")
        print(f"   Risk score: {privacy_results.get('overall_risk_score', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Privacy audit failed: {e}")
        privacy_results = {}
    
    # Save results
    print("\n9. Saving results...")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save synthetic data
    synthetic_data.to_csv(results_dir / "synthetic_data.csv", index=False)
    
    # Save evaluation results
    import json
    with open(results_dir / "evaluation_results.json", 'w') as f:
        json.dump({
            'quality_results': quality_results,
            'fairness_results': fairness_results,
            'privacy_results': privacy_results
        }, f, indent=2, default=str)
    
    print(f"   ✅ Results saved to: {results_dir}")
    
    print("\n" + "=" * 70)
    print("🎉 Basic usage example completed successfully!")
    print("=" * 70)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"• Original data: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"• Synthetic data: {len(synthetic_data)} samples")
    print(f"• Quality score: {quality_results.get('overall_quality', {}).get('overall_score', 'N/A')}")
    print(f"• Fairness audit: {'✅ Passed' if fairness_results.get('audit_passed') else '❌ Failed'}")
    print(f"• Privacy risk: {privacy_results.get('risk_level', 'Unknown')}")


if __name__ == "__main__":
    main()
