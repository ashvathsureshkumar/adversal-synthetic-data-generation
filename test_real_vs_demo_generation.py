#!/usr/bin/env python3
"""
Test to show the difference between demo synthetic data generation 
and actual GAN model training/generation
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

# Import our models and training
from models.wgan_gp import WGAN_GP
from models.generator import Generator
from models.discriminator import Discriminator
from training.trainer import WGANGPTrainer
from data.preprocessor import DataPreprocessor
from main import SyntheticDataPipeline

def test_current_demo_approach():
    """Test the current demo approach (statistical sampling)"""
    print(" TESTING CURRENT DEMO APPROACH")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples).clip(18, 80).astype(int),
        'income': np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'approved': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    print(f"Original data shape: {data.shape}")
    print(f"Original gender distribution:\n{data['gender'].value_counts(normalize=True)}")
    print(f"Original approval rate: {data['approved'].mean():.2%}")
    
    # Current demo approach (from strands_agent.py)
    start_time = time.time()
    
    synthetic_data = pd.DataFrame()
    num_synthetic = 500
    
    for col in data.columns:
        if data[col].dtype in ['object', 'category']:
            # Categorical: sample with replacement
            synthetic_data[col] = np.random.choice(
                data[col].dropna().unique(), 
                size=num_synthetic, 
                replace=True
            )
        else:
            # Numerical: add noise
            mean = data[col].mean()
            std = data[col].std()
            noise_scale = 0.1
            synthetic_data[col] = np.random.normal(
                mean, 
                std * noise_scale, 
                size=num_synthetic
            )
    
    demo_time = time.time() - start_time
    
    print(f"\n DEMO RESULTS:")
    print(f"Generation time: {demo_time:.3f} seconds")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Synthetic gender distribution:\n{synthetic_data['gender'].value_counts(normalize=True)}")
    print(f"Synthetic approval rate: {synthetic_data['approved'].mean():.2%}")
    
    return data, synthetic_data, demo_time

def test_real_gan_approach():
    """Test actual GAN model training and generation"""
    print("\n TESTING REAL GAN APPROACH")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples).clip(18, 80).astype(int),
        'income': np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'approved': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Preprocess data for GAN
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.fit_transform(data)
    
    print(f"Processed data shape: {processed_data.shape}")
    
    # Create actual GAN model
    input_dim = processed_data.shape[1]
    generator = Generator(noise_dim=100, output_dim=input_dim, hidden_dims=[128, 256])
    discriminator = Discriminator(input_dim=input_dim, hidden_dims=[256, 128])
    
    # Create WGAN-GP model
    wgan_gp = WGAN_GP(
        input_dim=input_dim,
        noise_dim=100,
        generator_hidden_dims=[128, 256],
        discriminator_hidden_dims=[256, 128]
    )
    
    print(f"Created WGAN-GP model with {input_dim}D input")
    
    # Quick training simulation (just a few steps for demo)
    start_time = time.time()
    
    # Convert to tensor
    data_tensor = torch.FloatTensor(processed_data.values)
    
    # Simulate a few training steps
    print("Training GAN model...")
    for epoch in range(5):  # Just 5 epochs for demo
        # Training step simulation
        step_metrics = wgan_gp.train_step(data_tensor[:64])  # Small batch
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: D_loss={step_metrics.get('discriminator_loss', 0):.3f}")
    
    # Generate synthetic data
    print("Generating synthetic data with trained GAN...")
    with torch.no_grad():
        synthetic_tensor = wgan_gp.generate_samples(500, return_numpy=True)
    
    # Convert back to DataFrame
    synthetic_processed = pd.DataFrame(synthetic_tensor, columns=processed_data.columns)
    synthetic_data = preprocessor.inverse_transform(synthetic_processed)
    
    gan_time = time.time() - start_time
    
    print(f"\n GAN RESULTS:")
    print(f"Training + generation time: {gan_time:.3f} seconds")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    if 'gender' in synthetic_data.columns:
        print(f"Synthetic gender distribution:\n{synthetic_data['gender'].value_counts(normalize=True)}")
    if 'approved' in synthetic_data.columns:
        print(f"Synthetic approval rate: {synthetic_data['approved'].mean():.2%}")
    
    return synthetic_data, gan_time

def test_full_pipeline():
    """Test the complete SyntheticDataPipeline"""
    print("\n TESTING COMPLETE PIPELINE")
    print("=" * 50)
    
    try:
        # Create a temporary config
        config = {
            'model': {
                'type': 'wgan_gp',
                'noise_dim': 100,
                'generator_hidden_dims': [128, 256],
                'discriminator_hidden_dims': [256, 128]
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 10,
                'learning_rate': 0.0002,
                'n_critic': 5
            },
            'fairness': {
                'enabled': True,
                'protected_attributes': ['gender'],
                'fairness_lambda': 0.1
            },
            'privacy': {
                'enabled': True,
                'epsilon': 1.0,
                'delta': 1e-5
            },
            'databases': {
                'weaviate': {'enabled': False},
                'neo4j': {'enabled': False}
            }
        }
        
        # Save config temporarily
        import yaml
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        # Initialize pipeline
        pipeline = SyntheticDataPipeline(config_path)
        
        # Create sample data file
        np.random.seed(42)
        n_samples = 500  # Smaller for faster demo
        
        data = pd.DataFrame({
            'age': np.random.normal(35, 10, n_samples).clip(18, 80).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'approved': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            data_path = f.name
        
        print(f"Running pipeline on {len(data)} samples...")
        start_time = time.time()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(data_path, num_synthetic_samples=200)
        
        pipeline_time = time.time() - start_time
        
        print(f"\n PIPELINE RESULTS:")
        print(f"Total time: {pipeline_time:.3f} seconds")
        print(f"Status: {results.get('status', 'unknown')}")
        print(f"Original dataset: {results.get('dataset_shape', 'unknown')}")
        print(f"Synthetic samples: {results.get('synthetic_samples_generated', 'unknown')}")
        
        quality = results.get('quality_metrics', {})
        if quality:
            print(f"Quality score: {quality.get('overall_quality', {}).get('overall_score', 'N/A')}")
        
        fairness = results.get('fairness_audit', {})
        if fairness:
            print(f"Fairness violations: {len(fairness.get('violations', []))}")
        
        # Cleanup
        os.unlink(config_path)
        os.unlink(data_path)
        
        return results, pipeline_time
        
    except Exception as e:
        print(f" Pipeline test failed: {e}")
        return None, 0

def main():
    print(" ADVERSARIAL-AWARE SYNTHETIC DATA GENERATION")
    print(" DEMO vs REAL IMPLEMENTATION COMPARISON")
    print("=" * 60)
    
    # Test current demo approach
    original_data, demo_synthetic, demo_time = test_current_demo_approach()
    
    # Test real GAN approach
    try:
        gan_synthetic, gan_time = test_real_gan_approach()
    except Exception as e:
        print(f" GAN test failed: {e}")
        gan_time = 0
    
    # Test complete pipeline
    try:
        pipeline_results, pipeline_time = test_full_pipeline()
    except Exception as e:
        print(f" Pipeline test failed: {e}")
        pipeline_time = 0
    
    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    
    print(f"\n CURRENT STATUS:")
    print(f"  • Demo approach (statistical sampling):  Working ({demo_time:.3f}s)")
    print(f"  • Real GAN training: {' Working' if gan_time > 0 else ' Failed'} ({gan_time:.3f}s)")
    print(f"  • Complete pipeline: {' Working' if pipeline_time > 0 else ' Failed'} ({pipeline_time:.3f}s)")
    
    print(f"\n WHAT'S ACTUALLY HAPPENING:")
    print(f"  • Streamlit dashboard: Shows demo interface")
    print(f"  • Strands agent tools: Uses statistical sampling (NOT real GANs)")
    print(f"  • Cloud storage: Real (AWS S3, Neo4j, Weaviate)")
    print(f"  • GAN models: Implemented but not used in current workflow")
    
    print(f"\n TO GET REAL GAN GENERATION:")
    print(f"  • Need to replace statistical sampling in strands_agent.py")
    print(f"  • Use SyntheticDataPipeline.train_model() and generate_synthetic_data()")
    print(f"  • Training takes longer but produces higher quality synthetic data")
    
    print(f"\n RECOMMENDATION:")
    print(f"  Current setup is perfect for hackathon demo (fast + cloud integration)")
    print(f"  For production: Switch to real GAN training for higher quality")

if __name__ == "__main__":
    main()
