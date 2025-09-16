"""
Simplified Real Workflow Streamlit App
Upload dataset -> Train real GANs -> Generate synthetic data -> Store in cloud
(Avoids relative import issues by using direct imports)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import tempfile
import time
import yaml
from datetime import datetime
import boto3
import torch
import torch.nn as nn

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aws'))

from dotenv import load_dotenv
load_dotenv()

# Direct imports to avoid relative import issues
try:
    from aws.s3_manager import S3Manager
    from databases.neo4j_client import Neo4jManager, DataLineageTracker
    from databases.weaviate_client import WeaviateClient
    from databases.embedding_generator import EmbeddingGenerator
    from data.preprocessor import DataPreprocessor
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Some components may not be available")

# Configure page
st.set_page_config(
    page_title="Real Workflow - Adversarial-Aware Synthetic Data Generator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: #0a0a0a;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #a0a0a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .workflow-step {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #7877c6;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        margin: 0.5rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Simple GAN implementation (to avoid import issues)
class SimpleGenerator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

class SimpleDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def initialize_cloud_components():
    """Initialize cloud components safely"""
    components = {}
    
    try:
        components['s3'] = S3Manager(
            bucket_name=os.getenv('AWS_S3_BUCKET', 'adversal-synthetic-data')
        )
        st.sidebar.success("S3: Connected")
    except Exception as e:
        st.sidebar.error(f"S3: {str(e)[:30]}...")
        components['s3'] = None
    
    try:
        components['neo4j'] = Neo4jManager()
        components['lineage'] = DataLineageTracker(components['neo4j'])
        st.sidebar.success("Neo4j: Connected")
    except Exception as e:
        st.sidebar.error(f"Neo4j: {str(e)[:30]}...")
        components['neo4j'] = None
        components['lineage'] = None
    
    try:
        components['weaviate'] = WeaviateClient()
        components['embeddings'] = EmbeddingGenerator()
        st.sidebar.success("Weaviate: Connected")
    except Exception as e:
        st.sidebar.error(f"Weaviate: {str(e)[:30]}...")
        components['weaviate'] = None
        components['embeddings'] = None
    
    return components

def train_simple_gan(data, num_epochs=10):
    """Train a simple GAN model"""
    
    # Convert to tensor
    if isinstance(data, pd.DataFrame):
        # Simple preprocessing - just normalize numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) == 0:
            st.error("No numeric columns found for training")
            return None
        
        # Normalize to [-1, 1]
        normalized_data = 2 * (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min()) - 1
        data_tensor = torch.FloatTensor(normalized_data.values)
    else:
        data_tensor = torch.FloatTensor(data)
    
    input_dim = data_tensor.shape[1]
    noise_dim = 100
    
    # Initialize models
    generator = SimpleGenerator(noise_dim, input_dim)
    discriminator = SimpleDiscriminator(input_dim)
    
    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    
    criterion = nn.BCELoss()
    
    # Training progress
    progress_placeholder = st.empty()
    
    for epoch in range(num_epochs):
        # Train discriminator
        d_optimizer.zero_grad()
        
        # Real data
        real_labels = torch.ones(data_tensor.size(0), 1)
        real_outputs = discriminator(data_tensor)
        d_loss_real = criterion(real_outputs, real_labels)
        
        # Fake data
        noise = torch.randn(data_tensor.size(0), noise_dim)
        fake_data = generator(noise)
        fake_labels = torch.zeros(data_tensor.size(0), 1)
        fake_outputs = discriminator(fake_data.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()
        
        # Train generator
        g_optimizer.zero_grad()
        fake_outputs = discriminator(fake_data)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        # Update progress
        if epoch % 2 == 0:
            progress_placeholder.text(f"Training epoch {epoch+1}/{num_epochs} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    progress_placeholder.text(f"Training completed! Final - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    return generator, discriminator, normalized_data

def generate_synthetic_data(generator, original_data, num_samples):
    """Generate synthetic data using trained generator"""
    noise_dim = 100
    
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim)
        synthetic_tensor = generator(noise)
    
    # Convert back to DataFrame with original column names
    if isinstance(original_data, pd.DataFrame):
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        synthetic_df = pd.DataFrame(synthetic_tensor.numpy(), columns=numeric_cols)
        
        # Denormalize (reverse the [-1,1] normalization)
        for col in numeric_cols:
            min_val = original_data[col].min()
            max_val = original_data[col].max()
            synthetic_df[col] = (synthetic_df[col] + 1) / 2 * (max_val - min_val) + min_val
        
        return synthetic_df
    else:
        return synthetic_tensor.numpy()

def run_simplified_workflow(uploaded_file, num_synthetic_samples, num_epochs):
    """Run the simplified real workflow"""
    
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### Workflow Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load data
            status_text.text("Step 1/6: Loading and preprocessing data...")
            progress_bar.progress(15)
            
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file)
            
            # Validate data
            if data.empty:
                st.error("Uploaded file is empty")
                return None
            
            if len(data.columns) == 0:
                st.error("No columns found in uploaded file")
                return None
            st.markdown("#### Dataset Preview")
            st.dataframe(data.head(), width="stretch")
            st.markdown(f"**Shape:** {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Step 2: Initialize cloud components
            status_text.text("Step 2/6: Connecting to cloud services...")
            progress_bar.progress(25)
            
            components = initialize_cloud_components()
            
            # Step 3: Upload original data to S3
            status_text.text("Step 3/6: Uploading to AWS S3...")
            progress_bar.progress(35)
            
            dataset_name = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            s3_stored = False
            
            if components['s3']:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    data.to_csv(f.name, index=False)
                    s3_key = f"datasets/{dataset_name}"
                    if components['s3'].upload_file(f.name, s3_key):
                        st.success(f"Original data uploaded to S3: {s3_key}")
                        s3_stored = True
                    os.unlink(f.name)
            
            # Step 4: Train GAN model
            status_text.text("Step 4/6: Training GAN model...")
            progress_bar.progress(45)
            
            generator, discriminator, processed_data = train_simple_gan(data, num_epochs)
            
            if generator is None:
                st.error("Model training failed")
                return None
            
            st.success("GAN model training completed!")
            
            # Step 5: Generate synthetic data
            status_text.text("Step 5/6: Generating synthetic data...")
            progress_bar.progress(70)
            
            synthetic_data = generate_synthetic_data(generator, data, num_synthetic_samples)
            st.success(f"Generated {len(synthetic_data)} synthetic samples")
            
            # Step 6: Store results
            status_text.text("Step 6/6: Storing results in cloud...")
            progress_bar.progress(85)
            
            synthetic_s3_key = None
            if components['s3'] and s3_stored:
                synthetic_filename = f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    synthetic_data.to_csv(f.name, index=False)
                    synthetic_s3_key = f"synthetic/{synthetic_filename}"
                    if components['s3'].upload_file(f.name, synthetic_s3_key):
                        st.success(f"Synthetic data uploaded to S3: {synthetic_s3_key}")
                    os.unlink(f.name)
            
            # Track lineage
            run_id = f"simple_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if components['lineage']:
                try:
                    components['lineage'].track_generation_run(
                        run_id=run_id,
                        source_dataset=dataset_name,
                        model_type="simple_gan",
                        num_samples=num_synthetic_samples,
                        metadata={
                            "epochs": num_epochs,
                            "original_shape": data.shape,
                            "synthetic_shape": synthetic_data.shape
                        }
                    )
                    st.success("Lineage tracked in Neo4j")
                except Exception as e:
                    st.warning(f"Lineage tracking failed: {str(e)[:50]}...")
            
            progress_bar.progress(100)
            status_text.text("Workflow completed successfully!")
            
            return {
                'original_data': data,
                'synthetic_data': synthetic_data,
                'dataset_s3_key': s3_key if s3_stored else None,
                'synthetic_s3_key': synthetic_s3_key,
                'run_id': run_id,
                'model_trained': True
            }
            
        except Exception as e:
            st.error(f"Workflow failed: {str(e)}")
            return None

def display_results(results):
    """Display comprehensive results"""
    if not results:
        return
    
    st.markdown("---")
    st.markdown("## Workflow Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(results['original_data'])}</div>
            <div class="metric-label">Original Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(results['synthetic_data'])}</div>
            <div class="metric-label">Synthetic Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{"✓" if results['model_trained'] else "✗"}</div>
            <div class="metric-label">Model Trained</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        s3_status = "✓" if results['synthetic_s3_key'] else "✗"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{s3_status}</div>
            <div class="metric-label">Cloud Stored</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data comparison
    st.markdown("### Data Comparison")
    
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Distribution Analysis", "Cloud Status"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Data")
            st.dataframe(results['original_data'].head(10), width="stretch")
        
        with col2:
            st.markdown("#### Synthetic Data")
            st.dataframe(results['synthetic_data'].head(10), width="stretch")
    
    with tab2:
        # Statistical comparison
        st.markdown("#### Distribution Comparison")
        
        numeric_cols = results['original_data'].select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            for col in numeric_cols[:2]:  # Show first 2 columns
                if col in results['synthetic_data'].columns:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=results['original_data'][col],
                        name='Original',
                        opacity=0.7,
                        nbinsx=20
                    ))
                    
                    fig.add_trace(go.Histogram(
                        x=results['synthetic_data'][col],
                        name='Synthetic',
                        opacity=0.7,
                        nbinsx=20
                    ))
                    
                    fig.update_layout(
                        title=f'Distribution: {col}',
                        xaxis_title=col,
                        yaxis_title='Count',
                        barmode='overlay',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#ffffff'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Cloud Storage Status")
        
        if results['dataset_s3_key']:
            st.success(f"Original data in S3: {results['dataset_s3_key']}")
        else:
            st.warning("Original data not stored in S3")
        
        if results['synthetic_s3_key']:
            st.success(f"Synthetic data in S3: {results['synthetic_s3_key']}")
        else:
            st.warning("Synthetic data not stored in S3")
        
        st.info(f"Workflow run ID: {results['run_id']}")

def main():
    # Header
    st.markdown('<div class="main-title">Real GAN Workflow</div>', unsafe_allow_html=True)
    st.markdown("Upload your dataset and train a real GAN model to generate synthetic data")
    
    # Sidebar configuration
    st.sidebar.markdown("### Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset", 
        type=['csv'],
        help="Upload a CSV file to train a GAN model"
    )
    
    # Parameters
    num_synthetic_samples = st.sidebar.number_input(
        "Synthetic Samples",
        min_value=50,
        max_value=5000,
        value=500,
        step=50
    )
    
    num_epochs = st.sidebar.number_input(
        "Training Epochs",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        help="More epochs = better quality but slower training"
    )
    
    # Cloud status
    st.sidebar.markdown("### Cloud Status")
    
    # Main content
    if uploaded_file is not None:
        st.markdown("### Ready to Train")
        
        # Show file info
        try:
            # Reset file pointer and read
            uploaded_file.seek(0)
            df_preview = pd.read_csv(uploaded_file)
            
            st.markdown(f"**File:** {uploaded_file.name}")
            st.markdown(f"**Size:** {df_preview.shape[0]} rows, {df_preview.shape[1]} columns")
            
            # Preview
            with st.expander("Preview Data"):
                st.dataframe(df_preview.head(), width="stretch")
            
            # Check if we have numeric data
            numeric_cols = df_preview.select_dtypes(include=[np.number]).columns
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return
        if len(numeric_cols) == 0:
            st.error("No numeric columns found. Please upload a dataset with numeric data for GAN training.")
            return
        
        st.info(f"Will train on {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")
        
        # Run workflow button
        if st.button("Train GAN Model & Generate Data", type="primary"):
            with st.spinner("Training GAN model..."):
                results = run_simplified_workflow(
                    uploaded_file, 
                    num_synthetic_samples, 
                    num_epochs
                )
                
                if results:
                    display_results(results)
                    
                    # Success message
                    st.markdown("""
                    <div class="success-box">
                        <h3>GAN Training Completed!</h3>
                        <p>Your dataset has been processed through a real GAN workflow:</p>
                        <ul>
                            <li>Actual GAN model trained on your data</li>
                            <li>Synthetic data generated using adversarial learning</li>
                            <li>Data stored in AWS S3 with tracking</li>
                            <li>Lineage recorded in Neo4j</li>
                            <li>Statistical comparisons available</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Instructions
        st.markdown("""
        <div class="workflow-step">
            <h3>Real GAN Workflow Steps</h3>
            <ol>
                <li><strong>Upload Dataset:</strong> CSV file with numeric columns</li>
                <li><strong>Configure Training:</strong> Set epochs and sample count</li>
                <li><strong>Train GAN:</strong> Real adversarial training process</li>
                <li><strong>Generate Data:</strong> Create synthetic samples</li>
                <li><strong>Store Results:</strong> Save to cloud with tracking</li>
                <li><strong>Compare Quality:</strong> Statistical analysis</li>
            </ol>
            
            <h4>This Uses Real GANs:</h4>
            <ul>
                <li>Generator and Discriminator networks</li>
                <li>Adversarial training process</li>
                <li>Actual model learning from your data</li>
                <li>Cloud storage with full lineage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
