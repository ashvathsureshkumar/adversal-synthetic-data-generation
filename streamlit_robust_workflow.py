"""
Robust Real Workflow Streamlit App
Fixed file upload handling and improved error handling
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
import io

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aws'))

from dotenv import load_dotenv
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Real GAN Workflow",
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

# Simple GAN implementation
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

@st.cache_data
def load_csv_data(uploaded_file_content):
    """Load CSV data with caching to avoid re-reading"""
    try:
        # Create a StringIO object from the file content
        csv_string = io.StringIO(uploaded_file_content)
        data = pd.read_csv(csv_string)
        return data, None
    except Exception as e:
        return None, str(e)

def safe_cloud_init():
    """Safely initialize cloud components"""
    components = {'s3': None, 'neo4j': None, 'lineage': None}
    
    # S3
    try:
        from aws.s3_manager import S3Manager
        components['s3'] = S3Manager(
            bucket_name=os.getenv('AWS_S3_BUCKET', 'adversal-synthetic-data')
        )
        st.sidebar.success("AWS S3: Connected")
    except Exception as e:
        st.sidebar.error(f"AWS S3: Connection failed")
    
    # Neo4j
    try:
        from databases.neo4j_client import Neo4jManager, DataLineageTracker
        components['neo4j'] = Neo4jManager()
        components['lineage'] = DataLineageTracker(components['neo4j'])
        st.sidebar.success("Neo4j: Connected")
    except Exception as e:
        st.sidebar.error(f"Neo4j: Connection failed")
    
    return components

def train_simple_gan(data, num_epochs=15):
    """Train a simple GAN model with progress tracking"""
    
    # Prepare data
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) == 0:
        return None, None, None, "No numeric columns found"
    
    # Normalize to [-1, 1]
    data_min = numeric_data.min()
    data_max = numeric_data.max()
    normalized_data = 2 * (numeric_data - data_min) / (data_max - data_min) - 1
    data_tensor = torch.FloatTensor(normalized_data.values)
    
    input_dim = data_tensor.shape[1]
    noise_dim = 100
    
    # Initialize models
    generator = SimpleGenerator(noise_dim, input_dim)
    discriminator = SimpleDiscriminator(input_dim)
    
    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()
    
    # Training containers
    progress_placeholder = st.empty()
    loss_placeholder = st.empty()
    
    d_losses = []
    g_losses = []
    
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
        
        # Track losses
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
        # Update progress
        progress = (epoch + 1) / num_epochs
        progress_placeholder.progress(progress, f"Training epoch {epoch+1}/{num_epochs}")
        
        if epoch % 3 == 0:
            loss_placeholder.text(f"Discriminator Loss: {d_loss.item():.4f} | Generator Loss: {g_loss.item():.4f}")
    
    progress_placeholder.success(f"Training completed! Final losses - D: {d_losses[-1]:.4f}, G: {g_losses[-1]:.4f}")
    
    return generator, discriminator, (data_min, data_max, numeric_data.columns), None

def generate_synthetic_data(generator, normalization_info, num_samples):
    """Generate synthetic data and denormalize"""
    data_min, data_max, columns = normalization_info
    noise_dim = 100
    
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim)
        synthetic_tensor = generator(noise)
    
    # Denormalize
    synthetic_normalized = synthetic_tensor.numpy()
    
    # Ensure we have the right shape
    if synthetic_normalized.shape[1] != len(columns):
        st.error(f"Model output dimension {synthetic_normalized.shape[1]} doesn't match expected {len(columns)} columns")
        return None
    
    # Convert Series to numpy arrays if needed
    if hasattr(data_min, 'values'):
        data_min = data_min.values
    if hasattr(data_max, 'values'):
        data_max = data_max.values
    
    synthetic_denormalized = (synthetic_normalized + 1) / 2 * (data_max - data_min) + data_min
    
    # Create DataFrame
    synthetic_df = pd.DataFrame(synthetic_denormalized, columns=columns)
    
    return synthetic_df

def run_complete_workflow(uploaded_file_content, num_synthetic_samples, num_epochs):
    """Run the complete workflow"""
    
    st.markdown("### Training Progress")
    
    try:
        # Step 1: Load data
        st.info("Step 1/6: Loading dataset...")
        data, error = load_csv_data(uploaded_file_content)
        
        if error:
            st.error(f"Failed to load data: {error}")
            return None
        
        if data is None or data.empty:
            st.error("No data loaded")
            return None
        
        st.success(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Step 2: Initialize cloud
        st.info("Step 2/6: Connecting to cloud services...")
        components = safe_cloud_init()
        
        # Step 3: Upload to S3
        st.info("Step 3/6: Uploading to AWS S3...")
        dataset_name = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if components['s3']:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                data.to_csv(f.name, index=False)
                s3_key = f"datasets/{dataset_name}"
                if components['s3'].upload_file(f.name, s3_key):
                    st.success(f"Uploaded to S3: {s3_key}")
                os.unlink(f.name)
        
        # Step 4: Train GAN
        st.info("Step 4/6: Training GAN model...")
        generator, discriminator, norm_info, error = train_simple_gan(data, num_epochs)
        
        if error:
            st.error(error)
            return None
        
        # Step 5: Generate synthetic data
        st.info("Step 5/6: Generating synthetic data...")
        synthetic_data = generate_synthetic_data(generator, norm_info, num_synthetic_samples)
        
        if synthetic_data is None:
            st.error("Failed to generate synthetic data")
            return None
        
        st.success(f"Generated {len(synthetic_data)} synthetic samples")
        
        # Step 6: Store results
        st.info("Step 6/6: Storing results...")
        
        # Store synthetic data in S3
        synthetic_s3_key = None
        if components['s3']:
            synthetic_filename = f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                synthetic_data.to_csv(f.name, index=False)
                synthetic_s3_key = f"synthetic/{synthetic_filename}"
                if components['s3'].upload_file(f.name, synthetic_s3_key):
                    st.success(f"Synthetic data saved to S3: {synthetic_s3_key}")
                os.unlink(f.name)
        
        # Track lineage
        run_id = f"gan_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if components['lineage']:
            try:
                components['lineage'].track_generation_run(
                    run_id=run_id,
                    source_dataset=dataset_name,
                    model_type="simple_gan",
                    num_samples=num_synthetic_samples,
                    metadata={
                        "epochs": num_epochs,
                        "original_shape": list(data.shape),
                        "synthetic_shape": list(synthetic_data.shape)
                    }
                )
                st.success("Lineage tracked in Neo4j")
            except Exception as e:
                st.warning(f"Lineage tracking failed: {str(e)[:50]}...")
        
        return {
            'original_data': data,
            'synthetic_data': synthetic_data,
            'run_id': run_id,
            'synthetic_s3_key': synthetic_s3_key
        }
        
    except Exception as e:
        st.error(f"Workflow failed: {str(e)}")
        return None

def display_results(results):
    """Display results with comparisons"""
    
    st.markdown("---")
    st.markdown("## Results")
    
    # Metrics
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
            <div class="metric-value">Real GAN</div>
            <div class="metric-label">Model Type</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        stored = "✓" if results['synthetic_s3_key'] else "✗"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stored}</div>
            <div class="metric-label">Cloud Stored</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data comparison tabs
    tab1, tab2 = st.tabs(["Data Preview", "Statistical Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Data")
            st.dataframe(results['original_data'].head(8), width="stretch")
        
        with col2:
            st.markdown("#### Synthetic Data")
            st.dataframe(results['synthetic_data'].head(8), width="stretch")
    
    with tab2:
        st.markdown("#### Distribution Comparison")
        
        numeric_cols = results['original_data'].select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:2]:  # Show first 2 columns
            if col in results['synthetic_data'].columns:
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=results['original_data'][col],
                    name='Original',
                    opacity=0.7,
                    nbinsx=15
                ))
                
                fig.add_trace(go.Histogram(
                    x=results['synthetic_data'][col],
                    name='Synthetic',
                    opacity=0.7,
                    nbinsx=15
                ))
                
                fig.update_layout(
                    title=f'{col} Distribution',
                    barmode='overlay',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.markdown('<div class="main-title">Real GAN Training Workflow</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV dataset and train a real GAN model to generate synthetic data")
    
    # Sidebar
    st.sidebar.markdown("### Configuration")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="CSV file with numeric columns for GAN training"
    )
    
    num_synthetic_samples = st.sidebar.number_input(
        "Synthetic Samples",
        min_value=50,
        max_value=2000,
        value=300,
        step=50
    )
    
    num_epochs = st.sidebar.number_input(
        "Training Epochs",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )
    
    st.sidebar.markdown("### Cloud Status")
    
    # Main content
    if uploaded_file is not None:
        # Read the file content once
        file_content = uploaded_file.getvalue().decode('utf-8')
        
        # Load and preview data
        data_preview, error = load_csv_data(file_content)
        
        if error:
            st.error(f"Error reading file: {error}")
            return
        
        if data_preview is None:
            st.error("No data found in file")
            return
        
        st.markdown("### Dataset Ready")
        st.markdown(f"**File:** {uploaded_file.name}")
        st.markdown(f"**Size:** {data_preview.shape[0]} rows, {data_preview.shape[1]} columns")
        
        # Show preview
        with st.expander("Preview Data"):
            st.dataframe(data_preview.head(), width="stretch")
        
        # Check numeric columns
        numeric_cols = data_preview.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            st.error("No numeric columns found. Please upload a dataset with numeric data.")
            return
        
        st.info(f"Will train on {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")
        
        # Run workflow
        if st.button("Train Real GAN Model", type="primary"):
            results = run_complete_workflow(file_content, num_synthetic_samples, num_epochs)
            
            if results:
                display_results(results)
                
                st.markdown("""
                <div class="success-box">
                    <h3>GAN Training Complete!</h3>
                    <p>Successfully trained a real adversarial network and generated synthetic data:</p>
                    <ul>
                        <li>Real Generator and Discriminator networks trained</li>
                        <li>Adversarial training process completed</li>
                        <li>Synthetic data generated from learned patterns</li>
                        <li>Results stored in cloud infrastructure</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        ### How to Use
        
        1. **Upload CSV**: Use the sidebar to upload a CSV file with numeric data
        2. **Configure**: Set the number of synthetic samples and training epochs
        3. **Train**: Click "Train Real GAN Model" to start the process
        4. **Results**: View synthetic data and statistical comparisons
        
        **This uses real adversarial training** - not statistical sampling!
        """)

if __name__ == "__main__":
    main()
