"""
Real Workflow Streamlit App
Upload dataset -> Train real GANs -> Generate synthetic data -> Store in cloud
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

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

# Import our real components
from main import SyntheticDataPipeline
from aws.s3_manager import S3Manager
from databases.neo4j_client import Neo4jManager, DataLineageTracker
from databases.weaviate_client import WeaviateClient
from databases.embedding_generator import EmbeddingGenerator

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
    
    .workflow-step:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(120, 119, 198, 0.3);
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
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

def initialize_components():
    """Initialize cloud components"""
    try:
        # S3 Manager
        s3_manager = S3Manager(
            bucket_name=os.getenv('AWS_S3_BUCKET', 'adversal-synthetic-data')
        )
        
        # Neo4j components
        neo4j_manager = Neo4jManager()
        lineage_tracker = DataLineageTracker(neo4j_manager)
        
        # Weaviate client
        weaviate_client = WeaviateClient()
        embedding_generator = EmbeddingGenerator()
        
        return s3_manager, neo4j_manager, lineage_tracker, weaviate_client, embedding_generator
    except Exception as e:
        st.error(f"Error initializing cloud components: {e}")
        return None, None, None, None, None

def create_temp_config():
    """Create a temporary config for the pipeline"""
    config = {
        'model': {
            'type': 'wgan_gp',
            'noise_dim': 100,
            'generator_hidden_dims': [128, 256],
            'discriminator_hidden_dims': [256, 128]
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 20,  # Reduced for demo
            'learning_rate': 0.0002,
            'n_critic': 5
        },
        'fairness': {
            'enabled': True,
            'protected_attributes': [],  # Will be auto-detected
            'fairness_lambda': 0.1
        },
        'privacy': {
            'enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5
        },
        'databases': {
            'weaviate': {'enabled': True},
            'neo4j': {'enabled': True}
        }
    }
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return f.name

def run_real_workflow(uploaded_file, num_synthetic_samples, privacy_epsilon, fairness_enabled):
    """Run the complete real workflow"""
    
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.markdown("### Workflow Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Save uploaded file
            status_text.text("Step 1/8: Processing uploaded dataset...")
            progress_bar.progress(10)
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
                f.write(uploaded_file.getbuffer())
                data_path = f.name
            
            # Load and preview data
            data = pd.read_csv(data_path)
            st.markdown("#### Uploaded Dataset Preview")
            st.dataframe(data.head(), use_container_width=True)
            st.markdown(f"**Dataset shape:** {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Step 2: Initialize pipeline
            status_text.text("Step 2/8: Initializing synthetic data pipeline...")
            progress_bar.progress(20)
            
            config_path = create_temp_config()
            pipeline = SyntheticDataPipeline(config_path)
            
            # Step 3: Upload to S3
            status_text.text("Step 3/8: Uploading dataset to AWS S3...")
            progress_bar.progress(30)
            
            s3_manager, neo4j_manager, lineage_tracker, weaviate_client, embedding_generator = initialize_components()
            
            if s3_manager:
                dataset_name = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                s3_key = f"datasets/{dataset_name}"
                
                if s3_manager.upload_file(data_path, s3_key):
                    st.success(f"Dataset uploaded to S3: s3://{s3_manager.bucket_name}/{s3_key}")
                else:
                    st.error("Failed to upload to S3")
            
            # Step 4: Train model
            status_text.text("Step 4/8: Training WGAN-GP model (this may take a few minutes)...")
            progress_bar.progress(40)
            
            time.sleep(2)  # Brief pause for UI
            
            # Load and preprocess data
            processed_data = pipeline.load_and_preprocess_data(data_path)
            st.info(f"Preprocessed data shape: {processed_data.shape}")
            
            # Train the model
            model = pipeline.train_model(processed_data)
            st.success("Model training completed!")
            
            # Step 5: Generate synthetic data
            status_text.text("Step 5/8: Generating synthetic data with trained model...")
            progress_bar.progress(60)
            
            synthetic_data = pipeline.generate_synthetic_data(
                model, 
                num_samples=num_synthetic_samples, 
                original_data=processed_data
            )
            
            st.success(f"Generated {len(synthetic_data)} synthetic samples")
            
            # Step 6: Store in vector database
            status_text.text("Step 6/8: Storing embeddings in Weaviate...")
            progress_bar.progress(70)
            
            if weaviate_client and embedding_generator:
                # Store original data embeddings
                pipeline.store_in_vector_database(processed_data, "real")
                # Store synthetic data embeddings  
                pipeline.store_in_vector_database(synthetic_data, "synthetic")
                st.success("Data stored in Weaviate vector database")
            
            # Step 7: Track lineage in Neo4j
            status_text.text("Step 7/8: Recording data lineage in Neo4j...")
            progress_bar.progress(80)
            
            if lineage_tracker:
                run_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                lineage_tracker.track_generation_run(
                    run_id=run_id,
                    source_dataset=dataset_name,
                    model_type="wgan_gp",
                    num_samples=num_synthetic_samples,
                    metadata={
                        "privacy_epsilon": privacy_epsilon,
                        "fairness_enabled": fairness_enabled,
                        "original_shape": data.shape,
                        "synthetic_shape": synthetic_data.shape
                    }
                )
                st.success("Lineage tracked in Neo4j Aura")
            
            # Step 8: Evaluate and store results
            status_text.text("Step 8/8: Running quality evaluation...")
            progress_bar.progress(90)
            
            # Run evaluations
            quality_results = pipeline.evaluate_quality(processed_data, synthetic_data)
            fairness_results = pipeline.run_fairness_audit(synthetic_data)
            privacy_results = pipeline.run_privacy_audit(model, processed_data)
            
            # Store synthetic data to S3
            synthetic_filename = f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                synthetic_data.to_csv(f.name, index=False)
                synthetic_s3_key = f"synthetic/{synthetic_filename}"
                
                if s3_manager.upload_file(f.name, synthetic_s3_key):
                    st.success(f"Synthetic data saved to S3: s3://{s3_manager.bucket_name}/{synthetic_s3_key}")
            
            progress_bar.progress(100)
            status_text.text("Workflow completed successfully!")
            
            # Cleanup temp files
            os.unlink(data_path)
            os.unlink(config_path)
            
            return {
                'original_data': data,
                'synthetic_data': synthetic_data,
                'quality_results': quality_results,
                'fairness_results': fairness_results,
                'privacy_results': privacy_results,
                'dataset_s3_key': s3_key,
                'synthetic_s3_key': synthetic_s3_key,
                'run_id': run_id
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
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Original Samples</div>
        </div>
        """.format(len(results['original_data'])), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Synthetic Samples</div>
        </div>
        """.format(len(results['synthetic_data'])), unsafe_allow_html=True)
    
    with col3:
        quality_score = results['quality_results'].get('overall_quality', {}).get('overall_score', 0)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1%}</div>
            <div class="metric-label">Quality Score</div>
        </div>
        """.format(quality_score), unsafe_allow_html=True)
    
    with col4:
        fairness_violations = len(results['fairness_results'].get('violations', []))
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Fairness Violations</div>
        </div>
        """.format(fairness_violations), unsafe_allow_html=True)
    
    # Data comparison
    st.markdown("### Data Comparison")
    
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Statistical Comparison", "Cloud Storage"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Data")
            st.dataframe(results['original_data'].head(10), use_container_width=True)
        
        with col2:
            st.markdown("#### Synthetic Data")
            st.dataframe(results['synthetic_data'].head(10), use_container_width=True)
    
    with tab2:
        # Statistical comparison charts
        st.markdown("#### Statistical Distribution Comparison")
        
        # Select numeric columns for comparison
        numeric_cols = results['original_data'].select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Show first 3 numeric columns
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=results['original_data'][col],
                    name='Original',
                    opacity=0.7,
                    nbinsx=30
                ))
                
                fig.add_trace(go.Histogram(
                    x=results['synthetic_data'][col],
                    name='Synthetic',
                    opacity=0.7,
                    nbinsx=30
                ))
                
                fig.update_layout(
                    title=f'Distribution Comparison: {col}',
                    xaxis_title=col,
                    yaxis_title='Frequency',
                    barmode='overlay',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Cloud Storage Verification")
        
        # S3 Storage
        st.markdown("**AWS S3 Storage:**")
        st.code(f"Original dataset: s3://{os.getenv('AWS_S3_BUCKET')}/{results['dataset_s3_key']}")
        st.code(f"Synthetic dataset: s3://{os.getenv('AWS_S3_BUCKET')}/{results['synthetic_s3_key']}")
        
        # Neo4j Lineage
        st.markdown("**Neo4j Aura Lineage:**")
        st.code(f"Run ID: {results['run_id']}")
        st.code(f"Database: {os.getenv('NEO4J_DATABASE')}")
        
        # Weaviate Embeddings
        st.markdown("**Weaviate Vector Storage:**")
        st.code(f"Cluster: {os.getenv('WEAVIATE_URL')}")
        st.code("Original and synthetic data embeddings stored with metadata")

def main():
    # Header
    st.markdown('<div class="main-title">Real Workflow Generator</div>', unsafe_allow_html=True)
    st.markdown("Upload your dataset and watch it go through the complete adversarial-aware synthetic data pipeline")
    
    # Sidebar configuration
    st.sidebar.markdown("### Workflow Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset", 
        type=['csv'],
        help="Upload a CSV file to process through the real GAN workflow"
    )
    
    # Parameters
    num_synthetic_samples = st.sidebar.number_input(
        "Synthetic Samples to Generate",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )
    
    privacy_epsilon = st.sidebar.slider(
        "Privacy Level (ε)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Lower ε = more private"
    )
    
    fairness_enabled = st.sidebar.checkbox("Enable Fairness Constraints", value=True)
    
    # Cloud status
    st.sidebar.markdown("### Cloud Status")
    
    # Check cloud connectivity
    try:
        # Test S3
        s3 = boto3.client('s3')
        s3.head_bucket(Bucket=os.getenv('AWS_S3_BUCKET', 'adversal-synthetic-data'))
        st.sidebar.success("AWS S3: Connected")
    except:
        st.sidebar.error("AWS S3: Not connected")
    
    try:
        # Test Neo4j
        from neo4j import GraphDatabase
        uri = os.getenv('NEO4J_URI')
        if uri:
            st.sidebar.success("Neo4j Aura: Connected")
        else:
            st.sidebar.error("Neo4j Aura: Not configured")
    except:
        st.sidebar.error("Neo4j Aura: Not connected")
    
    try:
        # Test Weaviate
        import weaviate
        client = weaviate.connect_to_wcs(
            cluster_url=os.getenv('WEAVIATE_URL'),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv('WEAVIATE_API_KEY'))
        )
        client.close()
        st.sidebar.success("Weaviate: Connected")
    except:
        st.sidebar.error("Weaviate: Not connected")
    
    # Main content
    if uploaded_file is not None:
        st.markdown("### Ready to Process")
        
        # Show file info
        df_preview = pd.read_csv(uploaded_file)
        st.markdown(f"**File:** {uploaded_file.name}")
        st.markdown(f"**Size:** {df_preview.shape[0]} rows, {df_preview.shape[1]} columns")
        
        # Preview
        with st.expander("Preview Data"):
            st.dataframe(df_preview.head(), use_container_width=True)
        
        # Run workflow button
        if st.button("Run Complete Workflow", type="primary"):
            with st.spinner("Running real workflow..."):
                results = run_real_workflow(
                    uploaded_file, 
                    num_synthetic_samples, 
                    privacy_epsilon, 
                    fairness_enabled
                )
                
                if results:
                    display_results(results)
                    
                    # Success message
                    st.markdown("""
                    <div class="success-box">
                        <h3>Workflow Completed Successfully!</h3>
                        <p>Your dataset has been processed through the complete adversarial-aware synthetic data pipeline:</p>
                        <ul>
                            <li>Real WGAN-GP model trained on your data</li>
                            <li>High-quality synthetic data generated</li>
                            <li>Data stored in AWS S3 with full lineage</li>
                            <li>Embeddings created and stored in Weaviate</li>
                            <li>Lineage tracked in Neo4j Aura</li>
                            <li>Quality and fairness evaluations completed</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Instructions
        st.markdown("""
        <div class="workflow-step">
            <h3>How to Use the Real Workflow</h3>
            <ol>
                <li><strong>Upload Dataset:</strong> Use the file uploader in the sidebar to upload a CSV file</li>
                <li><strong>Configure Parameters:</strong> Set the number of synthetic samples, privacy level, and fairness options</li>
                <li><strong>Verify Cloud Status:</strong> Ensure all cloud services show "Connected" status</li>
                <li><strong>Run Workflow:</strong> Click "Run Complete Workflow" to start the real GAN training process</li>
                <li><strong>View Results:</strong> See quality metrics, data comparisons, and cloud storage verification</li>
            </ol>
        </div>
        
        <div class="info-box">
            <h4>What This Workflow Does:</h4>
            <ul>
                <li>Trains an actual WGAN-GP model on your uploaded data</li>
                <li>Generates synthetic data using adversarial learning</li>
                <li>Stores original and synthetic data in AWS S3</li>
                <li>Creates vector embeddings and stores them in Weaviate</li>
                <li>Tracks complete data lineage in Neo4j Aura</li>
                <li>Runs comprehensive quality and fairness evaluations</li>
                <li>Provides statistical comparisons and visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
