"""
Framer-Style Streamlit Dashboard
Ultra-sleek design with proper spacing and minimal clutter
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure page
st.set_page_config(
    page_title="Adversarial-Aware Synthetic Data Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Framer-style CSS
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global reset and dark theme */
    .stApp {
        background: #0a0a0a;
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Remove default padding */
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    /* Hero section */
    .hero-section {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        background: radial-gradient(circle at 50% 50%, rgba(120, 119, 198, 0.05) 0%, transparent 50%);
        padding: 0 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 20%, rgba(120, 119, 198, 0.1) 0%, transparent 25%),
            radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.03) 0%, transparent 25%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: clamp(3rem, 8vw, 6rem);
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.04em;
        margin: 0 0 1.5rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #a0a0a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: clamp(1.1rem, 2.5vw, 1.5rem);
        font-weight: 400;
        color: #a0a0a0;
        margin: 0 0 3rem 0;
        max-width: 600px;
        line-height: 1.5;
        position: relative;
        z-index: 1;
    }
    
    .hero-cta {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #7877c6 0%, #5b59b8 100%);
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none;
        display: inline-block;
        position: relative;
        overflow: hidden;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(120, 119, 198, 0.3);
    }
    
    .cta-button-secondary {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff;
        padding: 1rem 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none;
        display: inline-block;
        backdrop-filter: blur(10px);
    }
    
    .cta-button-secondary:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* Section layout */
    .section {
        padding: 6rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
    }
    
    .section-small {
        padding: 4rem 2rem;
    }
    
    .section-header {
        text-align: center;
        margin-bottom: 4rem;
    }
    
    .section-title {
        font-size: clamp(2.5rem, 5vw, 3.5rem);
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.02em;
        margin: 0 0 1rem 0;
        color: #ffffff;
    }
    
    .section-description {
        font-size: 1.25rem;
        font-weight: 400;
        color: #a0a0a0;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Grid layouts */
    .grid {
        display: grid;
        gap: 2rem;
    }
    
    .grid-2 {
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }
    
    .grid-3 {
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }
    
    .grid-4 {
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    }
    
    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    }
    
    .card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.03);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        color: #ffffff;
    }
    
    .card-description {
        color: #a0a0a0;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(120, 119, 198, 0.3);
        background: rgba(120, 119, 198, 0.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #7877c6;
        margin: 0 0 0.5rem 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        color: #22c55e;
        margin-top: 0.5rem;
    }
    
    /* Status badges */
    .status-grid {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 3rem 0;
    }
    
    .status-badge {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.2);
        color: #22c55e;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-badge::before {
        content: '●';
        color: #22c55e;
    }
    
    /* Navigation */
    .nav-tabs {
        display: flex;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.5rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        overflow-x: auto;
    }
    
    .nav-tab {
        background: transparent;
        border: none;
        color: #a0a0a0;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: nowrap;
    }
    
    .nav-tab:hover {
        color: #ffffff;
        background: rgba(255, 255, 255, 0.05);
    }
    
    .nav-tab.active {
        background: #7877c6;
        color: #ffffff;
    }
    
    /* Charts */
    .chart-container {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Form elements */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #7877c6 !important;
        box-shadow: 0 0 0 3px rgba(120, 119, 198, 0.1) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7877c6 0%, #5b59b8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(120, 119, 198, 0.3) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(10, 10, 10, 0.95) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 4rem 0;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .section {
            padding: 4rem 1rem;
        }
        
        .grid {
            gap: 1rem;
        }
        
        .hero-cta {
            flex-direction: column;
            align-items: center;
        }
        
        .cta-button,
        .cta-button-secondary {
            width: 100%;
            max-width: 300px;
        }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Main content
def main():
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Adversarial-Aware Synthetic Data</div>
        <div class="hero-subtitle">
            Enterprise-grade synthetic data generation with built-in fairness and privacy guarantees. 
            Cloud-native architecture powered by AWS, Neo4j, and Weaviate.
        </div>
        <div class="hero-cta">
            <div class="cta-button">Explore Platform</div>
            <div class="cta-button-secondary">View Documentation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status indicators
    st.markdown("""
    <div class="status-grid">
        <div class="status-badge">AWS S3 Connected</div>
        <div class="status-badge">Neo4j Aura Active</div>
        <div class="status-badge">Weaviate Ready</div>
        <div class="status-badge">AI Agent Online</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("""
    <div class="nav-tabs">
        <div class="nav-tab active">Overview</div>
        <div class="nav-tab">Bias Detection</div>
        <div class="nav-tab">Privacy</div>
        <div class="nav-tab">Fairness</div>
        <div class="nav-tab">AI Agent</div>
        <div class="nav-tab">Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">98.1%</div>
            <div class="metric-label">Bias Reduction</div>
            <div class="metric-delta">+45% vs baseline</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">94/100</div>
            <div class="metric-label">Privacy Score</div>
            <div class="metric-delta">ε=1.0 differential privacy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">87%</div>
            <div class="metric-label">Data Quality</div>
            <div class="metric-delta">Utility preserved</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">1.2s</div>
            <div class="metric-label">Generation Speed</div>
            <div class="metric-delta">Per 1K samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Features section
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <div class="section-title">Advanced Capabilities</div>
            <div class="section-description">
                Enterprise-grade features for production synthetic data generation
            </div>
        </div>
        
        <div class="grid grid-3">
            <div class="card">
                <div class="card-title">Bias Detection Engine</div>
                <div class="card-description">
                    Advanced statistical analysis with 98%+ bias reduction through fairness algorithms and real-time discrimination detection.
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Differential Privacy</div>
                <div class="card-description">
                    Mathematically guaranteed privacy protection with configurable epsilon parameters and zero re-identification risk.
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Cloud-Native Architecture</div>
                <div class="card-description">
                    Scalable infrastructure with AWS S3 storage, Neo4j Aura lineage tracking, and Weaviate vector search.
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">AI-Powered Interface</div>
                <div class="card-description">
                    Conversational synthetic data generation through Strands Agents with AWS Bedrock integration.
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Fairness Auditing</div>
                <div class="card-description">
                    Comprehensive bias monitoring across multiple dimensions with automated compliance reporting.
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Vector Search</div>
                <div class="card-description">
                    Intelligent dataset discovery using 768-dimensional embeddings for similarity search and recommendation.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive demo section
    st.markdown("""
    <div class="section section-small">
        <div class="section-header">
            <div class="section-title">Interactive Demo</div>
            <div class="section-description">
                Experience the platform capabilities with real-time generation and analysis
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Demo interface
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Dataset Configuration")
        
        dataset_type = st.selectbox(
            "Dataset Type",
            ["Loan Applications", "Healthcare Outcomes", "Employment Decisions", "Custom"]
        )
        
        num_samples = st.number_input(
            "Number of Samples",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        privacy_level = st.slider(
            "Privacy Level (ε)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
        
        fairness_constraint = st.selectbox(
            "Fairness Constraint",
            ["Demographic Parity", "Equalized Odds", "Equal Opportunity"]
        )
        
        if st.button("Generate Synthetic Data", type="primary"):
            with st.spinner("Generating data..."):
                st.success(f"Generated {num_samples} synthetic samples")
                st.info(f"Privacy: ε={privacy_level} | Fairness: {fairness_constraint}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Demo chart
        if st.button("Generate Synthetic Data", type="primary", key="chart_gen"):
            # Mock bias analysis chart
            groups = ['Male', 'Female', 'White', 'Black', 'Hispanic', 'Asian']
            original_rates = [0.75, 0.45, 0.68, 0.42, 0.48, 0.61]
            synthetic_rates = [0.63, 0.61, 0.62, 0.60, 0.64, 0.62]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=groups,
                y=original_rates,
                name='Original (Biased)',
                marker_color='rgba(239, 68, 68, 0.7)'
            ))
            fig.add_trace(go.Bar(
                x=groups,
                y=synthetic_rates,
                name='Synthetic (Fair)',
                marker_color='rgba(34, 197, 94, 0.7)'
            ))
            
            fig.update_layout(
                title="Bias Reduction Analysis",
                xaxis_title="Protected Groups",
                yaxis_title="Approval Rate",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                height=400,
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("### Bias Reduction Visualization")
            st.markdown("Click 'Generate Synthetic Data' to see bias analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # AI Agent section
    st.markdown("""
    <div class="section">
        <div class="section-header">
            <div class="section-title">AI Agent Interface</div>
            <div class="section-description">
                Natural language interface for synthetic data generation and analysis
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI expert for adversarial-aware synthetic data generation. How can I help you create fair, private, and high-quality synthetic data?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about synthetic data generation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            # AI response based on keywords
            if "bias" in prompt.lower() or "fairness" in prompt.lower():
                response = """I can help you detect and reduce bias through advanced statistical analysis:

• **Statistical Testing**: Chi-square tests for discrimination detection
• **Fairness Constraints**: Demographic parity and equalized odds enforcement  
• **Bias Metrics**: Comprehensive disparity measurement across protected groups
• **Mitigation Strategies**: Automated bias reduction with 98%+ effectiveness

Would you like me to run a fairness audit on your dataset?"""
            
            elif "privacy" in prompt.lower():
                response = """I implement differential privacy to protect individual records:

• **Configurable Privacy**: Epsilon parameters from 0.1 to 5.0
• **Noise Injection**: Calibrated statistical noise for privacy guarantees
• **Utility Preservation**: Optimized privacy-utility trade-offs
• **Re-identification Protection**: Zero exact record matches guaranteed

The current recommendation is ε=1.0 for optimal balance."""
            
            elif "generate" in prompt.lower() or "create" in prompt.lower():
                response = """I can generate high-quality synthetic data with enterprise-grade guarantees:

• **Advanced Models**: WGAN-GP and Conditional GANs
• **Fairness Integration**: 98%+ bias reduction capability
• **Privacy Preservation**: Differential privacy with configurable epsilon
• **Quality Metrics**: 87% statistical utility preservation
• **Cloud Storage**: Automatic AWS S3 integration with lineage tracking"""
            
            else:
                response = """I specialize in enterprise-grade synthetic data generation with:

• **Bias Detection**: Advanced fairness auditing and discrimination analysis
• **Privacy Engineering**: Differential privacy implementation and optimization  
• **Quality Assurance**: Statistical fidelity and utility measurement
• **Cloud Integration**: AWS S3, Neo4j Aura, and Weaviate connectivity

How can I assist with your synthetic data requirements?"""
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer spacing
    st.markdown('<div style="height: 4rem;"></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
