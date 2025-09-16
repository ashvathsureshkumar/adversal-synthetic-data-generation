# Adversarial-Aware Synthetic Data Generator

A comprehensive, hackathon-ready synthetic data generator that combines state-of-the-art adversarial training with fairness constraints, privacy preservation, and modern data infrastructure.

##  Features

- ** Advanced GAN Models**: cGAN/WGAN-GP with gradient penalty for high-quality synthetic tabular data
- **️ Fairness & Privacy**: Built-in constraints during training to ensure responsible AI
- **️ Vector Database**: Weaviate integration for embedding storage and similarity search
- ** Graph Database**: Neo4j for comprehensive data lineage tracking and audit trails
- **️ Cloud Ready**: Full AWS integration for scalable training and deployment (SageMaker, S3, EC2)
- ** Interactive Dashboard**: Beautiful Streamlit-powered UI for easy data generation and visualization
- ** Agent Framework**: Strands Agents SDK integration for workflow automation
- ** Privacy Preservation**: Differential privacy with comprehensive audit capabilities
- ** Quality Evaluation**: Extensive metrics for synthetic data quality assessment

## ‍️ Quick Start

### Option 1: Interactive Dashboard (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch dashboard
python run_dashboard.py
```
Navigate to `http://localhost:8501` and upload your dataset to get started!

### Option 2: Command Line Interface
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with sample data
python src/main.py --data examples/sample_data.csv --config config/config.yaml --samples 1000
```

### Option 3: Python API
```python
from src.main import SyntheticDataPipeline

# Initialize pipeline
pipeline = SyntheticDataPipeline("config/config.yaml")

# Run complete workflow
results = pipeline.run_complete_pipeline("your_data.csv", num_synthetic_samples=1000)
```

##  Prerequisites

- Python 3.8+
- Optional: Docker (for Weaviate/Neo4j)
- Optional: AWS account (for cloud features)

## Project Structure

```
adversal-synthetic-data/
├── src/                    # Core implementation
│   ├── models/            # GAN architectures
│   ├── training/          # Training loops and fairness constraints
│   ├── data/              # Data processing utilities
│   ├── databases/         # Weaviate and Neo4j integrations
│   └── agents/            # Strands Agents integration
├── aws/                   # AWS deployment scripts
├── notebooks/             # Jupyter notebooks for experimentation
├── tests/                 # Unit tests
├── config/                # Configuration files
└── streamlit_app.py       # Main dashboard application
```

##  Advanced Setup (Optional)

### Database Services (For Full Features)
```bash
# Start Weaviate (Vector Database)
docker run -d -p 8080:8080 --name weaviate semitechnologies/weaviate:latest

# Start Neo4j (Graph Database)
docker run -d -p 7474:7474 -p 7687:7687 --name neo4j \
  -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### AWS Configuration (For Cloud Features)
```bash
# Configure AWS credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
```

##  Key Components

### 1. **GAN Models** (`src/models/`)
- **WGAN-GP**: Wasserstein GAN with Gradient Penalty for stable training
- **Conditional GAN**: For controlled generation with specific conditions
- **Fairness Integration**: Built-in demographic parity and equalized odds constraints

### 2. **Training Pipeline** (`src/training/`)
- **Differential Privacy**: Opacus integration for privacy-preserving training
- **Fairness Constraints**: Multiple fairness metrics and constraint types
- **Quality Evaluation**: Comprehensive synthetic data quality assessment

### 3. **Database Integration** (`src/databases/`)
- **Weaviate**: Vector embeddings for similarity search and data exploration
- **Neo4j**: Complete data lineage tracking and audit trails
- **Embedding Generation**: Multiple methods including autoencoders and PCA

### 4. **Cloud Infrastructure** (`aws/`)
- **SageMaker**: Scalable model training and deployment
- **S3**: Secure data and model storage with versioning
- **EC2**: Custom training clusters for large datasets

### 5. **Dashboard** (`streamlit_app.py`)
- **Interactive UI**: Upload, train, generate, and evaluate
- **Real-time Visualization**: Training progress and quality metrics
- **Fairness Audits**: Built-in bias detection and mitigation tools

##  Example Usage

### Basic Synthetic Data Generation
```python
import pandas as pd
from src.main import SyntheticDataPipeline

# Load your data
df = pd.read_csv("your_dataset.csv")

# Create and run pipeline
pipeline = SyntheticDataPipeline("config/config.yaml")
results = pipeline.run_complete_pipeline("your_dataset.csv")

# Access generated data
synthetic_data = results['synthetic_data']
quality_score = results['quality_metrics']['overall_score']
fairness_passed = results['fairness_audit']['audit_passed']
```

### Advanced Configuration
```python
# Custom model configuration
config = {
    'model': {
        'type': 'wgan_gp',
        'noise_dim': 100,
        'lambda_gp': 10.0
    },
    'fairness': {
        'enabled': True,
        'protected_attributes': ['gender', 'race'],
        'fairness_metric': 'demographic_parity'
    },
    'privacy': {
        'enabled': True,
        'epsilon': 1.0  # Differential privacy budget
    }
}
```

##  Hackathon Ready Features

- ** Quick Setup**: Get running in under 5 minutes
- ** Beautiful UI**: Professional Streamlit dashboard
- ** End-to-End**: Complete workflow from upload to download
- ** Rich Visualizations**: Training curves, quality metrics, fairness audits
- **️ Cloud Integration**: Deploy to AWS with one command
- ** Agent Integration**: Ready for workflow automation
- ** Demo Ready**: Built-in sample data and configurations

## ️ Technology Stack

- ** ML Framework**: PyTorch with GPU support
- **️ Vector Database**: Weaviate for embeddings
- ** Graph Database**: Neo4j for lineage tracking
- **️ Cloud Platform**: AWS (SageMaker, EC2, S3)
- ** Frontend**: Streamlit with modern UI
- ** Agent Framework**: Strands Agents SDK
- ** Privacy**: Opacus for differential privacy
- **️ Fairness**: Fairlearn and custom implementations

##  Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

##  License

MIT License - See [LICENSE](LICENSE) file for details.

##  Acknowledgments

Built for hackathons with ️ using state-of-the-art ML and modern data infrastructure.
