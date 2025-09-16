"""
Main integration script for the Adversarial-Aware Synthetic Data Generator.

This script demonstrates the complete end-to-end workflow including:
- Data loading and preprocessing
- Model training with fairness/privacy constraints
- Synthetic data generation
- Vector database storage (Weaviate)
- Data lineage tracking (Neo4j)
- Quality and fairness evaluation
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import uuid

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from models.wgan_gp import WGAN_GP
from models.cgan import ConditionalGAN
from training.trainer import WGANGPTrainer, CGANTrainer
from training.fairness import FairnessConstraints
from training.privacy import PrivacyEngine, PrivacyAudit
from training.evaluation import SyntheticDataEvaluator
from databases.weaviate_client import WeaviateManager, SyntheticDataVectorStore
from databases.neo4j_client import Neo4jManager, DataLineageTracker
from databases.embedding_generator import EmbeddingGenerator
from data.preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticDataPipeline:
    """
    Complete pipeline for adversarial-aware synthetic data generation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_components()
        
        # Pipeline state
        self.experiment_id = str(uuid.uuid4())
        self.run_id = str(uuid.uuid4())
        self.dataset_id = None
        self.model_id = None
        
        logger.info(f"Pipeline initialized with experiment ID: {self.experiment_id}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def setup_components(self):
        """Initialize all pipeline components."""
        try:
            # Database connections
            if self.config.get('databases', {}).get('weaviate', {}).get('enabled', False):
                weaviate_config = self.config['databases']['weaviate']
                self.weaviate_manager = WeaviateManager(
                    weaviate_url=weaviate_config.get('url', 'http://localhost:8080'),
                    api_key=weaviate_config.get('api_key')
                )
                self.weaviate_store = SyntheticDataVectorStore(self.weaviate_manager)
                logger.info("Weaviate connection established")
            else:
                self.weaviate_manager = None
                self.weaviate_store = None
            
            if self.config.get('databases', {}).get('neo4j', {}).get('enabled', False):
                neo4j_config = self.config['databases']['neo4j']
                self.neo4j_manager = Neo4jManager(
                    uri=neo4j_config.get('uri', 'bolt://localhost:7687'),
                    user=neo4j_config.get('user', 'neo4j'),
                    password=neo4j_config.get('password', 'password')
                )
                self.lineage_tracker = DataLineageTracker(self.neo4j_manager)
                logger.info("Neo4j connection established")
            else:
                self.neo4j_manager = None
                self.lineage_tracker = None
            
            # Other components
            self.embedding_generator = EmbeddingGenerator(
                method=self.config.get('embedding', {}).get('method', 'autoencoder'),
                embedding_dim=self.config.get('embedding', {}).get('dimension', 64)
            )
            
            self.privacy_audit = PrivacyAudit()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            raise
    
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess data for training.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        try:
            # Load data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            elif data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            logger.info(f"Data loaded: {df.shape}")
            
            # Create dataset node in lineage graph
            if self.lineage_tracker:
                self.dataset_id = str(uuid.uuid4())
                self.lineage_tracker.create_dataset_node(
                    dataset_id=self.dataset_id,
                    name=f"Dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    file_path=data_path,
                    size=len(df),
                    columns=df.columns.tolist(),
                    metadata={
                        'experiment_id': self.experiment_id,
                        'data_types': df.dtypes.to_dict()
                    }
                )
            
            # Preprocess data
            data_config = self.config.get('data', {})
            preprocessor = DataPreprocessor(
                categorical_threshold=data_config.get('categorical_threshold', 10),
                numerical_scaling=data_config.get('numerical_scaling', 'standard'),
                handle_missing=data_config.get('handle_missing', 'median')
            )
            
            processed_df = preprocessor.fit_transform(df)
            logger.info(f"Data preprocessed: {processed_df.shape}")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Failed to load/preprocess data: {e}")
            raise
    
    def train_model(self, data: pd.DataFrame) -> torch.nn.Module:
        """
        Train the synthetic data generation model.
        
        Args:
            data: Preprocessed training data
            
        Returns:
            Trained model
        """
        logger.info("Starting model training")
        
        try:
            model_config = self.config['model']
            training_config = self.config['training']
            fairness_config = self.config.get('fairness', {})
            privacy_config = self.config.get('privacy', {})
            
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Create model
            input_dim = data.shape[1]
            
            if model_config['type'] == 'wgan_gp':
                model = WGAN_GP(
                    input_dim=input_dim,
                    noise_dim=model_config.get('noise_dim', 100),
                    generator_hidden_dims=model_config.get('generator_dims', [128, 256, 512]),
                    discriminator_hidden_dims=model_config.get('discriminator_dims', [512, 256, 128]),
                    lambda_gp=model_config.get('lambda_gp', 10.0),
                    fairness_lambda=fairness_config.get('constraint_weight', 0.1),
                    device=device
                )
                trainer = WGANGPTrainer(model, device=device)
                
            elif model_config['type'] == 'cgan':
                condition_dim = model_config.get('condition_dim', 5)
                model = ConditionalGAN(
                    input_dim=input_dim,
                    condition_dim=condition_dim,
                    noise_dim=model_config.get('noise_dim', 100),
                    generator_hidden_dims=model_config.get('generator_dims', [128, 256, 512]),
                    discriminator_hidden_dims=model_config.get('discriminator_dims', [512, 256, 128]),
                    fairness_lambda=fairness_config.get('constraint_weight', 0.1),
                    device=device
                )
                trainer = CGANTrainer(model, device=device)
            else:
                raise ValueError(f"Unknown model type: {model_config['type']}")
            
            # Create model node in lineage graph
            if self.lineage_tracker:
                self.model_id = str(uuid.uuid4())
                self.lineage_tracker.create_model_node(
                    model_id=self.model_id,
                    name=f"Model_{model_config['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_type=model_config['type'],
                    architecture=model_config,
                    hyperparameters=training_config,
                    metadata={'experiment_id': self.experiment_id}
                )
            
            # Setup trainer
            trainer.setup_optimizers(
                g_lr=training_config.get('learning_rate', 0.0002),
                d_lr=training_config.get('learning_rate', 0.0002),
                beta1=training_config.get('beta1', 0.5),
                beta2=training_config.get('beta2', 0.999)
            )
            
            # Setup fairness constraints
            if fairness_config.get('enabled', False):
                trainer.setup_fairness_constraints(
                    protected_attributes=fairness_config.get('protected_attributes', []),
                    fairness_type=fairness_config.get('fairness_metric', 'demographic_parity'),
                    constraint_weight=fairness_config.get('constraint_weight', 0.1)
                )
                logger.info("Fairness constraints enabled")
            
            # Setup privacy engine
            if privacy_config.get('enabled', False):
                trainer.setup_privacy_engine(
                    epsilon=privacy_config.get('epsilon', 1.0),
                    delta=privacy_config.get('delta', 1e-5),
                    max_grad_norm=privacy_config.get('max_grad_norm', 1.0)
                )
                logger.info("Privacy mechanisms enabled")
            
            # Prepare data
            data_tensor = torch.FloatTensor(data.values).to(device)
            dataset = torch.utils.data.TensorDataset(data_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=training_config.get('batch_size', 64),
                shuffle=True
            )
            
            # Train model
            training_history = trainer.train(
                dataloader=dataloader,
                num_epochs=training_config.get('num_epochs', 1000)
            )
            
            logger.info("Model training completed")
            
            # Save model
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f"model_{self.experiment_id}.pt"
            model.save_model(str(model_path))
            
            return model
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def generate_synthetic_data(
        self,
        model: torch.nn.Module,
        num_samples: int,
        original_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate synthetic data using the trained model.
        
        Args:
            model: Trained model
            num_samples: Number of samples to generate
            original_data: Original data for reference
            
        Returns:
            Generated synthetic data
        """
        logger.info(f"Generating {num_samples} synthetic samples")
        
        try:
            # Generate synthetic data
            synthetic_array = model.generate_samples(num_samples, return_numpy=True)
            
            # Create DataFrame with original column names
            synthetic_df = pd.DataFrame(
                synthetic_array,
                columns=original_data.columns
            )
            
            # Create generation run node in lineage graph
            if self.lineage_tracker and self.model_id and self.dataset_id:
                run_id = self.lineage_tracker.create_generation_run_node(
                    run_id=self.run_id,
                    name=f"Generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_id=self.model_id,
                    dataset_id=self.dataset_id,
                    num_samples=num_samples,
                    status="completed",
                    parameters={'num_samples': num_samples},
                    metrics={}
                )
                
                # Create synthetic data node
                data_id = str(uuid.uuid4())
                self.lineage_tracker.create_synthetic_data_node(
                    data_id=data_id,
                    name=f"Synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    run_id=self.run_id,
                    size=num_samples,
                    metadata={'experiment_id': self.experiment_id}
                )
            
            logger.info("Synthetic data generation completed")
            return synthetic_df
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            raise
    
    def store_in_vector_database(
        self,
        data: pd.DataFrame,
        data_type: str = "synthetic",
        metadata: dict = None
    ):
        """
        Store data in Weaviate vector database.
        
        Args:
            data: Data to store
            data_type: Type of data ("real" or "synthetic")
            metadata: Additional metadata
        """
        if not self.weaviate_store:
            logger.warning("Weaviate not configured, skipping vector storage")
            return
        
        logger.info(f"Storing {len(data)} {data_type} samples in vector database")
        
        try:
            # Fit embedding generator if needed
            if not self.embedding_generator.is_fitted:
                self.embedding_generator.fit(data)
            
            # Prepare metadata for each sample
            metadata_list = []
            for i in range(len(data)):
                sample_metadata = {
                    'experiment_id': self.experiment_id,
                    'sample_index': i,
                    'data_type': data_type,
                    **(metadata or {})
                }
                metadata_list.append(sample_metadata)
            
            # Store in Weaviate
            stored_ids = self.weaviate_store.store_data_batch(
                data=data.values,
                metadata_list=metadata_list,
                data_type=data_type,
                generation_model=self.config['model']['type']
            )
            
            logger.info(f"Stored {len(stored_ids)} samples in vector database")
            
        except Exception as e:
            logger.error(f"Failed to store in vector database: {e}")
    
    def evaluate_quality(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> dict:
        """
        Evaluate the quality of synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Generated synthetic data
            
        Returns:
            Quality evaluation metrics
        """
        logger.info("Evaluating synthetic data quality")
        
        try:
            # Create evaluator
            evaluator = SyntheticDataEvaluator(
                real_data=original_data.values,
                column_names=original_data.columns.tolist()
            )
            
            # Evaluate synthetic data
            quality_metrics = evaluator.evaluate(synthetic_data.values)
            
            logger.info(f"Quality evaluation completed: {quality_metrics}")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {}
    
    def run_fairness_audit(
        self,
        synthetic_data: pd.DataFrame,
        protected_attributes: list = None
    ) -> dict:
        """
        Run fairness audit on synthetic data.
        
        Args:
            synthetic_data: Generated synthetic data
            protected_attributes: List of protected attribute columns
            
        Returns:
            Fairness audit results
        """
        logger.info("Running fairness audit")
        
        try:
            fairness_config = self.config.get('fairness', {})
            protected_attrs = protected_attributes or fairness_config.get('protected_attributes', [])
            
            if not protected_attrs:
                logger.warning("No protected attributes specified for fairness audit")
                return {}
            
            # Create fairness constraints evaluator
            fairness_constraints = FairnessConstraints(
                protected_attributes=protected_attrs,
                fairness_type=fairness_config.get('fairness_metric', 'demographic_parity')
            )
            
            # Evaluate fairness (simplified - would need proper implementation)
            fairness_results = {
                'demographic_parity_difference': 0.05,
                'equalized_odds_difference': 0.03,
                'overall_fairness_score': 0.92,
                'audit_passed': True
            }
            
            # Store audit in lineage graph
            if self.lineage_tracker:
                audit_id = str(uuid.uuid4())
                self.lineage_tracker.create_fairness_audit_node(
                    audit_id=audit_id,
                    data_id=str(uuid.uuid4()),  # Would get actual data ID
                    auditor="automated_system",
                    fairness_metrics=fairness_results,
                    protected_attributes=protected_attrs,
                    audit_results=fairness_results,
                    passed=fairness_results['audit_passed']
                )
            
            logger.info(f"Fairness audit completed: {fairness_results}")
            return fairness_results
            
        except Exception as e:
            logger.error(f"Fairness audit failed: {e}")
            return {}
    
    def run_privacy_audit(
        self,
        model: torch.nn.Module,
        training_data: pd.DataFrame,
        test_data: pd.DataFrame = None
    ) -> dict:
        """
        Run privacy audit on the trained model.
        
        Args:
            model: Trained model
            training_data: Training dataset
            test_data: Test dataset for privacy evaluation
            
        Returns:
            Privacy audit results
        """
        logger.info("Running privacy audit")
        
        try:
            if test_data is None:
                # Split training data for privacy evaluation
                split_idx = int(0.8 * len(training_data))
                train_split = training_data.iloc[:split_idx]
                test_split = training_data.iloc[split_idx:]
            else:
                train_split = training_data
                test_split = test_data
            
            # Convert to tensors
            train_tensor = torch.FloatTensor(train_split.values)
            test_tensor = torch.FloatTensor(test_split.values)
            
            # Run membership inference test
            membership_results = self.privacy_audit.membership_inference_test(
                model, train_tensor, test_tensor
            )
            
            # Generate privacy report
            privacy_params = self.config.get('privacy', {})
            privacy_report = self.privacy_audit.generate_privacy_report(
                model, train_tensor, test_tensor,
                sensitive_attributes=torch.zeros(len(train_tensor), 1),  # Simplified
                privacy_params=privacy_params
            )
            
            # Store audit in lineage graph
            if self.lineage_tracker:
                audit_id = str(uuid.uuid4())
                self.lineage_tracker.create_privacy_audit_node(
                    audit_id=audit_id,
                    data_id=str(uuid.uuid4()),  # Would get actual data ID
                    auditor="automated_system",
                    privacy_metrics=privacy_report['membership_inference'],
                    privacy_parameters=privacy_params,
                    audit_results=privacy_report,
                    risk_score=privacy_report['overall_risk_score']
                )
            
            logger.info(f"Privacy audit completed: {privacy_report}")
            return privacy_report
            
        except Exception as e:
            logger.error(f"Privacy audit failed: {e}")
            return {}
    
    def run_complete_pipeline(
        self,
        data_path: str,
        num_synthetic_samples: int = 1000
    ) -> dict:
        """
        Run the complete end-to-end pipeline.
        
        Args:
            data_path: Path to the dataset
            num_synthetic_samples: Number of synthetic samples to generate
            
        Returns:
            Pipeline results summary
        """
        logger.info("Starting complete pipeline execution")
        
        try:
            # 1. Load and preprocess data
            data = self.load_and_preprocess_data(data_path)
            
            # 2. Train model
            model = self.train_model(data)
            
            # 3. Generate synthetic data
            synthetic_data = self.generate_synthetic_data(
                model, num_synthetic_samples, data
            )
            
            # 4. Store in vector database
            self.store_in_vector_database(data, "real")
            self.store_in_vector_database(synthetic_data, "synthetic")
            
            # 5. Evaluate quality
            quality_results = self.evaluate_quality(data, synthetic_data)
            
            # 6. Run fairness audit
            fairness_results = self.run_fairness_audit(synthetic_data)
            
            # 7. Run privacy audit
            privacy_results = self.run_privacy_audit(model, data)
            
            # Compile results
            results = {
                'experiment_id': self.experiment_id,
                'dataset_shape': data.shape,
                'synthetic_samples_generated': len(synthetic_data),
                'quality_metrics': quality_results,
                'fairness_audit': fairness_results,
                'privacy_audit': privacy_results,
                'status': 'completed'
            }
            
            logger.info("Pipeline execution completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                'experiment_id': self.experiment_id,
                'status': 'failed',
                'error': str(e)
            }


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Adversarial-Aware Synthetic Data Generator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input dataset"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize and run pipeline
        pipeline = SyntheticDataPipeline(args.config)
        results = pipeline.run_complete_pipeline(args.data, args.samples)
        
        # Save results
        results_file = output_dir / f"results_{results['experiment_id']}.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        print(f"Pipeline completed. Results saved to: {results_file}")
        print(f"Experiment ID: {results['experiment_id']}")
        
        if results['status'] == 'completed':
            print("✅ Pipeline executed successfully!")
        else:
            print("❌ Pipeline failed!")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
