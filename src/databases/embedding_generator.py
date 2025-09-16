"""
Embedding generation for tabular data vectorization.

This module provides various methods for converting tabular data into vector
embeddings suitable for storage in vector databases like Weaviate.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from dataclasses import dataclass
import warnings
import logging


@dataclass
class DataEmbedding:
    """Container for data embeddings with metadata."""
    embeddings: np.ndarray
    method: str
    dimension: int
    metadata: Dict[str, Any]
    preprocessing_info: Dict[str, Any]


class TabularAutoEncoder(nn.Module):
    """
    Neural network autoencoder for learning tabular data embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 96],
        dropout_rate: float = 0.1
    ):
        super(TabularAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final encoding layer
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final decoding layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input data
            
        Returns:
            Tuple of (embeddings, reconstructed_data)
        """
        embeddings = self.encoder(x)
        reconstructed = self.decoder(embeddings)
        return embeddings, reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings only."""
        return self.encoder(x)


class EmbeddingGenerator:
    """
    Main class for generating embeddings from tabular data using various methods.
    """
    
    def __init__(
        self,
        method: str = "autoencoder",
        embedding_dim: int = 64,
        random_state: int = 42
    ):
        self.method = method
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        
        # Components
        self.scaler = None
        self.label_encoders = {}
        self.autoencoder = None
        self.pca = None
        self.tsne = None
        
        # Training state
        self.is_fitted = False
        self.preprocessing_info = {}
        
        self.logger = logging.getLogger(__name__)
    
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        categorical_columns: Optional[List[str]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> 'EmbeddingGenerator':
        """
        Fit the embedding generator on training data.
        
        Args:
            data: Training data
            categorical_columns: List of categorical column names (if DataFrame)
            epochs: Training epochs for autoencoder
            batch_size: Batch size for training
            learning_rate: Learning rate for autoencoder
            
        Returns:
            Self for method chaining
        """
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
        
        # Store preprocessing info
        self.preprocessing_info = {
            "original_shape": data.shape,
            "column_names": list(data.columns),
            "categorical_columns": categorical_columns or [],
            "method": self.method
        }
        
        # Preprocess data
        processed_data = self._preprocess_data(data, categorical_columns, fit_transformers=True)
        
        # Fit embedding method
        if self.method == "autoencoder":
            self._fit_autoencoder(processed_data, epochs, batch_size, learning_rate)
        elif self.method == "pca":
            self._fit_pca(processed_data)
        elif self.method == "tsne":
            self._fit_tsne(processed_data)
        elif self.method == "statistical":
            # Statistical method doesn't require fitting
            pass
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")
        
        self.is_fitted = True
        self.logger.info(f"EmbeddingGenerator fitted with method: {self.method}")
        return self
    
    def _preprocess_data(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        fit_transformers: bool = False
    ) -> np.ndarray:
        """
        Preprocess tabular data for embedding generation.
        
        Args:
            data: Input data
            categorical_columns: List of categorical columns
            fit_transformers: Whether to fit transformers or use existing ones
            
        Returns:
            Preprocessed data as numpy array
        """
        processed_data = data.copy()
        
        # Handle categorical columns
        if categorical_columns:
            for col in categorical_columns:
                if col in processed_data.columns:
                    if fit_transformers:
                        if col not in self.label_encoders:
                            self.label_encoders[col] = LabelEncoder()
                        processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col])
                    else:
                        if col in self.label_encoders:
                            # Handle unseen categories
                            unique_vals = set(processed_data[col].unique())
                            known_vals = set(self.label_encoders[col].classes_)
                            unseen_vals = unique_vals - known_vals
                            
                            if unseen_vals:
                                warnings.warn(f"Unseen categories in {col}: {unseen_vals}")
                                # Map unseen values to most frequent class
                                most_frequent = self.label_encoders[col].classes_[0]
                                processed_data[col] = processed_data[col].replace(list(unseen_vals), most_frequent)
                            
                            processed_data[col] = self.label_encoders[col].transform(processed_data[col])
        
        # Handle missing values
        processed_data = processed_data.fillna(processed_data.mean(numeric_only=True))
        
        # Scale numerical features
        if fit_transformers:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(processed_data)
        else:
            if self.scaler is not None:
                scaled_data = self.scaler.transform(processed_data)
            else:
                scaled_data = processed_data.values
        
        return scaled_data
    
    def _fit_autoencoder(
        self,
        data: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float
    ):
        """Fit autoencoder for embedding generation."""
        input_dim = data.shape[1]
        
        # Initialize autoencoder
        hidden_dims = [min(256, input_dim * 2), max(64, input_dim)]
        self.autoencoder = TabularAutoEncoder(
            input_dim=input_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=hidden_dims
        )
        
        # Training setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Convert data to tensor
        data_tensor = torch.FloatTensor(data).to(device)
        
        # Training loop
        self.autoencoder.train()
        for epoch in range(epochs):
            # Create batches
            indices = torch.randperm(len(data_tensor))
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(data_tensor), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_data = data_tensor[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                embeddings, reconstructed = self.autoencoder(batch_data)
                loss = criterion(reconstructed, batch_data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if epoch % 20 == 0:
                avg_loss = epoch_loss / num_batches
                self.logger.debug(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.autoencoder.eval()
    
    def _fit_pca(self, data: np.ndarray):
        """Fit PCA for dimensionality reduction."""
        self.pca = PCA(n_components=self.embedding_dim, random_state=self.random_state)
        self.pca.fit(data)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        self.logger.info(f"PCA explained variance: {explained_variance:.3f}")
    
    def _fit_tsne(self, data: np.ndarray):
        """Fit t-SNE for nonlinear dimensionality reduction."""
        # For large datasets, use PCA as preprocessing
        if data.shape[1] > 50:
            pca = PCA(n_components=50, random_state=self.random_state)
            data = pca.fit_transform(data)
        
        self.tsne = TSNE(
            n_components=self.embedding_dim,
            random_state=self.random_state,
            perplexity=min(30, len(data) - 1)
        )
        
        # Note: t-SNE doesn't have a direct transform method,
        # so we'll store the fitted data for reference
        self._tsne_training_data = data
    
    def generate_embeddings(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        categorical_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate embeddings for new data.
        
        Args:
            data: Input data to embed
            categorical_columns: List of categorical columns
            
        Returns:
            Generated embeddings
        """
        if not self.is_fitted:
            raise RuntimeError("EmbeddingGenerator must be fitted before generating embeddings")
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.preprocessing_info["column_names"])
        
        # Preprocess data using fitted transformers
        processed_data = self._preprocess_data(data, categorical_columns, fit_transformers=False)
        
        # Generate embeddings based on method
        if self.method == "autoencoder":
            return self._generate_autoencoder_embeddings(processed_data)
        elif self.method == "pca":
            return self._generate_pca_embeddings(processed_data)
        elif self.method == "tsne":
            return self._generate_tsne_embeddings(processed_data)
        elif self.method == "statistical":
            return self._generate_statistical_embeddings(processed_data)
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")
    
    def _generate_autoencoder_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Generate embeddings using trained autoencoder."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_tensor = torch.FloatTensor(data).to(device)
        
        with torch.no_grad():
            embeddings = self.autoencoder.encode(data_tensor)
            return embeddings.cpu().numpy()
    
    def _generate_pca_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Generate embeddings using PCA."""
        return self.pca.transform(data)
    
    def _generate_tsne_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Generate embeddings using t-SNE."""
        # For t-SNE, we need to refit for new data since it doesn't have transform
        # This is computationally expensive, so we'll use a simplified approach
        warnings.warn("t-SNE doesn't support transform on new data. Using approximate method.")
        
        # Combine with training data and transform, then extract new samples
        combined_data = np.vstack([self._tsne_training_data, data])
        
        tsne = TSNE(
            n_components=self.embedding_dim,
            random_state=self.random_state,
            perplexity=min(30, len(combined_data) - 1)
        )
        combined_embeddings = tsne.fit_transform(combined_data)
        
        # Return only the embeddings for new data
        return combined_embeddings[len(self._tsne_training_data):]
    
    def _generate_statistical_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Generate embeddings using statistical features."""
        n_samples = data.shape[0]
        
        # Calculate various statistical features
        features = []
        
        # Basic statistics
        features.append(np.mean(data, axis=1))  # Mean
        features.append(np.std(data, axis=1))   # Standard deviation
        features.append(np.median(data, axis=1))  # Median
        features.append(np.percentile(data, 25, axis=1))  # Q1
        features.append(np.percentile(data, 75, axis=1))  # Q3
        features.append(np.min(data, axis=1))   # Min
        features.append(np.max(data, axis=1))   # Max
        
        # Higher-order moments
        from scipy.stats import skew, kurtosis
        features.append(skew(data, axis=1))     # Skewness
        features.append(kurtosis(data, axis=1)) # Kurtosis
        
        # Combine features
        statistical_embeddings = np.column_stack(features)
        
        # Pad or truncate to desired embedding dimension
        if statistical_embeddings.shape[1] < self.embedding_dim:
            # Pad with zeros
            padding = np.zeros((n_samples, self.embedding_dim - statistical_embeddings.shape[1]))
            statistical_embeddings = np.hstack([statistical_embeddings, padding])
        elif statistical_embeddings.shape[1] > self.embedding_dim:
            # Use PCA to reduce dimensionality
            pca = PCA(n_components=self.embedding_dim, random_state=self.random_state)
            statistical_embeddings = pca.fit_transform(statistical_embeddings)
        
        return statistical_embeddings
    
    def create_data_embedding(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> DataEmbedding:
        """
        Create a DataEmbedding object with embeddings and metadata.
        
        Args:
            data: Input data
            metadata: Additional metadata
            categorical_columns: List of categorical columns
            
        Returns:
            DataEmbedding object
        """
        embeddings = self.generate_embeddings(data, categorical_columns)
        
        return DataEmbedding(
            embeddings=embeddings,
            method=self.method,
            dimension=self.embedding_dim,
            metadata=metadata or {},
            preprocessing_info=self.preprocessing_info
        )
    
    def save_model(self, filepath: str):
        """Save the fitted embedding generator."""
        import pickle
        
        state = {
            "method": self.method,
            "embedding_dim": self.embedding_dim,
            "random_state": self.random_state,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "preprocessing_info": self.preprocessing_info,
            "is_fitted": self.is_fitted
        }
        
        # Save method-specific models
        if self.method == "autoencoder" and self.autoencoder is not None:
            state["autoencoder_state"] = self.autoencoder.state_dict()
        elif self.method == "pca" and self.pca is not None:
            state["pca"] = self.pca
        elif self.method == "tsne" and hasattr(self, "_tsne_training_data"):
            state["tsne_training_data"] = self._tsne_training_data
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        
        self.logger.info(f"EmbeddingGenerator saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted embedding generator."""
        import pickle
        
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        
        self.method = state["method"]
        self.embedding_dim = state["embedding_dim"]
        self.random_state = state["random_state"]
        self.scaler = state["scaler"]
        self.label_encoders = state["label_encoders"]
        self.preprocessing_info = state["preprocessing_info"]
        self.is_fitted = state["is_fitted"]
        
        # Load method-specific models
        if self.method == "autoencoder" and "autoencoder_state" in state:
            input_dim = len(self.preprocessing_info["column_names"])
            hidden_dims = [min(256, input_dim * 2), max(64, input_dim)]
            self.autoencoder = TabularAutoEncoder(
                input_dim=input_dim,
                embedding_dim=self.embedding_dim,
                hidden_dims=hidden_dims
            )
            self.autoencoder.load_state_dict(state["autoencoder_state"])
            self.autoencoder.eval()
        elif self.method == "pca" and "pca" in state:
            self.pca = state["pca"]
        elif self.method == "tsne" and "tsne_training_data" in state:
            self._tsne_training_data = state["tsne_training_data"]
        
        self.logger.info(f"EmbeddingGenerator loaded from {filepath}")
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding generator."""
        info = {
            "method": self.method,
            "embedding_dimension": self.embedding_dim,
            "is_fitted": self.is_fitted,
            "preprocessing_info": self.preprocessing_info
        }
        
        if self.method == "pca" and self.pca is not None:
            info["explained_variance_ratio"] = self.pca.explained_variance_ratio_.sum()
        elif self.method == "autoencoder" and self.autoencoder is not None:
            info["autoencoder_params"] = sum(p.numel() for p in self.autoencoder.parameters())
        
        return info
