"""
Discriminator networks for adversarial training.

Implements both unconditional and conditional discriminators with advanced
architectures including gradient penalty support for WGAN-GP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class Discriminator(nn.Module):
    """
    Unconditional Discriminator for adversarial training.
    
    Designed specifically for tabular data with spectral normalization
    and advanced regularization techniques.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.3,
        spectral_norm: bool = True,
        leaky_relu_slope: float = 0.2
    ):
        super(Discriminator, self).__init__()
        
        self.input_dim = input_dim
        self.spectral_norm = spectral_norm
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            
            # Apply spectral normalization for training stability
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
                
            layers.append(linear)
            
            # Skip batch norm in first layer and use LeakyReLU
            if i > 0:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
            
        # Output layer (no activation for WGAN-GP)
        output_linear = nn.Linear(prev_dim, 1)
        if spectral_norm:
            output_linear = nn.utils.spectral_norm(output_linear)
        layers.append(output_linear)
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.
        
        Args:
            x: Input data tensor of shape (batch_size, input_dim)
            
        Returns:
            Discriminator output (real/fake score)
        """
        return self.network(x)


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator for supervised adversarial training.
    
    Takes both data and conditioning information to make more informed
    real/fake decisions.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        condition_dim: int = 5,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.3,
        spectral_norm: bool = True,
        leaky_relu_slope: float = 0.2,
        embedding_dim: Optional[int] = None
    ):
        super(ConditionalDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.spectral_norm = spectral_norm
        
        # Embedding for categorical conditions (optional)
        if embedding_dim is not None:
            self.condition_embedding = nn.Embedding(condition_dim, embedding_dim)
            total_input_dim = input_dim + embedding_dim
        else:
            self.condition_embedding = None
            total_input_dim = input_dim + condition_dim
        
        # Build the network
        layers = []
        prev_dim = total_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(prev_dim, hidden_dim)
            
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
                
            layers.append(linear)
            
            if i > 0:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
            
        # Output layer
        output_linear = nn.Linear(prev_dim, 1)
        if spectral_norm:
            output_linear = nn.utils.spectral_norm(output_linear)
        layers.append(output_linear)
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the conditional discriminator.
        
        Args:
            x: Input data tensor of shape (batch_size, input_dim)
            conditions: Condition tensor of shape (batch_size, condition_dim)
            
        Returns:
            Discriminator output conditioned on the inputs
        """
        if self.condition_embedding is not None:
            conditions = self.condition_embedding(conditions.long())
            conditions = conditions.view(conditions.size(0), -1)
        
        # Concatenate data and conditions
        discriminator_input = torch.cat([x, conditions], dim=1)
        return self.network(discriminator_input)


class FairnessAwareDiscriminator(nn.Module):
    """
    Fairness-aware discriminator that includes additional heads for
    protected attribute prediction to enforce fairness constraints.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        protected_attrs_dims: List[int] = [2],  # Number of classes for each protected attribute
        hidden_dims: List[int] = [512, 256, 128],
        shared_layers: int = 2,  # Number of shared layers before branching
        dropout_rate: float = 0.3,
        spectral_norm: bool = True,
        leaky_relu_slope: float = 0.2
    ):
        super(FairnessAwareDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.protected_attrs_dims = protected_attrs_dims
        self.shared_layers = shared_layers
        
        # Shared feature extractor
        shared_layers_list = []
        prev_dim = input_dim
        
        for i in range(shared_layers):
            if i < len(hidden_dims):
                hidden_dim = hidden_dims[i]
                linear = nn.Linear(prev_dim, hidden_dim)
                
                if spectral_norm:
                    linear = nn.utils.spectral_norm(linear)
                    
                shared_layers_list.append(linear)
                
                if i > 0:
                    shared_layers_list.append(nn.BatchNorm1d(hidden_dim))
                shared_layers_list.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
                shared_layers_list.append(nn.Dropout(dropout_rate))
                
                prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers_list)
        
        # Real/fake discriminator head
        discriminator_layers = []
        for i in range(shared_layers, len(hidden_dims)):
            hidden_dim = hidden_dims[i]
            linear = nn.Linear(prev_dim, hidden_dim)
            
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
                
            discriminator_layers.extend([
                linear,
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(leaky_relu_slope, inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # Final discriminator output
        final_linear = nn.Linear(prev_dim, 1)
        if spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        discriminator_layers.append(final_linear)
        
        self.discriminator_head = nn.Sequential(*discriminator_layers)
        
        # Protected attribute prediction heads
        self.protected_heads = nn.ModuleList()
        for attr_dim in protected_attrs_dims:
            head = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.BatchNorm1d(prev_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(prev_dim // 2, attr_dim)
            )
            self.protected_heads.append(head)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning both discriminator and protected attribute predictions.
        
        Args:
            x: Input data tensor
            
        Returns:
            Tuple of (discriminator_output, protected_attribute_predictions)
        """
        # Shared feature extraction
        shared_features = self.shared_network(x)
        
        # Real/fake discrimination
        discriminator_output = self.discriminator_head(shared_features)
        
        # Protected attribute predictions
        protected_outputs = []
        for head in self.protected_heads:
            protected_outputs.append(head(shared_features))
        
        return discriminator_output, protected_outputs


class SelfAttentionDiscriminator(nn.Module):
    """
    Advanced discriminator with self-attention mechanism for capturing
    complex feature relationships in tabular data.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout_rate: float = 0.3,
        spectral_norm: bool = True
    ):
        super(SelfAttentionDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        if spectral_norm:
            self.input_projection = nn.utils.spectral_norm(self.input_projection)
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        if spectral_norm:
            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    layer = nn.utils.spectral_norm(layer)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Discriminator output
        """
        # Project to hidden dimension and add sequence dimension
        x = self.input_projection(x)  # (batch_size, hidden_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Apply self-attention layers
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention with residual connection
            attn_output, _ = attention(x, x, x)
            x = layer_norm(x + attn_output)
        
        # Remove sequence dimension and classify
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        return self.classifier(x)
