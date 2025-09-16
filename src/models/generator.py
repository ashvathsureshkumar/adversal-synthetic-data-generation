"""
Generator networks for synthetic data generation.

Implements both unconditional and conditional generators optimized for tabular data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class Generator(nn.Module):
    """
    Unconditional Generator for tabular data synthesis.
    
    Args:
        noise_dim: Dimension of input noise vector
        output_dim: Dimension of generated data (number of features)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
        batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        output_dim: int = 10,
        hidden_dims: List[int] = [128, 256, 512],
        dropout_rate: float = 0.2,
        batch_norm: bool = True
    ):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        
        # Build the network layers
        layers = []
        prev_dim = noise_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Assuming normalized data in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.
        
        Args:
            noise: Random noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            Generated data tensor of shape (batch_size, output_dim)
        """
        return self.network(noise)
    
    def generate_samples(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples tensor
        """
        self.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim, device=device)
            samples = self.forward(noise)
        return samples


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator for controlled synthetic data generation.
    
    Allows conditioning on specific attributes for targeted data synthesis.
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        output_dim: int = 10,
        condition_dim: int = 5,
        hidden_dims: List[int] = [128, 256, 512],
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        embedding_dim: Optional[int] = None
    ):
        super(ConditionalGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        
        # Embedding for categorical conditions (optional)
        if embedding_dim is not None:
            self.condition_embedding = nn.Embedding(condition_dim, embedding_dim)
            input_dim = noise_dim + embedding_dim
        else:
            self.condition_embedding = None
            input_dim = noise_dim + condition_dim
        
        # Build the network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
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
    
    def forward(self, noise: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the conditional generator.
        
        Args:
            noise: Random noise tensor of shape (batch_size, noise_dim)
            conditions: Condition tensor of shape (batch_size, condition_dim)
            
        Returns:
            Generated data tensor conditioned on the inputs
        """
        if self.condition_embedding is not None:
            # Assume conditions are categorical indices
            conditions = self.condition_embedding(conditions.long())
            conditions = conditions.view(conditions.size(0), -1)
        
        # Concatenate noise and conditions
        generator_input = torch.cat([noise, conditions], dim=1)
        return self.network(generator_input)
    
    def generate_conditional_samples(
        self,
        num_samples: int,
        conditions: torch.Tensor,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Generate samples conditioned on specific attributes.
        
        Args:
            num_samples: Number of samples to generate
            conditions: Condition tensor for each sample
            device: Device to generate samples on
            
        Returns:
            Conditionally generated samples
        """
        self.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim, device=device)
            if conditions.size(0) == 1:
                conditions = conditions.repeat(num_samples, 1)
            samples = self.forward(noise, conditions)
        return samples


class ResidualBlock(nn.Module):
    """Residual block for deeper generator architectures."""
    
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class DeepGenerator(Generator):
    """
    Deep Generator with residual connections for complex data distributions.
    """
    
    def __init__(
        self,
        noise_dim: int = 100,
        output_dim: int = 10,
        hidden_dim: int = 512,
        num_residual_blocks: int = 3,
        dropout_rate: float = 0.2
    ):
        super(Generator, self).__init__()  # Skip parent __init__
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        
        # Initial projection
        self.initial = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(num_residual_blocks)
        ])
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = self.initial(noise)
        
        for block in self.residual_blocks:
            x = block(x)
            
        return self.output(x)
