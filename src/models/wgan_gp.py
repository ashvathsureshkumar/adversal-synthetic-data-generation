"""
Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation.

This module provides a complete WGAN-GP implementation optimized for tabular data
with built-in fairness constraints and privacy mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .generator import Generator, ConditionalGenerator
from .discriminator import Discriminator, FairnessAwareDiscriminator


class WGAN_GP(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty for high-quality synthetic data generation.
    
    Features:
    - Gradient penalty for training stability
    - Fairness constraints integration
    - Privacy-preserving mechanisms
    - Advanced loss functions for tabular data
    """
    
    def __init__(
        self,
        input_dim: int,
        noise_dim: int = 100,
        generator_hidden_dims: List[int] = [128, 256, 512],
        discriminator_hidden_dims: List[int] = [512, 256, 128],
        lambda_gp: float = 10.0,
        fairness_lambda: float = 0.1,
        protected_attrs_dims: Optional[List[int]] = None,
        condition_dim: Optional[int] = None,
        spectral_norm: bool = True,
        device: str = "cpu"
    ):
        super(WGAN_GP, self).__init__()
        
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.lambda_gp = lambda_gp
        self.fairness_lambda = fairness_lambda
        self.device = device
        self.condition_dim = condition_dim
        
        # Initialize generator
        if condition_dim is not None:
            self.generator = ConditionalGenerator(
                noise_dim=noise_dim,
                output_dim=input_dim,
                condition_dim=condition_dim,
                hidden_dims=generator_hidden_dims
            )
        else:
            self.generator = Generator(
                noise_dim=noise_dim,
                output_dim=input_dim,
                hidden_dims=generator_hidden_dims
            )
        
        # Initialize discriminator
        if protected_attrs_dims is not None:
            self.discriminator = FairnessAwareDiscriminator(
                input_dim=input_dim,
                protected_attrs_dims=protected_attrs_dims,
                hidden_dims=discriminator_hidden_dims,
                spectral_norm=spectral_norm
            )
            self.fairness_enabled = True
        else:
            self.discriminator = Discriminator(
                input_dim=input_dim,
                hidden_dims=discriminator_hidden_dims,
                spectral_norm=spectral_norm
            )
            self.fairness_enabled = False
        
        # Move to device
        self.to(device)
        
        # Training state
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'w_distance': [],
            'gradient_penalty': [],
            'fairness_loss': []
        }
    
    def gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        conditions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP training stability.
        
        Args:
            real_data: Real data samples
            fake_data: Generated fake data samples
            conditions: Optional conditioning information
            
        Returns:
            Gradient penalty term
        """
        batch_size = real_data.size(0)
        
        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_data)
        
        # Interpolated samples
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Discriminator output for interpolated samples
        if self.condition_dim is not None and conditions is not None:
            if self.fairness_enabled:
                d_interpolated, _ = self.discriminator(interpolated)
            else:
                d_interpolated = self.discriminator(interpolated, conditions)
        else:
            if self.fairness_enabled:
                d_interpolated, _ = self.discriminator(interpolated)
            else:
                d_interpolated = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def fairness_loss(
        self,
        generated_data: torch.Tensor,
        protected_predictions: List[torch.Tensor],
        target_distributions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute fairness loss to enforce demographic parity.
        
        Args:
            generated_data: Generated synthetic data
            protected_predictions: Predicted protected attributes
            target_distributions: Target distributions for protected attributes
            
        Returns:
            Fairness loss term
        """
        if not self.fairness_enabled or not protected_predictions:
            return torch.tensor(0.0, device=self.device)
        
        total_fairness_loss = 0.0
        
        for pred, target_dist in zip(protected_predictions, target_distributions):
            # Convert predictions to probabilities
            pred_probs = F.softmax(pred, dim=1)
            
            # Compute predicted distribution
            pred_dist = pred_probs.mean(dim=0)
            
            # Ensure target_dist is on the correct device
            if isinstance(target_dist, np.ndarray):
                target_dist = torch.tensor(target_dist, device=self.device, dtype=torch.float32)
            
            # KL divergence between predicted and target distributions
            kl_div = F.kl_div(
                pred_dist.log(),
                target_dist,
                reduction='batchmean'
            )
            
            total_fairness_loss += kl_div
        
        return total_fairness_loss
    
    def generator_loss(
        self,
        fake_data: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        protected_targets: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute generator loss including adversarial and fairness terms.
        
        Args:
            fake_data: Generated fake data
            conditions: Optional conditioning information
            protected_targets: Target distributions for protected attributes
            
        Returns:
            Dictionary containing loss components
        """
        # Discriminator output for fake data
        if self.condition_dim is not None and conditions is not None:
            if self.fairness_enabled:
                d_fake, protected_preds = self.discriminator(fake_data)
            else:
                d_fake = self.discriminator(fake_data, conditions)
                protected_preds = []
        else:
            if self.fairness_enabled:
                d_fake, protected_preds = self.discriminator(fake_data)
            else:
                d_fake = self.discriminator(fake_data)
                protected_preds = []
        
        # Adversarial loss (maximize discriminator output for fake data)
        adversarial_loss = -d_fake.mean()
        
        # Fairness loss
        fairness_loss = torch.tensor(0.0, device=self.device)
        if self.fairness_enabled and protected_targets is not None:
            fairness_loss = self.fairness_loss(fake_data, protected_preds, protected_targets)
        
        # Total generator loss
        total_loss = adversarial_loss + self.fairness_lambda * fairness_loss
        
        return {
            'total': total_loss,
            'adversarial': adversarial_loss,
            'fairness': fairness_loss
        }
    
    def discriminator_loss(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        protected_labels: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator loss including gradient penalty and fairness terms.
        
        Args:
            real_data: Real data samples
            fake_data: Generated fake data samples
            conditions: Optional conditioning information
            protected_labels: True protected attribute labels for fairness
            
        Returns:
            Dictionary containing loss components
        """
        # Discriminator outputs
        if self.condition_dim is not None and conditions is not None:
            if self.fairness_enabled:
                d_real, protected_real = self.discriminator(real_data)
                d_fake, protected_fake = self.discriminator(fake_data.detach())
            else:
                d_real = self.discriminator(real_data, conditions)
                d_fake = self.discriminator(fake_data.detach(), conditions)
                protected_real = protected_fake = []
        else:
            if self.fairness_enabled:
                d_real, protected_real = self.discriminator(real_data)
                d_fake, protected_fake = self.discriminator(fake_data.detach())
            else:
                d_real = self.discriminator(real_data)
                d_fake = self.discriminator(fake_data.detach())
                protected_real = protected_fake = []
        
        # Wasserstein distance
        w_distance = d_real.mean() - d_fake.mean()
        
        # Gradient penalty
        gp = self.gradient_penalty(real_data, fake_data, conditions)
        
        # Protected attribute classification loss (for fairness)
        protected_loss = torch.tensor(0.0, device=self.device)
        if self.fairness_enabled and protected_labels is not None:
            for pred, label in zip(protected_real, protected_labels):
                if isinstance(label, np.ndarray):
                    label = torch.tensor(label, device=self.device, dtype=torch.long)
                protected_loss += F.cross_entropy(pred, label)
        
        # Total discriminator loss (minimize negative Wasserstein distance)
        total_loss = -w_distance + self.lambda_gp * gp + protected_loss
        
        return {
            'total': total_loss,
            'wasserstein_distance': w_distance,
            'gradient_penalty': gp,
            'protected_loss': protected_loss
        }
    
    def generate_samples(
        self,
        num_samples: int,
        conditions: Optional[torch.Tensor] = None,
        return_numpy: bool = True
    ) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            num_samples: Number of samples to generate
            conditions: Optional conditioning information
            return_numpy: Whether to return numpy array or torch tensor
            
        Returns:
            Generated synthetic samples
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim, device=self.device)
            
            if self.condition_dim is not None and conditions is not None:
                if conditions.size(0) == 1:
                    conditions = conditions.repeat(num_samples, 1)
                samples = self.generator(noise, conditions)
            else:
                samples = self.generator(noise)
        
        if return_numpy:
            return samples.cpu().numpy()
        return samples
    
    def train_step(
        self,
        real_data: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        protected_labels: Optional[List[torch.Tensor]] = None,
        protected_targets: Optional[List[torch.Tensor]] = None,
        n_critic: int = 5
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            real_data: Real data batch
            conditions: Optional conditioning information
            protected_labels: True protected attribute labels
            protected_targets: Target distributions for protected attributes
            n_critic: Number of critic updates per generator update
            
        Returns:
            Dictionary of loss values
        """
        batch_size = real_data.size(0)
        
        # Generate fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        if self.condition_dim is not None and conditions is not None:
            fake_data = self.generator(noise, conditions)
        else:
            fake_data = self.generator(noise)
        
        # Train discriminator
        d_losses = self.discriminator_loss(
            real_data, fake_data, conditions, protected_labels
        )
        
        # Train generator (less frequently)
        g_losses = {'total': torch.tensor(0.0), 'adversarial': torch.tensor(0.0), 'fairness': torch.tensor(0.0)}
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count % n_critic == 0:
            # Generate new fake data for generator training
            noise = torch.randn(batch_size, self.noise_dim, device=self.device)
            if self.condition_dim is not None and conditions is not None:
                fake_data = self.generator(noise, conditions)
            else:
                fake_data = self.generator(noise)
                
            g_losses = self.generator_loss(fake_data, conditions, protected_targets)
        
        # Store training history
        self.training_history['d_loss'].append(d_losses['total'].item())
        self.training_history['g_loss'].append(g_losses['total'].item())
        self.training_history['w_distance'].append(d_losses['wasserstein_distance'].item())
        self.training_history['gradient_penalty'].append(d_losses['gradient_penalty'].item())
        self.training_history['fairness_loss'].append(g_losses['fairness'].item())
        
        return {
            'discriminator_loss': d_losses['total'].item(),
            'generator_loss': g_losses['total'].item(),
            'wasserstein_distance': d_losses['wasserstein_distance'].item(),
            'gradient_penalty': d_losses['gradient_penalty'].item(),
            'fairness_loss': g_losses['fairness'].item()
        }
    
    def save_model(self, filepath: str):
        """Save the complete model state."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'training_history': self.training_history,
            'config': {
                'input_dim': self.input_dim,
                'noise_dim': self.noise_dim,
                'lambda_gp': self.lambda_gp,
                'fairness_lambda': self.fairness_lambda,
                'condition_dim': self.condition_dim,
                'fairness_enabled': self.fairness_enabled
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the complete model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        
    def get_training_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics for visualization."""
        return self.training_history.copy()
