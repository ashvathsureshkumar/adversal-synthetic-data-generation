"""
Conditional GAN (cGAN) implementation for controlled synthetic data generation.

This module provides a complete cGAN implementation with fairness constraints
and privacy mechanisms for tabular data synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .generator import ConditionalGenerator
from .discriminator import ConditionalDiscriminator, FairnessAwareDiscriminator


class ConditionalGAN(nn.Module):
    """
    Conditional Generative Adversarial Network for controlled data synthesis.
    
    Features:
    - Conditional generation based on specified attributes
    - Fairness constraints integration
    - Privacy-preserving mechanisms
    - Label smoothing and other training improvements
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        noise_dim: int = 100,
        generator_hidden_dims: List[int] = [128, 256, 512],
        discriminator_hidden_dims: List[int] = [512, 256, 128],
        fairness_lambda: float = 0.1,
        protected_attrs_dims: Optional[List[int]] = None,
        embedding_dim: Optional[int] = None,
        label_smoothing: float = 0.1,
        device: str = "cpu"
    ):
        super(ConditionalGAN, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.noise_dim = noise_dim
        self.fairness_lambda = fairness_lambda
        self.label_smoothing = label_smoothing
        self.device = device
        
        # Initialize conditional generator
        self.generator = ConditionalGenerator(
            noise_dim=noise_dim,
            output_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dims=generator_hidden_dims,
            embedding_dim=embedding_dim
        )
        
        # Initialize discriminator
        if protected_attrs_dims is not None:
            self.discriminator = FairnessAwareDiscriminator(
                input_dim=input_dim,
                protected_attrs_dims=protected_attrs_dims,
                hidden_dims=discriminator_hidden_dims
            )
            self.fairness_enabled = True
        else:
            self.discriminator = ConditionalDiscriminator(
                input_dim=input_dim,
                condition_dim=condition_dim,
                hidden_dims=discriminator_hidden_dims,
                embedding_dim=embedding_dim
            )
            self.fairness_enabled = False
        
        # Move to device
        self.to(device)
        
        # Training state
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'g_loss_adv': [],
            'g_loss_fairness': [],
            'd_loss_real': [],
            'd_loss_fake': [],
            'd_accuracy': []
        }
    
    def _smooth_labels(self, labels: torch.Tensor, smoothing: float = None) -> torch.Tensor:
        """
        Apply label smoothing for more stable training.
        
        Args:
            labels: Original labels (0 or 1)
            smoothing: Smoothing factor
            
        Returns:
            Smoothed labels
        """
        if smoothing is None:
            smoothing = self.label_smoothing
            
        if smoothing > 0:
            # Smooth real labels: 1 -> [1-smoothing, 1]
            # Smooth fake labels: 0 -> [0, smoothing]
            smoothed = labels.clone()
            if labels.max() > 0.5:  # Real labels
                smoothed = labels - smoothing + torch.rand_like(labels) * 2 * smoothing
            else:  # Fake labels
                smoothed = labels + torch.rand_like(labels) * smoothing
            return smoothed
        return labels
    
    def discriminator_loss(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        conditions: torch.Tensor,
        protected_labels: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator loss with label smoothing and fairness terms.
        
        Args:
            real_data: Real data samples
            fake_data: Generated fake data samples
            conditions: Conditioning information
            protected_labels: True protected attribute labels for fairness
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = real_data.size(0)
        
        # Real and fake labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Apply label smoothing
        real_labels_smooth = self._smooth_labels(real_labels)
        fake_labels_smooth = self._smooth_labels(fake_labels)
        
        # Discriminator outputs
        if self.fairness_enabled:
            d_real, protected_real = self.discriminator(real_data)
            d_fake, protected_fake = self.discriminator(fake_data.detach())
        else:
            d_real = self.discriminator(real_data, conditions)
            d_fake = self.discriminator(fake_data.detach(), conditions)
            protected_real = protected_fake = []
        
        # Adversarial losses
        loss_real = F.binary_cross_entropy_with_logits(d_real, real_labels_smooth)
        loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels_smooth)
        
        # Protected attribute classification loss (for fairness)
        protected_loss = torch.tensor(0.0, device=self.device)
        if self.fairness_enabled and protected_labels is not None:
            for pred, label in zip(protected_real, protected_labels):
                if isinstance(label, np.ndarray):
                    label = torch.tensor(label, device=self.device, dtype=torch.long)
                protected_loss += F.cross_entropy(pred, label)
        
        # Total discriminator loss
        total_loss = loss_real + loss_fake + protected_loss
        
        # Calculate accuracy
        with torch.no_grad():
            d_real_pred = torch.sigmoid(d_real) > 0.5
            d_fake_pred = torch.sigmoid(d_fake) > 0.5
            accuracy = (d_real_pred.float().mean() + (1 - d_fake_pred.float()).mean()) / 2
        
        return {
            'total': total_loss,
            'real': loss_real,
            'fake': loss_fake,
            'protected': protected_loss,
            'accuracy': accuracy
        }
    
    def generator_loss(
        self,
        fake_data: torch.Tensor,
        conditions: torch.Tensor,
        protected_targets: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute generator loss including adversarial and fairness terms.
        
        Args:
            fake_data: Generated fake data
            conditions: Conditioning information
            protected_targets: Target distributions for protected attributes
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = fake_data.size(0)
        
        # We want the discriminator to classify generated data as real
        real_labels = torch.ones(batch_size, 1, device=self.device)
        
        # Discriminator output for fake data
        if self.fairness_enabled:
            d_fake, protected_preds = self.discriminator(fake_data)
        else:
            d_fake = self.discriminator(fake_data, conditions)
            protected_preds = []
        
        # Adversarial loss
        adversarial_loss = F.binary_cross_entropy_with_logits(d_fake, real_labels)
        
        # Fairness loss
        fairness_loss = torch.tensor(0.0, device=self.device)
        if self.fairness_enabled and protected_targets is not None:
            fairness_loss = self._fairness_loss(fake_data, protected_preds, protected_targets)
        
        # Total generator loss
        total_loss = adversarial_loss + self.fairness_lambda * fairness_loss
        
        return {
            'total': total_loss,
            'adversarial': adversarial_loss,
            'fairness': fairness_loss
        }
    
    def _fairness_loss(
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
        if not protected_predictions:
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
    
    def generate_samples(
        self,
        num_samples: int,
        conditions: torch.Tensor,
        return_numpy: bool = True
    ) -> torch.Tensor:
        """
        Generate conditional synthetic samples.
        
        Args:
            num_samples: Number of samples to generate
            conditions: Conditioning information for each sample
            return_numpy: Whether to return numpy array or torch tensor
            
        Returns:
            Generated synthetic samples
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim, device=self.device)
            
            # Expand conditions if necessary
            if conditions.size(0) == 1:
                conditions = conditions.repeat(num_samples, 1)
            elif conditions.size(0) != num_samples:
                raise ValueError(f"Conditions size {conditions.size(0)} doesn't match num_samples {num_samples}")
            
            samples = self.generator(noise, conditions)
        
        if return_numpy:
            return samples.cpu().numpy()
        return samples
    
    def train_step(
        self,
        real_data: torch.Tensor,
        conditions: torch.Tensor,
        protected_labels: Optional[List[torch.Tensor]] = None,
        protected_targets: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            real_data: Real data batch
            conditions: Conditioning information
            protected_labels: True protected attribute labels
            protected_targets: Target distributions for protected attributes
            
        Returns:
            Dictionary of loss values
        """
        batch_size = real_data.size(0)
        
        # Generate fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_data = self.generator(noise, conditions)
        
        # Train discriminator
        d_losses = self.discriminator_loss(
            real_data, fake_data, conditions, protected_labels
        )
        
        # Train generator
        g_losses = self.generator_loss(fake_data, conditions, protected_targets)
        
        # Store training history
        self.training_history['d_loss'].append(d_losses['total'].item())
        self.training_history['g_loss'].append(g_losses['total'].item())
        self.training_history['g_loss_adv'].append(g_losses['adversarial'].item())
        self.training_history['g_loss_fairness'].append(g_losses['fairness'].item())
        self.training_history['d_loss_real'].append(d_losses['real'].item())
        self.training_history['d_loss_fake'].append(d_losses['fake'].item())
        self.training_history['d_accuracy'].append(d_losses['accuracy'].item())
        
        return {
            'discriminator_loss': d_losses['total'].item(),
            'generator_loss': g_losses['total'].item(),
            'generator_adv_loss': g_losses['adversarial'].item(),
            'generator_fairness_loss': g_losses['fairness'].item(),
            'discriminator_real_loss': d_losses['real'].item(),
            'discriminator_fake_loss': d_losses['fake'].item(),
            'discriminator_accuracy': d_losses['accuracy'].item()
        }
    
    def conditional_generation_analysis(
        self,
        conditions_list: List[torch.Tensor],
        num_samples_per_condition: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze generation quality across different conditions.
        
        Args:
            conditions_list: List of different condition tensors to test
            num_samples_per_condition: Number of samples per condition
            
        Returns:
            Analysis results dictionary
        """
        self.generator.eval()
        results = {}
        
        with torch.no_grad():
            for i, condition in enumerate(conditions_list):
                # Generate samples for this condition
                samples = self.generate_samples(
                    num_samples_per_condition, 
                    condition.unsqueeze(0),
                    return_numpy=True
                )
                
                # Basic statistics
                results[f'condition_{i}'] = {
                    'mean': np.mean(samples, axis=0),
                    'std': np.std(samples, axis=0),
                    'min': np.min(samples, axis=0),
                    'max': np.max(samples, axis=0),
                    'samples': samples
                }
        
        return results
    
    def save_model(self, filepath: str):
        """Save the complete model state."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'training_history': self.training_history,
            'config': {
                'input_dim': self.input_dim,
                'condition_dim': self.condition_dim,
                'noise_dim': self.noise_dim,
                'fairness_lambda': self.fairness_lambda,
                'label_smoothing': self.label_smoothing,
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
