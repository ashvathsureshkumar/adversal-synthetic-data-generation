"""
Comprehensive training module for GAN-based synthetic data generation.

This module provides advanced training loops with fairness constraints,
privacy mechanisms, and comprehensive logging and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from tqdm import tqdm
import os
import time
from pathlib import Path

from ..models.wgan_gp import WGAN_GP
from ..models.cgan import ConditionalGAN
from .fairness import FairnessConstraints
from .privacy import PrivacyEngine
from .evaluation import SyntheticDataEvaluator
from .utils import EarlyStopping, ModelCheckpoint, TrainingLogger


class BaseGANTrainer:
    """
    Base trainer class with common functionality for all GAN variants.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        logging_level: str = "INFO"
    ):
        self.model = model
        self.device = device
        self.logger = TrainingLogger(logging_level)
        
        # Training state
        self.current_epoch = 0
        self.best_score = float('inf')
        self.training_start_time = None
        
        # Components
        self.fairness_constraints = None
        self.privacy_engine = None
        self.evaluator = None
        self.early_stopping = None
        self.checkpoint = None
        
    def setup_optimizers(
        self,
        g_lr: float = 0.0002,
        d_lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        weight_decay: float = 0.0
    ):
        """Setup optimizers for generator and discriminator."""
        self.g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=g_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
        self.d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=d_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
    
    def setup_fairness_constraints(
        self,
        protected_attributes: List[str],
        fairness_type: str = "demographic_parity",
        constraint_weight: float = 0.1
    ):
        """Setup fairness constraints."""
        self.fairness_constraints = FairnessConstraints(
            protected_attributes=protected_attributes,
            fairness_type=fairness_type,
            constraint_weight=constraint_weight
        )
    
    def setup_privacy_engine(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0
    ):
        """Setup differential privacy engine."""
        self.privacy_engine = PrivacyEngine(
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm
        )
    
    def setup_evaluation(self, real_data: np.ndarray, column_names: List[str]):
        """Setup evaluation metrics."""
        self.evaluator = SyntheticDataEvaluator(
            real_data=real_data,
            column_names=column_names
        )
    
    def setup_early_stopping(
        self,
        patience: int = 50,
        min_delta: float = 0.001,
        monitor: str = "generator_loss"
    ):
        """Setup early stopping."""
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            monitor=monitor
        )
    
    def setup_checkpoint(self, save_dir: str, save_freq: int = 100):
        """Setup model checkpointing."""
        self.checkpoint = ModelCheckpoint(
            save_dir=save_dir,
            save_freq=save_freq
        )
    
    def save_training_state(self, filepath: str):
        """Save complete training state."""
        state = {
            'model_state': self.model.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'best_score': self.best_score,
            'training_history': self.model.get_training_metrics()
        }
        torch.save(state, filepath)
        self.logger.info(f"Training state saved to {filepath}")
    
    def load_training_state(self, filepath: str):
        """Load complete training state."""
        state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.g_optimizer.load_state_dict(state['g_optimizer_state'])
        self.d_optimizer.load_state_dict(state['d_optimizer_state'])
        self.current_epoch = state['current_epoch']
        self.best_score = state['best_score']
        self.logger.info(f"Training state loaded from {filepath}")


class WGANGPTrainer(BaseGANTrainer):
    """
    Trainer for Wasserstein GAN with Gradient Penalty.
    """
    
    def __init__(
        self,
        model: WGAN_GP,
        device: str = "cpu",
        n_critic: int = 5,
        logging_level: str = "INFO"
    ):
        super().__init__(model, device, logging_level)
        self.n_critic = n_critic
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        conditions: Optional[torch.Tensor] = None,
        protected_labels: Optional[List[torch.Tensor]] = None,
        protected_targets: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for real data
            conditions: Optional conditioning information
            protected_labels: True protected attribute labels
            protected_targets: Target distributions for fairness
            
        Returns:
            Dictionary of average loss values for the epoch
        """
        self.model.train()
        epoch_metrics = {
            'discriminator_loss': 0.0,
            'generator_loss': 0.0,
            'wasserstein_distance': 0.0,
            'gradient_penalty': 0.0,
            'fairness_loss': 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {self.current_epoch}")):
            if isinstance(batch_data, list):
                real_data = batch_data[0].to(self.device)
            else:
                real_data = batch_data.to(self.device)
            
            # Prepare batch conditions and labels
            batch_conditions = None
            batch_protected_labels = None
            batch_protected_targets = None
            
            if conditions is not None:
                batch_size = real_data.size(0)
                if conditions.size(0) >= batch_size:
                    batch_conditions = conditions[:batch_size]
                else:
                    # Repeat conditions if needed
                    repeat_factor = (batch_size + conditions.size(0) - 1) // conditions.size(0)
                    batch_conditions = conditions.repeat(repeat_factor, 1)[:batch_size]
            
            if protected_labels is not None:
                batch_protected_labels = [
                    label[:real_data.size(0)] for label in protected_labels
                ]
            
            if protected_targets is not None:
                batch_protected_targets = protected_targets
            
            # Training step
            step_metrics = self.model.train_step(
                real_data=real_data,
                conditions=batch_conditions,
                protected_labels=batch_protected_labels,
                protected_targets=batch_protected_targets,
                n_critic=self.n_critic
            )
            
            # Update discriminator
            if batch_idx % self.n_critic == 0 or batch_idx == num_batches - 1:
                self.d_optimizer.zero_grad()
                # Get discriminator loss from the step
                d_loss = self.model.discriminator_loss(
                    real_data, 
                    self.model.generate_samples(real_data.size(0), batch_conditions, return_numpy=False),
                    batch_conditions,
                    batch_protected_labels
                )['total']
                d_loss.backward()
                
                # Apply privacy constraints if enabled
                if self.privacy_engine is not None:
                    self.privacy_engine.clip_gradients(self.model.discriminator)
                
                self.d_optimizer.step()
            
            # Update generator (less frequently)
            if batch_idx % self.n_critic == 0:
                self.g_optimizer.zero_grad()
                # Generate new samples for generator training
                noise = torch.randn(real_data.size(0), self.model.noise_dim, device=self.device)
                if batch_conditions is not None:
                    fake_data = self.model.generator(noise, batch_conditions)
                else:
                    fake_data = self.model.generator(noise)
                    
                g_loss = self.model.generator_loss(
                    fake_data, 
                    batch_conditions, 
                    batch_protected_targets
                )['total']
                g_loss.backward()
                
                # Apply privacy constraints if enabled
                if self.privacy_engine is not None:
                    self.privacy_engine.clip_gradients(self.model.generator)
                
                self.g_optimizer.step()
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
        
        # Average metrics over the epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        conditions: Optional[torch.Tensor] = None,
        protected_labels: Optional[List[torch.Tensor]] = None,
        protected_targets: Optional[List[torch.Tensor]] = None,
        validation_data: Optional[np.ndarray] = None,
        validation_freq: int = 10
    ) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of training epochs
            conditions: Optional conditioning information
            protected_labels: True protected attribute labels for fairness
            protected_targets: Target distributions for fairness
            validation_data: Optional validation data for evaluation
            validation_freq: Frequency of validation evaluation
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting WGAN-GP training for {num_epochs} epochs")
        self.training_start_time = time.time()
        
        training_history = {
            'epoch': [],
            'discriminator_loss': [],
            'generator_loss': [],
            'wasserstein_distance': [],
            'gradient_penalty': [],
            'fairness_loss': [],
            'validation_score': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            epoch_metrics = self.train_epoch(
                dataloader, conditions, protected_labels, protected_targets
            )
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"D_loss={epoch_metrics['discriminator_loss']:.4f}, "
                f"G_loss={epoch_metrics['generator_loss']:.4f}, "
                f"W_dist={epoch_metrics['wasserstein_distance']:.4f}"
            )
            
            # Store training history
            training_history['epoch'].append(epoch)
            for key, value in epoch_metrics.items():
                if key in training_history:
                    training_history[key].append(value)
            
            # Validation evaluation
            validation_score = 0.0
            if validation_data is not None and epoch % validation_freq == 0:
                validation_score = self.evaluate_on_validation(validation_data)
                training_history['validation_score'].append(validation_score)
                self.logger.info(f"Validation score: {validation_score:.4f}")
            else:
                training_history['validation_score'].append(0.0)
            
            # Early stopping check
            if self.early_stopping is not None:
                should_stop = self.early_stopping.check(
                    epoch_metrics.get(self.early_stopping.monitor, validation_score)
                )
                if should_stop:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Checkpoint saving
            if self.checkpoint is not None:
                self.checkpoint.save_if_needed(epoch, self.model, epoch_metrics)
        
        training_time = time.time() - self.training_start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return training_history
    
    def evaluate_on_validation(self, validation_data: np.ndarray) -> float:
        """Evaluate model on validation data."""
        if self.evaluator is None:
            return 0.0
        
        # Generate synthetic data
        num_samples = len(validation_data)
        synthetic_data = self.model.generate_samples(num_samples)
        
        # Compute evaluation metrics
        metrics = self.evaluator.evaluate(synthetic_data)
        
        # Return a composite score (lower is better)
        return metrics.get('wasserstein_distance', 0.0)


class CGANTrainer(BaseGANTrainer):
    """
    Trainer for Conditional GAN.
    """
    
    def __init__(
        self,
        model: ConditionalGAN,
        device: str = "cpu",
        logging_level: str = "INFO"
    ):
        super().__init__(model, device, logging_level)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        conditions: torch.Tensor,
        protected_labels: Optional[List[torch.Tensor]] = None,
        protected_targets: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for real data
            conditions: Conditioning information (required for cGAN)
            protected_labels: True protected attribute labels
            protected_targets: Target distributions for fairness
            
        Returns:
            Dictionary of average loss values for the epoch
        """
        self.model.train()
        epoch_metrics = {
            'discriminator_loss': 0.0,
            'generator_loss': 0.0,
            'generator_adv_loss': 0.0,
            'generator_fairness_loss': 0.0,
            'discriminator_accuracy': 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {self.current_epoch}")):
            if isinstance(batch_data, list):
                real_data = batch_data[0].to(self.device)
            else:
                real_data = batch_data.to(self.device)
            
            batch_size = real_data.size(0)
            
            # Prepare batch conditions and labels
            if conditions.size(0) >= batch_size:
                batch_conditions = conditions[:batch_size].to(self.device)
            else:
                # Repeat conditions if needed
                repeat_factor = (batch_size + conditions.size(0) - 1) // conditions.size(0)
                batch_conditions = conditions.repeat(repeat_factor, 1)[:batch_size].to(self.device)
            
            batch_protected_labels = None
            if protected_labels is not None:
                batch_protected_labels = [
                    label[:batch_size] for label in protected_labels
                ]
            
            # Update discriminator
            self.d_optimizer.zero_grad()
            
            # Generate fake data
            noise = torch.randn(batch_size, self.model.noise_dim, device=self.device)
            fake_data = self.model.generator(noise, batch_conditions)
            
            # Discriminator loss
            d_losses = self.model.discriminator_loss(
                real_data, fake_data, batch_conditions, batch_protected_labels
            )
            d_losses['total'].backward()
            
            # Apply privacy constraints if enabled
            if self.privacy_engine is not None:
                self.privacy_engine.clip_gradients(self.model.discriminator)
            
            self.d_optimizer.step()
            
            # Update generator
            self.g_optimizer.zero_grad()
            
            # Generate new fake data for generator training
            noise = torch.randn(batch_size, self.model.noise_dim, device=self.device)
            fake_data = self.model.generator(noise, batch_conditions)
            
            # Generator loss
            g_losses = self.model.generator_loss(
                fake_data, batch_conditions, protected_targets
            )
            g_losses['total'].backward()
            
            # Apply privacy constraints if enabled
            if self.privacy_engine is not None:
                self.privacy_engine.clip_gradients(self.model.generator)
            
            self.g_optimizer.step()
            
            # Accumulate metrics
            epoch_metrics['discriminator_loss'] += d_losses['total'].item()
            epoch_metrics['generator_loss'] += g_losses['total'].item()
            epoch_metrics['generator_adv_loss'] += g_losses['adversarial'].item()
            epoch_metrics['generator_fairness_loss'] += g_losses['fairness'].item()
            epoch_metrics['discriminator_accuracy'] += d_losses['accuracy'].item()
        
        # Average metrics over the epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(
        self,
        dataloader: DataLoader,
        conditions: torch.Tensor,
        num_epochs: int,
        protected_labels: Optional[List[torch.Tensor]] = None,
        protected_targets: Optional[List[torch.Tensor]] = None,
        validation_data: Optional[Tuple[np.ndarray, torch.Tensor]] = None,
        validation_freq: int = 10
    ) -> Dict[str, List[float]]:
        """
        Complete training loop for conditional GAN.
        
        Args:
            dataloader: Training data loader
            conditions: Conditioning information for all training data
            num_epochs: Number of training epochs
            protected_labels: True protected attribute labels for fairness
            protected_targets: Target distributions for fairness
            validation_data: Optional (data, conditions) tuple for validation
            validation_freq: Frequency of validation evaluation
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting cGAN training for {num_epochs} epochs")
        self.training_start_time = time.time()
        
        training_history = {
            'epoch': [],
            'discriminator_loss': [],
            'generator_loss': [],
            'generator_adv_loss': [],
            'generator_fairness_loss': [],
            'discriminator_accuracy': [],
            'validation_score': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            epoch_metrics = self.train_epoch(
                dataloader, conditions, protected_labels, protected_targets
            )
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"D_loss={epoch_metrics['discriminator_loss']:.4f}, "
                f"G_loss={epoch_metrics['generator_loss']:.4f}, "
                f"D_acc={epoch_metrics['discriminator_accuracy']:.4f}"
            )
            
            # Store training history
            training_history['epoch'].append(epoch)
            for key, value in epoch_metrics.items():
                if key in training_history:
                    training_history[key].append(value)
            
            # Validation evaluation
            validation_score = 0.0
            if validation_data is not None and epoch % validation_freq == 0:
                validation_score = self.evaluate_on_validation(*validation_data)
                training_history['validation_score'].append(validation_score)
                self.logger.info(f"Validation score: {validation_score:.4f}")
            else:
                training_history['validation_score'].append(0.0)
            
            # Early stopping check
            if self.early_stopping is not None:
                should_stop = self.early_stopping.check(
                    epoch_metrics.get(self.early_stopping.monitor, validation_score)
                )
                if should_stop:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Checkpoint saving
            if self.checkpoint is not None:
                self.checkpoint.save_if_needed(epoch, self.model, epoch_metrics)
        
        training_time = time.time() - self.training_start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return training_history
    
    def evaluate_on_validation(
        self, 
        validation_data: np.ndarray, 
        validation_conditions: torch.Tensor
    ) -> float:
        """Evaluate model on validation data with conditions."""
        if self.evaluator is None:
            return 0.0
        
        # Generate synthetic data with validation conditions
        synthetic_data = self.model.generate_samples(
            len(validation_data), 
            validation_conditions
        )
        
        # Compute evaluation metrics
        metrics = self.evaluator.evaluate(synthetic_data)
        
        # Return a composite score (lower is better)
        return metrics.get('wasserstein_distance', 0.0)


# Alias for backward compatibility
GANTrainer = WGANGPTrainer
