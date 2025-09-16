"""
Fairness constraints and metrics for synthetic data generation.

This module implements various fairness constraints and evaluation metrics
to ensure generated synthetic data maintains fair representation across
protected attributes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
import warnings


class BaseFairnessLoss(ABC, nn.Module):
    """
    Abstract base class for fairness loss functions.
    """
    
    def __init__(self, weight: float = 1.0):
        super(BaseFairnessLoss, self).__init__()
        self.weight = weight
    
    @abstractmethod
    def forward(
        self,
        predictions: torch.Tensor,
        protected_attributes: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute fairness loss.
        
        Args:
            predictions: Model predictions
            protected_attributes: Protected attribute values
            labels: True labels (if available)
            
        Returns:
            Fairness loss value
        """
        pass


class DemographicParityLoss(BaseFairnessLoss):
    """
    Demographic Parity fairness constraint.
    
    Ensures that the probability of positive prediction is equal across
    all groups defined by protected attributes.
    """
    
    def __init__(self, weight: float = 1.0, epsilon: float = 1e-8):
        super(DemographicParityLoss, self).__init__(weight)
        self.epsilon = epsilon
    
    def forward(
        self,
        predictions: torch.Tensor,
        protected_attributes: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute demographic parity loss.
        
        Args:
            predictions: Binary predictions or probabilities [batch_size, 1]
            protected_attributes: Protected group indicators [batch_size, 1]
            labels: Not used for demographic parity
            
        Returns:
            Demographic parity loss
        """
        # Convert to probabilities if needed
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Get unique protected groups
        unique_groups = torch.unique(protected_attributes)
        
        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute positive prediction rates for each group
        group_rates = []
        for group in unique_groups:
            group_mask = (protected_attributes == group).float()
            group_size = group_mask.sum() + self.epsilon
            group_positive_rate = (predictions * group_mask).sum() / group_size
            group_rates.append(group_positive_rate)
        
        # Compute pairwise differences
        total_loss = torch.tensor(0.0, device=predictions.device)
        for i in range(len(group_rates)):
            for j in range(i + 1, len(group_rates)):
                total_loss += torch.abs(group_rates[i] - group_rates[j])
        
        return self.weight * total_loss


class EqualizedOddsLoss(BaseFairnessLoss):
    """
    Equalized Odds fairness constraint.
    
    Ensures that both true positive rate and false positive rate are equal
    across all groups defined by protected attributes.
    """
    
    def __init__(self, weight: float = 1.0, epsilon: float = 1e-8):
        super(EqualizedOddsLoss, self).__init__(weight)
        self.epsilon = epsilon
    
    def forward(
        self,
        predictions: torch.Tensor,
        protected_attributes: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute equalized odds loss.
        
        Args:
            predictions: Binary predictions or probabilities [batch_size, 1]
            protected_attributes: Protected group indicators [batch_size, 1]
            labels: True binary labels [batch_size, 1]
            
        Returns:
            Equalized odds loss
        """
        if labels is None:
            raise ValueError("Labels are required for Equalized Odds constraint")
        
        # Convert to probabilities if needed
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).float()
        
        # Get unique protected groups
        unique_groups = torch.unique(protected_attributes)
        
        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute TPR and FPR for each group
        group_tprs = []
        group_fprs = []
        
        for group in unique_groups:
            group_mask = (protected_attributes == group).float()
            
            # True positives and negatives for this group
            group_labels = labels * group_mask
            group_preds = binary_predictions * group_mask
            
            # TPR = TP / (TP + FN)
            true_positives = (group_labels * group_preds).sum()
            actual_positives = group_labels.sum() + self.epsilon
            tpr = true_positives / actual_positives
            
            # FPR = FP / (FP + TN)
            false_positives = ((1 - group_labels) * group_preds).sum()
            actual_negatives = (1 - group_labels).sum() + self.epsilon
            fpr = false_positives / actual_negatives
            
            group_tprs.append(tpr)
            group_fprs.append(fpr)
        
        # Compute pairwise differences for both TPR and FPR
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        for i in range(len(group_tprs)):
            for j in range(i + 1, len(group_tprs)):
                tpr_diff = torch.abs(group_tprs[i] - group_tprs[j])
                fpr_diff = torch.abs(group_fprs[i] - group_fprs[j])
                total_loss += tpr_diff + fpr_diff
        
        return self.weight * total_loss


class EqualOpportunityLoss(BaseFairnessLoss):
    """
    Equal Opportunity fairness constraint.
    
    Ensures that true positive rate is equal across all groups.
    """
    
    def __init__(self, weight: float = 1.0, epsilon: float = 1e-8):
        super(EqualOpportunityLoss, self).__init__(weight)
        self.epsilon = epsilon
    
    def forward(
        self,
        predictions: torch.Tensor,
        protected_attributes: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute equal opportunity loss.
        
        Args:
            predictions: Binary predictions or probabilities [batch_size, 1]
            protected_attributes: Protected group indicators [batch_size, 1]
            labels: True binary labels [batch_size, 1]
            
        Returns:
            Equal opportunity loss
        """
        if labels is None:
            raise ValueError("Labels are required for Equal Opportunity constraint")
        
        # Convert to probabilities if needed
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).float()
        
        # Get unique protected groups
        unique_groups = torch.unique(protected_attributes)
        
        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute TPR for each group
        group_tprs = []
        
        for group in unique_groups:
            group_mask = (protected_attributes == group).float()
            
            # True positives for this group
            group_labels = labels * group_mask
            group_preds = binary_predictions * group_mask
            
            # TPR = TP / (TP + FN)
            true_positives = (group_labels * group_preds).sum()
            actual_positives = group_labels.sum() + self.epsilon
            tpr = true_positives / actual_positives
            
            group_tprs.append(tpr)
        
        # Compute pairwise differences in TPR
        total_loss = torch.tensor(0.0, device=predictions.device)
        for i in range(len(group_tprs)):
            for j in range(i + 1, len(group_tprs)):
                total_loss += torch.abs(group_tprs[i] - group_tprs[j])
        
        return self.weight * total_loss


class DistributionMatchingLoss(BaseFairnessLoss):
    """
    Distribution matching loss for ensuring synthetic data matches
    target demographic distributions.
    """
    
    def __init__(
        self,
        target_distributions: Dict[str, torch.Tensor],
        weight: float = 1.0,
        distance_metric: str = "kl_divergence"
    ):
        super(DistributionMatchingLoss, self).__init__(weight)
        self.target_distributions = target_distributions
        self.distance_metric = distance_metric
    
    def forward(
        self,
        predictions: torch.Tensor,
        protected_attributes: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distribution matching loss.
        
        Args:
            predictions: Generated data or predictions
            protected_attributes: Protected attribute predictions or values
            labels: Not used for distribution matching
            
        Returns:
            Distribution matching loss
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        # Compute empirical distribution from protected_attributes
        if protected_attributes.dim() > 1 and protected_attributes.size(1) > 1:
            # Multi-class protected attributes
            empirical_dist = F.softmax(protected_attributes, dim=1).mean(dim=0)
        else:
            # Binary protected attributes
            empirical_dist = torch.stack([
                1 - protected_attributes.mean(),
                protected_attributes.mean()
            ])
        
        # Find matching target distribution
        target_key = f"protected_attr_0"  # Simplification for demo
        if target_key in self.target_distributions:
            target_dist = self.target_distributions[target_key]
            
            if isinstance(target_dist, np.ndarray):
                target_dist = torch.tensor(target_dist, device=predictions.device)
            
            # Compute distance
            if self.distance_metric == "kl_divergence":
                # Add small epsilon to avoid log(0)
                epsilon = 1e-8
                empirical_dist = empirical_dist + epsilon
                target_dist = target_dist + epsilon
                
                # Normalize
                empirical_dist = empirical_dist / empirical_dist.sum()
                target_dist = target_dist / target_dist.sum()
                
                loss = F.kl_div(
                    empirical_dist.log(),
                    target_dist,
                    reduction='batchmean'
                )
            elif self.distance_metric == "wasserstein":
                # Simplified 1D Wasserstein distance
                loss = torch.abs(empirical_dist - target_dist).sum()
            else:
                # L2 distance
                loss = F.mse_loss(empirical_dist, target_dist)
            
            total_loss += loss
        
        return self.weight * total_loss


class FairnessConstraints:
    """
    Main fairness constraints manager that combines multiple fairness metrics.
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        fairness_type: str = "demographic_parity",
        constraint_weight: float = 0.1,
        target_distributions: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None
    ):
        self.protected_attributes = protected_attributes
        self.fairness_type = fairness_type
        self.constraint_weight = constraint_weight
        self.target_distributions = target_distributions or {}
        
        # Initialize fairness loss functions
        self.fairness_losses = self._create_fairness_losses()
    
    def _create_fairness_losses(self) -> Dict[str, BaseFairnessLoss]:
        """Create fairness loss functions based on specified type."""
        losses = {}
        
        if self.fairness_type == "demographic_parity":
            losses['demographic_parity'] = DemographicParityLoss(self.constraint_weight)
        elif self.fairness_type == "equalized_odds":
            losses['equalized_odds'] = EqualizedOddsLoss(self.constraint_weight)
        elif self.fairness_type == "equal_opportunity":
            losses['equal_opportunity'] = EqualOpportunityLoss(self.constraint_weight)
        elif self.fairness_type == "distribution_matching":
            losses['distribution_matching'] = DistributionMatchingLoss(
                self.target_distributions, 
                self.constraint_weight
            )
        elif self.fairness_type == "combined":
            # Use multiple fairness constraints
            losses['demographic_parity'] = DemographicParityLoss(self.constraint_weight * 0.5)
            losses['distribution_matching'] = DistributionMatchingLoss(
                self.target_distributions, 
                self.constraint_weight * 0.5
            )
        else:
            warnings.warn(f"Unknown fairness type: {self.fairness_type}")
        
        return losses
    
    def compute_fairness_loss(
        self,
        predictions: torch.Tensor,
        protected_attributes: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total fairness loss.
        
        Args:
            predictions: Model predictions or generated data
            protected_attributes: Protected attribute values or predictions
            labels: True labels (if available)
            
        Returns:
            Total fairness loss
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        for loss_name, loss_fn in self.fairness_losses.items():
            try:
                loss_value = loss_fn(predictions, protected_attributes, labels)
                total_loss += loss_value
            except Exception as e:
                warnings.warn(f"Error computing {loss_name}: {e}")
        
        return total_loss
    
    def evaluate_fairness_metrics(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate fairness metrics on numpy arrays.
        
        Args:
            predictions: Model predictions (binary or probabilities)
            protected_attributes: Protected attribute values
            labels: True labels (if available)
            
        Returns:
            Dictionary of fairness metric values
        """
        metrics = {}
        
        # Convert to binary predictions if probabilities
        if predictions.min() >= 0 and predictions.max() <= 1:
            binary_predictions = (predictions > 0.5).astype(int)
        else:
            binary_predictions = predictions
        
        # Demographic Parity
        metrics.update(self._compute_demographic_parity(binary_predictions, protected_attributes))
        
        # Equalized Odds (if labels available)
        if labels is not None:
            metrics.update(self._compute_equalized_odds(binary_predictions, protected_attributes, labels))
        
        # Distribution matching
        metrics.update(self._compute_distribution_metrics(protected_attributes))
        
        return metrics
    
    def _compute_demographic_parity(
        self, 
        predictions: np.ndarray, 
        protected_attributes: np.ndarray
    ) -> Dict[str, float]:
        """Compute demographic parity metrics."""
        metrics = {}
        
        unique_groups = np.unique(protected_attributes)
        group_rates = []
        
        for group in unique_groups:
            group_mask = (protected_attributes == group)
            if group_mask.sum() > 0:
                group_rate = predictions[group_mask].mean()
                group_rates.append(group_rate)
                metrics[f'positive_rate_group_{group}'] = group_rate
        
        if len(group_rates) >= 2:
            metrics['demographic_parity_difference'] = max(group_rates) - min(group_rates)
            metrics['demographic_parity_ratio'] = min(group_rates) / (max(group_rates) + 1e-8)
        
        return metrics
    
    def _compute_equalized_odds(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute equalized odds metrics."""
        metrics = {}
        
        unique_groups = np.unique(protected_attributes)
        tprs = []
        fprs = []
        
        for group in unique_groups:
            group_mask = (protected_attributes == group)
            if group_mask.sum() > 0:
                group_labels = labels[group_mask]
                group_preds = predictions[group_mask]
                
                if len(np.unique(group_labels)) > 1:  # Both classes present
                    tn, fp, fn, tp = confusion_matrix(group_labels, group_preds).ravel()
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    tprs.append(tpr)
                    fprs.append(fpr)
                    
                    metrics[f'tpr_group_{group}'] = tpr
                    metrics[f'fpr_group_{group}'] = fpr
        
        if len(tprs) >= 2:
            metrics['equalized_odds_tpr_difference'] = max(tprs) - min(tprs)
            metrics['equalized_odds_fpr_difference'] = max(fprs) - min(fprs)
        
        return metrics
    
    def _compute_distribution_metrics(self, protected_attributes: np.ndarray) -> Dict[str, float]:
        """Compute distribution-related metrics."""
        metrics = {}
        
        unique_groups, counts = np.unique(protected_attributes, return_counts=True)
        empirical_dist = counts / counts.sum()
        
        # Store empirical distribution
        for i, group in enumerate(unique_groups):
            metrics[f'group_{group}_proportion'] = empirical_dist[i]
        
        # Compare with target distribution if available
        target_key = "protected_attr_0"
        if target_key in self.target_distributions:
            target_dist = self.target_distributions[target_key]
            if isinstance(target_dist, torch.Tensor):
                target_dist = target_dist.cpu().numpy()
            
            # KL divergence
            epsilon = 1e-8
            empirical_dist_smooth = empirical_dist + epsilon
            target_dist_smooth = target_dist + epsilon
            
            empirical_dist_smooth /= empirical_dist_smooth.sum()
            target_dist_smooth /= target_dist_smooth.sum()
            
            kl_div = np.sum(empirical_dist_smooth * np.log(empirical_dist_smooth / target_dist_smooth))
            metrics['distribution_kl_divergence'] = kl_div
            
            # Wasserstein distance (1D)
            metrics['distribution_wasserstein_distance'] = np.sum(np.abs(empirical_dist - target_dist))
        
        return metrics
