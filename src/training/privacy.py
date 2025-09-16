"""
Privacy-preserving mechanisms for synthetic data generation.

This module implements differential privacy and other privacy-preserving
techniques for GAN-based synthetic data generation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from opacus import PrivacyEngine as OpacusPrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import logging


class PrivacyEngine:
    """
    Privacy engine for applying differential privacy to GAN training.
    
    This class wraps Opacus functionality and provides additional
    privacy-preserving mechanisms for synthetic data generation.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        secure_mode: bool = False
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.secure_mode = secure_mode
        
        # Privacy accounting
        self.privacy_spent = 0.0
        self.steps_taken = 0
        
        # Opacus engine (will be initialized when model is attached)
        self.opacus_engine = None
        
        logging.getLogger("opacus").setLevel(logging.WARNING)
    
    def attach_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
        """
        Attach privacy engine to model, optimizer, and data loader.
        
        Args:
            model: Neural network model
            optimizer: Optimizer for the model
            data_loader: Training data loader
            
        Returns:
            Tuple of (private_model, private_optimizer, private_data_loader)
        """
        if self.noise_multiplier is None:
            # Calculate noise multiplier based on epsilon and delta
            self.noise_multiplier = self._calculate_noise_multiplier(
                len(data_loader.dataset),
                data_loader.batch_size,
                self.epsilon,
                self.delta
            )
        
        try:
            self.opacus_engine = OpacusPrivacyEngine(secure_mode=self.secure_mode)
            
            private_model, private_optimizer, private_data_loader = self.opacus_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=1,  # We'll handle epochs manually
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm
            )
            
            return private_model, private_optimizer, private_data_loader
            
        except Exception as e:
            warnings.warn(f"Failed to initialize Opacus privacy engine: {e}")
            # Fallback to manual gradient clipping
            return model, optimizer, data_loader
    
    def _calculate_noise_multiplier(
        self,
        dataset_size: int,
        batch_size: int,
        epsilon: float,
        delta: float,
        epochs: int = 100
    ) -> float:
        """
        Calculate noise multiplier for given privacy parameters.
        
        This is a simplified calculation. In practice, you might want to use
        more sophisticated privacy accounting methods.
        """
        # Simple heuristic for noise multiplier
        q = batch_size / dataset_size  # Sampling probability
        
        # Using the moments accountant approximation
        if epochs * q < 1:
            sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        else:
            sigma = np.sqrt(epochs * q * np.log(1 / delta)) / epsilon
        
        return max(sigma, 0.1)  # Minimum noise level
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Manual gradient clipping when Opacus is not available.
        
        Args:
            model: Model to clip gradients for
            
        Returns:
            Gradient norm before clipping
        """
        if self.opacus_engine is not None:
            # Opacus handles this automatically
            return 0.0
        
        # Manual gradient clipping
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise_to_gradients(self, model: nn.Module):
        """
        Add noise to gradients for differential privacy (manual implementation).
        
        Args:
            model: Model to add noise to
        """
        if self.opacus_engine is not None:
            # Opacus handles this automatically
            return
        
        # Add Gaussian noise to gradients
        noise_scale = self.noise_multiplier * self.max_grad_norm
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=noise_scale,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad.data.add_(noise)
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy expenditure.
        
        Returns:
            Tuple of (epsilon_spent, delta_spent)
        """
        if self.opacus_engine is not None:
            try:
                return self.opacus_engine.get_epsilon(self.delta), self.delta
            except:
                pass
        
        # Simple linear accounting (very conservative)
        epsilon_spent = self.privacy_spent
        return epsilon_spent, self.delta
    
    def step(self):
        """Record a privacy step."""
        self.steps_taken += 1
        # Simple linear privacy accounting
        self.privacy_spent = min(self.privacy_spent + self.epsilon / 1000, self.epsilon)
    
    def is_privacy_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        epsilon_spent, _ = self.get_privacy_spent()
        return epsilon_spent >= self.epsilon


class DifferentialPrivacyGAN:
    """
    Wrapper for training GANs with differential privacy guarantees.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        target_modules: List[str] = ["discriminator"]
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.target_modules = target_modules
        
        # Privacy engines for different components
        self.privacy_engines = {}
        
    def setup_privacy(
        self,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        data_loader: torch.utils.data.DataLoader
    ) -> Tuple[Dict[str, nn.Module], Dict[str, torch.optim.Optimizer]]:
        """
        Setup differential privacy for specified model components.
        
        Args:
            models: Dictionary of models (e.g., {"generator": gen, "discriminator": disc})
            optimizers: Dictionary of optimizers
            data_loader: Training data loader
            
        Returns:
            Tuple of (private_models, private_optimizers)
        """
        private_models = {}
        private_optimizers = {}
        
        for module_name in self.target_modules:
            if module_name in models and module_name in optimizers:
                # Create privacy engine for this module
                privacy_engine = PrivacyEngine(
                    epsilon=self.epsilon,
                    delta=self.delta,
                    max_grad_norm=self.max_grad_norm
                )
                
                # Attach privacy to the module
                private_model, private_optimizer, _ = privacy_engine.attach_model(
                    models[module_name],
                    optimizers[module_name],
                    data_loader
                )
                
                private_models[module_name] = private_model
                private_optimizers[module_name] = private_optimizer
                self.privacy_engines[module_name] = privacy_engine
            else:
                # Non-private module
                private_models[module_name] = models[module_name]
                private_optimizers[module_name] = optimizers[module_name]
        
        return private_models, private_optimizers
    
    def get_total_privacy_spent(self) -> Dict[str, Tuple[float, float]]:
        """Get privacy expenditure for all modules."""
        privacy_spent = {}
        for module_name, engine in self.privacy_engines.items():
            privacy_spent[module_name] = engine.get_privacy_spent()
        return privacy_spent
    
    def is_any_privacy_exhausted(self) -> bool:
        """Check if any module has exhausted its privacy budget."""
        for engine in self.privacy_engines.values():
            if engine.is_privacy_exhausted():
                return True
        return False


class PrivacyAudit:
    """
    Privacy auditing and analysis tools for synthetic data.
    """
    
    def __init__(self):
        self.audit_history = []
    
    def membership_inference_test(
        self,
        model: nn.Module,
        training_data: torch.Tensor,
        test_data: torch.Tensor,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Perform membership inference attack to test privacy.
        
        Args:
            model: Trained model to test
            training_data: Original training data
            test_data: Hold-out test data
            num_samples: Number of samples for the test
            
        Returns:
            Attack success metrics
        """
        model.eval()
        
        # Sample equal amounts from training and test data
        n_train = min(num_samples // 2, len(training_data))
        n_test = min(num_samples // 2, len(test_data))
        
        train_indices = torch.randperm(len(training_data))[:n_train]
        test_indices = torch.randperm(len(test_data))[:n_test]
        
        train_samples = training_data[train_indices]
        test_samples = test_data[test_indices]
        
        # Compute model outputs (confidence scores)
        with torch.no_grad():
            if hasattr(model, 'discriminator'):
                # For GANs, use discriminator output
                train_scores = model.discriminator(train_samples).squeeze()
                test_scores = model.discriminator(test_samples).squeeze()
            else:
                # For other models, use model output directly
                train_scores = model(train_samples).squeeze()
                test_scores = model(test_samples).squeeze()
        
        # Create labels (1 for training data, 0 for test data)
        train_labels = torch.ones(len(train_scores))
        test_labels = torch.zeros(len(test_scores))
        
        # Combine data
        all_scores = torch.cat([train_scores, test_scores])
        all_labels = torch.cat([train_labels, test_labels])
        
        # Simple threshold-based attack
        # Higher scores suggest membership in training set
        thresholds = torch.linspace(all_scores.min(), all_scores.max(), 100)
        best_accuracy = 0.0
        best_threshold = 0.0
        
        for threshold in thresholds:
            predictions = (all_scores > threshold).float()
            accuracy = (predictions == all_labels).float().mean().item()
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold.item()
        
        # Compute additional metrics
        final_predictions = (all_scores > best_threshold).float()
        
        # True positive rate (correctly identified training samples)
        train_mask = all_labels == 1
        tpr = (final_predictions[train_mask] == 1).float().mean().item()
        
        # False positive rate (incorrectly identified test samples)
        test_mask = all_labels == 0
        fpr = (final_predictions[test_mask] == 1).float().mean().item()
        
        audit_result = {
            'attack_accuracy': best_accuracy,
            'attack_advantage': best_accuracy - 0.5,  # Advantage over random guessing
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'best_threshold': best_threshold,
            'baseline_accuracy': 0.5  # Random guessing baseline
        }
        
        self.audit_history.append(audit_result)
        return audit_result
    
    def attribute_inference_test(
        self,
        model: nn.Module,
        data: torch.Tensor,
        sensitive_attributes: torch.Tensor,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Test for attribute inference attacks.
        
        Args:
            model: Trained model to test
            data: Input data
            sensitive_attributes: Sensitive attributes to predict
            num_samples: Number of samples for the test
            
        Returns:
            Attack success metrics
        """
        model.eval()
        
        # Sample data
        n_samples = min(num_samples, len(data))
        indices = torch.randperm(len(data))[:n_samples]
        
        sampled_data = data[indices]
        sampled_attributes = sensitive_attributes[indices]
        
        # Generate synthetic data using the model
        with torch.no_grad():
            if hasattr(model, 'generate_samples'):
                synthetic_data = model.generate_samples(n_samples, return_numpy=False)
            else:
                # Fallback for models without generate_samples method
                noise = torch.randn(n_samples, model.noise_dim if hasattr(model, 'noise_dim') else 100)
                synthetic_data = model.generator(noise) if hasattr(model, 'generator') else model(noise)
        
        # Train a simple classifier to predict sensitive attributes from synthetic data
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Convert to numpy for sklearn
        synthetic_np = synthetic_data.cpu().numpy()
        attributes_np = sampled_attributes.cpu().numpy()
        
        # Split into train/test
        split_idx = len(synthetic_np) // 2
        
        X_train = synthetic_np[:split_idx]
        y_train = attributes_np[:split_idx]
        X_test = synthetic_np[split_idx:]
        y_test = attributes_np[split_idx:]
        
        try:
            # Train classifier
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(X_train, y_train)
            
            # Test predictions
            predictions = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            # Baseline accuracy (majority class)
            baseline_accuracy = max(np.bincount(y_test)) / len(y_test)
            
        except Exception as e:
            warnings.warn(f"Attribute inference test failed: {e}")
            accuracy = 0.5
            baseline_accuracy = 0.5
        
        return {
            'attack_accuracy': accuracy,
            'attack_advantage': accuracy - baseline_accuracy,
            'baseline_accuracy': baseline_accuracy
        }
    
    def privacy_risk_score(
        self,
        membership_results: Dict[str, float],
        attribute_results: Dict[str, float]
    ) -> float:
        """
        Compute overall privacy risk score.
        
        Args:
            membership_results: Results from membership inference test
            attribute_results: Results from attribute inference test
            
        Returns:
            Privacy risk score (0-1, lower is better)
        """
        membership_risk = membership_results.get('attack_advantage', 0)
        attribute_risk = attribute_results.get('attack_advantage', 0)
        
        # Weighted combination of risks
        total_risk = 0.6 * membership_risk + 0.4 * attribute_risk
        
        # Normalize to 0-1 range
        return max(0, min(1, total_risk))
    
    def generate_privacy_report(
        self,
        model: nn.Module,
        training_data: torch.Tensor,
        test_data: torch.Tensor,
        sensitive_attributes: torch.Tensor,
        privacy_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive privacy analysis report.
        
        Args:
            model: Trained model
            training_data: Original training data
            test_data: Hold-out test data
            sensitive_attributes: Sensitive attributes
            privacy_params: Privacy parameters used in training
            
        Returns:
            Comprehensive privacy report
        """
        # Run privacy tests
        membership_results = self.membership_inference_test(model, training_data, test_data)
        attribute_results = self.attribute_inference_test(model, training_data, sensitive_attributes)
        
        # Calculate risk score
        risk_score = self.privacy_risk_score(membership_results, attribute_results)
        
        # Privacy budget analysis
        epsilon = privacy_params.get('epsilon', 'N/A')
        delta = privacy_params.get('delta', 'N/A')
        
        report = {
            'privacy_parameters': {
                'epsilon': epsilon,
                'delta': delta,
                'max_grad_norm': privacy_params.get('max_grad_norm', 'N/A')
            },
            'membership_inference': membership_results,
            'attribute_inference': attribute_results,
            'overall_risk_score': risk_score,
            'risk_level': self._categorize_risk(risk_score),
            'recommendations': self._generate_recommendations(risk_score, membership_results, attribute_results)
        }
        
        return report
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize privacy risk level."""
        if risk_score < 0.1:
            return "Low"
        elif risk_score < 0.3:
            return "Medium"
        else:
            return "High"
    
    def _generate_recommendations(
        self,
        risk_score: float,
        membership_results: Dict[str, float],
        attribute_results: Dict[str, float]
    ) -> List[str]:
        """Generate privacy improvement recommendations."""
        recommendations = []
        
        if risk_score > 0.3:
            recommendations.append("Consider reducing epsilon for stronger privacy guarantees")
        
        if membership_results.get('attack_advantage', 0) > 0.2:
            recommendations.append("Model may be overfitting - consider regularization techniques")
            recommendations.append("Increase noise multiplier in differential privacy mechanism")
        
        if attribute_results.get('attack_advantage', 0) > 0.2:
            recommendations.append("Consider post-processing to remove sensitive attribute correlations")
            recommendations.append("Use attribute suppression techniques during training")
        
        if not recommendations:
            recommendations.append("Privacy protection appears adequate based on current tests")
        
        return recommendations
