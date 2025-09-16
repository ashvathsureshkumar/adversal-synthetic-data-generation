"""
Evaluation metrics and utilities for synthetic data quality assessment.

This module provides comprehensive evaluation of synthetic data quality including
statistical fidelity, distributional similarity, and privacy preservation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
import logging


class SyntheticDataEvaluator:
    """
    Comprehensive evaluator for synthetic data quality assessment.
    
    Provides statistical, distributional, and utility-based evaluation metrics
    to assess the quality and privacy of generated synthetic data.
    """
    
    def __init__(
        self,
        real_data: np.ndarray,
        column_names: Optional[List[str]] = None,
        categorical_columns: Optional[List[int]] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            real_data: Original real dataset
            column_names: Names of the columns
            categorical_columns: Indices of categorical columns
        """
        self.real_data = real_data
        self.column_names = column_names or [f"feature_{i}" for i in range(real_data.shape[1])]
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = [i for i in range(real_data.shape[1]) 
                                 if i not in self.categorical_columns]
        
        self.logger = logging.getLogger(__name__)
        
        # Pre-compute real data statistics
        self._compute_real_data_stats()
    
    def _compute_real_data_stats(self):
        """Pre-compute statistics for real data."""
        self.real_stats = {}
        
        # Overall statistics
        self.real_stats['mean'] = np.mean(self.real_data, axis=0)
        self.real_stats['std'] = np.std(self.real_data, axis=0)
        self.real_stats['min'] = np.min(self.real_data, axis=0)
        self.real_stats['max'] = np.max(self.real_data, axis=0)
        
        # Correlation matrix
        self.real_stats['correlation'] = np.corrcoef(self.real_data.T)
        
        # Column-wise statistics
        self.real_stats['columns'] = {}
        for i, col_name in enumerate(self.column_names):
            col_data = self.real_data[:, i]
            self.real_stats['columns'][col_name] = {
                'mean': np.mean(col_data),
                'std': np.std(col_data),
                'skewness': stats.skew(col_data),
                'kurtosis': stats.kurtosis(col_data),
                'percentiles': np.percentile(col_data, [25, 50, 75])
            }
    
    def evaluate(self, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive evaluation of synthetic data quality.
        
        Args:
            synthetic_data: Generated synthetic dataset
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.logger.info(f"Evaluating synthetic data quality: {synthetic_data.shape}")
        
        results = {
            'basic_statistics': self._evaluate_basic_statistics(synthetic_data),
            'distributional_similarity': self._evaluate_distributional_similarity(synthetic_data),
            'correlation_similarity': self._evaluate_correlation_similarity(synthetic_data),
            'machine_learning_efficacy': self._evaluate_ml_efficacy(synthetic_data),
            'privacy_metrics': self._evaluate_privacy_metrics(synthetic_data),
            'overall_quality': {}
        }
        
        # Compute overall quality score
        results['overall_quality'] = self._compute_overall_quality(results)
        
        return results
    
    def _evaluate_basic_statistics(self, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate basic statistical properties."""
        synth_stats = {
            'mean': np.mean(synthetic_data, axis=0),
            'std': np.std(synthetic_data, axis=0),
            'min': np.min(synthetic_data, axis=0),
            'max': np.max(synthetic_data, axis=0)
        }
        
        # Compute differences
        mean_error = np.mean(np.abs(synth_stats['mean'] - self.real_stats['mean']))
        std_error = np.mean(np.abs(synth_stats['std'] - self.real_stats['std']))
        
        # Compute relative errors
        mean_relative_error = np.mean(
            np.abs(synth_stats['mean'] - self.real_stats['mean']) / 
            (np.abs(self.real_stats['mean']) + 1e-8)
        )
        
        std_relative_error = np.mean(
            np.abs(synth_stats['std'] - self.real_stats['std']) / 
            (self.real_stats['std'] + 1e-8)
        )
        
        return {
            'mean_absolute_error': mean_error,
            'std_absolute_error': std_error,
            'mean_relative_error': mean_relative_error,
            'std_relative_error': std_relative_error,
            'synthetic_stats': synth_stats,
            'score': max(0, 1 - (mean_relative_error + std_relative_error) / 2)
        }
    
    def _evaluate_distributional_similarity(self, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate distributional similarity using various metrics."""
        results = {
            'wasserstein_distances': [],
            'ks_test_pvalues': [],
            'js_divergences': [],
            'mean_wasserstein_distance': 0.0,
            'mean_ks_pvalue': 0.0,
            'mean_js_divergence': 0.0
        }
        
        for i in range(self.real_data.shape[1]):
            real_col = self.real_data[:, i]
            synth_col = synthetic_data[:, i]
            
            # Wasserstein distance
            try:
                wd = wasserstein_distance(real_col, synth_col)
                results['wasserstein_distances'].append(wd)
            except Exception:
                results['wasserstein_distances'].append(float('inf'))
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_pvalue = stats.ks_2samp(real_col, synth_col)
                results['ks_test_pvalues'].append(ks_pvalue)
            except Exception:
                results['ks_test_pvalues'].append(0.0)
            
            # Jensen-Shannon divergence (for discretized data)
            try:
                # Discretize data into bins
                bins = 50
                real_hist, bin_edges = np.histogram(real_col, bins=bins, density=True)
                synth_hist, _ = np.histogram(synth_col, bins=bin_edges, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-8
                real_hist = real_hist + epsilon
                synth_hist = synth_hist + epsilon
                
                # Normalize
                real_hist = real_hist / np.sum(real_hist)
                synth_hist = synth_hist / np.sum(synth_hist)
                
                # Jensen-Shannon divergence
                m = 0.5 * (real_hist + synth_hist)
                js_div = 0.5 * stats.entropy(real_hist, m) + 0.5 * stats.entropy(synth_hist, m)
                results['js_divergences'].append(js_div)
            except Exception:
                results['js_divergences'].append(float('inf'))
        
        # Compute averages
        results['mean_wasserstein_distance'] = np.mean([d for d in results['wasserstein_distances'] 
                                                       if not np.isinf(d)])
        results['mean_ks_pvalue'] = np.mean(results['ks_test_pvalues'])
        results['mean_js_divergence'] = np.mean([d for d in results['js_divergences'] 
                                               if not np.isinf(d)])
        
        # Overall distributional similarity score
        # Higher KS p-value is better, lower JS divergence is better
        ks_score = min(1.0, results['mean_ks_pvalue'] * 2)  # Scale p-value
        js_score = max(0.0, 1.0 - results['mean_js_divergence'])  # Invert JS divergence
        
        results['score'] = (ks_score + js_score) / 2
        
        return results
    
    def _evaluate_correlation_similarity(self, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate similarity of correlation structures."""
        synth_corr = np.corrcoef(synthetic_data.T)
        real_corr = self.real_stats['correlation']
        
        # Handle NaN values in correlation matrices
        real_corr = np.nan_to_num(real_corr)
        synth_corr = np.nan_to_num(synth_corr)
        
        # Frobenius norm of the difference
        frobenius_distance = np.linalg.norm(real_corr - synth_corr, 'fro')
        
        # Mean absolute error of correlations
        correlation_mae = np.mean(np.abs(real_corr - synth_corr))
        
        # Correlation of correlations (flattened upper triangular parts)
        mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
        real_corr_flat = real_corr[mask]
        synth_corr_flat = synth_corr[mask]
        
        try:
            correlation_of_correlations = stats.pearsonr(real_corr_flat, synth_corr_flat)[0]
            if np.isnan(correlation_of_correlations):
                correlation_of_correlations = 0.0
        except Exception:
            correlation_of_correlations = 0.0
        
        # Score based on correlation preservation
        score = max(0, correlation_of_correlations)
        
        return {
            'frobenius_distance': frobenius_distance,
            'correlation_mae': correlation_mae,
            'correlation_of_correlations': correlation_of_correlations,
            'synthetic_correlation_matrix': synth_corr,
            'score': score
        }
    
    def _evaluate_ml_efficacy(self, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate utility for machine learning tasks."""
        results = {
            'train_on_synthetic_test_on_real': {},
            'train_on_real_test_on_synthetic': {},
            'discriminator_test': {}
        }
        
        try:
            # Test 1: Train on synthetic, test on real
            # Use a simple classification task based on data quartiles
            real_target = self._create_classification_target(self.real_data)
            synth_target = self._create_classification_target(synthetic_data)
            
            # Train on synthetic, test on real
            if len(np.unique(synth_target)) > 1 and len(np.unique(real_target)) > 1:
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(synthetic_data, synth_target)
                
                real_pred = clf.predict(self.real_data)
                real_pred_proba = clf.predict_proba(self.real_data)[:, 1] if real_pred_proba.shape[1] == 2 else None
                
                train_synth_accuracy = accuracy_score(real_target, real_pred)
                train_synth_auc = roc_auc_score(real_target, real_pred_proba) if real_pred_proba is not None else 0.5
                
                results['train_on_synthetic_test_on_real'] = {
                    'accuracy': train_synth_accuracy,
                    'auc': train_synth_auc
                }
            
            # Test 2: Train on real, test on synthetic
            if len(np.unique(real_target)) > 1 and len(np.unique(synth_target)) > 1:
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(self.real_data, real_target)
                
                synth_pred = clf.predict(synthetic_data)
                synth_pred_proba = clf.predict_proba(synthetic_data)[:, 1] if clf.predict_proba(synthetic_data).shape[1] == 2 else None
                
                train_real_accuracy = accuracy_score(synth_target, synth_pred)
                train_real_auc = roc_auc_score(synth_target, synth_pred_proba) if synth_pred_proba is not None else 0.5
                
                results['train_on_real_test_on_synthetic'] = {
                    'accuracy': train_real_accuracy,
                    'auc': train_real_auc
                }
            
            # Test 3: Discriminator test (can we distinguish real from synthetic?)
            discriminator_result = self._discriminator_test(synthetic_data)
            results['discriminator_test'] = discriminator_result
            
        except Exception as e:
            self.logger.warning(f"ML efficacy evaluation failed: {e}")
            results['error'] = str(e)
        
        # Compute overall ML efficacy score
        scores = []
        if 'accuracy' in results['train_on_synthetic_test_on_real']:
            scores.append(results['train_on_synthetic_test_on_real']['accuracy'])
        if 'accuracy' in results['train_on_real_test_on_synthetic']:
            scores.append(results['train_on_real_test_on_synthetic']['accuracy'])
        if 'accuracy' in results['discriminator_test']:
            # For discriminator, lower accuracy is better (harder to distinguish)
            scores.append(1.0 - results['discriminator_test']['accuracy'])
        
        results['score'] = np.mean(scores) if scores else 0.0
        
        return results
    
    def _create_classification_target(self, data: np.ndarray) -> np.ndarray:
        """Create a binary classification target based on data characteristics."""
        # Use the first principal component to create binary target
        try:
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(data).flatten()
            median_pc1 = np.median(pc1)
            return (pc1 > median_pc1).astype(int)
        except Exception:
            # Fallback: use mean of first column
            return (data[:, 0] > np.mean(data[:, 0])).astype(int)
    
    def _discriminator_test(self, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Test how well a classifier can distinguish real from synthetic data."""
        try:
            # Combine real and synthetic data
            combined_data = np.vstack([self.real_data, synthetic_data])
            combined_labels = np.hstack([
                np.ones(len(self.real_data)),  # Real = 1
                np.zeros(len(synthetic_data))  # Synthetic = 0
            ])
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                combined_data, combined_labels, test_size=0.3, random_state=42, stratify=combined_labels
            )
            
            # Train discriminator
            discriminator = LogisticRegression(random_state=42, max_iter=1000)
            discriminator.fit(X_train, y_train)
            
            # Evaluate
            train_accuracy = discriminator.score(X_train, y_train)
            test_accuracy = discriminator.score(X_test, y_test)
            
            # Get predictions and probabilities
            y_pred = discriminator.predict(X_test)
            y_pred_proba = discriminator.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            return {
                'train_accuracy': train_accuracy,
                'accuracy': test_accuracy,
                'auc': auc_score,
                'interpretation': self._interpret_discriminator_score(test_accuracy)
            }
            
        except Exception as e:
            return {'error': str(e), 'accuracy': 0.5, 'auc': 0.5}
    
    def _interpret_discriminator_score(self, accuracy: float) -> str:
        """Interpret discriminator accuracy score."""
        if accuracy < 0.55:
            return "Excellent - Very hard to distinguish real from synthetic"
        elif accuracy < 0.65:
            return "Good - Moderately hard to distinguish"
        elif accuracy < 0.75:
            return "Fair - Somewhat easy to distinguish"
        else:
            return "Poor - Easy to distinguish real from synthetic"
    
    def _evaluate_privacy_metrics(self, synthetic_data: np.ndarray) -> Dict[str, Any]:
        """Evaluate privacy preservation metrics."""
        results = {
            'nearest_neighbor_distance': {},
            'dcr_score': 0.0,  # Distance to Closest Record
            'nndr_score': 0.0  # Nearest Neighbor Distance Ratio
        }
        
        try:
            # Distance to Closest Record (DCR)
            dcr_distances = []
            
            for i in range(min(1000, len(synthetic_data))):  # Sample for efficiency
                synth_sample = synthetic_data[i:i+1]
                
                # Find distance to closest real record
                distances = np.linalg.norm(self.real_data - synth_sample, axis=1)
                min_distance = np.min(distances)
                dcr_distances.append(min_distance)
            
            results['nearest_neighbor_distance']['mean_dcr'] = np.mean(dcr_distances)
            results['nearest_neighbor_distance']['std_dcr'] = np.std(dcr_distances)
            results['nearest_neighbor_distance']['min_dcr'] = np.min(dcr_distances)
            
            # DCR score (higher is better for privacy)
            # Normalize by the average distance between real samples
            real_distances = []
            for i in range(min(500, len(self.real_data))):
                other_indices = np.arange(len(self.real_data))
                other_indices = other_indices[other_indices != i]
                if len(other_indices) > 0:
                    distances = np.linalg.norm(
                        self.real_data[other_indices] - self.real_data[i:i+1], axis=1
                    )
                    real_distances.append(np.min(distances))
            
            avg_real_distance = np.mean(real_distances) if real_distances else 1.0
            results['dcr_score'] = results['nearest_neighbor_distance']['mean_dcr'] / avg_real_distance
            
            # Privacy risk assessment
            if results['nearest_neighbor_distance']['min_dcr'] < avg_real_distance * 0.1:
                results['privacy_risk'] = "High - Some synthetic samples very close to real data"
            elif results['nearest_neighbor_distance']['min_dcr'] < avg_real_distance * 0.5:
                results['privacy_risk'] = "Medium - Some synthetic samples moderately close"
            else:
                results['privacy_risk'] = "Low - Synthetic samples well separated from real data"
            
        except Exception as e:
            results['error'] = str(e)
            self.logger.warning(f"Privacy metrics evaluation failed: {e}")
        
        # Overall privacy score
        results['score'] = min(1.0, results['dcr_score'])
        
        return results
    
    def _compute_overall_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall quality score from individual metrics."""
        scores = []
        weights = []
        
        # Statistical fidelity (weight: 0.3)
        if 'score' in results['basic_statistics']:
            scores.append(results['basic_statistics']['score'])
            weights.append(0.3)
        
        # Distributional similarity (weight: 0.3)
        if 'score' in results['distributional_similarity']:
            scores.append(results['distributional_similarity']['score'])
            weights.append(0.3)
        
        # Correlation preservation (weight: 0.2)
        if 'score' in results['correlation_similarity']:
            scores.append(results['correlation_similarity']['score'])
            weights.append(0.2)
        
        # Machine learning efficacy (weight: 0.15)
        if 'score' in results['machine_learning_efficacy']:
            scores.append(results['machine_learning_efficacy']['score'])
            weights.append(0.15)
        
        # Privacy preservation (weight: 0.05)
        if 'score' in results['privacy_metrics']:
            scores.append(results['privacy_metrics']['score'])
            weights.append(0.05)
        
        if scores and weights:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 0.0
        
        # Quality assessment
        if overall_score >= 0.8:
            assessment = "Excellent synthetic data quality"
        elif overall_score >= 0.6:
            assessment = "Good synthetic data quality"
        elif overall_score >= 0.4:
            assessment = "Fair synthetic data quality - improvements needed"
        else:
            assessment = "Poor synthetic data quality - significant improvements needed"
        
        return {
            'overall_score': overall_score,
            'assessment': assessment,
            'component_scores': dict(zip(['statistics', 'distributions', 'correlations', 'ml_efficacy', 'privacy'], scores)),
            'weights_used': dict(zip(['statistics', 'distributions', 'correlations', 'ml_efficacy', 'privacy'], weights))
        }
    
    def generate_report(self, synthetic_data: np.ndarray, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            synthetic_data: Generated synthetic dataset
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        results = self.evaluate(synthetic_data)
        
        report_lines = [
            "=" * 80,
            "SYNTHETIC DATA QUALITY EVALUATION REPORT",
            "=" * 80,
            "",
            f"Dataset Information:",
            f"  Real data shape: {self.real_data.shape}",
            f"  Synthetic data shape: {synthetic_data.shape}",
            f"  Number of features: {len(self.column_names)}",
            f"  Categorical features: {len(self.categorical_columns)}",
            f"  Numerical features: {len(self.numerical_columns)}",
            "",
            "=" * 50,
            "OVERALL QUALITY ASSESSMENT",
            "=" * 50,
            f"Overall Score: {results['overall_quality']['overall_score']:.3f}",
            f"Assessment: {results['overall_quality']['assessment']}",
            "",
        ]
        
        # Component scores
        if 'component_scores' in results['overall_quality']:
            report_lines.extend([
                "Component Scores:",
                f"  Statistical Fidelity: {results['overall_quality']['component_scores'].get('statistics', 0):.3f}",
                f"  Distributional Similarity: {results['overall_quality']['component_scores'].get('distributions', 0):.3f}",
                f"  Correlation Preservation: {results['overall_quality']['component_scores'].get('correlations', 0):.3f}",
                f"  ML Efficacy: {results['overall_quality']['component_scores'].get('ml_efficacy', 0):.3f}",
                f"  Privacy Preservation: {results['overall_quality']['component_scores'].get('privacy', 0):.3f}",
                "",
            ])
        
        # Detailed results
        report_lines.extend([
            "=" * 50,
            "DETAILED EVALUATION RESULTS",
            "=" * 50,
            "",
            "1. Basic Statistics:",
            f"   Mean Absolute Error: {results['basic_statistics']['mean_absolute_error']:.6f}",
            f"   Std Absolute Error: {results['basic_statistics']['std_absolute_error']:.6f}",
            f"   Mean Relative Error: {results['basic_statistics']['mean_relative_error']:.6f}",
            f"   Std Relative Error: {results['basic_statistics']['std_relative_error']:.6f}",
            "",
            "2. Distributional Similarity:",
            f"   Mean Wasserstein Distance: {results['distributional_similarity']['mean_wasserstein_distance']:.6f}",
            f"   Mean KS Test P-value: {results['distributional_similarity']['mean_ks_pvalue']:.6f}",
            f"   Mean JS Divergence: {results['distributional_similarity']['mean_js_divergence']:.6f}",
            "",
            "3. Correlation Similarity:",
            f"   Correlation MAE: {results['correlation_similarity']['correlation_mae']:.6f}",
            f"   Correlation of Correlations: {results['correlation_similarity']['correlation_of_correlations']:.6f}",
            "",
            "4. Machine Learning Efficacy:",
        ])
        
        if 'train_on_synthetic_test_on_real' in results['machine_learning_efficacy']:
            tstr = results['machine_learning_efficacy']['train_on_synthetic_test_on_real']
            report_lines.append(f"   Train on Synthetic, Test on Real - Accuracy: {tstr.get('accuracy', 0):.3f}")
        
        if 'discriminator_test' in results['machine_learning_efficacy']:
            disc = results['machine_learning_efficacy']['discriminator_test']
            report_lines.extend([
                f"   Discriminator Test - Accuracy: {disc.get('accuracy', 0):.3f}",
                f"   Interpretation: {disc.get('interpretation', 'N/A')}",
            ])
        
        report_lines.extend([
            "",
            "5. Privacy Metrics:",
            f"   DCR Score: {results['privacy_metrics']['dcr_score']:.6f}",
            f"   Privacy Risk: {results['privacy_metrics'].get('privacy_risk', 'Unknown')}",
            "",
            "=" * 80,
        ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Evaluation report saved to: {save_path}")
        
        return report


class QualityMetrics:
    """
    Standalone quality metrics calculator for quick evaluations.
    """
    
    @staticmethod
    def wasserstein_distance_score(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
        """Calculate average Wasserstein distance across all features."""
        distances = []
        for i in range(real_data.shape[1]):
            try:
                wd = wasserstein_distance(real_data[:, i], synthetic_data[:, i])
                distances.append(wd)
            except Exception:
                continue
        return np.mean(distances) if distances else float('inf')
    
    @staticmethod
    def correlation_difference(real_data: np.ndarray, synthetic_data: np.ndarray) -> float:
        """Calculate difference in correlation matrices."""
        real_corr = np.corrcoef(real_data.T)
        synth_corr = np.corrcoef(synthetic_data.T)
        
        real_corr = np.nan_to_num(real_corr)
        synth_corr = np.nan_to_num(synth_corr)
        
        return np.mean(np.abs(real_corr - synth_corr))
    
    @staticmethod
    def statistical_distance(real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical distances."""
        real_mean = np.mean(real_data, axis=0)
        synth_mean = np.mean(synthetic_data, axis=0)
        real_std = np.std(real_data, axis=0)
        synth_std = np.std(synthetic_data, axis=0)
        
        return {
            'mean_absolute_error': np.mean(np.abs(real_mean - synth_mean)),
            'std_absolute_error': np.mean(np.abs(real_std - synth_std)),
            'mean_relative_error': np.mean(np.abs(real_mean - synth_mean) / (np.abs(real_mean) + 1e-8)),
            'std_relative_error': np.mean(np.abs(real_std - synth_std) / (real_std + 1e-8))
        }
