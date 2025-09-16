"""
Data preprocessing utilities for synthetic data generation.

This module provides comprehensive data preprocessing capabilities including
categorical encoding, numerical scaling, missing value handling, and data validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import logging
import warnings


class DataPreprocessor:
    """
    Comprehensive data preprocessor for synthetic data generation.
    
    Handles categorical encoding, numerical scaling, missing values,
    and data type conversions to prepare data for GAN training.
    """
    
    def __init__(
        self,
        categorical_threshold: int = 10,
        numerical_scaling: str = "standard",
        handle_missing: str = "median",
        encode_categorical: str = "label",
        detect_types: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            categorical_threshold: Max unique values to consider categorical
            numerical_scaling: Type of scaling ("standard", "minmax", "robust", "none")
            handle_missing: Missing value strategy ("mean", "median", "mode", "drop", "knn")
            encode_categorical: Categorical encoding ("label", "onehot", "target")
            detect_types: Whether to auto-detect data types
        """
        self.categorical_threshold = categorical_threshold
        self.numerical_scaling = numerical_scaling
        self.handle_missing = handle_missing
        self.encode_categorical = encode_categorical
        self.detect_types = detect_types
        
        # Fitted components
        self.scaler = None
        self.imputer = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.column_info = {}
        self.feature_names = []
        
        # State
        self.is_fitted = False
        
        self.logger = logging.getLogger(__name__)
    
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            data: Training dataset
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting preprocessor on data with shape: {data.shape}")
        
        # Detect data types
        self.column_info = self._analyze_columns(data)
        
        # Prepare data for fitting
        processed_data = data.copy()
        
        # Handle missing values first
        processed_data = self._fit_missing_value_handler(processed_data)
        
        # Encode categorical variables
        processed_data = self._fit_categorical_encoder(processed_data)
        
        # Scale numerical variables
        processed_data = self._fit_numerical_scaler(processed_data)
        
        # Store feature names
        self.feature_names = processed_data.columns.tolist()
        
        self.is_fitted = True
        self.logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names)}")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        self.logger.info(f"Transforming data with shape: {data.shape}")
        
        processed_data = data.copy()
        
        # Apply transformations in the same order as fitting
        processed_data = self._transform_missing_values(processed_data)
        processed_data = self._transform_categorical(processed_data)
        processed_data = self._transform_numerical(processed_data)
        
        # Ensure consistent feature order
        if set(processed_data.columns) == set(self.feature_names):
            processed_data = processed_data[self.feature_names]
        
        self.logger.info(f"Data transformed to shape: {processed_data.shape}")
        return processed_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data in one step."""
        return self.fit(data).transform(data)
    
    def _analyze_columns(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze columns to determine data types and characteristics.
        
        Args:
            data: Dataset to analyze
            
        Returns:
            Dictionary with column information
        """
        column_info = {}
        
        for col in data.columns:
            col_data = data[col]
            info = {
                'dtype': str(col_data.dtype),
                'unique_values': col_data.nunique(),
                'missing_count': col_data.isnull().sum(),
                'missing_percentage': col_data.isnull().sum() / len(col_data),
                'is_categorical': False,
                'is_numerical': False,
                'is_binary': False
            }
            
            # Determine if categorical
            if self.detect_types:
                if (col_data.dtype == 'object' or 
                    info['unique_values'] <= self.categorical_threshold):
                    info['is_categorical'] = True
                else:
                    info['is_numerical'] = True
                
                # Check if binary
                if info['unique_values'] == 2:
                    info['is_binary'] = True
            else:
                # Use dtype to determine type
                if col_data.dtype in ['int64', 'float64']:
                    info['is_numerical'] = True
                else:
                    info['is_categorical'] = True
            
            # Additional statistics for numerical columns
            if info['is_numerical']:
                info.update({
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                })
            
            column_info[col] = info
        
        return column_info
    
    def _fit_missing_value_handler(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit missing value handler."""
        if self.handle_missing == "drop":
            # Don't fit anything for drop strategy
            return data
        elif self.handle_missing == "knn":
            self.imputer = KNNImputer(n_neighbors=5)
            # Convert categorical columns to numeric for KNN
            temp_data = data.copy()
            for col in data.columns:
                if self.column_info[col]['is_categorical']:
                    le = LabelEncoder()
                    temp_data[col] = le.fit_transform(temp_data[col].astype(str))
            
            self.imputer.fit(temp_data)
            return data
        else:
            # Simple imputation strategies
            strategy_map = {
                "mean": "mean",
                "median": "median", 
                "mode": "most_frequent"
            }
            
            if self.handle_missing in strategy_map:
                self.imputer = SimpleImputer(strategy=strategy_map[self.handle_missing])
                
                # Fit on numerical columns only for mean/median
                if self.handle_missing in ["mean", "median"]:
                    numerical_cols = [col for col, info in self.column_info.items() 
                                    if info['is_numerical']]
                    if numerical_cols:
                        self.imputer.fit(data[numerical_cols])
                else:
                    # Mode can work on all columns
                    self.imputer.fit(data)
        
        return data
    
    def _transform_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values using fitted imputer."""
        if self.handle_missing == "drop":
            return data.dropna()
        elif self.imputer is not None:
            if self.handle_missing == "knn":
                # Apply KNN imputation
                temp_data = data.copy()
                categorical_encoders = {}
                
                # Encode categorical columns temporarily
                for col in data.columns:
                    if self.column_info[col]['is_categorical']:
                        le = LabelEncoder()
                        # Handle unseen categories
                        unique_vals = temp_data[col].dropna().unique()
                        le.fit(unique_vals.astype(str))
                        temp_data[col] = temp_data[col].astype(str)
                        temp_data[col] = temp_data[col].map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                        categorical_encoders[col] = le
                
                # Apply KNN imputation
                imputed_data = self.imputer.transform(temp_data)
                result_data = pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
                
                # Decode categorical columns
                for col, le in categorical_encoders.items():
                    result_data[col] = result_data[col].round().astype(int)
                    result_data[col] = result_data[col].map(
                        lambda x: le.inverse_transform([max(0, min(x, len(le.classes_)-1))])[0]
                        if x >= 0 else le.classes_[0]
                    )
                
                return result_data
            else:
                # Simple imputation
                if self.handle_missing in ["mean", "median"]:
                    numerical_cols = [col for col, info in self.column_info.items() 
                                    if info['is_numerical']]
                    data[numerical_cols] = self.imputer.transform(data[numerical_cols])
                else:
                    data = pd.DataFrame(
                        self.imputer.transform(data),
                        columns=data.columns,
                        index=data.index
                    )
        
        return data
    
    def _fit_categorical_encoder(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit categorical encoder."""
        categorical_cols = [col for col, info in self.column_info.items() 
                          if info['is_categorical']]
        
        if not categorical_cols:
            return data
        
        if self.encode_categorical == "label":
            # Label encoding
            for col in categorical_cols:
                le = LabelEncoder()
                le.fit(data[col].astype(str))
                self.label_encoders[col] = le
        
        elif self.encode_categorical == "onehot":
            # One-hot encoding
            self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.onehot_encoder.fit(data[categorical_cols])
        
        return data
    
    def _transform_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables."""
        categorical_cols = [col for col, info in self.column_info.items() 
                          if info['is_categorical']]
        
        if not categorical_cols:
            return data
        
        result_data = data.copy()
        
        if self.encode_categorical == "label":
            # Apply label encoding
            for col in categorical_cols:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    result_data[col] = result_data[col].astype(str)
                    result_data[col] = result_data[col].map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
        
        elif self.encode_categorical == "onehot":
            # Apply one-hot encoding
            if self.onehot_encoder is not None:
                encoded_data = self.onehot_encoder.transform(data[categorical_cols])
                
                # Create feature names for one-hot encoded columns
                feature_names = []
                for i, col in enumerate(categorical_cols):
                    n_categories = len(self.onehot_encoder.categories_[i])
                    for j in range(n_categories):
                        feature_names.append(f"{col}_{j}")
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=feature_names,
                    index=data.index
                )
                
                # Drop original categorical columns and add encoded ones
                result_data = result_data.drop(columns=categorical_cols)
                result_data = pd.concat([result_data, encoded_df], axis=1)
        
        return result_data
    
    def _fit_numerical_scaler(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit numerical scaler."""
        if self.numerical_scaling == "none":
            return data
        
        numerical_cols = [col for col, info in self.column_info.items() 
                         if info['is_numerical']]
        
        if not numerical_cols:
            return data
        
        # Create scaler based on strategy
        if self.numerical_scaling == "standard":
            self.scaler = StandardScaler()
        elif self.numerical_scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif self.numerical_scaling == "robust":
            self.scaler = RobustScaler()
        else:
            warnings.warn(f"Unknown scaling method: {self.numerical_scaling}")
            return data
        
        # Fit scaler on numerical columns
        self.scaler.fit(data[numerical_cols])
        
        return data
    
    def _transform_numerical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical variables."""
        if self.numerical_scaling == "none" or self.scaler is None:
            return data
        
        numerical_cols = [col for col, info in self.column_info.items() 
                         if info['is_numerical']]
        
        if not numerical_cols:
            return data
        
        result_data = data.copy()
        
        # Apply scaling
        scaled_data = self.scaler.transform(data[numerical_cols])
        result_data[numerical_cols] = scaled_data
        
        return result_data
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform processed data back to original format.
        
        Args:
            data: Processed data to inverse transform
            
        Returns:
            Data in original format
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse transform")
        
        result_data = data.copy()
        
        # Inverse transform numerical scaling
        if self.scaler is not None:
            numerical_cols = [col for col, info in self.column_info.items() 
                             if info['is_numerical'] and col in result_data.columns]
            if numerical_cols:
                result_data[numerical_cols] = self.scaler.inverse_transform(
                    result_data[numerical_cols]
                )
        
        # Inverse transform categorical encoding
        if self.encode_categorical == "label":
            for col, le in self.label_encoders.items():
                if col in result_data.columns:
                    # Round to nearest integer and clip to valid range
                    result_data[col] = result_data[col].round().astype(int)
                    result_data[col] = result_data[col].clip(0, len(le.classes_) - 1)
                    result_data[col] = le.inverse_transform(result_data[col])
        
        elif self.encode_categorical == "onehot" and self.onehot_encoder is not None:
            # This is more complex for one-hot encoding
            # Would need to reconstruct original categorical columns
            pass
        
        return result_data
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing steps applied."""
        return {
            'is_fitted': self.is_fitted,
            'column_info': self.column_info,
            'feature_names': self.feature_names,
            'categorical_threshold': self.categorical_threshold,
            'numerical_scaling': self.numerical_scaling,
            'handle_missing': self.handle_missing,
            'encode_categorical': self.encode_categorical,
            'n_features_original': len(self.column_info),
            'n_features_processed': len(self.feature_names)
        }


class DataValidator:
    """
    Data validation utilities for synthetic data generation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return a report.
        
        Args:
            data: Dataset to validate
            
        Returns:
            Validation report
        """
        report = {
            'shape': data.shape,
            'missing_values': {},
            'duplicate_rows': 0,
            'data_types': {},
            'numerical_outliers': {},
            'categorical_distribution': {},
            'quality_score': 0.0,
            'issues': [],
            'warnings': []
        }
        
        # Check missing values
        missing_counts = data.isnull().sum()
        report['missing_values'] = missing_counts.to_dict()
        
        total_missing = missing_counts.sum()
        if total_missing > 0:
            report['warnings'].append(f"Found {total_missing} missing values")
        
        # Check duplicate rows
        duplicate_count = data.duplicated().sum()
        report['duplicate_rows'] = duplicate_count
        
        if duplicate_count > 0:
            report['warnings'].append(f"Found {duplicate_count} duplicate rows")
        
        # Analyze data types
        report['data_types'] = data.dtypes.value_counts().to_dict()
        
        # Check for numerical outliers (using IQR method)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count = len(outliers)
            
            report['numerical_outliers'][col] = {
                'count': outlier_count,
                'percentage': outlier_count / len(data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if outlier_count > len(data) * 0.05:  # More than 5% outliers
                report['warnings'].append(f"High outlier count in {col}: {outlier_count}")
        
        # Analyze categorical distributions
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            total_count = len(data)
            
            report['categorical_distribution'][col] = {
                'unique_values': len(value_counts),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'most_frequent_percentage': value_counts.iloc[0] / total_count * 100 if len(value_counts) > 0 else 0
            }
            
            # Check for high cardinality
            if len(value_counts) > total_count * 0.5:
                report['warnings'].append(f"High cardinality in {col}: {len(value_counts)} unique values")
            
            # Check for imbalanced distribution
            if len(value_counts) > 1:
                max_freq = value_counts.iloc[0]
                min_freq = value_counts.iloc[-1]
                imbalance_ratio = max_freq / min_freq
                
                if imbalance_ratio > 100:
                    report['warnings'].append(f"Highly imbalanced distribution in {col}")
        
        # Calculate overall quality score
        quality_factors = []
        
        # Missing values factor (0-1, higher is better)
        missing_percentage = total_missing / (data.shape[0] * data.shape[1])
        missing_factor = max(0, 1 - missing_percentage * 2)  # Penalize missing values
        quality_factors.append(missing_factor)
        
        # Duplicate rows factor
        duplicate_percentage = duplicate_count / data.shape[0]
        duplicate_factor = max(0, 1 - duplicate_percentage * 2)
        quality_factors.append(duplicate_factor)
        
        # Data type consistency factor
        object_percentage = (data.dtypes == 'object').sum() / len(data.columns)
        type_factor = 1 - abs(object_percentage - 0.3)  # Assume ~30% categorical is good
        quality_factors.append(max(0, type_factor))
        
        report['quality_score'] = np.mean(quality_factors)
        
        # Determine overall assessment
        if report['quality_score'] >= 0.8:
            report['assessment'] = "Good quality data"
        elif report['quality_score'] >= 0.6:
            report['assessment'] = "Moderate quality data - consider preprocessing"
        else:
            report['assessment'] = "Poor quality data - significant preprocessing needed"
            report['issues'].append("Low overall quality score")
        
        return report
    
    def validate_for_synthetic_generation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data specifically for synthetic data generation.
        
        Args:
            data: Dataset to validate
            
        Returns:
            Validation report with synthetic generation recommendations
        """
        base_report = self.validate_data_quality(data)
        
        # Additional checks for synthetic generation
        synthetic_report = {
            **base_report,
            'synthetic_generation_ready': True,
            'recommendations': []
        }
        
        # Check minimum data size
        if len(data) < 1000:
            synthetic_report['synthetic_generation_ready'] = False
            synthetic_report['recommendations'].append(
                "Dataset is too small for reliable synthetic generation (< 1000 rows)"
            )
        
        # Check for too many categorical columns with high cardinality
        high_cardinality_cols = 0
        for col, info in synthetic_report['categorical_distribution'].items():
            if info['unique_values'] > 50:
                high_cardinality_cols += 1
        
        if high_cardinality_cols > len(data.columns) * 0.3:
            synthetic_report['recommendations'].append(
                "Consider reducing cardinality of categorical columns or using different encoding"
            )
        
        # Check for extreme class imbalance
        for col, info in synthetic_report['categorical_distribution'].items():
            if info['most_frequent_percentage'] > 95:
                synthetic_report['recommendations'].append(
                    f"Extreme class imbalance in {col} may affect generation quality"
                )
        
        # Check for too many missing values
        total_missing_percentage = sum(synthetic_report['missing_values'].values()) / (
            data.shape[0] * data.shape[1]
        )
        
        if total_missing_percentage > 0.2:
            synthetic_report['synthetic_generation_ready'] = False
            synthetic_report['recommendations'].append(
                "Too many missing values (>20%) - implement missing value handling"
            )
        
        return synthetic_report
