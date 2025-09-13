import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering for network traffic anomaly detection."""
    
    def __init__(self, n_components=50, k_best=100):
        self.n_components = n_components
        self.k_best = k_best
        self.pca = None
        self.feature_selector = None
        self.scaler = RobustScaler()
        
    def engineer_features(self, df):
        """Create advanced features from raw network data."""
        print("[INFO] Engineering advanced features...")
        
        # Create copy to avoid modifying original
        df_enhanced = df.copy()
        
        # 1. Statistical Features
        df_enhanced = self._add_statistical_features(df_enhanced)
        
        # 2. Protocol-Specific Features
        df_enhanced = self._add_protocol_features(df_enhanced)
        
        # 3. Connection Pattern Features
        df_enhanced = self._add_connection_features(df_enhanced)
        
        # 4. Time-Based Features
        df_enhanced = self._add_temporal_features(df_enhanced)
        
        # 5. Ratio Features
        df_enhanced = self._add_ratio_features(df_enhanced)
        
        # 6. Interaction Features
        df_enhanced = self._add_interaction_features(df_enhanced)
        
        # 7. Aggregated Features
        df_enhanced = self._add_aggregated_features(df_enhanced)
        
        print(f"[SUCCESS] Enhanced features: {df_enhanced.shape[1]} total features")
        return df_enhanced
    
    def _add_statistical_features(self, df):
        """Add statistical aggregations."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Rolling statistics for time-series like behavior
        for col in numeric_cols[:10]:  # Limit to avoid explosion
            if col not in ['label', 'target_8']:
                # Moving averages
                df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_rolling_std_3'] = df[col].rolling(window=3, min_periods=1).std()
                
                # Percentile features
                df[f'{col}_percentile_25'] = df[col].rolling(window=10, min_periods=1).quantile(0.25)
                df[f'{col}_percentile_75'] = df[col].rolling(window=10, min_periods=1).quantile(0.75)
                
                # Z-score normalization
                df[f'{col}_zscore'] = (df[col] - df[col].rolling(window=100, min_periods=1).mean()) / \
                                    (df[col].rolling(window=100, min_periods=1).std() + 1e-8)
        
        return df
    
    def _add_protocol_features(self, df):
        """Add protocol-specific features."""
        if 'protocol_type' in df.columns:
            # Protocol frequency encoding
            protocol_counts = df['protocol_type'].value_counts()
            df['protocol_frequency'] = df['protocol_type'].map(protocol_counts)
            
            # Protocol entropy
            df['protocol_entropy'] = -np.log(df['protocol_frequency'] / len(df))
            
            # Protocol combinations with service
            if 'service' in df.columns:
                df['protocol_service_combo'] = df['protocol_type'] + '_' + df['service']
                
        if 'service' in df.columns:
            # Service frequency
            service_counts = df['service'].value_counts()
            df['service_frequency'] = df['service'].map(service_counts)
            
            # Service entropy
            df['service_entropy'] = -np.log(df['service_frequency'] / len(df))
            
        return df
    
    def _add_connection_features(self, df):
        """Add connection pattern features."""
        # Connection duration patterns
        if 'duration' in df.columns:
            df['duration_log'] = np.log1p(df['duration'])
            df['duration_squared'] = df['duration'] ** 2
            df['duration_cubed'] = df['duration'] ** 3
            
        # Byte ratio features
        if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
            df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1e-8)
            df['bytes_sum'] = df['src_bytes'] + df['dst_bytes']
            df['bytes_diff'] = df['src_bytes'] - df['dst_bytes']
            df['bytes_product'] = df['src_bytes'] * df['dst_bytes']
            
        # Rate-based features
        if 'count' in df.columns and 'srv_count' in df.columns:
            df['count_ratio'] = df['count'] / (df['srv_count'] + 1e-8)
            df['count_sum'] = df['count'] + df['srv_count']
            df['count_diff'] = df['count'] - df['srv_count']
            
        return df
    
    def _add_temporal_features(self, df):
        """Add time-based features."""
        # Create synthetic time index for temporal patterns
        df['time_index'] = range(len(df))
        
        # Cyclical encoding
        df['time_sin'] = np.sin(2 * np.pi * df['time_index'] / 1000)
        df['time_cos'] = np.cos(2 * np.pi * df['time_index'] / 1000)
        
        # Time-based rolling features
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        for col in numeric_cols:
            if col not in ['label', 'target_8', 'time_index', 'time_sin', 'time_cos']:
                df[f'{col}_time_trend'] = df[col].rolling(window=50, min_periods=1).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
        
        return df
    
    def _add_ratio_features(self, df):
        """Add ratio and proportion features."""
        # Error rate ratios
        if 'serror_rate' in df.columns and 'srv_serror_rate' in df.columns:
            df['error_rate_ratio'] = df['serror_rate'] / (df['srv_serror_rate'] + 1e-8)
            df['error_rate_sum'] = df['serror_rate'] + df['srv_serror_rate']
            
        if 'rerror_rate' in df.columns and 'srv_rerror_rate' in df.columns:
            df['rerror_rate_ratio'] = df['rerror_rate'] / (df['srv_rerror_rate'] + 1e-8)
            df['rerror_rate_sum'] = df['rerror_rate'] + df['srv_rerror_rate']
            
        # Host-based ratios
        if 'dst_host_count' in df.columns and 'dst_host_srv_count' in df.columns:
            df['host_srv_ratio'] = df['dst_host_count'] / (df['dst_host_srv_count'] + 1e-8)
            df['host_srv_sum'] = df['dst_host_count'] + df['dst_host_srv_count']
            
        return df
    
    def _add_interaction_features(self, df):
        """Add interaction features between variables."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['label', 'target_8']][:8]
        
        # Create pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
        return df
    
    def _add_aggregated_features(self, df):
        """Add aggregated statistical features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['label', 'target_8']]
        
        # Global statistics
        for col in numeric_cols[:10]:  # Limit to avoid explosion
            df[f'{col}_global_mean'] = df[col].mean()
            df[f'{col}_global_std'] = df[col].std()
            df[f'{col}_global_min'] = df[col].min()
            df[f'{col}_global_max'] = df[col].max()
            df[f'{col}_global_median'] = df[col].median()
            
            # Normalized features
            df[f'{col}_normalized'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            
        return df
    
    def select_best_features(self, X, y):
        """Select the best features using optimized selection methods."""
        print("[INFO] Selecting best features (optimized)...")
        
        # Use only F-statistic for speed (mutual_info_classif is slower)
        f_selector = SelectKBest(score_func=f_classif, k=self.k_best)
        X_selected = f_selector.fit_transform(X, y)
        
        # Store selector for later use
        self.feature_selector = f_selector
        
        print(f"[SUCCESS] Selected {X_selected.shape[1]} best features out of {X.shape[1]}")
        return X_selected, f_selector.get_support()
    
    def apply_pca(self, X):
        """Apply PCA for dimensionality reduction (optimized)."""
        print("[INFO] Applying PCA (optimized)...")
        
        # Use smaller number of components for speed
        n_components = min(self.n_components, X.shape[1] // 2)
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"[SUCCESS] PCA completed with {n_components} components, explained variance: {explained_variance:.3f}")
        
        return X_pca
    
    def transform(self, X):
        """Transform features using fitted transformers."""
        if self.pca is not None:
            X = self.pca.transform(X)
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        return X
    
    def fit_transform(self, X, y=None):
        """Fit and transform features (optimized)."""
        print(f"[INFO] Processing {X.shape[0]} samples with {X.shape[1]} features...")
        
        # Scale features
        print("[INFO] Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Select best features if y is provided
        if y is not None:
            X_selected, _ = self.select_best_features(X_scaled, y)
            X_final = X_selected
        else:
            X_final = X_scaled
        
        # Apply PCA
        X_pca = self.apply_pca(X_final)
        
        print(f"[SUCCESS] Feature engineering completed: {X.shape[1]} → {X_pca.shape[1]} features")
        return X_pca
