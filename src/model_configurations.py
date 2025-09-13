import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class OptimizedModels:
    """Optimized model configurations for 3 specific models: LightGBM, XGBoost, RandomForest."""
    
    def __init__(self, n_classes, input_dim):
        self.n_classes = n_classes
        self.input_dim = input_dim
        
    def get_random_forest(self):
        """Optimized Random Forest for high accuracy and speed."""
        return RandomForestClassifier(
            n_estimators=200,           # Optimized for speed while maintaining accuracy
            max_depth=20,               # Balanced depth for speed and accuracy
            min_samples_split=8,        # Optimized for speed
            min_samples_leaf=3,         # Optimized for speed
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,             # Out-of-bag scoring
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',    # Handle class imbalance
            criterion='entropy'         # Better for classification
        )
    
    def get_xgboost(self):
        """Optimized XGBoost for high accuracy and speed."""
        return XGBClassifier(
            n_estimators=300,          # Optimized for speed while maintaining accuracy
            max_depth=7,                # Balanced depth
            learning_rate=0.05,         # Faster convergence
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=1.0,            # L2 regularization
            objective="multi:softprob",
            num_class=self.n_classes,
            n_jobs=-1,
            random_state=42,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            use_label_encoder=False
        )
    
    def get_lightgbm(self):
        """Optimized LightGBM for high accuracy and speed."""
        return LGBMClassifier(
            n_estimators=400,          # Optimized for speed while maintaining accuracy
            learning_rate=0.05,         # Faster convergence
            max_depth=8,                # Balanced depth
            num_leaves=255,             # Optimized leaf count
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=1.0,            # L2 regularization
            min_child_samples=20,
            min_child_weight=1e-3,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
            objective="multiclass",
            metric="multi_logloss",
            verbose=-1
        )
    

    
    def get_all_models(self):
        """Get all 3 optimized models."""
        return {
            'random_forest': self.get_random_forest(),
            'xgboost': self.get_xgboost(),
            'lightgbm': self.get_lightgbm()
        }
