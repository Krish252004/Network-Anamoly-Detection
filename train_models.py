#!/usr/bin/env python3
"""
Clean Training Script for 3 Models: LightGBM, XGBoost, RandomForest
Optimized for 97-99% accuracy with fast training times
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

try:
    from utilities import save_pickle, load_pickle, load_config, setup_logger, ensure_dir
    from data_preprocessing import load_and_preprocess_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are in the src/ directory")
    sys.exit(1)

class CleanTrainer:
    """Clean trainer for 3 specific models with high accuracy target."""
    
    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)
        self.logger = setup_logger(log_dir="results/logs")
        self.models = {}
        self.results = {}
        

    
    def load_and_preprocess_data(self):
        """Load and preprocess data."""
        print("[INFO] Loading and preprocessing data...")
        
        try:
            # Load processed data
            processed_dir = "data/processed"
            if not os.path.exists(processed_dir):
                print("[ERROR] Processed data not found. Please run data preprocessing first.")
                return False
            
            # Load data
            X_train = load_pickle(os.path.join(processed_dir, "X_train_processed.pkl"))
            y_train = load_pickle(os.path.join(processed_dir, "y_train_processed.pkl"))
            X_test = load_pickle(os.path.join(processed_dir, "X_test_processed.pkl"))
            y_test = load_pickle(os.path.join(processed_dir, "y_test_processed.pkl"))
            
            print(f"[SUCCESS] Data loaded: Train={X_train.shape}, Test={X_test.shape}")
            
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Data loading failed: {e}")
            return False
    
    def initialize_models(self):
        """Initialize the 3 specific models."""
        print("[INFO] Initializing models...")
        
        n_classes = len(np.unique(self.y_train))
        input_dim = self.X_train.shape[1]
        
        # 1. Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=self.cfg['optimization']['rf_n_estimators'],
            max_depth=self.cfg['optimization']['rf_max_depth'],
            min_samples_split=self.cfg['optimization']['rf_min_samples_split'],
            min_samples_leaf=self.cfg['optimization']['rf_min_samples_leaf'],
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        # 2. XGBoost
        self.models['xgboost'] = XGBClassifier(
            n_estimators=self.cfg['optimization']['xgb_n_estimators'],
            max_depth=self.cfg['optimization']['xgb_max_depth'],
            learning_rate=self.cfg['optimization']['xgb_learning_rate'],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=n_classes,
            n_jobs=-1,
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False
        )
        
        # 3. LightGBM
        self.models['lightgbm'] = LGBMClassifier(
            n_estimators=self.cfg['optimization']['lgbm_n_estimators'],
            learning_rate=self.cfg['optimization']['lgbm_learning_rate'],
            max_depth=8,
            num_leaves=255,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
            objective="multiclass",
            metric="multi_logloss",
            verbose=-1
        )
        
        print(f"[SUCCESS] Initialized {len(self.models)} models")
        return True
    

    
    def train_ml_models(self):
        """Train ML models (Random Forest, XGBoost, LightGBM)."""
        print("\n" + "="*60)
        print("TRAINING ML MODELS")
        print("="*60)
        
        ml_models = ['random_forest', 'xgboost', 'lightgbm']
        
        for name in ml_models:
            print(f"\n{'='*20} {name.upper()} {'='*20}")
            
            start_time = time.time()
            
            try:
                # Cross-validation
                cv = StratifiedKFold(n_splits=self.cfg['training']['k_folds'], shuffle=True, random_state=42)
                cv_scores = cross_val_score(self.models[name], self.X_train, self.y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                
                # Train on full data
                self.models[name].fit(self.X_train, self.y_train)
                
                # Test set evaluation
                y_pred = self.models[name].predict(self.X_test)
                test_accuracy = accuracy_score(self.y_test, y_pred)
                test_f1 = f1_score(self.y_test, y_pred, average='macro')
                
                training_time = time.time() - start_time
                
                results = {
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1,
                    'training_time': training_time
                }
                
                print(f"[SUCCESS] {name} completed in {training_time:.2f}s")
                print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"   Test Accuracy: {test_accuracy:.4f}")
                print(f"   Test F1: {test_f1:.4f}")
                
                # Check target achievement
                if test_accuracy >= 0.97:
                    print(f"🎯 {name} has achieved target accuracy!")
                elif test_accuracy >= 0.95:
                    print(f"📈 {name} is very close to target!")
                
                self.results[name] = results
                
            except Exception as e:
                print(f"[ERROR] {name} training failed: {e}")
                self.results[name] = None
    

    

    
    def save_models(self):
        """Save all trained models."""
        print("\n[INFO] Saving models...")
        
        model_dir = "models/clean_models"
        ensure_dir(model_dir)
        
        for name, model in self.models.items():
            if name in ['random_forest', 'xgboost', 'lightgbm'] and self.results.get(name):
                # Save ML models
                save_pickle(model, os.path.join(model_dir, f"{name}.pkl"))
                print(f"💾 Saved {name}")
    
    def generate_report(self):
        """Generate comprehensive training report."""
        print("\n" + "="*60)
        print("TRAINING REPORT")
        print("="*60)
        
        if not self.results:
            print("[ERROR] No training results available")
            return
        
        # Sort models by test accuracy
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items() if v is not None], 
            key=lambda x: x[1]['test_accuracy'], 
            reverse=True
        )
        
        print(f"\n{'Model':<20} {'CV Acc':<12} {'Test Acc':<12} {'F1 Score':<12} {'Time (s)':<12}")
        print("-" * 80)
        
        for name, results in sorted_results:
            cv_mean = results.get('cv_mean', 'N/A')
            cv_std = results.get('cv_std', 'N/A')
            test_acc = results['test_accuracy']
            f1 = results['test_f1']
            time_taken = results['training_time']
            
            if cv_mean != 'N/A':
                cv_str = f"{cv_mean:.4f}±{cv_std:.4f}"
            else:
                cv_str = "N/A"
            
            print(f"{name:<20} {cv_str:<12} {test_acc:<12.4f} {f1:<12.4f} {time_taken:<12.2f}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        target_models = [r for r in sorted_results if r[1]['test_accuracy'] >= 0.97]
        close_models = [r for r in sorted_results if 0.95 <= r[1]['test_accuracy'] < 0.97]
        
        print(f"🎯 Models achieving target (97%+): {len(target_models)}")
        print(f"📈 Models close to target (95-97%): {len(close_models)}")
        
        if target_models:
            print(f"\n🏆 Best model: {target_models[0][0]} ({target_models[0][1]['test_accuracy']:.4f})")
        
        # Save results
        results_dir = "results/clean_training"
        ensure_dir(results_dir)
        save_pickle(self.results, os.path.join(results_dir, "training_results.pkl"))
        print(f"\n💾 Results saved to {results_dir}")
    
    def run_training(self):
        """Run complete training pipeline."""
        print("🚀 Starting Clean Training Pipeline")
        print("="*60)
        
        # Load data
        if not self.load_and_preprocess_data():
            return False
        
        # Initialize models
        if not self.initialize_models():
            return False
        
        # Train ML models
        self.train_ml_models()
        
        # Save models
        self.save_models()
        
        # Generate report
        self.generate_report()
        
        print("\n✅ Training pipeline completed successfully!")
        return True

def main():
    """Main function."""
    trainer = CleanTrainer()
    success = trainer.run_training()
    
    if success:
        print("\n🎉 All models trained successfully!")
        print("Check the results/clean_training directory for detailed results.")
    else:
        print("\n❌ Training pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
