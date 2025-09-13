import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import sys

# Add parent directory to path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .utilities import save_pickle, load_pickle, load_config, setup_logger, get_class_weights, ensure_dir
    from .model_configurations import OptimizedModels
except ImportError:
    from utilities import save_pickle, load_pickle, load_config, setup_logger, get_class_weights, ensure_dir
    from model_configurations import OptimizedModels



# ---------- TRAINING ----------

def train_all_models(cfg: dict, data: dict):
    """Train all models with production-ready validation strategy."""
    logger = setup_logger(log_dir=cfg["paths"].get("results_dir", "results"))
    
    # Extract data
    X_train_full = data["X_train_full"]
    y_train_full = data["y_train_full"]
    cv_splits = data["cv_splits"]
    
    # Create holdout validation set for production testing
    logger.info("[INFO] Creating holdout validation set for production testing...")
    from sklearn.model_selection import train_test_split
    
    # Split training data: 80% for training, 20% for holdout validation
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        stratify=y_train_full, 
        random_state=cfg["random_seed"]
    )
    
    logger.info(f"[INFO] Training set: {X_train.shape[0]} samples")
    logger.info(f"[INFO] Holdout validation set: {X_holdout.shape[0]} samples")
    logger.info(f"[INFO] Cross-validation folds: {len(cv_splits)}")
    
    # Save holdout set for production evaluation
    holdout_dir = os.path.join(cfg["paths"]["results_dir"], "holdout")
    ensure_dir(holdout_dir)
    save_pickle(X_holdout, os.path.join(holdout_dir, "X_holdout.pkl"))
    save_pickle(y_holdout, os.path.join(holdout_dir, "y_holdout.pkl"))
    logger.info(f"[INFO] Holdout set saved to: {holdout_dir}")
    
    model_dir = cfg["paths"]["model_dir"]
    ensure_dir(model_dir)
    
    logger.info(f"[INFO] Starting training of 3 models...")
    logger.info(f"[INFO] Training data shape: {data['X_train_full'].shape}")
    logger.info(f"[INFO] Models will be saved to: {model_dir}")

    if trained_models is None:
        trained_models = {}

    X = data["X_train_full"]
    y = data["y_train_full"]
    splits = data["cv_splits"]
    classes = data["classes"]
    num_classes = len(classes)

    results_summary = {}
    artifacts = {}

    # Try to reload pretrained models if available
    for model_name, model_path in trained_models.items():
        try:
            model = load_pickle(model_path)
            artifacts[model_name] = model
            logger.info(f"[LOADED] {model_name} from {model_path}")
        except Exception as e:
            logger.warning(f"[WARNING] Could not load {model_name}: {e}")

    # ---------- 1. RANDOM FOREST (OPTIMIZED) ----------
    if cfg["models"].get("train_random_forest", True):
        try:
            logger.info("[INFO] [1/3] Training optimized Random Forest...")
            
            rf = RandomForestClassifier(
                n_estimators=cfg["optimization"]["rf_n_estimators"],
                max_depth=cfg["optimization"]["rf_max_depth"],
                min_samples_split=cfg["optimization"]["rf_min_samples_split"],
                min_samples_leaf=cfg["optimization"]["rf_min_samples_leaf"],
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=cfg["random_seed"],
                class_weight='balanced'
            )
            
            # Train with cross-validation
            cv_scores = []
            logger.info(f"[INFO] Running {len(splits)}-fold cross-validation...")
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                rf.fit(X_train_fold, y_train_fold)
                y_pred = rf.predict(X_val_fold)
                fold_score = accuracy_score(y_val_fold, y_pred)
                cv_scores.append(fold_score)
                logger.info(f"  Fold {fold+1}/{len(splits)}: {fold_score:.4f}")
            
            # Train on full data
            rf.fit(X, y)
            
            # Save model
            save_pickle(rf, os.path.join(model_dir, "random_forest.pkl"))
            artifacts["random_forest"] = rf
            results_summary["random_forest"] = {
                "cv_scores": cv_scores,
                "cv_mean": np.mean(cv_scores),
                "cv_std": np.std(cv_scores)
            }
            logger.info("[SAVED] Random Forest model")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to train Random Forest: {e}")
            results_summary["random_forest"] = None

    # ---------- 2. XGBOOST (OPTIMIZED) ----------
    if cfg["models"].get("train_xgboost", True):
        try:
            logger.info("[INFO] [2/3] Training optimized XGBoost...")
            
            xgb = XGBClassifier(
                n_estimators=cfg["optimization"]["xgb_n_estimators"],
                max_depth=cfg["optimization"]["xgb_max_depth"],
                learning_rate=cfg["optimization"]["xgb_learning_rate"],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=num_classes,
                n_jobs=-1,
                random_state=cfg["random_seed"],
                eval_metric="mlogloss",
                use_label_encoder=False
            )
            
            # Train with cross-validation
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                xgb.fit(X_train_fold, y_train_fold)
                y_pred = xgb.predict(X_val_fold)
                fold_score = accuracy_score(y_val_fold, y_pred)
                cv_scores.append(fold_score)
                logger.info(f"Fold {fold+1}: {fold_score:.4f}")
            
            # Train on full data
            xgb.fit(X, y)
            
            # Save model
            save_pickle(xgb, os.path.join(model_dir, "xgboost.pkl"))
            artifacts["xgboost"] = xgb
            results_summary["xgboost"] = {
                "cv_scores": cv_scores,
                "cv_mean": np.mean(cv_scores),
                "cv_std": np.std(cv_scores)
            }
            logger.info("[SAVED] XGBoost model")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to train XGBoost: {e}")
            results_summary["xgboost"] = None

    # ---------- 3. LIGHTGBM (OPTIMIZED) ----------
    if cfg["models"].get("train_lightgbm", True):
        try:
            logger.info("[INFO] [3/3] Training optimized LightGBM...")
            
            lgbm = LGBMClassifier(
                n_estimators=cfg["optimization"]["lgbm_n_estimators"],
                learning_rate=cfg["optimization"]["lgbm_learning_rate"],
                max_depth=8,
                num_leaves=255,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                n_jobs=-1,
                random_state=cfg["random_seed"],
                class_weight='balanced',
                objective="multiclass",
                metric="multi_logloss",
                verbose=-1
            )
            
            # Train with cross-validation
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                lgbm.fit(X_train_fold, y_train_fold)
                y_pred = lgbm.predict(X_val_fold)
                fold_score = accuracy_score(y_val_fold, y_pred)
                cv_scores.append(fold_score)
                logger.info(f"Fold {fold+1}: {fold_score:.4f}")
            
            # Train on full data
            lgbm.fit(X, y)
            
            # Save model
            save_pickle(lgbm, os.path.join(model_dir, "lightgbm.pkl"))
            artifacts["lightgbm"] = lgbm
            results_summary["lightgbm"] = {
                "cv_scores": cv_scores,
                "cv_mean": np.mean(cv_scores),
                "cv_std": np.std(cv_scores)
            }
            logger.info("[SAVED] LightGBM model")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to train LightGBM: {e}")
            results_summary["lightgbm"] = None



    # ---------- FINAL SUMMARY ----------
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    successful_models = 0
    for model_name, results in results_summary.items():
        if results:
            cv_mean = results["cv_mean"]
            cv_std = results["cv_std"]
            logger.info(f"{model_name.upper()}: CV Accuracy = {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Check target achievement
            if cv_mean >= 0.97:
                logger.info(f"🎯 {model_name} has achieved target accuracy!")
            elif cv_mean >= 0.95:
                logger.info(f"📈 {model_name} is very close to target!")
            successful_models += 1
        else:
            logger.warning(f"{model_name.upper()}: Training failed")
    
    # Save training summary
    logger.info(f"\n[INFO] Successfully trained {successful_models}/{len(results_summary)} models")
    logger.info(f"[INFO] Models saved to: {model_dir}")
    
    # List all saved models
    saved_models = []
    for model_name in artifacts.keys():
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        
        if os.path.exists(model_path):
            saved_models.append(f"{model_name}: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    logger.info(f"[INFO] Saved models: {', '.join(saved_models)}")
    
    return artifacts, results_summary



def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--config", default="../config.yaml", help="Path to config file")
    parser.add_argument("--data-dir", default="../data/processed", help="Directory containing processed data")
    args = parser.parse_args()
    
    try:
        # Load configuration
        cfg = load_config(args.config)
        
        # Check if processed data exists
        if not os.path.exists(args.data_dir):
            print(f"[ERROR] Processed data directory not found: {args.data_dir}")
            print("Please run data preprocessing first or specify correct --data-dir")
            return 1
        
        # Load processed data
        print("Loading processed data...")
        data = {
            "X_train_full": load_pickle(os.path.join(args.data_dir, "X_train_processed.pkl")),
            "y_train_full": load_pickle(os.path.join(args.data_dir, "y_train_processed.pkl")),
            "cv_splits": load_pickle(os.path.join(args.data_dir, "cv_splits.pkl")),
            "classes": load_pickle(os.path.join(args.data_dir, "metadata.pkl"))["classes"]
        }
        
        print(f"[SUCCESS] Loaded data: X_train shape={data['X_train_full'].shape}, classes={len(data['classes'])}")
        print(f"[INFO] Starting training with {len(data['cv_splits'])} cross-validation folds")
        print(f"[INFO] Training {len([k for k, v in cfg['models'].items() if v])} models")
        
        # Train models
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING")
        print("="*60)
        artifacts, results_summary = train_all_models(cfg, data)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"[SUCCESS] Model training completed successfully!")
        print(f"Trained models: {list(artifacts.keys())}")
        
    except Exception as e:
        print(f"[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
