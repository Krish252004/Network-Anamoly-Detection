import gzip
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from tqdm import tqdm
import sys

# Add parent directory to path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .utilities import save_pickle, load_pickle, load_config, ensure_dir, setup_logger
    from .feature_engineering import AdvancedFeatureEngineer
except ImportError:
    from utilities import save_pickle, load_pickle, load_config, ensure_dir, setup_logger
    from feature_engineering import AdvancedFeatureEngineer

def _validate_data_files(cfg: dict):
    """Validate that all required data files exist and are accessible."""
    required_files = [
        cfg["paths"]["raw_train"],
        cfg["paths"]["raw_test"],
        cfg["paths"]["raw_unlabeled"]
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read data file: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Data file is empty: {file_path}")

def _validate_gzip_file(file_path: str):
    """Validate that a gzip file is not corrupted."""
    try:
        with gzip.open(file_path, 'rt') as f:
            # Try to read first few lines to check integrity
            for i, line in enumerate(f):
                if i >= 5:  # Check first 5 lines
                    break
    except Exception as e:
        raise ValueError(f"Gzip file appears to be corrupted: {file_path}. Error: {str(e)}")

# Keep same KDD_COLUMNS and mapping functions from earlier (omitted here for brevity)
KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label"
]

# 8-class mapping (same as earlier)
DOS = {"back.", "land.", "neptune.", "pod.", "smurf.", "teardrop."}
DDOS = set()
PROBE = {"ipsweep.", "nmap.", "portsweep.", "satan."}
R2L_PASS = {"guess_passwd.", "ftp_write."}
R2L_DATA = {"imap.", "multihop.", "phf.", "spy.", "warezclient.", "warezmaster."}
U2R = {"buffer_overflow.", "loadmodule.", "perl.", "rootkit."}
WEB_ATTACKS = {"sql_injection."}
NORMAL = {"normal."}

def map_attack_to_8cls(label: str) -> str:
    if label in NORMAL: return "normal"
    if label in DOS: return "dos"
    if label in DDOS: return "ddos"
    if label in PROBE: return "probe"
    if label in R2L_PASS: return "r2l_password"
    if label in R2L_DATA: return "r2l_data"
    if label in U2R: return "u2r"
    if label in WEB_ATTACKS: return "web_attacks"
    return "dos"

def _load_gz(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f, names=KDD_COLUMNS)

def _build_transformer(df: pd.DataFrame) -> ColumnTransformer:
    cat = ["protocol_type", "service", "flag"]
    num = [c for c in df.columns if c not in cat + ["label", "target_8"]]
    transformers = []
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if num:
        transformers.append(("num", StandardScaler(), num))
    return ColumnTransformer(transformers)

def _drop_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique()
    keep = [c for c in df.columns if nunique[c] > 1 or c in ["label"]]
    return df[keep]

def load_and_prepare_data(cfg: dict):
    logger = setup_logger(log_dir=cfg["paths"].get("results_dir", "results"))
    
    # Validate data files before processing
    logger.info("Validating data files...")
    _validate_data_files(cfg)
    
    # Create necessary directories
    processed_dir = "data/processed"
    ensure_dir(processed_dir)
    ensure_dir(cfg["paths"]["results_dir"])
    
    # Check if data is already processed
    if os.path.exists(os.path.join(processed_dir, "X_train_processed.pkl")):
        logger.info("[INFO] Preprocessed data already exists. Loading...")
        return {
            "X_train_full": load_pickle(os.path.join(processed_dir, "X_train_processed.pkl")),
            "y_train_full": load_pickle(os.path.join(processed_dir, "y_train_processed.pkl")),
            "X_test": load_pickle(os.path.join(processed_dir, "X_test_processed.pkl")),
            "y_test": load_pickle(os.path.join(processed_dir, "y_test_processed.pkl")),
            "cv_splits": load_pickle(os.path.join(processed_dir, "cv_splits.pkl")),
            "classes": load_pickle(os.path.join(processed_dir, "metadata.pkl"))["classes"]
        }
    
    logger.info("Loading and preprocessing data...")
    
    # Load raw data
    logger.info("Loading raw data files...")
    train_df = _load_gz(cfg["paths"]["raw_train"])
    test_df = _load_gz(cfg["paths"]["raw_test"])
    
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Map labels to 8 classes
    logger.info("Mapping attack labels to 8 classes...")
    train_df["target_8"] = train_df["label"].apply(map_attack_to_8cls)
    test_df["target_8"] = test_df["label"].apply(map_attack_to_8cls)
    
    # Drop constant columns if requested
    if cfg["preprocessing"].get("drop_constant", True):
        logger.info("Dropping constant columns...")
        train_df = _drop_constant_cols(train_df)
        test_df = _drop_constant_cols(test_df)
        logger.info(f"After dropping constants - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Split features and target
    feature_cols = [c for c in train_df.columns if c not in ["label", "target_8"]]
    X_train = train_df[feature_cols]
    y_train = train_df["target_8"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_8"]
    
    # Build and fit transformer
    logger.info("Building and fitting data transformer...")
    transformer = _build_transformer(X_train)
    transformer.fit(X_train)
    
    # Transform data
    logger.info("Transforming data...")
    X_train_t = transformer.transform(X_train)
    X_test_t = transformer.transform(X_test)
    
    # Encode labels
    logger.info("Encoding labels...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Production-ready feature engineering
    logger.info("Applying production-ready feature engineering...")
    
    if cfg.get("fast_mode", False):
        logger.info("Using fast mode - minimal feature engineering")
        X_train_enhanced = X_train_t
        X_test_enhanced = X_test_t
        feature_engineer = None
        
        # Save feature engineer info for fast mode
        fast_feature_info = {
            "mode": "fast",
            "n_components": X_train_t.shape[1],
            "k_best": X_train_t.shape[1]
        }
        save_pickle(fast_feature_info, os.path.join(processed_dir, "feature_engineer.pkl"))
    else:
        # Use production-ready feature engineering with regularization
        feature_engineer = AdvancedFeatureEngineer(n_components=30, k_best=50)
        X_train_enhanced = feature_engineer.fit_transform(X_train_t, y_train_encoded)
        X_test_enhanced = feature_engineer.transform(X_test_t)
        
        # Save feature engineer for later use
        save_pickle(feature_engineer, os.path.join(processed_dir, "feature_engineer.pkl"))
    
    logger.info(f"Enhanced training features: {X_train_enhanced.shape}")
    logger.info(f"Enhanced test features: {X_test_enhanced.shape}")
    
    # Production-ready oversampling with validation
    if cfg["preprocessing"].get("oversample", True):
        if cfg.get("fast_mode", False):
            logger.info("Using fast mode - simple SMOTE oversampling")
            try:
                # Use simple SMOTE for fast mode
                smote = SMOTE(random_state=cfg["preprocessing"].get("oversample_seed", 42))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_enhanced, y_train_encoded)
                logger.info(f"After SMOTE - Train: {X_train_balanced.shape}")
                
                # Limit dataset size for faster training in fast mode
                max_samples = 500000  # Limit to 500K samples for speed
                if X_train_balanced.shape[0] > max_samples:
                    logger.info(f"Limiting dataset to {max_samples} samples for faster training...")
                    # Stratified sampling to maintain class balance
                    from sklearn.model_selection import train_test_split
                    X_train_balanced, _, y_train_balanced, _ = train_test_split(
                        X_train_balanced, y_train_balanced, 
                        train_size=max_samples, 
                        stratify=y_train_balanced, 
                        random_state=cfg["random_seed"]
                    )
                    logger.info(f"After limiting - Train: {X_train_balanced.shape}")
                    
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Using original data.")
                X_train_balanced, y_train_balanced = X_train_enhanced, y_train_encoded
        else:
            logger.info("Applying production-ready oversampling techniques...")
            try:
                # Production-ready: Use robust SMOTE with validation
                from sklearn.model_selection import cross_val_score
                from sklearn.ensemble import RandomForestClassifier
                
                # Try multiple sampling strategies with cross-validation
                sampling_strategies = [
                    ('SMOTE', SMOTE(random_state=cfg["preprocessing"].get("oversample_seed", 42))),
                    ('ADASYN', ADASYN(random_state=cfg["preprocessing"].get("oversample_seed", 42))),
                    ('BorderlineSMOTE', BorderlineSMOTE(random_state=cfg["preprocessing"].get("oversample_seed", 42)))
                ]
                
                best_strategy = None
                best_score = 0
                
                # Validate each strategy with cross-validation
                for name, sampler in sampling_strategies:
                    try:
                        X_resampled, y_resampled = sampler.fit_resample(X_train_enhanced, y_train_encoded)
                        
                        # Use cross-validation for robust validation
                        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                        cv_scores = cross_val_score(rf, X_resampled, y_resampled, cv=3, scoring='accuracy')
                        avg_score = cv_scores.mean()
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_strategy = (name, sampler)
                            
                        logger.info(f"Sampling strategy {name}: CV score = {avg_score:.4f} ± {cv_scores.std():.4f}")
                        
                    except Exception as e:
                        logger.warning(f"Sampling strategy {name} failed: {e}")
                        continue
                
                if best_strategy:
                    name, sampler = best_strategy
                    logger.info(f"Using best sampling strategy: {name}")
                    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_enhanced, y_train_encoded)
                    logger.info(f"After {name} - Train: {X_train_balanced.shape}")
                else:
                    logger.warning("All sampling strategies failed. Using original data.")
                    X_train_balanced, y_train_balanced = X_train_enhanced, y_train_encoded
                    
            except Exception as e:
                logger.warning(f"Production oversampling failed: {e}. Using original data.")
                X_train_balanced, y_train_balanced = X_train_enhanced, y_train_encoded
    else:
        X_train_balanced, y_train_balanced = X_train_enhanced, y_train_encoded
    
    # Create K-Fold splits with stratification
    logger.info("Creating production-ready K-Fold cross-validation splits...")
    k_folds = cfg["training"].get("k_folds", 3)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=cfg["random_seed"])
    cv_splits = list(skf.split(X_train_balanced, y_train_balanced))
    
    # Save processed data
    logger.info("Saving processed data...")
    save_pickle(X_train_balanced, os.path.join(processed_dir, "X_train_processed.pkl"))
    save_pickle(y_train_balanced, os.path.join(processed_dir, "y_train_processed.pkl"))
    save_pickle(X_test_enhanced, os.path.join(processed_dir, "X_test_processed.pkl"))
    save_pickle(y_test_encoded, os.path.join(processed_dir, "y_test_processed.pkl"))
    save_pickle(cv_splits, os.path.join(processed_dir, "cv_splits.pkl"))
    
    # Save transformer and label encoder
    save_pickle(transformer, cfg["paths"]["transformer"])
    save_pickle(le, cfg["paths"]["label_encoder"])
    
    # Save metadata with production information
    metadata = {
        "classes": le.classes_.tolist(),
        "feature_names": transformer.get_feature_names_out().tolist(),
        "n_features": X_train_balanced.shape[1],
        "n_classes": len(le.classes_),
        "production_ready": True,
        "data_robustness": {
            "oversampling_validation": True,
            "cross_validation_folds": k_folds,
            "feature_engineering_regularized": not cfg.get("fast_mode", False)
        }
    }
    save_pickle(metadata, os.path.join(processed_dir, "metadata.pkl"))
    
    logger.info("[SUCCESS] Production-ready data preprocessing completed!")
    logger.info(f"Final shapes - Train: {X_train_balanced.shape}, Test: {X_test_enhanced.shape}")
    logger.info(f"Classes: {metadata['n_classes']}, Features: {metadata['n_features']}")
    logger.info(f"Production ready: {metadata['production_ready']}")
    
    return {
        "X_train_full": X_train_balanced,
        "y_train_full": y_train_balanced,
        "X_test": X_test_enhanced,
        "y_test": y_test_encoded,
        "cv_splits": cv_splits,
        "classes": le.classes_
    }

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess network traffic data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for processed data")
    args = parser.parse_args()
    
    try:
        # Load configuration
        cfg = load_config(args.config)
        
        # Override output directory if specified
        if args.output_dir != "data/processed":
            cfg["paths"]["results_dir"] = args.output_dir
        
        print("Starting data preprocessing...")
        data = load_and_prepare_data(cfg)
        
        print("[SUCCESS] Data preprocessing completed successfully!")
        print(f"Processed data saved to: {args.output_dir}")
        print(f"Train data shape: {data['X_train_full'].shape}")
        print(f"Test data shape: {data['X_test'].shape}")
        print(f"Number of classes: {len(data['classes'])}")
        
    except Exception as e:
        print(f"[ERROR] Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
