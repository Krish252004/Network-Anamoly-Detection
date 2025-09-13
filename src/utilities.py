import os
import random
import yaml
import numpy as np
import logging
from joblib import dump, load

def load_config(path="config.yaml"):
    """Load configuration from YAML file with error handling."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate required config sections
        required_sections = ["paths", "preprocessing", "training", "models"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate paths
        required_paths = ["raw_train", "raw_test", "raw_unlabeled", "model_dir", "results_dir"]
        for path_key in required_paths:
            if path_key not in config["paths"]:
                raise ValueError(f"Missing required path configuration: {path_key}")
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in config file: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")

def set_global_seed(seed: int = 42):
    """Set global random seeds for reproducibility."""
    try:
        random.seed(seed)
        np.random.seed(seed)
        
        # Try to set TensorFlow seed
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass  # TensorFlow not available
        
        # Try to set other common ML library seeds
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        except ImportError:
            pass  # PyTorch not available
            
    except Exception as e:
        print(f"Warning: Could not set all random seeds: {e}")

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {path}: {e}")

def save_pickle(obj, path):
    """Save object to pickle file with error handling."""
    try:
        ensure_dir(os.path.dirname(path))
        dump(obj, path)
    except Exception as e:
        raise RuntimeError(f"Failed to save pickle file {path}: {e}")

def load_pickle(path):
    """Load object from pickle file with error handling."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pickle file not found: {path}")
        return load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle file {path}: {e}")

def setup_logger(name="nad", log_dir="results", log_file="training.log", level=logging.INFO):
    """Setup logger with file and console handlers."""
    try:
        ensure_dir(log_dir)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        
        return logger
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        basic_logger = logging.getLogger(name)
        basic_logger.setLevel(level)
        if not basic_logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            basic_logger.addHandler(ch)
        return basic_logger

def get_class_weights(y):
    """Return class weights in sklearn dict format for imbalanced classification."""
    try:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        return {int(c): float(w) for c, w in zip(classes, weights)}
    except Exception as e:
        print(f"Warning: Could not compute class weights: {e}")
        return None

def validate_data_integrity(data_path: str, expected_size: int = None):
    """Validate data file integrity."""
    try:
        if not os.path.exists(data_path):
            return False, f"File does not exist: {data_path}"
        
        if not os.access(data_path, os.R_OK):
            return False, f"File not readable: {data_path}"
        
        if os.path.getsize(data_path) == 0:
            return False, f"File is empty: {data_path}"
        
        if expected_size and os.path.getsize(data_path) < expected_size:
            return False, f"File size too small: {data_path}"
        
        return True, "File validation passed"
        
    except Exception as e:
        return False, f"Validation error: {e}"

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 
        'tensorflow', 'minisom', 'yaml', 'joblib', 'tqdm', 
        'imblearn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        return False, f"Missing packages: {', '.join(missing_packages)}"
    
    return True, "All dependencies available"
