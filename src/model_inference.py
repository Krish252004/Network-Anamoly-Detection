import os
import gzip
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import sys

# Add parent directory to path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .data_preprocessing import KDD_COLUMNS, map_attack_to_8cls  # mapping used for robustness
    from .utilities import load_pickle, load_config
except ImportError:
    from data_preprocessing import KDD_COLUMNS, map_attack_to_8cls  # mapping used for robustness
    from utilities import load_pickle, load_config

def _load_unlabeled(path: str) -> pd.DataFrame:
    # Unlabeled test file often lacks 'label'; still has 41 cols; add placeholder if needed.
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, header=None)
    # If 41 columns: add label col to align
    if df.shape[1] == 41:
        df.columns = KDD_COLUMNS[:-1]
        df["label"] = "unknown."
    else:
        df.columns = KDD_COLUMNS
    return df

def _som_predict(som_art, X):
    som, labels = som_art["som"], som_art["labels"]
    y_pred = []
    for xi in X:
        w = som.winner(xi)
        y_pred.append(labels.get(w, 0))
    return np.array(y_pred)

def load_models_for_inference(cfg: dict):
    """Load trained models for production inference with confidence scoring."""
    logger = setup_logger(log_dir=cfg["paths"].get("results_dir", "results"))
    
    model_dir = cfg["paths"]["model_dir"]
    models = {}
    
    logger.info(f"[LOADING] Loading models from {model_dir} for production inference...")
    
    # Load tree-based models
    try:
        models["random_forest"] = load_pickle(os.path.join(model_dir, "random_forest.pkl"))
        logger.info("[LOADED] Random Forest model")
    except Exception as e:
        logger.warning(f"[WARNING] Could not load Random Forest: {e}")
    
    try:
        models["xgboost"] = load_pickle(os.path.join(model_dir, "xgboost.pkl"))
        logger.info("[LOADED] XGBoost model")
    except Exception as e:
        logger.warning(f"[WARNING] Could not load XGBoost: {e}")
    
    try:
        models["lightgbm"] = load_pickle(os.path.join(model_dir, "lightgbm.pkl"))
        logger.info("[LOADED] LightGBM model")
    except Exception as e:
        logger.warning(f"[WARNING] Could not load LightGBM: {e}")
    
    # Load neural network models
    try:
        import tensorflow as tf
        models["nn"] = tf.keras.models.load_model(os.path.join(model_dir, "nn_model.keras"))
        models["nn_label_binarizer"] = load_pickle(os.path.join(model_dir, "nn_label_binarizer.pkl"))
        logger.info("[LOADED] Neural Network model")
    except Exception as e:
        logger.warning(f"[WARNING] Could not load Neural Network: {e}")
    
    try:
        models["lstm"] = tf.keras.models.load_model(os.path.join(model_dir, "lstm_model.keras"))
        models["lstm_label_binarizer"] = load_pickle(os.path.join(model_dir, "lstm_label_binarizer.pkl"))
        logger.info("[LOADED] LSTM model")
    except Exception as e:
        logger.warning(f"[WARNING] Could not load LSTM: {e}")
    
    # Load SOM model
    try:
        models["som"] = load_pickle(os.path.join(model_dir, "som.pkl"))
        logger.info("[LOADED] SOM model")
    except Exception as e:
        logger.warning(f"[WARNING] Could not load SOM: {e}")
    
    # Load preprocessing components
    try:
        models["transformer"] = load_pickle(cfg["paths"]["transformer"])
        models["label_encoder"] = load_pickle(cfg["paths"]["label_encoder"])
        logger.info("[LOADED] Preprocessing components")
    except Exception as e:
        logger.error(f"[ERROR] Could not load preprocessing components: {e}")
        return None
    
    logger.info(f"[SUCCESS] Loaded {len([k for k in models.keys() if k not in ['transformer', 'label_encoder', 'nn_label_binarizer', 'lstm_label_binarizer']])} models")
    return models

def predict_with_confidence(models: dict, X: np.ndarray, cfg: dict):
    """Make production-ready predictions with confidence scoring and ensemble support."""
    logger = setup_logger(log_dir=cfg["paths"].get("results_dir", "results"))
    
    if not models or "transformer" not in models:
        logger.error("[ERROR] No models or preprocessing components loaded")
        return None
    
    # Preprocess input data
    try:
        X_transformed = models["transformer"].transform(X)
        logger.info(f"[PREPROCESSED] Input data: {X.shape} → {X_transformed.shape}")
    except Exception as e:
        logger.error(f"[ERROR] Preprocessing failed: {e}")
        return None
    
    # Get predictions from each model
    predictions = {}
    probabilities = {}
    confidence_scores = {}
    
    # Tree-based models
    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        if model_name in models:
            try:
                y_pred = models[model_name].predict(X_transformed)
                y_proba = models[model_name].predict_proba(X_transformed)
                confidence = np.max(y_proba, axis=1)
                
                predictions[model_name] = y_pred
                probabilities[model_name] = y_proba
                confidence_scores[model_name] = confidence
                
                logger.info(f"[PREDICTED] {model_name}: avg confidence = {np.mean(confidence):.4f}")
                
            except Exception as e:
                logger.warning(f"[WARNING] {model_name} prediction failed: {e}")
    
    # Neural Network models
    if "nn" in models and "nn_label_binarizer" in models:
        try:
            y_proba = models["nn"].predict(X_transformed, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
            confidence = np.max(y_proba, axis=1)
            
            predictions["nn"] = y_pred
            probabilities["nn"] = y_proba
            confidence_scores["nn"] = confidence
            
            logger.info(f"[PREDICTED] Neural Network: avg confidence = {np.mean(confidence):.4f}")
            
        except Exception as e:
            logger.warning(f"[WARNING] Neural Network prediction failed: {e}")
    
    # LSTM model
    if "lstm" in models and "lstm_label_binarizer" in models:
        try:
            X_lstm = X_transformed.reshape((X_transformed.shape[0], X_transformed.shape[1], 1))
            y_proba = models["lstm"].predict(X_lstm, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
            confidence = np.max(y_proba, axis=1)
            
            predictions["lstm"] = y_pred
            probabilities["lstm"] = y_proba
            confidence_scores["lstm"] = confidence
            
            logger.info(f"[PREDICTED] LSTM: avg confidence = {np.mean(confidence):.4f}")
            
        except Exception as e:
            logger.warning(f"[WARNING] LSTM prediction failed: {e}")
    
    # Create ensemble prediction if enabled
    ensemble_results = None
    if cfg.get("production", {}).get("use_ensemble", True) and len(predictions) >= cfg.get("production", {}).get("min_models_for_ensemble", 2):
        try:
            # Soft voting ensemble
            ensemble_proba = np.mean(list(probabilities.values()), axis=0)
            ensemble_pred = np.argmax(ensemble_proba, axis=1)
            ensemble_confidence = np.max(ensemble_proba, axis=1)
            
            ensemble_results = {
                "predictions": ensemble_pred,
                "probabilities": ensemble_proba,
                "confidence_scores": ensemble_confidence,
                "model_agreement": _calculate_model_agreement(predictions)
            }
            
            logger.info(f"[ENSEMBLE] Created ensemble prediction with avg confidence = {np.mean(ensemble_confidence):.4f}")
            
        except Exception as e:
            logger.warning(f"[WARNING] Ensemble creation failed: {e}")
    
    # Production confidence analysis
    confidence_analysis = _analyze_production_confidence(confidence_scores, cfg)
    
    # Convert predictions to class labels
    label_encoder = models["label_encoder"]
    class_predictions = {}
    
    for model_name, pred in predictions.items():
        class_predictions[model_name] = label_encoder.inverse_transform(pred)
    
    if ensemble_results is not None:
        class_predictions["ensemble"] = label_encoder.inverse_transform(ensemble_results["predictions"])
    
    return {
        "predictions": predictions,
        "class_predictions": class_predictions,
        "probabilities": probabilities,
        "confidence_scores": confidence_scores,
        "ensemble": ensemble_results,
        "confidence_analysis": confidence_analysis,
        "production_ready": True
    }

def _calculate_model_agreement(predictions: dict):
    """Calculate agreement between different models for ensemble robustness."""
    if len(predictions) < 2:
        return 1.0
    
    # Calculate pairwise agreement
    agreement_scores = []
    model_names = list(predictions.keys())
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            agreement = np.mean(predictions[model1] == predictions[model2])
            agreement_scores.append(agreement)
    
    return np.mean(agreement_scores)

def _analyze_production_confidence(confidence_scores: dict, cfg: dict):
    """Analyze confidence scores for production monitoring."""
    production_cfg = cfg.get("production", {})
    
    high_threshold = production_cfg.get("high_confidence_threshold", 0.9)
    medium_threshold = production_cfg.get("medium_confidence_threshold", 0.7)
    low_threshold = production_cfg.get("low_confidence_threshold", 0.5)
    
    analysis = {}
    
    for model_name, confidence in confidence_scores.items():
        high_conf = np.sum(confidence >= high_threshold)
        medium_conf = np.sum((confidence >= medium_threshold) & (confidence < high_threshold))
        low_conf = np.sum((confidence >= low_threshold) & (confidence < medium_threshold))
        very_low_conf = np.sum(confidence < low_threshold)
        
        total = len(confidence)
        
        analysis[model_name] = {
            "high_confidence": {"count": high_conf, "percentage": (high_conf / total) * 100},
            "medium_confidence": {"count": medium_conf, "percentage": (medium_conf / total) * 100},
            "low_confidence": {"count": low_conf, "percentage": (low_conf / total) * 100},
            "very_low_confidence": {"count": very_low_conf, "percentage": (very_low_conf / total) * 100},
            "avg_confidence": np.mean(confidence),
            "confidence_std": np.std(confidence)
        }
    
    return analysis

def predict_unlabeled_data(cfg: dict, unlabeled_file: str = None):
    """Make predictions on unlabeled data with production-ready confidence scoring."""
    logger = setup_logger(log_dir=cfg["paths"].get("results_dir", "results"))
    
    # Load models
    models = load_models_for_inference(cfg)
    if not models:
        logger.error("[ERROR] Failed to load models for inference")
        return None
    
    # Load unlabeled data
    if unlabeled_file is None:
        unlabeled_file = cfg["paths"]["raw_unlabeled"]
    
    try:
        from .data_preprocessing import _load_gz, KDD_COLUMNS
        unlabeled_df = _load_gz(unlabeled_file)
        logger.info(f"[LOADED] Unlabeled data: {unlabeled_df.shape}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load unlabeled data: {e}")
        return None
    
    # Remove label column if present
    if "label" in unlabeled_df.columns:
        unlabeled_df = unlabeled_df.drop("label", axis=1)
    
    # Make predictions
    X_unlabeled = unlabeled_df.values
    results = predict_with_confidence(models, X_unlabeled, cfg)
    
    if results is None:
        logger.error("[ERROR] Prediction failed")
        return None
    
    # Save results
    output_path = cfg["paths"]["unlabeled_predictions"]
    ensure_dir(os.path.dirname(output_path))
    
    # Create results DataFrame
    import pandas as pd
    
    # Use ensemble predictions if available, otherwise use preferred model
    if "ensemble" in results["class_predictions"]:
        final_predictions = results["class_predictions"]["ensemble"]
        final_confidence = results["ensemble"]["confidence_scores"]
        prediction_source = "ensemble"
    else:
        preferred_model = cfg["inference"]["preferred_model"]
        if preferred_model in results["class_predictions"]:
            final_predictions = results["class_predictions"][preferred_model]
            final_confidence = results["confidence_scores"][preferred_model]
            prediction_source = preferred_model
        else:
            # Use first available model
            first_model = list(results["class_predictions"].keys())[0]
            final_predictions = results["class_predictions"][first_model]
            final_confidence = results["confidence_scores"][first_model]
            prediction_source = first_model
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "predicted_class": final_predictions,
        "confidence": final_confidence,
        "prediction_source": prediction_source
    })
    
    # Add individual model predictions if available
    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        if model_name in results["class_predictions"]:
            results_df[f"{model_name}_prediction"] = results["class_predictions"][model_name]
            results_df[f"{model_name}_confidence"] = results["confidence_scores"][model_name]
    
    # Save results
    results_df.to_csv(output_path, index=False)
    logger.info(f"[SAVED] Predictions saved to {output_path}")
    
    # Production confidence summary
    logger.info("\n" + "=" * 50)
    logger.info("PRODUCTION INFERENCE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total samples: {len(results_df)}")
    logger.info(f"Prediction source: {prediction_source}")
    logger.info(f"Average confidence: {np.mean(final_confidence):.4f}")
    logger.info(f"High confidence samples: {np.sum(final_confidence >= 0.9)} ({np.sum(final_confidence >= 0.9)/len(final_confidence)*100:.1f}%)")
    logger.info(f"Low confidence samples: {np.sum(final_confidence < 0.7)} ({np.sum(final_confidence < 0.7)/len(final_confidence)*100:.1f}%)")
    
    return results_df

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on unlabeled network traffic data")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained models")
    parser.add_argument("--output", help="Output file path for predictions")
    args = parser.parse_args()
    
    try:
        # Load configuration
        cfg = load_config(args.config)
        
        # Override paths if specified
        if args.models_dir != "models":
            cfg["paths"]["model_dir"] = args.models_dir
        if args.output:
            cfg["paths"]["unlabeled_predictions"] = args.output
        
        # Check if models directory exists
        if not os.path.exists(args.models_dir):
            print(f"❌ Models directory not found: {args.models_dir}")
            print("Please train models first or specify correct --models-dir")
            return 1
        
        # Check if required files exist
        required_files = [
            cfg["paths"]["transformer"],
            cfg["paths"]["label_encoder"],
            cfg["paths"]["raw_unlabeled"]
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Required file not found: {file_path}")
                return 1
        
        # Load trained models
        print("Loading trained models...")
        artifacts = {}
        model_files = {
            "random_forest": "random_forest.pkl",
            "xgboost": "xgboost.pkl", 
            "lightgbm": "lightgbm.pkl",
            "nn": "nn_model.keras",
            "lstm": "lstm_model.keras",
            "som": "som.pkl"
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(args.models_dir, filename)
            if os.path.exists(model_path):
                try:
                    if model_name in ["nn", "lstm"]:
                        model = load_model(model_path)
                    else:
                        model = load_pickle(model_path)
                    artifacts[model_name] = model
                    print(f"[SUCCESS] Loaded {model_name}")
                except Exception as e:
                    print(f"⚠️ Failed to load {model_name}: {e}")
            else:
                print(f"⚠️ Model file not found: {filename}")
        
        if not artifacts:
            print("❌ No models could be loaded. Please check the models directory.")
            return 1
        
        print(f"[SUCCESS] Loaded {len(artifacts)} models: {list(artifacts.keys())}")
        
        # Run inference
        print("Starting inference...")
        predict_unlabeled(cfg, artifacts)
        
        print("[SUCCESS] Inference completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
