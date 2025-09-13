# src/evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, f1_score
)
import sys

# Add parent directory to path for standalone execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .utilities import load_pickle, setup_logger, ensure_dir, load_config, save_pickle
except ImportError:
    from utilities import load_pickle, setup_logger, ensure_dir, load_config, save_pickle

def _predict_proba(model, X):
    """Get probability predictions from model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    # Fallback: predict and one-hot encode
    pred = model.predict(X)
    proba = np.zeros((len(pred), np.max(pred) + 1))
    for i, p in enumerate(pred):
        proba[i, int(p)] = 1.0
    return proba

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_path):
    """Create confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

def plot_roc_curves(y_true, y_pred_proba, classes, model_name, save_path):
    """Create ROC curves for all classes."""
    from sklearn.preprocessing import label_binarize
    
    # Binarize the output for ROC curves
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC for each class
    roc_auc_scores = {}
    for i, class_name in enumerate(classes):
        if y_true_bin[:, i].sum() > 0:  # Only if class exists in test set
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc_scores[class_name] = roc_auc
            
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc_scores

def plot_model_comparison(results_data, save_path):
    """Create model comparison bar chart."""
    models = list(results_data.keys())
    accuracies = [results_data[model]['accuracy'] for model in models]
    f1_scores = [results_data[model]['f1_score'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.8)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # F1 Score comparison
    bars2 = ax2.bar(models, f1_scores, color='lightgreen', alpha=0.8)
    ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_all(cfg: dict, data: dict, artifacts: dict):
    """Evaluate all models with basic visualizations."""
    logger = setup_logger(log_dir=cfg["paths"].get("results_dir", "results"))
    
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 50)
    
    # Load test data
    X_test = data["X_test"]
    y_test = data["y_test"]
    classes = data["classes"]
    
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Number of classes: {len(classes)}")
    
    # Create plots directory
    plots_dir = os.path.join(cfg["paths"]["results_dir"], "plots")
    ensure_dir(plots_dir)
    
    # Evaluation results
    evaluation_results = {}
    results_summary = {}
    
    # Evaluate each model
    for model_name, model in artifacts.items():
        logger.info(f"\n[EVALUATING] {model_name.upper()}")
        
        try:
            # Get predictions and probabilities
            y_pred = model.predict(X_test)
            y_pred_proba = _predict_proba(model, X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate confidence scores
            confidence_scores = np.max(y_pred_proba, axis=1)
            avg_confidence = np.mean(confidence_scores)
            
            # Save results
            evaluation_results[model_name] = {
                "predictions": y_pred,
                "probabilities": y_pred_proba,
                "confidence_scores": confidence_scores,
                "metrics": {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "avg_confidence": avg_confidence
                }
            }
            
            results_summary[model_name] = {
                "accuracy": accuracy,
                "f1_score": f1,
                "avg_confidence": avg_confidence
            }
            
            logger.info(f"[SUCCESS] {model_name.upper()}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            logger.info(f"[CONFIDENCE] Avg={avg_confidence:.4f}")
            
            # Create visualizations
            logger.info(f"Creating visualizations for {model_name}...")
            
            # Confusion Matrix
            cm_path = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
            cm = plot_confusion_matrix(y_test, y_pred, classes, model_name, cm_path)
            logger.info(f"   Confusion matrix saved: {cm_path}")
            
            # ROC Curves
            roc_path = os.path.join(plots_dir, f"{model_name}_roc_curves.png")
            roc_auc_scores = plot_roc_curves(y_test, y_pred_proba, classes, model_name, roc_path)
            logger.info(f"   ROC curves saved: {roc_path}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to evaluate {model_name}: {e}")
            evaluation_results[model_name] = {"error": str(e)}
    
    # Create model comparison
    if len(results_summary) > 1:
        logger.info(f"\nCreating model comparison...")
        comparison_path = os.path.join(plots_dir, "model_comparison.png")
        plot_model_comparison(results_summary, comparison_path)
        logger.info(f"   Model comparison saved: {comparison_path}")
    
    # Save evaluation results
    results_path = os.path.join(cfg["paths"]["results_dir"], "evaluation_results.pkl")
    save_pickle(evaluation_results, results_path)
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    
    for model_name, results in evaluation_results.items():
        if "error" in results:
            logger.info(f"FAILED {model_name.upper()}: {results['error']}")
            continue
            
        metrics = results["metrics"]
        logger.info(f"SUCCESS {model_name.upper()}:")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"   Avg Confidence: {metrics['avg_confidence']:.4f}")
    
    logger.info(f"\nVisualizations saved to: {plots_dir}")
    logger.info("[SUCCESS] Model evaluation completed!")
    
    return evaluation_results

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained anomaly detection models")
    parser.add_argument("--config", default="../config.yaml", help="Path to config file")
    parser.add_argument("--data-dir", default="../data/processed", help="Directory containing processed data")
    parser.add_argument("--models-dir", default="../models", help="Directory containing trained models")
    args = parser.parse_args()
    
    try:
        # Load configuration
        cfg = load_config(args.config)
        
        # Override paths if specified
        if args.data_dir != "../data/processed":
            cfg["paths"]["results_dir"] = args.data_dir
        if args.models_dir != "../models":
            cfg["paths"]["model_dir"] = args.models_dir
        
        # Check if processed data exists
        if not os.path.exists(args.data_dir):
            print(f"❌ Processed data directory not found: {args.data_dir}")
            print("Please run data preprocessing first or specify correct --data-dir")
            return 1
        
        # Check if models exist
        if not os.path.exists(args.models_dir):
            print(f"❌ Models directory not found: {args.models_dir}")
            print("Please train models first or specify correct --models-dir")
            return 1
        
        # Load processed data
        print("Loading processed data...")
        data = {
            "X_test": load_pickle(os.path.join(args.data_dir, "X_test_processed.pkl")),
            "y_test": load_pickle(os.path.join(args.data_dir, "y_test_processed.pkl")),
            "classes": load_pickle(os.path.join(args.data_dir, "metadata.pkl"))["classes"]
        }
        
        print(f"[SUCCESS] Loaded test data: X_test shape={data['X_test'].shape}")
        
        # Load trained models
        print("Loading trained models...")
        artifacts = {}
        model_files = {
            "random_forest": "random_forest.pkl",
            "xgboost": "xgboost.pkl", 
            "lightgbm": "lightgbm.pkl"
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(args.models_dir, filename)
            if os.path.exists(model_path):
                try:
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
        
        # Evaluate models
        print("Starting model evaluation...")
        results = evaluate_all(cfg, data, artifacts)
        
        print("[SUCCESS] Model evaluation completed successfully!")
        print(f"Results saved to: {cfg['paths']['results_dir']}")
        
    except Exception as e:
        print(f"[ERROR] Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
