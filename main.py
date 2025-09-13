#!/usr/bin/env python3
"""
Clean Main Run Script for Network Traffic Anomaly Detection
Trains 3 models: LightGBM, XGBoost, RandomForest
Optimized for 97-99% accuracy with fast training times
"""

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

try:
    from utilities import load_config, setup_logger, ensure_dir
    from data_preprocessing import load_and_prepare_data
    from model_training import train_all_models
    from model_evaluation import evaluate_all
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are in the src/ directory")
    sys.exit(1)

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, total_steps, description):
    """Print a formatted step description."""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 50)

def main():
    """Main execution function."""
    start_time = time.time()
    
    print_header("NETWORK TRAFFIC ANOMALY DETECTION")
    print("Clean Training Pipeline for 3 Models")
    print("Target: 97-99% Accuracy with Fast Training")
    
    # Load configuration
    print_step(1, 5, "Loading Configuration")
    try:
        cfg = load_config("config.yaml")
        logger = setup_logger(log_dir="results/logs")
        logger.info("Configuration loaded successfully")
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Data preprocessing
    print_step(2, 5, "Data Preprocessing")
    try:
        print("Loading and preprocessing data...")
        data = load_and_prepare_data(cfg)
        if data is None:
            print("❌ Data preprocessing failed")
            sys.exit(1)
        print("✅ Data preprocessing completed")
        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        logger.error(f"Data preprocessing failed: {e}")
        sys.exit(1)
    
    # Model training
    print_step(3, 5, "Model Training")
    try:
        print("Training 3 models with cross-validation...")
        print("Models: LightGBM, XGBoost, RandomForest")
        
        artifacts, results_summary = train_all_models(cfg, data)
        
        if not artifacts:
            print("❌ Model training failed")
            sys.exit(1)
        
        print("✅ Model training completed")
        logger.info("Model training completed successfully")
        
        # Print training summary
        print("\n" + "="*60)
        print("TRAINING RESULTS SUMMARY")
        print("="*60)
        
        for model_name, results in results_summary.items():
            if results:
                cv_mean = results["cv_mean"]
                cv_std = results["cv_std"]
                print(f"{model_name.upper():<20} CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
                
                # Check target achievement
                if cv_mean >= 0.97:
                    print(f"   🎯 Target accuracy achieved!")
                elif cv_mean >= 0.95:
                    print(f"   📈 Very close to target!")
                else:
                    print(f"   ⚠️  Below target (need improvement)")
            else:
                print(f"{model_name.upper():<20} ❌ Training failed")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        logger.error(f"Model training failed: {e}")
        sys.exit(1)
    
    # Model evaluation
    print_step(4, 5, "Model Evaluation")
    try:
        print("Evaluating all trained models...")
        
        # Load test data
        test_data = {
            "X_test": data["X_test"],
            "y_test": data["y_test"],
            "classes": data["classes"]
        }
        
        # Evaluate models
        evaluation_results = evaluate_all(cfg, test_data, artifacts)
        
        if evaluation_results:
            print("✅ Model evaluation completed")
            logger.info("Model evaluation completed successfully")
            
            # Print evaluation summary
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            
            for model_name, results in evaluation_results.items():
                if results:
                    accuracy = results.get("accuracy", "N/A")
                    f1_score = results.get("f1_score", "N/A")
                    print(f"{model_name.upper():<20} Accuracy: {accuracy}, F1: {f1_score}")
                else:
                    print(f"{model_name.upper():<20} ❌ Evaluation failed")
        else:
            print("❌ Model evaluation failed")
            
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        logger.error(f"Model evaluation failed: {e}")
    
    # Model saving and cleanup
    print_step(5, 5, "Finalizing and Cleanup")
    try:
        print("Saving models and cleaning up...")
        
        # Ensure model directory exists
        model_dir = Path(cfg["paths"]["model_dir"])
        ensure_dir(str(model_dir))
        
        # Save training results
        results_dir = Path("results/clean_training")
        ensure_dir(str(results_dir))
        
        from utilities import save_pickle
        save_pickle(results_summary, results_dir / "training_results.pkl")
        save_pickle(evaluation_results, results_dir / "evaluation_results.pkl")
        
        print("✅ Models and results saved")
        print(f"   Models: {model_dir}")
        print(f"   Results: {results_dir}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        print(f"❌ Finalization failed: {e}")
        logger.error(f"Finalization failed: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    
    print_header("PIPELINE COMPLETED")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Check target achievement
    target_models = 0
    close_models = 0
    
    for model_name, results in results_summary.items():
        if results:
            cv_mean = results["cv_mean"]
            if cv_mean >= 0.97:
                target_models += 1
            elif cv_mean >= 0.95:
                close_models += 1
    
    print(f"\n🎯 Models achieving target (97%+): {target_models}/3")
    print(f"📈 Models close to target (95-97%): {close_models}/3")
    
    if target_models >= 2:
        print("\n🎉 SUCCESS: Majority of models achieved target accuracy!")
    elif target_models + close_models >= 2:
        print("\n📈 GOOD: Most models are performing well!")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT: Consider tuning hyperparameters")
    
    print("\n📁 Check the following directories for results:")
    print(f"   - Models: {cfg['paths']['model_dir']}")
    print(f"   - Results: results/clean_training")
    print(f"   - Logs: results/logs")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
