# Network Traffic Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)

A **production-ready** machine learning system for detecting network traffic anomalies and cyber attacks using ensemble methods. This system achieves **92-93% accuracy** on the KDD Cup 1999 dataset and is optimized for real-world deployment.

## Key Features

- **Multi-Model Ensemble**: LightGBM, XGBoost, and Random Forest
- **High Accuracy**: 92-93% classification accuracy
- **Production Ready**: Confidence scoring and monitoring
- **Comprehensive Analytics**: Evaluation metrics and visualizations

## Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- 2GB+ disk space for data and models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Krish252004/Network-Anamoly-Detection.git
   cd Network-Anamoly-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** (KDD Cup 1999)
   ```bash
   # Download the KDD Cup 1999 dataset from:
   # https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
   # 
   # Create the following directory structure:
   # data/raw/
   #   - kddcup.data_10_percent.gz
   #   - corrected.gz
   #   - kddcup.testdata.unlabeled.gz
   ```

### Running the System

**Complete Pipeline (Training + Evaluation)**
```bash
python main.py
```

**Individual Components**
```bash
# Data preprocessing only
python -m src.data_preprocessing

# Model training only
python -m src.model_training

# Model evaluation only
python -m src.model_evaluation
```

## Dataset Information

This project uses the **KDD Cup 1999** dataset, which contains:

- **Training Data**: 494,021 samples (10% of full dataset)
- **Test Data**: 311,029 samples
- **Features**: 41 network connection features
- **Attack Types**: 4 main categories
  - **DoS**: Denial of Service attacks
  - **Probe**: Surveillance and probing
  - **R2L**: Remote to local attacks
  - **U2R**: User to root attacks

### Feature Categories
- **Basic Features**: Duration, protocol type, service, flag
- **Content Features**: Login attempts, file operations, etc.
- **Traffic Features**: Source/destination bytes, connection counts
- **Time-based Features**: Connection patterns and timing

## System Architecture

```
Project Structure
├── main.py                    # Main execution script
├── train_models.py           # Training script
├── config.yaml               # Configuration settings
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── LICENSE                  # License file
└── src/
    ├── __init__.py          # Package initialization
    ├── data_preprocessing.py # Data loading and preprocessing
    ├── feature_engineering.py # Advanced feature creation
    ├── model_training.py     # Model training pipeline
    ├── model_evaluation.py   # Evaluation and metrics
    ├── model_inference.py    # Production inference
    ├── model_configurations.py # Model configurations
    └── utilities.py          # Helper functions
```


## Model Performance

### Accuracy Results
| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| **LightGBM** | 92.48% | 92.29% | ~2 minutes |
| **XGBoost** | 92.41% | 92.22% | ~3 minutes |
| **Random Forest** | 92.36% | 92.13% | ~5 minutes |
| **Ensemble** | 92.50% | 92.31% | - |

### Attack Detection Performance
- **DoS Attacks**: High detection rate
- **Probe Attacks**: High detection rate
- **R2L Attacks**: Moderate detection rate
- **U2R Attacks**: Lower detection rate


## Usage

### Basic Usage
```python
from src.model_inference import ModelInference

# Initialize the model
inference = ModelInference()

# Make predictions on new data
predictions = inference.predict(new_network_data)
print(f"Detected anomalies: {predictions}")
```

### Model Training
```python
# Train all models
python main.py

# Train specific model
python train_models.py
```

## Requirements

- Python 3.8+
- scikit-learn, xgboost, lightgbm
- pandas, numpy
- matplotlib, seaborn (for visualizations)
- imbalanced-learn (for class balancing)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Acknowledgments

- KDD Cup 1999 Dataset for providing the benchmark dataset
- Scikit-learn, XGBoost, and LightGBM communities for excellent ML tools


*Built for cybersecurity and machine learning*