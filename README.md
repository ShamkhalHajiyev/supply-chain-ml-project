# Supply Chain Analytics: ML-Powered Delivery Prediction & Demand Forecasting

## Project Overview

This data science project tackles critical supply chain optimization challenges through advanced machine learning and deep learning techniques. Focused on two key objectives: predicting late deliveries and forecasting product demand, this solution helps e-commerce businesses enhance operational efficiency and customer satisfaction.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full ML pipeline
python main.py --all

# Or run individual steps
python main.py --train-classification   # Classification with ensemble models
python main.py --train-forecasting      # ML forecasting models
python main.py --train-lstm             # LSTM model (optional)
```

## Key Features

- **8 Classification Models** including Voting & Stacking Ensembles
- **7 Forecasting Models** (XGBoost, Random Forest, Gradient Boosting, etc.)
- **Overfitting Detection** - Train vs Test comparison for all models
- **SHAP Explanations** - Interpretable model insights
- **Comprehensive Notebooks** - 7 detailed step-by-step notebooks

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py --all` | Run complete pipeline |
| `python main.py --data` | Download/load raw data |
| `python main.py --preprocess` | Preprocess data |
| `python main.py --features` | Build features |
| `python main.py --train-classification` | Train classification models (with ensembles) |
| `python main.py --train-forecasting` | Train ML forecasting models |
| `python main.py --train-lstm` | Train LSTM model |
| `python main.py --evaluate` | Evaluate trained models |

## Project Structure

```
├── main.py                    # CLI entry point
├── data/
│   ├── raw/                   # Raw data from Kaggle
│   ├── interim/               # Intermediate processed data
│   └── processed/             # Final processed datasets
├── models/                    # Trained model artifacts (.pkl)
├── notebooks/                 # Step-by-step analysis notebooks
│   ├── 01_data_loading.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_classification_modeling.ipynb
│   ├── 06_demand_forecasting.ipynb
│   └── 07_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── data_manager.py    # Data loading & caching
│   │   └── preprocess.py      # Data preprocessing
│   ├── features/
│   │   └── build_features.py  # Feature engineering
│   └── models/
│       ├── classifier.py      # Classification (8 models)
│       ├── forecaster.py      # ML forecasting (7 models)
│       └── forecaster_lstm.py # LSTM forecasting
└── pyproject.toml             # Dependencies
```

## Notebooks Guide

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 01 | `01_data_loading.ipynb` | Load & explore raw data | Dataset structure, target variables |
| 02 | `02_exploratory_analysis.ipynb` | Comprehensive EDA | Distributions, correlations, patterns |
| 03 | `03_data_preprocessing.ipynb` | Data cleaning | Missing values, outliers, encoding |
| 04 | `04_feature_engineering.ipynb` | Feature creation | Temporal, customer, product features |
| 05 | `05_classification_modeling.ipynb` | Late delivery prediction | **Ensemble models, overfitting analysis, SHAP** |
| 06 | `06_demand_forecasting.ipynb` | Demand forecasting | **XGBoost, Random Forest vs LSTM** |
| 07 | `07_model_evaluation.ipynb` | Business insights | Impact analysis, recommendations |

## Models

### Classification (Late Delivery Prediction)
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- **Voting Ensemble**
- **Stacking Ensemble**

### Forecasting (Demand Prediction)
- Ridge/Lasso Regression
- Random Forest
- Gradient Boosting
- **XGBoost**
- Extra Trees
- ElasticNet
- LSTM (optional deep learning)

## Technical Stack

- **Python 3.10+** with uv package manager
- **Data**: Pandas, NumPy
- **ML**: Scikit-learn, XGBoost
- **Deep Learning**: PyTorch
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Explainability**: SHAP

## Getting Started

```bash
# 1. Clone and setup
git clone <repo>
cd supply-chain-ml-project

# 2. Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv sync

# 3. Run pipeline
python main.py --all

# 4. Or explore notebooks
jupyter notebook notebooks/01_data_loading.ipynb
```

## Dataset

[DataCo Supply Chain Dataset](https://www.kaggle.com/datasets/saicharankomati/dataco-supply-chain-dataset) - 180K e-commerce transactions with order, customer, product, and shipping data. Auto-downloaded on first run.

## Business Value

- **Reduce delivery delays** through predictive early warning
- **Optimize inventory** with accurate demand forecasting
- **Interpretable insights** via SHAP explanations
- **Actionable recommendations** for operations teams
