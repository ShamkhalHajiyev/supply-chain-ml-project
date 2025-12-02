# Supply Chain ML Pipeline - Architecture Documentation

## System Overview

End-to-end ML system for supply chain optimization with two components:
1. **Classification**: Late delivery prediction (8 models including ensembles)
2. **Forecasting**: Demand prediction (7 ML models + optional LSTM)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │   Kaggle    │───▶│ Data Manager │───▶│ Raw/Interim/Process │    │
│  │   Dataset   │    │  (Caching)   │    │    (Parquet)        │    │
│  └─────────────┘    └──────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                                │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │ Preprocessor│───▶│   Feature    │───▶│   Feature Store     │    │
│  │  (Cleaning) │    │  Engineer    │    │   (60+ features)    │    │
│  └─────────────┘    └──────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│     CLASSIFICATION        │   │       FORECASTING          │
│  ┌─────────────────────┐  │   │  ┌─────────────────────┐  │
│  │ • Logistic Reg      │  │   │  │ • Ridge/Lasso       │  │
│  │ • Decision Tree     │  │   │  │ • Random Forest     │  │
│  │ • Random Forest     │  │   │  │ • Gradient Boosting │  │
│  │ • Extra Trees       │  │   │  │ • XGBoost           │  │
│  │ • Gradient Boosting │  │   │  │ • Extra Trees       │  │
│  │ • AdaBoost          │  │   │  │ • ElasticNet        │  │
│  │ • Voting Ensemble   │  │   │  └─────────────────────┘  │
│  │ • Stacking Ensemble │  │   │  ┌─────────────────────┐  │
│  └─────────────────────┘  │   │  │ LSTM (Optional)     │  │
│  classifier.py            │   │  │ Deep Learning       │  │
└───────────────────────────┘   │  └─────────────────────┘  │
                │               │  forecaster.py             │
                │               │  forecaster_lstm.py        │
                │               └───────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION LAYER                                │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │ Overfitting │    │    SHAP      │    │   Business          │    │
│  │  Detection  │    │ Explainer    │    │   Insights          │    │
│  └─────────────┘    └──────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
supply-chain-ml-project/
├── main.py                         # CLI entry point
├── pyproject.toml                  # Dependencies
│
├── data/
│   ├── raw/                        # Raw CSV from Kaggle
│   ├── interim/                    # Intermediate data
│   └── processed/                  # ML-ready features
│
├── models/                         # Saved model artifacts (.pkl, .pt)
│
├── notebooks/                      # Step-by-step analysis
│   ├── 01_data_loading.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_classification_modeling.ipynb
│   ├── 06_demand_forecasting.ipynb
│   └── 07_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── data_manager.py         # Kaggle API, caching
│   │   └── preprocess.py           # Data cleaning
│   ├── features/
│   │   └── build_features.py       # Feature engineering
│   └── models/
│       ├── __init__.py             # Module exports
│       ├── classifier.py           # 8 classification models
│       ├── forecaster.py           # 7 ML forecasting models
│       └── forecaster_lstm.py      # LSTM deep learning
│
└── reports/
    ├── figures/
    └── logs/
```

## Model Specifications

### Classification Models (`classifier.py`)

| Model | Type | Key Parameters | Purpose |
|-------|------|----------------|---------|
| Logistic Regression | Linear | max_iter=1000, balanced | Baseline |
| Decision Tree | Tree | max_depth=10 | Interpretable |
| Random Forest | Ensemble | n=200, depth=15 | High accuracy |
| Extra Trees | Ensemble | n=200, depth=15 | Variance reduction |
| Gradient Boosting | Boosting | n=150, lr=0.1, depth=5 | Best single model |
| AdaBoost | Boosting | n=100, lr=0.1 | Alternative boosting |
| **Voting Ensemble** | Meta | RF + GB + ET, soft voting | Robust predictions |
| **Stacking Ensemble** | Meta | RF + GB + ET → LR | Best overall |

### Forecasting Models (`forecaster.py`)

| Model | Type | Key Parameters | Purpose |
|-------|------|----------------|---------|
| Ridge Regression | Linear | alpha=1.0 | Baseline |
| Lasso Regression | Linear | alpha=0.1 | Feature selection |
| ElasticNet | Linear | alpha=0.1, l1=0.5 | Mixed regularization |
| Random Forest | Ensemble | n=100, depth=10 | Non-linear patterns |
| Gradient Boosting | Boosting | n=100, lr=0.1 | Sequential learning |
| **XGBoost** | Boosting | n=100, depth=5 | Best for tabular |
| Extra Trees | Ensemble | n=100, depth=10 | Alternative RF |

### LSTM Model (`forecaster_lstm.py`)

```
Input (sequence_length=30, features=2)
    ↓
LSTM Layer 1 (hidden=64)
    ↓
LSTM Layer 2 (hidden=64)
    ↓
Dropout (0.2)
    ↓
Fully Connected (output=1)
```

## Feature Engineering

### Classification Features (60+)

| Category | Count | Examples |
|----------|-------|----------|
| Temporal | 5 | day_of_week, month, is_weekend |
| Customer | 2 | order_count, lifetime_value |
| Product | 4 | popularity, order_value, discount_rate |
| Shipping | 2 | urgency, region_encoded |
| Financial | 3 | profit_margin_pct, sales_per_item |
| Encoded | 40+ | Label-encoded categoricals |

### Forecasting Features

| Type | Examples |
|------|----------|
| Lag Features | demand_lag_1, demand_lag_7, demand_lag_30 |
| Rolling Stats | rolling_mean_7, rolling_std_14, rolling_max_30 |
| Temporal | day_of_week, month, is_weekend, is_month_end |

## Overfitting Detection

Both classifier and forecaster include automatic overfitting detection:

```
Train Accuracy vs Test Accuracy → Gap Analysis
    ├── Gap > 5%  → ⚠️ OVERFITTING
    ├── Gap < -2% → ⚠️ UNDERFITTING
    └── Otherwise → ✅ GOOD FIT
```

**Visualization:** Train vs Test bar charts for all models

## Model Evaluation

### Classification Metrics
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC, Confusion Matrix
- 5-Fold Cross-Validation
- **SHAP Feature Importance**

### Forecasting Metrics
- RMSE, MAE, R², MAPE
- Train vs Test comparison
- Feature importance

## CLI Usage

```bash
# Full pipeline
python main.py --all

# Individual steps
python main.py --data                   # Load data
python main.py --preprocess             # Clean data
python main.py --features               # Build features
python main.py --train-classification   # Train classifiers
python main.py --train-forecasting      # Train ML forecasters
python main.py --train-lstm             # Train LSTM
python main.py --evaluate               # Evaluate models
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Package Manager | uv |
| Data Processing | Pandas, NumPy, Polars |
| ML Framework | Scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |
| Storage | Parquet |
| Notebooks | Jupyter |

## Performance Benchmarks

### Classification (Expected)

| Model | Test F1 | Status |
|-------|---------|--------|
| Logistic Regression | ~0.79 | Baseline |
| Random Forest | ~0.85 | Good |
| Gradient Boosting | ~0.87 | Best single |
| Stacking Ensemble | ~0.88 | Best overall |

### Forecasting (Expected)

| Model | Test R² | Status |
|-------|---------|--------|
| Ridge | ~0.65 | Baseline |
| XGBoost | ~0.80 | Best ML |
| LSTM | ~0.82 | Best overall |

---
**Version:** 2.0 | **Updated:** December 2025
