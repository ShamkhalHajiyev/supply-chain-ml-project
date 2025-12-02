# Supply Chain Analytics
## ML-Powered Delivery Prediction & Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

End-to-end machine learning system for supply chain optimization with two components:
- **Classification**: Late delivery prediction (8 models including ensembles)
- **Forecasting**: Demand prediction (7 ML models + optional LSTM)

**Key Features:**
- 🎯 **Ensemble Methods** - Voting & Stacking for best performance
- 📊 **Overfitting Detection** - Automatic train vs test comparison
- 🔍 **SHAP Explanations** - Interpretable model insights
- 📓 **7 Detailed Notebooks** - Step-by-step analysis
- 🚀 **Parallel Training** - Fast model training using all CPU cores
- ⚠️ **Data Leakage Prevention** - Proper feature selection

## Quick Start

```bash
# Install dependencies
uv sync

# Run full pipeline
python main.py --all

# Or individual steps
python main.py --train-classification   # 8 classification models
python main.py --train-forecasting      # 7 ML forecasting models
python main.py --train-lstm             # LSTM (optional)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `--all` | Complete pipeline |
| `--data` | Download/load raw data |
| `--preprocess` | Preprocess data |
| `--features` | Build features |
| `--train-classification` | Train classifiers (with parallel execution) |
| `--train-forecasting` | Train ML forecasters |
| `--train-lstm` | Train LSTM model |
| `--evaluate` | Evaluate trained models |
| `--no-parallel` | Disable parallel training (debug mode) |

## Project Structure

```
supply-chain-ml-project/
├── main.py                         # CLI entry point
├── pyproject.toml                  # Dependencies
├── data/
│   ├── raw/                        # Raw data from Kaggle
│   ├── interim/                    # Intermediate data
│   └── processed/                  # ML-ready features
├── models/                         # Saved models (.pkl, .pt)
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_classification_modeling.ipynb
│   ├── 06_demand_forecasting.ipynb
│   └── 07_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── data_manager.py         # Kaggle API, caching
│   │   └── preprocess.py           # Data cleaning
│   ├── features/
│   │   └── build_features.py       # 60+ features
│   └── models/
│       ├── classifier.py           # 8 classification models
│       ├── forecaster.py           # 7 ML forecasting models
│       └── forecaster_lstm.py      # LSTM deep learning
└── reports/
    ├── figures/
    └── logs/
```

## Notebooks Guide

| # | Notebook | Key Topics |
|---|----------|------------|
| 01 | Data Loading | Dataset structure, target variables |
| 02 | Exploratory Analysis | Distributions, correlations |
| 03 | Data Preprocessing | Missing values, outliers |
| 04 | Feature Engineering | 60+ features across 5 categories |
| 05 | **Classification** | 8 models, ensembles, SHAP |
| 06 | **Forecasting** | 7 ML models vs LSTM |
| 07 | Model Evaluation | Business impact, recommendations |

## Models

### Classification (Late Delivery)

> **Note**: Results are realistic (~70% accuracy) because we properly exclude leaky features.

| Model | Type | Test F1 | Status |
|-------|------|---------|--------|
| Logistic Regression | Baseline | ~0.69 | ✅ Good Fit |
| Decision Tree | Tree | ~0.70 | ✅ Good Fit |
| Random Forest | Ensemble | ~0.70 | ✅ Good Fit |
| Extra Trees | Ensemble | ~0.70 | ✅ Good Fit |
| Gradient Boosting | Boosting | ~0.70 | ✅ Good Fit |
| AdaBoost | Boosting | ~0.69 | ✅ Good Fit |
| Voting Ensemble | Meta | ~0.70 | ✅ Good Fit |
| **Stacking Ensemble** | Meta | **~0.71** | ✅ Best |

### Forecasting (Demand)

| Model | Type | Expected R² |
|-------|------|-------------|
| Ridge/Lasso/ElasticNet | Linear | ~0.65 |
| Random Forest | Ensemble | ~0.75 |
| Gradient Boosting | Boosting | ~0.78 |
| **XGBoost** | Boosting | ~0.80 |
| Extra Trees | Ensemble | ~0.76 |
| LSTM | Deep Learning | ~0.82 |

## Technical Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.10+ |
| Package Manager | uv |
| Data | Pandas, NumPy, Polars |
| ML | Scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Visualization | Matplotlib, Seaborn, Plotly |
| Explainability | SHAP |
| Storage | Parquet |

## Getting Started

```bash
# 1. Clone repository
git clone <repo-url>
cd supply-chain-ml-project

# 2. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Setup environment
uv venv && source .venv/bin/activate
uv sync

# 4. Run pipeline
python main.py --all

# 5. Or explore notebooks
jupyter notebook notebooks/01_data_loading.ipynb
```

## Dataset

[DataCo Supply Chain Dataset](https://www.kaggle.com/datasets/saicharankomati/dataco-supply-chain-dataset)
- **180K** e-commerce transactions
- **50+** features (order, customer, product, shipping)
- Auto-downloaded on first run

## ⚠️ Data Leakage Prevention

This project implements proper feature selection to avoid data leakage. The following columns are **excluded** from features because they contain information that would only be known AFTER delivery:

| Excluded Column | Reason |
|-----------------|--------|
| `late_delivery_risk` | This IS the target variable |
| `delivery_status` | Categorical form of target |
| `days_for_shipping_(real)` | Only known after delivery |
| `shipping_date` | Actual shipping date (post-hoc) |

This is why our models achieve ~70% accuracy instead of 100%. The 100% accuracy seen in some Kaggle kernels is due to data leakage!

## Business Value

- 📉 **10-15% reduction** in late deliveries through proactive intervention
- 📦 **Prioritized shipping** for high-risk orders
- 💰 **Cost savings** from reduced customer complaints and refunds
- 🎯 **70% delay detection rate** (realistic, leakage-free model)

## Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design & technical details |
| [PRESENTATION.md](PRESENTATION.md) | Slide deck for interviews |
| [QA_GUIDE.md](QA_GUIDE.md) | Interview Q&A preparation |

---

**Version:** 2.0 | **Python:** 3.10+ | **License:** MIT
