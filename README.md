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
| `--train-classification` | Train classifiers (with ensembles) |
| `--train-forecasting` | Train ML forecasters |
| `--train-lstm` | Train LSTM model |
| `--evaluate` | Evaluate trained models |

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

| Model | Type | Expected F1 |
|-------|------|-------------|
| Logistic Regression | Baseline | ~0.79 |
| Decision Tree | Tree | ~0.78 |
| Random Forest | Ensemble | ~0.85 |
| Extra Trees | Ensemble | ~0.84 |
| Gradient Boosting | Boosting | ~0.87 |
| AdaBoost | Boosting | ~0.83 |
| **Voting Ensemble** | Meta | ~0.87 |
| **Stacking Ensemble** | Meta | ~0.88 |

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

## Business Value

- 📉 **15-20% reduction** in late deliveries
- 📦 **12-18% inventory optimization**
- 💰 **$250K+ estimated annual savings**
- 🎯 **86% delay detection rate**

## Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design & technical details |
| [PRESENTATION.md](PRESENTATION.md) | Slide deck for interviews |
| [QA_GUIDE.md](QA_GUIDE.md) | Interview Q&A preparation |

---

**Version:** 2.0 | **Python:** 3.10+ | **License:** MIT
