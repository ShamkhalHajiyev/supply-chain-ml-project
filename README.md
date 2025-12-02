# Supply Chain Analytics: ML-Powered Delivery Prediction & Demand Forecasting

## Project Overview

This data science project tackles critical supply chain optimization challenges through advanced machine learning and deep learning techniques. Focused on two key objectives: predicting late deliveries and forecasting product demand, this solution helps e-commerce businesses enhance operational efficiency and customer satisfaction. By leveraging historical order data, shipping details, and customer information, the project develops predictive models that enable proactive management of delivery risks and inventory optimization.

The solution employs a comprehensive machine learning pipeline that processes e-commerce transaction data to identify patterns leading to delivery delays and forecast future product demand. Through careful feature engineering and the application of both traditional ML models and deep learning approaches (LSTM networks), the system provides actionable insights for supply chain managers to optimize delivery reliability and inventory management decisions.

## Quick Start

### Running the Complete Pipeline

```bash
# Install dependencies
uv sync

# Run the full ML pipeline (data → preprocessing → features → training)
python main.py --all
```

### Running Individual Steps

```bash
# Step 1: Download/load raw data
python main.py --data

# Step 2: Preprocess data
python main.py --preprocess

# Step 3: Build features
python main.py --features

# Step 4: Train ML classification models
python main.py --train-ml

# Step 5: Train LSTM forecasting model
python main.py --train-lstm

# Train all models (ML + LSTM)
python main.py --train-all

# Evaluate trained models
python main.py --evaluate
```

## Key Features & Technical Implementation

- **Data Processing & Analysis**
  - Automated data loading from Kaggle with caching support
  - Comprehensive exploratory data analysis (EDA) notebooks
  - Data quality assessment (missing values, duplicates, outliers)
  - Data cleaning pipeline with missing value imputation and standardization
  - Feature engineering for temporal, categorical, and geographic variables

- **Machine Learning Models**
  - Classification models for delivery delay prediction (Logistic Regression, Random Forest, Gradient Boosting)
  - LSTM-based deep learning models for time series demand forecasting
  - Model evaluation and performance optimization
  - Reproducible training pipelines

- **Technical Stack**
  - Python 3.10+ with modern package management (uv)
  - Data Processing: Pandas, NumPy
  - Data Visualization: Plotly, Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
  - Deep Learning: PyTorch
  - Data Storage: Parquet format for efficient storage
  - Development: Jupyter Notebooks

## Project Structure

```
├── main.py                    # CLI entry point for running pipeline
├── data/                      # Dataset storage
│   ├── raw/                   # Raw data files
│   ├── interim/               # Intermediate processed data
│   ├── processed/             # Final processed datasets
│   └── external/              # External data sources
├── models/                    # Trained model artifacts
├── notebooks/                 # Jupyter notebooks for each step
│   ├── 01_data_loading.ipynb          # Data loading & exploration
│   ├── 02_preprocessing.ipynb          # Data cleaning & preprocessing
│   ├── 03_feature_engineering.ipynb    # Feature creation
│   ├── 04_ml_training.ipynb            # ML model training
│   ├── 05_lstm_training.ipynb          # LSTM forecasting
│   ├── eda.ipynb                       # Full exploratory analysis
│   └── model_evaluation.ipynb          # Model evaluation
├── reports/                   # Analysis reports and visualizations
│   ├── figures/              # Generated plots and charts
│   └── logs/                 # Training logs
├── src/
│   ├── data/
│   │   ├── data_manager.py   # Data loading and file management
│   │   └── preprocess.py     # Data preprocessing and cleaning
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   └── models/
│       ├── train_lstm.py     # LSTM model training
│       └── train_ml.py       # ML classification training
├── utils/                     # Utility functions
├── pyproject.toml            # Project dependencies
└── uv.lock                   # Locked dependencies
```

## Notebooks Guide

The project includes step-by-step notebooks with detailed interpretation:

| Notebook | Description | Key Learning |
|----------|-------------|--------------|
| `01_data_loading.ipynb` | Load and explore raw data | Data structure, quality issues |
| `02_preprocessing.ipynb` | Clean and transform data | Missing values, outliers, encoding |
| `03_feature_engineering.ipynb` | Create ML features | Temporal, customer, product features |
| `04_ml_training.ipynb` | Train classification models | Model comparison, feature importance |
| `05_lstm_training.ipynb` | Train LSTM forecaster | Time series, neural networks |

## Skills Demonstrated

- Data preprocessing and cleaning
- Feature engineering and selection
- Time series analysis and forecasting
- Deep learning model development
- Machine learning model evaluation
- Business impact analysis
- Supply chain optimization
- Production-ready code development
- CLI application development

## Business Impact

This project demonstrates significant real-world business value by addressing critical supply chain challenges:
- **Reduce delivery delays** through predictive analytics
- **Optimize inventory levels** with accurate demand forecasting
- **Enhance customer satisfaction** through reliable delivery predictions
- **Minimize operational costs** through data-driven decision making

## Getting Started

1. Clone the repository
2. Set up the environment using uv:
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate

   # Install dependencies from pyproject.toml
   uv sync
   ```

3. Run the pipeline:
   ```bash
   # Option A: Run complete pipeline
   python main.py --all

   # Option B: Run step by step
   python main.py --data
   python main.py --preprocess
   python main.py --features
   python main.py --train-all
   ```

4. Or explore the notebooks:
   ```bash
   jupyter notebook notebooks/01_data_loading.ipynb
   ```

## Dataset

The project uses the [DataCo Supply Chain Dataset](https://www.kaggle.com/datasets/saicharankomati/dataco-supply-chain-dataset) from Kaggle. The dataset includes:
- Order details (dates, items, quantities, prices)
- Customer information (location, segment, demographics)
- Product information (categories, descriptions, prices)
- Shipping details (delivery status, dates, modes, regions)
- Financial metrics (sales, benefits, profit ratios)

The data is automatically downloaded on first use via the `data_manager.py` module.

## Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Raw Data   │ ──▶ │ Preprocessing │ ──▶ │ Feature         │
│  (Kaggle)   │     │  (cleaning)   │     │ Engineering     │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             ▼                             ▼
           ┌───────────────┐            ┌───────────────┐             ┌───────────────┐
           │ Logistic Reg  │            │ Random Forest │             │ Gradient Boost│
           └───────────────┘            └───────────────┘             └───────────────┘
                    │                             │                             │
                    └─────────────────────────────┼─────────────────────────────┘
                                                  ▼
                                         ┌───────────────┐
                                         │ Best Model    │
                                         │ Selection     │
                                         └───────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                                                           ▼
           ┌───────────────┐                                          ┌───────────────┐
           │ Classification│                                          │     LSTM      │
           │ (Late Delivery)│                                          │ (Forecasting) │
           └───────────────┘                                          └───────────────┘
```

## Portfolio Value

This project showcases advanced data science capabilities in solving real-world business problems, combining technical expertise in machine learning, deep learning, and data analysis with practical business acumen in supply chain optimization. It demonstrates proficiency in end-to-end ML project development, from data processing to model deployment, making it a valuable addition to any data science portfolio.
