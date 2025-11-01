# Supply Chain Analytics: ML-Powered Delivery Prediction & Demand Forecasting

## Project Overview

This data science project tackles critical supply chain optimization challenges through advanced machine learning and deep learning techniques. Focused on two key objectives: predicting late deliveries and forecasting product demand, this solution helps e-commerce businesses enhance operational efficiency and customer satisfaction. By leveraging historical order data, shipping details, and customer information, the project develops predictive models that enable proactive management of delivery risks and inventory optimization.

The solution employs a comprehensive machine learning pipeline that processes e-commerce transaction data to identify patterns leading to delivery delays and forecast future product demand. Through careful feature engineering and the application of both traditional ML models and deep learning approaches (LSTM networks), the system provides actionable insights for supply chain managers to optimize delivery reliability and inventory management decisions.

## Key Features & Technical Implementation

- **Data Processing & Analysis**
  - Automated data loading from Kaggle with caching support
  - Comprehensive exploratory data analysis (EDA) notebook with detailed visualizations
  - Data quality assessment (missing values, duplicates, outliers, encoding issues)
  - Data cleaning pipeline with missing value imputation and standardization
  - Feature engineering for temporal, categorical, and geographic variables
  - Feature selection analysis for classification and forecasting models
  - Efficient data handling using Pandas with parquet storage for processed data

- **Machine Learning Models**
  - Classification models for delivery delay prediction
  - LSTM-based deep learning models for time series demand forecasting
  - Model evaluation and performance optimization
  - Reproducible training pipelines

- **Technical Stack**
  - Python 3.10+ with modern package management (uv)
  - Data Processing: Pandas, NumPy, Polars
  - Data Visualization: Plotly, Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
  - Deep Learning: PyTorch
  - Data Storage: Parquet format for efficient storage
  - Development: Jupyter Notebooks
  - Version Control: Git

## Project Structure

```
├── data/                      # Dataset storage
│   ├── raw/                   # Raw data files
│   ├── interim/               # Intermediate processed data
│   ├── processed/             # Final processed datasets
│   └── external/              # External data sources
├── models/                    # Trained model artifacts
├── notebooks/                 # Jupyter notebooks for EDA
│   └── eda.ipynb             # Comprehensive exploratory data analysis
├── reports/                   # Analysis reports and visualizations
│   ├── figures/              # Generated plots and charts
│   └── logs/                 # Training logs and reports
├── src/
│   ├── data/
│   │   ├── data_manager.py   # Data loading and file management utilities
│   │   └── preprocess.py     # Data preprocessing and cleaning
│   ├── features/
│   │   └── build_features.py # Feature engineering and transformation
│   └── models/
│       ├── train_lstm.py     # LSTM model training pipeline
│       └── train_ml.py       # Machine learning model training
└── utils/                     # Utility functions and helpers
```

## Skills Demonstrated

- Data preprocessing and cleaning
- Feature engineering and selection
- Time series analysis and forecasting
- Deep learning model development
- Machine learning model evaluation
- Business impact analysis
- Supply chain optimization
- Production-ready code development
- Project organization and documentation

## Business Impact

This project demonstrates significant real-world business value by addressing critical supply chain challenges. The models enable businesses to:
- Reduce delivery delays through predictive analytics
- Optimize inventory levels with accurate demand forecasting
- Enhance customer satisfaction through reliable delivery predictions
- Minimize operational costs through data-driven decision making

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
3. Run the EDA notebook:
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```
   The EDA notebook includes:
   - Data overview and structure analysis
   - Data quality assessment
   - Exploratory analysis with visualizations
   - Data cleaning operations
   - Feature selection recommendations for ML models
4. Execute training pipelines in `src/models/` for model development

Note: This project uses uv for dependency management and virtual environments, ensuring reproducible environments and faster package installations.

## Dataset

The project uses the [DataCo Supply Chain Dataset](https://www.kaggle.com/datasets/saicharankomati/dataco-supply-chain-dataset) from Kaggle. The dataset includes:
- Order details (dates, items, quantities, prices)
- Customer information (location, segment, demographics)
- Product information (categories, descriptions, prices)
- Shipping details (delivery status, dates, modes, regions)
- Financial metrics (sales, benefits, profit ratios)

The data is automatically downloaded on first use via the `data_manager.py` module.

## Portfolio Value

This project showcases advanced data science capabilities in solving real-world business problems, combining technical expertise in machine learning, deep learning, and data analysis with practical business acumen in supply chain optimization. It demonstrates proficiency in end-to-end ML project development, from data processing to model deployment, making it a valuable addition to any data science portfolio.
