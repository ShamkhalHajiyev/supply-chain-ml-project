"""
ML Forecasting Models for Demand Prediction
Trains multiple regression models for time series forecasting.

Usage:
    from src.models.forecaster import DemandForecaster
    forecaster = DemandForecaster()
    forecaster.initialize_models()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import joblib
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    """
    Train and evaluate ML models for demand forecasting.

    Models:
    - Ridge/Lasso/ElasticNet (linear baselines)
    - Random Forest, Gradient Boosting, XGBoost, Extra Trees

    For LSTM, use forecaster_lstm.py
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.model_dir = Path(__file__).parents[2] / "models"
        self.model_dir.mkdir(exist_ok=True)

    def initialize_models(self):
        """Initialize forecasting models."""
        self.models = {
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=0.1, random_state=self.random_state),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=10,
                random_state=self.random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=self.random_state
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=self.random_state, n_jobs=-1
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=100, max_depth=10,
                random_state=self.random_state, n_jobs=-1
            )
        }
        print(f"✅ Initialized {len(self.models)} forecasting models")
        for name in self.models.keys():
            print(f"   • {name}")

    def prepare_time_series_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features: lag, rolling stats, temporal."""
        print("\n" + "=" * 60 + "\nPREPARING TIME SERIES FEATURES\n" + "=" * 60)

        # Find date column
        date_col = next((c for c in df.columns if 'date' in c.lower() and 'order' in c.lower()), None)
        if date_col is None:
            raise ValueError("No date column found")

        # Daily aggregation
        daily = df.groupby(date_col).agg({
            'order_item_quantity': 'sum', 'sales': 'sum'
        }).reset_index().sort_values(date_col).reset_index(drop=True)
        print(f"📊 Daily data: {len(daily)} days")

        features = daily.copy()
        target = 'order_item_quantity'

        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            features[f'demand_lag_{lag}'] = features[target].shift(lag)
            features[f'sales_lag_{lag}'] = features['sales'].shift(lag)

        # Rolling stats
        for w in [7, 14, 30]:
            features[f'demand_rolling_mean_{w}'] = features[target].rolling(w).mean()
            features[f'demand_rolling_std_{w}'] = features[target].rolling(w).std()
            features[f'demand_rolling_min_{w}'] = features[target].rolling(w).min()
            features[f'demand_rolling_max_{w}'] = features[target].rolling(w).max()

        # Temporal features
        dt = pd.to_datetime(features[date_col])
        features['day_of_week'] = dt.dt.dayofweek
        features['day_of_month'] = dt.dt.day
        features['month'] = dt.dt.month
        features['quarter'] = dt.dt.quarter
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)

        # Clean
        features = features.dropna().reset_index(drop=True)
        feature_cols = [c for c in features.columns if c not in [date_col, target, 'sales']]

        X = features[feature_cols]
        y = features[target]
        print(f"📊 Features: {X.shape}")
        return X, y, features[date_col]

    def split_time_series(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Temporal split (no shuffle)."""
        idx = int(len(X) * (1 - test_size))
        print(f"\n📊 Split: Train={idx}, Test={len(X)-idx}")
        return X.iloc[:idx], X.iloc[idx:], y.iloc[:idx], y.iloc[idx:]

    def train_model(self, name: str, X_train: pd.DataFrame, y_train: pd.Series):
        """Train single model."""
        print(f"\nTraining {name}...")
        self.models[name].fit(X_train, y_train)
        print(f"✅ {name} complete")
        return self.models[name]

    def evaluate_model(self, name: str, model: Any, X_train: pd.DataFrame,
                       y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate with train vs test comparison."""
        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_tr_pred)
        test_r2 = r2_score(y_test, y_te_pred)
        gap = train_r2 - test_r2
        status = "⚠️ OVERFITTING" if gap > 0.1 else ("⚠️ UNDERFITTING" if test_r2 < 0.3 else "✅ GOOD FIT")

        results = {
            'model_name': name,
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_tr_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_te_pred)),
            'train_mae': mean_absolute_error(y_train, y_tr_pred),
            'test_mae': mean_absolute_error(y_test, y_te_pred),
            'train_r2': train_r2, 'test_r2': test_r2,
            'r2_gap': gap, 'fit_status': status, 'predictions': y_te_pred
        }
        self.results[name] = results

        print(f"\n{name}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, {status}")
        return results

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series):
        """Train and evaluate all models."""
        print("\n" + "=" * 60 + "\nTRAINING ALL FORECASTING MODELS\n" + "=" * 60)
        for name in self.models:
            model = self.train_model(name, X_train, y_train)
            self.evaluate_model(name, model, X_train, y_train, X_test, y_test)
        self.select_best_model()

    def select_best_model(self):
        """Select best model by test R²."""
        best = max(self.results.items(), key=lambda x: x[1]['test_r2'])
        self.best_model_name, self.best_model = best[0], self.models[best[0]]
        print(f"\n🏆 BEST MODEL: {best[0]} (R²={best[1]['test_r2']:.4f})")

    def get_feature_importance(self, feature_names: list, top_n: int = 15) -> pd.DataFrame:
        """Get feature importance."""
        if hasattr(self.best_model, 'feature_importances_'):
            imp = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            imp = np.abs(self.best_model.coef_)
        else:
            return None
        df = pd.DataFrame({'feature': feature_names, 'importance': imp})
        return df.sort_values('importance', ascending=False).head(top_n)

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Comparison DataFrame."""
        return pd.DataFrame([{
            'Model': n, 'Train R²': r['train_r2'], 'Test R²': r['test_r2'],
            'Train RMSE': r['train_rmse'], 'Test RMSE': r['test_rmse'],
            'R² Gap': r['r2_gap'], 'Fit Status': r['fit_status']
        } for n, r in self.results.items()]).sort_values('Test R²', ascending=False)

    def save_models(self):
        """Save models."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        for name, model in self.models.items():
            path = self.model_dir / f"forecast_{name.replace(' ', '_').lower()}_{ts}.pkl"
            joblib.dump(model, path)
            print(f"  ✅ Saved {name}")
        joblib.dump(self.best_model, self.model_dir / f"forecast_best_{ts}.pkl")


def run_forecasting_pipeline():
    """Complete forecasting pipeline."""
    print("\n" + "=" * 80 + "\nFORECASTING PIPELINE\n" + "=" * 80)
    from src.data.preprocess import load_and_preprocess

    df = load_and_preprocess()
    forecaster = DemandForecaster()
    forecaster.initialize_models()

    X, y, _ = forecaster.prepare_time_series_features(df)
    X_train, X_test, y_train, y_test = forecaster.split_time_series(X, y)
    forecaster.train_all_models(X_train, y_train, X_test, y_test)
    forecaster.save_models()

    print("\n✅ FORECASTING PIPELINE COMPLETE!")
    return forecaster


if __name__ == "__main__":
    run_forecasting_pipeline()

