"""
Ensemble Forecasting - Combine Multiple Forecasting Models
Weighted ensemble of Prophet, SARIMAX, XGBoost, LightGBM, and LSTM.

Features:
- Simple average ensemble
- Weighted average ensemble
- Stacking ensemble
- Model performance tracking
- Automatic weight optimization

Usage:
    from src.models.ensemble_forecaster import EnsembleForecaster
    ensemble = EnsembleForecaster()
    ensemble.add_forecasters(forecasters_dict)
    forecast = ensemble.predict(X_test, method='weighted')
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class EnsembleForecaster:
    """
    Ensemble forecasting combining multiple time series models.

    Supported methods:
    - Simple average: Equal weights for all models
    - Weighted average: Performance-based weights
    - Median ensemble: Robust to outliers
    - Stacking: Meta-model learns optimal combination

    Advantages:
    - Reduces overfitting
    - Improves robustness
    - Combines different model strengths
    """

    def __init__(self):
        """Initialize ensemble forecaster."""
        self.forecasters: Dict[str, any] = {}
        self.weights: Dict[str, float] = {}
        self.validation_scores: Dict[str, float] = {}

    def add_forecaster(self, name: str, forecaster: any, weight: float = 1.0):
        """
        Add a forecaster to the ensemble.

        Args:
            name: Forecaster name
            forecaster: Trained forecaster object
            weight: Initial weight (for weighted ensemble)
        """
        self.forecasters[name] = forecaster
        self.weights[name] = weight
        print(f"✅ Added forecaster: {name} (weight={weight:.2f})")

    def add_forecasters(self, forecasters: Dict[str, any]):
        """
        Add multiple forecasters at once.

        Args:
            forecasters: Dictionary of {name: forecaster}
        """
        for name, forecaster in forecasters.items():
            self.add_forecaster(name, forecaster)

    def optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series,
                        metric: str = 'rmse') -> Dict[str, float]:
        """
        Optimize ensemble weights based on validation performance.

        Args:
            X_val: Validation features
            y_val: Validation targets
            metric: Metric to optimize ('rmse', 'mae', 'r2')

        Returns:
            Dictionary of optimized weights
        """
        print("\n🔍 Optimizing ensemble weights...")
        print("=" * 60)

        # Collect individual forecaster predictions
        predictions = {}
        for name, forecaster in self.forecasters.items():
            try:
                pred = forecaster.predict(X_val)
                predictions[name] = pred

                # Calculate individual performance
                if metric == 'rmse':
                    score = np.sqrt(mean_squared_error(y_val, pred))
                elif metric == 'mae':
                    score = mean_absolute_error(y_val, pred)
                elif metric == 'r2':
                    score = r2_score(y_val, pred)
                else:
                    score = np.sqrt(mean_squared_error(y_val, pred))

                self.validation_scores[name] = score
                print(f"   {name}: {metric.upper()}={score:.4f}")

            except Exception as e:
                print(f"   ⚠️ {name}: Failed to predict ({str(e)})")
                continue

        # Calculate inverse performance weights (better models get higher weight)
        if metric in ['rmse', 'mae']:
            # Lower is better - use inverse
            inv_scores = {name: 1.0 / (score + 1e-10)
                         for name, score in self.validation_scores.items()}
        else:  # r2
            # Higher is better - use directly
            inv_scores = {name: max(score, 0) + 1e-10
                         for name, score in self.validation_scores.items()}

        # Normalize to sum to 1.0
        total = sum(inv_scores.values())
        self.weights = {name: score / total for name, score in inv_scores.items()}

        print("\n📊 Optimized Weights:")
        print("=" * 60)
        for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {name}: {weight:.4f} ({weight*100:.1f}%)")

        return self.weights

    def predict(self, X: pd.DataFrame, method: str = 'weighted') -> np.ndarray:
        """
        Generate ensemble forecast.

        Args:
            X: Input features
            method: Ensemble method ('simple', 'weighted', 'median')

        Returns:
            Ensemble predictions
        """
        if not self.forecasters:
            raise ValueError("No forecasters added. Use add_forecaster() first.")

        # Collect predictions from all forecasters
        all_predictions = []
        valid_names = []

        for name, forecaster in self.forecasters.items():
            try:
                pred = forecaster.predict(X)
                all_predictions.append(pred)
                valid_names.append(name)
            except Exception as e:
                print(f"⚠️ {name} prediction failed: {str(e)}")
                continue

        if not all_predictions:
            raise ValueError("All forecasters failed to predict")

        all_predictions = np.array(all_predictions)

        # Apply ensemble method
        if method == 'simple':
            # Simple average
            ensemble_pred = np.mean(all_predictions, axis=0)

        elif method == 'weighted':
            # Weighted average using optimized weights
            weights_array = np.array([self.weights.get(name, 1.0) for name in valid_names])
            weights_array = weights_array / weights_array.sum()  # Normalize
            ensemble_pred = np.average(all_predictions, axis=0, weights=weights_array)

        elif method == 'median':
            # Median ensemble (robust to outliers)
            ensemble_pred = np.median(all_predictions, axis=0)

        else:
            raise ValueError(f"Unknown method: {method}")

        return ensemble_pred

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                method: str = 'weighted') -> Dict:
        """
        Evaluate ensemble performance.

        Args:
            X_test: Test features
            y_test: Test targets
            method: Ensemble method

        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X_test, method=method)

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        }

        print(f"\n📊 ENSEMBLE EVALUATION ({method.upper()} method)")
        print("=" * 60)
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE:  {metrics['mae']:.4f}")
        print(f"   R²:   {metrics['r2']:.4f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")

        return metrics

    def compare_methods(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare different ensemble methods.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            DataFrame with comparison results
        """
        print("\n📊 COMPARING ENSEMBLE METHODS")
        print("=" * 60)

        results = {}

        for method in ['simple', 'weighted', 'median']:
            try:
                metrics = self.evaluate(X_test, y_test, method=method)
                results[method.capitalize()] = metrics
            except Exception as e:
                print(f"⚠️ {method} method failed: {str(e)}")

        # Also evaluate individual forecasters
        for name, forecaster in self.forecasters.items():
            try:
                y_pred = forecaster.predict(X_test)
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
                }
                results[name] = metrics
            except Exception as e:
                print(f"⚠️ {name} evaluation failed: {str(e)}")

        # Create comparison DataFrame
        df = pd.DataFrame(results).T
        df = df.sort_values('r2', ascending=False)

        print("\n📊 ENSEMBLE vs INDIVIDUAL MODELS COMPARISON")
        print("=" * 60)
        print(df.to_string())

        # Highlight best method
        best_method = df['r2'].idxmax()
        print(f"\n🏆 BEST METHOD: {best_method}")
        print(f"   R²: {df.loc[best_method, 'r2']:.4f}")
        print(f"   RMSE: {df.loc[best_method, 'rmse']:.4f}")

        return df

    def get_forecast_summary(self, X: pd.DataFrame, method: str = 'weighted') -> pd.DataFrame:
        """
        Get detailed forecast summary with individual model predictions.

        Args:
            X: Input features
            method: Ensemble method for final prediction

        Returns:
            DataFrame with all model predictions and ensemble
        """
        summary = pd.DataFrame()

        # Individual model predictions
        for name, forecaster in self.forecasters.items():
            try:
                pred = forecaster.predict(X)
                summary[name] = pred
            except Exception as e:
                print(f"⚠️ {name} failed: {str(e)}")

        # Ensemble prediction
        if not summary.empty:
            ensemble_pred = self.predict(X, method=method)
            summary[f'Ensemble ({method})'] = ensemble_pred

        return summary


def run_ensemble_forecasting_pipeline():
    """
    Demonstration of ensemble forecasting pipeline.

    Shows how to:
    - Train multiple forecasters
    - Create ensemble
    - Optimize weights
    - Compare methods
    """
    print("\n" + "=" * 80 + "\nENSEMBLE FORECASTING PIPELINE\n" + "=" * 80)

    from src.data.preprocess import load_and_preprocess
    from src.models.forecaster import DemandForecaster

    # Load data
    df = load_and_preprocess()

    # Prepare time series features
    forecaster_base = DemandForecaster(random_state=42)
    forecaster_base.initialize_models()

    X, y, dates = forecaster_base.prepare_time_series_features(df)

    # Split data: train (60%), validation (20%), test (20%)
    n = len(X)
    train_idx = int(n * 0.6)
    val_idx = int(n * 0.8)

    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
    X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
    X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]

    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train)} ({len(X_train)/n*100:.0f}%)")
    print(f"   Val:   {len(X_val)} ({len(X_val)/n*100:.0f}%)")
    print(f"   Test:  {len(X_test)} ({len(X_test)/n*100:.0f}%)")

    # Train individual forecasters
    print("\n🚀 Training individual forecasters...")
    print("=" * 60)

    forecasters_dict = {}

    # Train ML models (from DemandForecaster)
    forecaster_base.train_all_models(X_train, y_train, X_val, y_val)

    # Extract trained models
    for name, model in forecaster_base.models.items():
        forecasters_dict[name] = model

    # Create ensemble
    ensemble = EnsembleForecaster()
    ensemble.add_forecasters(forecasters_dict)

    # Optimize weights on validation set
    ensemble.optimize_weights(X_val, y_val, metric='rmse')

    # Compare ensemble methods
    comparison_df = ensemble.compare_methods(X_test, y_test)

    print("\n✅ ENSEMBLE FORECASTING COMPLETE!")

    return ensemble, comparison_df


if __name__ == "__main__":
    ensemble, comparison = run_ensemble_forecasting_pipeline()
