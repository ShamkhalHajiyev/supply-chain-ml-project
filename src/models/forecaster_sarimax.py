"""
SARIMAX Forecasting Model for Demand Prediction
Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors.

Features:
- Automatic order selection (auto-ARIMA)
- External regressors (holidays, promotions, weather)
- Seasonal patterns
- Statistical rigor
- Diagnostic tests

Usage:
    from src.models.forecaster_sarimax import SARIMAXForecaster
    forecaster = SARIMAXForecaster()
    forecaster.train(y_train, exog=exog_train)
    forecast = forecaster.forecast(steps=30, exog_future=exog_test)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import joblib
from typing import Tuple, Dict, Optional
import warnings
import itertools
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels not available. Install with: pip install statsmodels")


class SARIMAXForecaster:
    """
    SARIMAX forecasting model for demand prediction.

    Advantages:
    - Statistical foundation
    - Handles seasonality explicitly
    - External regressors
    - Confidence intervals
    - Diagnostic tests

    Best for:
    - When external variables matter (weather, holidays, promotions)
    - Need statistical inference
    - Medium-sized datasets
    - Clear seasonal patterns
    """

    def __init__(self, seasonal_period: int = 7):
        """
        Initialize SARIMAX forecaster.

        Args:
            seasonal_period: Period of seasonality (7 for weekly, 12 for monthly)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels not installed. Install with: pip install statsmodels")

        self.seasonal_period = seasonal_period
        self.model = None
        self.fitted_model = None
        self.best_order = None
        self.best_seasonal_order = None
        self.model_dir = Path(__file__).parents[2] / "models"
        self.model_dir.mkdir(exist_ok=True)

    def check_stationarity(self, y: pd.Series) -> Dict:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.

        Args:
            y: Time series data

        Returns:
            Dictionary with test results
        """
        result = adfuller(y.dropna())

        is_stationary = result[1] < 0.05  # p-value < 0.05

        print("\n📊 STATIONARITY TEST (Augmented Dickey-Fuller)")
        print("=" * 60)
        print(f"   ADF Statistic: {result[0]:.4f}")
        print(f"   p-value: {result[1]:.4f}")
        print(f"   Critical Values:")
        for key, value in result[4].items():
            print(f"      {key}: {value:.4f}")
        print(f"   Result: {'✅ STATIONARY' if is_stationary else '⚠️ NON-STATIONARY'}")

        if not is_stationary:
            print("   💡 Recommendation: Use differencing (d=1 or d=2)")

        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': is_stationary,
            'critical_values': result[4]
        }

    def decompose_series(self, y: pd.Series, period: Optional[int] = None,
                        save_path: Optional[str] = None):
        """
        Decompose time series into trend, seasonal, and residual components.

        Args:
            y: Time series data
            period: Seasonal period (defaults to self.seasonal_period)
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt

        if period is None:
            period = self.seasonal_period

        decomposition = seasonal_decompose(y.dropna(), model='multiplicative', period=period)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        decomposition.observed.plot(ax=axes[0], title='Original', color='steelblue')
        axes[0].set_ylabel('Observed')

        decomposition.trend.plot(ax=axes[1], title='Trend', color='coral')
        axes[1].set_ylabel('Trend')

        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='green')
        axes[2].set_ylabel('Seasonal')

        decomposition.resid.plot(ax=axes[3], title='Residual', color='purple')
        axes[3].set_ylabel('Residual')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Decomposition plot saved to {save_path}")

        plt.show()

    def auto_arima(self, y: pd.Series, exog: Optional[pd.DataFrame] = None,
                   max_p: int = 3, max_q: int = 3, max_P: int = 2, max_Q: int = 2,
                   max_d: int = 2, max_D: int = 1) -> Tuple[Tuple, Tuple]:
        """
        Automatic ARIMA order selection using AIC.

        Args:
            y: Time series data
            exog: External regressors
            max_p: Maximum AR order
            max_q: Maximum MA order
            max_P: Maximum seasonal AR order
            max_Q: Maximum seasonal MA order
            max_d: Maximum differencing order
            max_D: Maximum seasonal differencing order

        Returns:
            Tuple of (best_order, best_seasonal_order)
        """
        print("\n🔍 AUTO-ARIMA: Searching for best parameters...")
        print("=" * 60)

        best_aic = np.inf
        best_order = None
        best_seasonal_order = None

        # Define parameter ranges
        p = range(0, max_p + 1)
        d = range(0, max_d + 1)
        q = range(0, max_q + 1)
        P = range(0, max_P + 1)
        D = range(0, max_D + 1)
        Q = range(0, max_Q + 1)

        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], self.seasonal_period)
                       for x in itertools.product(P, D, Q)]

        total_combinations = len(pdq) * len(seasonal_pdq)
        print(f"   Testing {total_combinations} parameter combinations...")

        tested = 0
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model = SARIMAX(y, exog=exog,
                                   order=param,
                                   seasonal_order=param_seasonal,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)

                    results = model.fit(disp=False, maxiter=50)

                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = param
                        best_seasonal_order = param_seasonal

                    tested += 1
                    if tested % 50 == 0:
                        print(f"   Progress: {tested}/{total_combinations} combinations tested...")

                except:
                    continue

        print(f"\n✅ AUTO-ARIMA COMPLETE")
        print("=" * 60)
        print(f"   Best Order: SARIMA{best_order}")
        print(f"   Best Seasonal Order: {best_seasonal_order}")
        print(f"   Best AIC: {best_aic:.2f}")
        print(f"   Tested: {tested}/{total_combinations} combinations")

        self.best_order = best_order
        self.best_seasonal_order = best_seasonal_order

        return best_order, best_seasonal_order

    def train(self, y: pd.Series, exog: Optional[pd.DataFrame] = None,
             auto_select: bool = True):
        """
        Train SARIMAX model.

        Args:
            y: Time series target variable
            exog: External regressors (optional)
            auto_select: If True, automatically select best parameters
        """
        print("\n" + "=" * 60)
        print("TRAINING SARIMAX MODEL")
        print("=" * 60)

        # Auto-select parameters if requested
        if auto_select and (self.best_order is None or self.best_seasonal_order is None):
            self.auto_arima(y, exog, max_p=2, max_q=2, max_P=1, max_Q=1)

        # Use default parameters if auto-select failed
        if self.best_order is None:
            self.best_order = (1, 1, 1)
        if self.best_seasonal_order is None:
            self.best_seasonal_order = (1, 1, 1, self.seasonal_period)

        # Create and fit model
        self.model = SARIMAX(y, exog=exog,
                            order=self.best_order,
                            seasonal_order=self.best_seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)

        self.fitted_model = self.model.fit(disp=False)

        print("\n✅ SARIMAX model trained successfully")
        print(f"   Order: SARIMA{self.best_order}")
        print(f"   Seasonal Order: {self.best_seasonal_order}")
        print(f"   AIC: {self.fitted_model.aic:.2f}")
        print(f"   BIC: {self.fitted_model.bic:.2f}")

    def forecast(self, steps: int = 30,
                exog_future: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate forecast for future periods.

        Args:
            steps: Number of periods to forecast
            exog_future: Future values of external regressors

        Returns:
            DataFrame with forecast and confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")

        forecast = self.fitted_model.get_forecast(steps=steps, exog=exog_future)

        forecast_df = pd.DataFrame({
            'forecast': forecast.predicted_mean.values,
            'lower_ci': forecast.conf_int().iloc[:, 0].values,
            'upper_ci': forecast.conf_int().iloc[:, 1].values
        })

        print(f"\n✅ Forecast generated for {steps} periods")
        print(f"   Mean forecast: {forecast_df['forecast'].mean():.2f}")
        print(f"   Forecast range: [{forecast_df['forecast'].min():.2f}, {forecast_df['forecast'].max():.2f}]")

        return forecast_df

    def evaluate(self, y_test: pd.Series,
                exog_test: Optional[pd.DataFrame] = None) -> Dict:
        """
        Evaluate SARIMAX on test set.

        Args:
            y_test: Test target values
            exog_test: Test external regressors

        Returns:
            Dictionary with evaluation metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.fitted_model.forecast(steps=len(y_test), exog=exog_test)

        y_true = y_test.values
        y_pred = predictions.values

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        }

        print("\n📊 SARIMAX EVALUATION METRICS")
        print("=" * 60)
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE:  {metrics['mae']:.4f}")
        print(f"   R²:   {metrics['r2']:.4f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")

        return metrics

    def plot_diagnostics(self, save_path: Optional[str] = None):
        """
        Plot diagnostic plots for model residuals.

        Args:
            save_path: Optional path to save figure
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")

        import matplotlib.pyplot as plt

        self.fitted_model.plot_diagnostics(figsize=(14, 10))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Diagnostics plot saved to {save_path}")

        plt.show()

    def save_model(self, filename: Optional[str] = None):
        """Save trained SARIMAX model."""
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call train() first.")

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"sarimax_forecaster_{ts}.pkl"

        path = self.model_dir / filename
        joblib.dump(self.fitted_model, path)
        print(f"✅ SARIMAX model saved to {path}")

    def load_model(self, filename: str):
        """Load trained SARIMAX model."""
        path = self.model_dir / filename
        self.fitted_model = joblib.load(path)
        print(f"✅ SARIMAX model loaded from {path}")


def run_sarimax_pipeline():
    """
    Complete SARIMAX forecasting pipeline.

    Demonstrates:
    - Stationarity testing
    - Seasonal decomposition
    - Auto-ARIMA
    - Model training
    - Forecasting
    - Evaluation
    - Diagnostics
    """
    print("\n" + "=" * 80 + "\nSARIMAX FORECASTING PIPELINE\n" + "=" * 80)

    if not STATSMODELS_AVAILABLE:
        print("❌ Statsmodels not installed. Install with: pip install statsmodels")
        return None

    from src.data.preprocess import load_and_preprocess

    # Load data
    df = load_and_preprocess()

    # Find date column
    date_col = next((c for c in df.columns if 'date' in c.lower() and 'order' in c.lower()), None)
    if date_col is None:
        raise ValueError("No date column found")

    # Aggregate to daily demand
    daily = df.groupby(date_col).agg({
        'order_item_quantity': 'sum'
    }).reset_index().sort_values(date_col)
    daily.set_index(date_col, inplace=True)

    y = daily['order_item_quantity']

    # Initialize forecaster
    forecaster = SARIMAXForecaster(seasonal_period=7)  # Weekly seasonality

    # Check stationarity
    stationarity_result = forecaster.check_stationarity(y)

    # Decompose series
    forecaster.decompose_series(y, save_path='reports/figures/sarimax_decomposition.png')

    # Split into train/test (hold out last 30 days)
    train_size = len(y) - 30
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    print(f"\n📊 Data split:")
    print(f"   Train: {len(y_train)} days")
    print(f"   Test:  {len(y_test)} days")

    # Train model with auto-selection
    forecaster.train(y_train, exog=None, auto_select=True)

    # Generate forecast
    forecast_df = forecaster.forecast(steps=30)

    # Evaluate on test set
    metrics = forecaster.evaluate(y_test)

    # Plot diagnostics
    forecaster.plot_diagnostics(save_path='reports/figures/sarimax_diagnostics.png')

    # Save model
    forecaster.save_model()

    print("\n✅ SARIMAX PIPELINE COMPLETE!")
    print(f"   Best metric - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

    return forecaster, metrics


if __name__ == "__main__":
    forecaster, metrics = run_sarimax_pipeline()
