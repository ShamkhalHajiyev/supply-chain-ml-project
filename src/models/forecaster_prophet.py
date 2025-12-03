"""
Prophet Forecasting Model for Demand Prediction
Facebook Prophet for time series with seasonality and holidays.

Features:
- Automatic seasonality detection (yearly, weekly, monthly)
- Holiday effects
- Trend changepoints
- Uncertainty intervals
- Visual diagnostics

Usage:
    from src.models.forecaster_prophet import ProphetForecaster
    forecaster = ProphetForecaster()
    forecaster.train(df)
    forecast = forecaster.forecast(periods=30)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import joblib
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not available. Install with: pip install prophet")


class ProphetForecaster:
    """
    Facebook Prophet forecasting model for demand prediction.

    Advantages:
    - Handles missing data and outliers
    - Multiple seasonality (daily, weekly, yearly)
    - Holiday effects
    - Intuitive parameters
    - Uncertainty quantification

    Best for:
    - Strong seasonal patterns
    - Multiple seasonality levels
    - Holiday effects
    - Missing data
    """

    def __init__(self,
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 seasonality_mode: str = 'multiplicative',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 growth: str = 'linear'):
        """
        Initialize Prophet forecaster.

        Args:
            yearly_seasonality: Enable yearly seasonality
            weekly_seasonality: Enable weekly seasonality
            daily_seasonality: Enable daily seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend (0.001-0.5)
            seasonality_prior_scale: Flexibility of seasonality (0.01-10)
            growth: 'linear' or 'logistic'
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed. Install with: pip install prophet")

        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.growth = growth
        self.model = None
        self.model_dir = Path(__file__).parents[2] / "models"
        self.model_dir.mkdir(exist_ok=True)

    def prepare_data(self, df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns).

        Args:
            df: DataFrame with date and target columns
            date_col: Name of date column
            target_col: Name of target column

        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

        print(f"✅ Prepared Prophet data: {len(prophet_df)} time points")
        print(f"   Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
        print(f"   Target stats: mean={prophet_df['y'].mean():.2f}, std={prophet_df['y'].std():.2f}")

        return prophet_df

    def train(self, df: pd.DataFrame, add_custom_seasonality: bool = True):
        """
        Train Prophet model.

        Args:
            df: DataFrame with 'ds' and 'y' columns
            add_custom_seasonality: Add monthly seasonality
        """
        print("\n" + "=" * 60)
        print("TRAINING PROPHET MODEL")
        print("=" * 60)

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            growth=self.growth
        )

        # Add custom seasonalities
        if add_custom_seasonality:
            # Monthly seasonality (30.5 days)
            self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Add country holidays (optional - uncomment if applicable)
        # self.model.add_country_holidays(country_name='US')

        # Fit model
        self.model.fit(df)

        print("✅ Prophet model trained successfully")
        print(f"   Seasonalities: ", end="")
        seasonalities = []
        if self.yearly_seasonality:
            seasonalities.append("yearly")
        if self.weekly_seasonality:
            seasonalities.append("weekly")
        if add_custom_seasonality:
            seasonalities.append("monthly")
        print(", ".join(seasonalities))
        print(f"   Seasonality mode: {self.seasonality_mode}")
        print(f"   Growth: {self.growth}")

    def forecast(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        Generate forecast for future periods.

        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)

        Returns:
            DataFrame with forecast including uncertainty intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)

        print(f"\n✅ Forecast generated for {periods} periods")
        print(f"   Forecast range: {forecast['ds'].iloc[-periods]} to {forecast['ds'].iloc[-1]}")

        return forecast

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate Prophet on test set.

        Args:
            test_df: Test DataFrame with 'ds' and 'y' columns

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(test_df[['ds']])

        y_true = test_df['y'].values
        y_pred = predictions['yhat'].values

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
            'coverage': self._calculate_coverage(test_df, predictions)
        }

        print("\n📊 PROPHET EVALUATION METRICS")
        print("=" * 60)
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE:  {metrics['mae']:.4f}")
        print(f"   R²:   {metrics['r2']:.4f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        print(f"   95% Coverage: {metrics['coverage']:.2f}%")

        return metrics

    def _calculate_coverage(self, test_df: pd.DataFrame, predictions: pd.DataFrame) -> float:
        """
        Calculate percentage of actual values within 95% prediction interval.

        Args:
            test_df: Test data with 'y' column
            predictions: Prophet predictions with yhat_lower, yhat_upper

        Returns:
            Coverage percentage
        """
        y_true = test_df['y'].values
        lower = predictions['yhat_lower'].values
        upper = predictions['yhat_upper'].values

        within_interval = (y_true >= lower) & (y_true <= upper)
        coverage = within_interval.mean() * 100

        return coverage

    def cross_validate(self, df: pd.DataFrame, initial: str = '365 days',
                      period: str = '30 days', horizon: str = '30 days') -> pd.DataFrame:
        """
        Perform time series cross-validation.

        Args:
            df: Training data with 'ds' and 'y' columns
            initial: Size of initial training period
            period: Spacing between cutoff dates
            horizon: Forecast horizon

        Returns:
            DataFrame with cross-validation results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print("\n🔍 Performing time series cross-validation...")
        print(f"   Initial: {initial}, Period: {period}, Horizon: {horizon}")

        df_cv = cross_validation(self.model, initial=initial, period=period, horizon=horizon)
        df_metrics = performance_metrics(df_cv)

        print("\n📊 CROSS-VALIDATION METRICS")
        print("=" * 60)
        print(df_metrics[['horizon', 'rmse', 'mae', 'mape', 'coverage']].head(10).to_string())

        return df_cv, df_metrics

    def plot_forecast(self, forecast: pd.DataFrame, save_path: Optional[str] = None):
        """
        Visualize forecast with components.

        Args:
            forecast: Forecast DataFrame from predict()
            save_path: Optional path to save figure
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        import matplotlib.pyplot as plt

        # Plot forecast
        fig1 = self.model.plot(forecast, figsize=(12, 6))
        fig1.suptitle('Prophet Forecast with Uncertainty Intervals', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig1.savefig(save_path.replace('.png', '_forecast.png'), dpi=300, bbox_inches='tight')
            print(f"✅ Forecast plot saved to {save_path.replace('.png', '_forecast.png')}")

        # Plot components (trend, seasonality)
        fig2 = self.model.plot_components(forecast, figsize=(12, 10))
        fig2.suptitle('Prophet Forecast Components', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig2.savefig(save_path.replace('.png', '_components.png'), dpi=300, bbox_inches='tight')
            print(f"✅ Components plot saved to {save_path.replace('.png', '_components.png')}")

        plt.show()

    def save_model(self, filename: Optional[str] = None):
        """Save trained Prophet model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"prophet_forecaster_{ts}.pkl"

        path = self.model_dir / filename
        joblib.dump(self.model, path)
        print(f"✅ Prophet model saved to {path}")

    def load_model(self, filename: str):
        """Load trained Prophet model."""
        path = self.model_dir / filename
        self.model = joblib.load(path)
        print(f"✅ Prophet model loaded from {path}")


def run_prophet_pipeline():
    """
    Complete Prophet forecasting pipeline.

    Demonstrates:
    - Data preparation
    - Model training
    - Forecasting
    - Evaluation
    - Visualization
    """
    print("\n" + "=" * 80 + "\nPROPHET FORECASTING PIPELINE\n" + "=" * 80)

    if not PROPHET_AVAILABLE:
        print("❌ Prophet not installed. Install with: pip install prophet")
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
    }).reset_index()

    # Initialize forecaster
    forecaster = ProphetForecaster(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )

    # Prepare data
    prophet_df = forecaster.prepare_data(daily, date_col, 'order_item_quantity')

    # Split into train/test (hold out last 30 days)
    train_df = prophet_df.iloc[:-30]
    test_df = prophet_df.iloc[-30:]

    print(f"\n📊 Data split:")
    print(f"   Train: {len(train_df)} days")
    print(f"   Test:  {len(test_df)} days")

    # Train model
    forecaster.train(train_df, add_custom_seasonality=True)

    # Generate forecast
    forecast = forecaster.forecast(periods=30)

    # Evaluate on test set
    metrics = forecaster.evaluate(test_df)

    # Visualize forecast
    forecaster.plot_forecast(forecast, save_path='reports/figures/prophet_forecast.png')

    # Cross-validation (optional, can be slow)
    # df_cv, df_metrics = forecaster.cross_validate(prophet_df, initial='180 days',
    #                                                period='30 days', horizon='30 days')

    # Save model
    forecaster.save_model()

    print("\n✅ PROPHET PIPELINE COMPLETE!")
    print(f"   Best metric - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

    return forecaster, metrics


if __name__ == "__main__":
    forecaster, metrics = run_prophet_pipeline()
