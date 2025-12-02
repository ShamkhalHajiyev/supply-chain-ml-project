"""
LSTM Model Training Module
Trains deep learning models for demand forecasting (time series).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import joblib
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    """
    LSTM model for demand forecasting.

    Architecture:
    - Input layer
    - 2 LSTM layers with dropout
    - Fully connected output layer
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get output from last time step
        out = self.fc(out[:, -1, :])

        return out


class DemandForecaster:
    """
    Train and evaluate LSTM model for demand forecasting.

    Pipeline:
    1. Data preparation (create sequences)
    2. Normalization
    3. Train/test split
    4. Model training
    5. Evaluation
    """

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 50,
        random_state: int = 42
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_state = random_state

        # Model components
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model directory
        self.model_dir = Path(__file__).parents[2] / "models"
        self.model_dir.mkdir(exist_ok=True)

        # Training history
        self.train_losses = []
        self.val_losses = []

        print(f" Initialized DemandForecaster")
        print(f"  Device: {self.device}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Hidden size: {hidden_size}")

    def create_sequences(self, data: np.ndarray, target: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            data: Time series data (n_samples, n_features)
            target: Target values (optional)

        Returns:
            X: Sequences (n_sequences, sequence_length, n_features)
            y: Targets (n_sequences,)
        """
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])

            if target is not None:
                y.append(target[i + self.sequence_length])
            else:
                # Use next value in data as target
                y.append(data[i + self.sequence_length, 0])

        return np.array(X), np.array(y)

    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare time series data for LSTM training.

        Args:
            df: DataFrame with time series data

        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        print("\n" + "=" * 60)
        print("PREPARING TIME SERIES DATA")
        print("=" * 60)

        # Aggregate by date to get daily demand
        if 'order_date_(dateorders)' in df.columns:
            daily_demand = df.groupby('order_date_(dateorders)').agg({
                'order_item_quantity': 'sum',
                'sales': 'sum'
            }).reset_index()

            # Sort by date
            daily_demand = daily_demand.sort_values('order_date_(dateorders)')

            # Use demand as target
            target_col = 'order_item_quantity'
            data = daily_demand[[target_col, 'sales']].values

        else:
            raise ValueError("No date column found for time series")

        print(f" Daily aggregation: {data.shape[0]} days")

        # Normalize data
        data_scaled = self.scaler.fit_transform(data)

        # Create sequences
        X, y = self.create_sequences(data_scaled, target=data_scaled[:, 0])
        print(f" Created sequences: {X.shape[0]} sequences")

        # Train/test split (temporal split - no shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f" Train sequences: {X_train.shape[0]}")
        print(f" Test sequences:  {X_test.shape[0]}")
        print("=" * 60)

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Train LSTM model.

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
        """
        print("\n" + "=" * 60)
        print("TRAINING LSTM MODEL")
        print("=" * 60)

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        input_size = X_train.shape[2]  # Number of features
        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f" Model initialized")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device).unsqueeze(1)

                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        print("\n Training complete")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print("=" * 60)

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        self.model.eval()

        # Predictions
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_test_tensor).cpu().numpy().flatten()

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100

        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

        print(f"\nTest Set Performance:")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAE:   {mae:.6f}")
        print(f"  R²:    {r2:.4f}")
        print(f"  MAPE:  {mape:.2f}%")
        print("=" * 60)

        return results

    def save_model(self):
        """Save model and scaler."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Save PyTorch model
        model_path = self.model_dir / f"lstm_forecaster_{timestamp}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, model_path)
        print(f" Saved LSTM model ’ {model_path}")

        # Save scaler
        scaler_path = self.model_dir / f"lstm_scaler_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f" Saved scaler ’ {scaler_path}")


def run_lstm_training_pipeline():
    """
    Complete LSTM training pipeline for demand forecasting.

    Steps:
    1. Load preprocessed data
    2. Prepare time series data
    3. Train LSTM model
    4. Evaluate model
    5. Save model
    """
    print("\n" + "=" * 80)
    print("LSTM DEMAND FORECASTING PIPELINE")
    print("=" * 80)

    # Import required modules
    from src.data.preprocess import load_and_preprocess

    # Step 1: Load preprocessed data
    print("\n[1/5] Loading data...")
    df = load_and_preprocess()

    # Step 2: Initialize forecaster
    print("\n[2/5] Initializing LSTM forecaster...")
    forecaster = DemandForecaster(
        sequence_length=30,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100,
        random_state=42
    )

    # Step 3: Prepare data
    print("\n[3/5] Preparing time series data...")
    X_train, X_test, y_train, y_test = forecaster.prepare_data(df)

    # Step 4: Train model
    print("\n[4/5] Training LSTM model...")
    forecaster.train_model(X_train, y_train, X_test, y_test)

    # Step 5: Evaluate model
    print("\n[5/5] Evaluating model...")
    results = forecaster.evaluate_model(X_test, y_test)

    # Save model
    print("\n=æ Saving model...")
    forecaster.save_model()

    print("\n" + "=" * 80)
    print(" LSTM TRAINING PIPELINE COMPLETE!")
    print("=" * 80)

    return forecaster, results


if __name__ == "__main__":
    forecaster, results = run_lstm_training_pipeline()
