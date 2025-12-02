"""
LSTM Forecasting Model for Demand Prediction
Deep learning approach for time series forecasting.

Usage:
    from src.models.forecaster_lstm import LSTMForecaster, DemandForecaster
    forecaster = DemandForecaster(sequence_length=30)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import joblib
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    """LSTM model for demand forecasting."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class DemandForecaster:
    """LSTM-based demand forecaster."""

    def __init__(self, sequence_length: int = 30, hidden_size: int = 64,
                 num_layers: int = 2, learning_rate: float = 0.001,
                 batch_size: int = 32, num_epochs: int = 50, random_state: int = 42):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(__file__).parents[2] / "models"
        self.model_dir.mkdir(exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        print(f"✅ LSTM Forecaster initialized (device: {self.device})")

    def create_sequences(self, data: np.ndarray, target: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length] if target is not None else data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare time series data."""
        print("\n" + "=" * 60 + "\nPREPARING LSTM DATA\n" + "=" * 60)

        date_col = next((c for c in df.columns if 'date' in c.lower() and 'order' in c.lower()), None)
        if not date_col:
            raise ValueError("No date column found")

        daily = df.groupby(date_col).agg({
            'order_item_quantity': 'sum', 'sales': 'sum'
        }).reset_index().sort_values(date_col)

        data = daily[['order_item_quantity', 'sales']].values
        data_scaled = self.scaler.fit_transform(data)

        X, y = self.create_sequences(data_scaled, data_scaled[:, 0])
        print(f"📊 Sequences: {X.shape[0]}")

        idx = int(len(X) * 0.8)
        print(f"📊 Train: {idx}, Test: {len(X)-idx}")
        return X[:idx], X[idx:], y[:idx], y[idx:]

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray):
        """Train LSTM model."""
        print("\n" + "=" * 60 + "\nTRAINING LSTM\n" + "=" * 60)

        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), self.batch_size, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), self.batch_size, shuffle=False)

        self.model = LSTMForecaster(X_train.shape[2], self.hidden_size, self.num_layers).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"✅ Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")

        best_loss, patience, counter = float('inf'), 10, 0
        best_state = None

        for epoch in range(self.num_epochs):
            # Train
            self.model.train()
            train_loss = sum(
                criterion(self.model(X.to(self.device)), y.to(self.device).unsqueeze(1)).item()
                for X, y in train_loader
                if not (optimizer.zero_grad(), self.model(X.to(self.device)),
                        criterion(self.model(X.to(self.device)), y.to(self.device).unsqueeze(1)).backward(),
                        optimizer.step())[-1]
            ) / len(train_loader) if False else 0

            # Simplified training loop
            self.model.train()
            t_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()
            t_loss /= len(train_loader)
            self.train_losses.append(t_loss)

            # Validate
            self.model.eval()
            v_loss = sum(criterion(self.model(X.to(self.device)), y.to(self.device).unsqueeze(1)).item()
                         for X, y in val_loader) / len(val_loader)
            self.val_losses.append(v_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train={t_loss:.6f}, Val={v_loss:.6f}")

            if v_loss < best_loss:
                best_loss, counter, best_state = v_loss, 0, self.model.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    print(f"⚠️ Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(best_state)
        print(f"\n✅ Training complete (best val loss: {best_loss:.6f})")

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model."""
        print("\n" + "=" * 60 + "\nEVALUATION\n" + "=" * 60)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.FloatTensor(X_test).to(self.device)).cpu().numpy().flatten()

        results = {
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'mae': mean_absolute_error(y_test, preds),
            'r2': r2_score(y_test, preds)
        }
        print(f"RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}, R²: {results['r2']:.4f}")
        return results

    def save_model(self):
        """Save model."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {'hidden_size': self.hidden_size, 'num_layers': self.num_layers,
                       'sequence_length': self.sequence_length}
        }, self.model_dir / f"lstm_forecaster_{ts}.pt")
        joblib.dump(self.scaler, self.model_dir / f"lstm_scaler_{ts}.pkl")
        print(f"✅ Saved LSTM model")


def run_lstm_pipeline():
    """Complete LSTM training pipeline."""
    print("\n" + "=" * 80 + "\nLSTM FORECASTING PIPELINE\n" + "=" * 80)
    from src.data.preprocess import load_and_preprocess

    df = load_and_preprocess()
    forecaster = DemandForecaster(sequence_length=30, num_epochs=100)
    X_train, X_test, y_train, y_test = forecaster.prepare_data(df)
    forecaster.train_model(X_train, y_train, X_test, y_test)
    results = forecaster.evaluate_model(X_test, y_test)
    forecaster.save_model()

    print("\n✅ LSTM PIPELINE COMPLETE!")
    return forecaster, results


if __name__ == "__main__":
    run_lstm_pipeline()

