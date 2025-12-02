#!/usr/bin/env python3
"""
Supply Chain ML Project - Main Entry Point

Run the complete pipeline or individual steps for supply chain ML analysis.

Usage:
    python main.py --all                    # Run complete pipeline
    python main.py --data                   # Download/load raw data
    python main.py --preprocess             # Run data preprocessing
    python main.py --features               # Build features
    python main.py --train-classification   # Train classification models (with ensembles)
    python main.py --train-forecasting      # Train ML forecasting models
    python main.py --train-lstm             # Train LSTM model (optional)
    python main.py --evaluate               # Evaluate models
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_data_loading():
    """Step 1: Download and load raw data."""
    print_header("STEP 1: DATA LOADING")

    from src.data.data_manager import load_raw, RAW_FILE

    df = load_raw()
    print(f"\n✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def run_preprocessing(df=None):
    """Step 2: Run data preprocessing."""
    print_header("STEP 2: DATA PREPROCESSING")

    from src.data.preprocess import load_and_preprocess

    df_clean = load_and_preprocess()
    print(f"\n✅ Preprocessing complete: {df_clean.shape}")
    return df_clean


def run_feature_engineering(df=None):
    """Step 3: Build features for ML models."""
    print_header("STEP 3: FEATURE ENGINEERING")

    from src.features.build_features import build_features_pipeline
    from src.data.preprocess import load_and_preprocess

    if df is None:
        df = load_and_preprocess()

    X, y = build_features_pipeline(df)
    print(f"\n✅ Features built: {X.shape[1]} features for {X.shape[0]:,} samples")
    return X, y, df


def run_classification_training(X=None, y=None):
    """Step 4: Train classification models with ensemble."""
    print_header("STEP 4: CLASSIFICATION MODEL TRAINING")

    from src.models.classifier import SupplyChainClassifier
    from src.features.build_features import build_features_pipeline
    from src.data.preprocess import load_and_preprocess

    if X is None or y is None:
        df = load_and_preprocess()
        X, y = build_features_pipeline(df)

    classifier = SupplyChainClassifier(random_state=42)
    classifier.initialize_models(include_ensemble=True)

    X_train, X_test, y_train, y_test = classifier.split_data(X, y, test_size=0.2)
    classifier.train_all_models(X_train, y_train, X_test, y_test)
    classifier.get_feature_importance(X.columns.tolist(), top_n=15)
    classifier.save_models()
    classifier.generate_report()

    print(f"\n✅ Classification training complete!")
    print(f"   Best model: {classifier.best_model_name}")
    return classifier


def run_forecasting_training(df=None):
    """Step 5: Train ML forecasting models."""
    print_header("STEP 5: FORECASTING MODEL TRAINING (ML)")

    from src.models.forecaster import DemandForecaster
    from src.data.preprocess import load_and_preprocess

    if df is None:
        df = load_and_preprocess()

    forecaster = DemandForecaster(random_state=42)
    forecaster.initialize_models()

    X, y, dates = forecaster.prepare_time_series_features(df)
    X_train, X_test, y_train, y_test = forecaster.split_time_series(X, y)
    forecaster.train_all_models(X_train, y_train, X_test, y_test)
    forecaster.get_feature_importance(X.columns.tolist(), top_n=15)
    forecaster.save_models()

    print(f"\n✅ Forecasting training complete!")
    print(f"   Best model: {forecaster.best_model_name}")
    return forecaster


def run_lstm_training(df=None):
    """Step 6: Train LSTM model (optional)."""
    print_header("STEP 6: LSTM MODEL TRAINING")

    from src.models.forecaster_lstm import DemandForecaster as LSTMForecaster
    from src.data.preprocess import load_and_preprocess

    if df is None:
        df = load_and_preprocess()

    forecaster = LSTMForecaster(
        sequence_length=30,
        hidden_size=64,
        num_layers=2,
        num_epochs=100,
        random_state=42
    )

    X_train, X_test, y_train, y_test = forecaster.prepare_data(df)
    forecaster.train_model(X_train, y_train, X_test, y_test)
    results = forecaster.evaluate_model(X_test, y_test)
    forecaster.save_model()

    print(f"\n✅ LSTM training complete!")
    print(f"   RMSE: {results['rmse']:.4f}, R²: {results['r2']:.4f}")
    return forecaster, results


def run_evaluation():
    """Step 7: Evaluate trained models."""
    print_header("STEP 7: MODEL EVALUATION")

    import joblib
    from sklearn.metrics import classification_report, accuracy_score
    from src.features.build_features import build_features_pipeline
    from src.data.preprocess import load_and_preprocess
    from sklearn.model_selection import train_test_split

    model_dir = Path(__file__).parent / "models"
    best_model_files = sorted(model_dir.glob('best_model*.pkl'))

    if not best_model_files:
        print("⚠️ No trained models found. Run training first.")
        return

    best_model = joblib.load(best_model_files[-1])
    print(f"Loaded: {best_model_files[-1].name}")

    df = load_and_preprocess()
    X, y = build_features_pipeline(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = best_model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['On-time', 'Late']))
    print(f"\n✅ Evaluation complete! Accuracy: {accuracy_score(y_test, y_pred):.4f}")


def run_full_pipeline():
    """Run the complete pipeline."""
    print_header("SUPPLY CHAIN ML - FULL PIPELINE")

    start_time = datetime.now()

    # Step 1-3: Data
    df = run_data_loading()
    df_clean = run_preprocessing(df)
    X, y, _ = run_feature_engineering(df_clean)

    # Step 4-5: Training
    classifier = run_classification_training(X, y)
    forecaster = run_forecasting_training(df_clean)

    # Summary
    duration = datetime.now() - start_time
    print_header("PIPELINE COMPLETE")
    print(f"✅ Duration: {duration}")
    print(f"   Best Classification: {classifier.best_model_name}")
    print(f"   Best Forecasting: {forecaster.best_model_name}")

    return {'classifier': classifier, 'forecaster': forecaster}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Supply Chain ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --all                    # Full pipeline
    python main.py --train-classification   # Classification only
    python main.py --train-forecasting      # Forecasting only
    python main.py --train-lstm             # LSTM only
        """
    )

    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--data', action='store_true', help='Load raw data')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess data')
    parser.add_argument('--features', action='store_true', help='Build features')
    parser.add_argument('--train-classification', action='store_true', help='Train classification models')
    parser.add_argument('--train-forecasting', action='store_true', help='Train ML forecasting models')
    parser.add_argument('--train-lstm', action='store_true', help='Train LSTM model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    try:
        if args.all:
            run_full_pipeline()
        else:
            if args.data:
                run_data_loading()
            if args.preprocess:
                run_preprocessing()
            if args.features:
                run_feature_engineering()
            if args.train_classification:
                run_classification_training()
            if args.train_forecasting:
                run_forecasting_training()
            if args.train_lstm:
                run_lstm_training()
            if args.evaluate:
                run_evaluation()

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
