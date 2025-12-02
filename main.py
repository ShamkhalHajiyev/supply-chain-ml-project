#!/usr/bin/env python3
"""
Supply Chain ML Project - Main Entry Point

This script provides a command-line interface to run the complete pipeline
or individual steps of the supply chain machine learning project.

Usage:
    python main.py --all                    # Run complete pipeline
    python main.py --data                   # Download/load raw data only
    python main.py --preprocess             # Run data preprocessing
    python main.py --features               # Build features
    python main.py --train-ml               # Train ML classification models
    python main.py --train-lstm             # Train LSTM forecasting model
    python main.py --train-all              # Train all models (ML + LSTM)
    python main.py --evaluate               # Evaluate trained models
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_step(step_num: int, total: int, description: str):
    """Print step progress."""
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 60)


def run_data_loading():
    """Step 1: Download and load raw data."""
    print_header("STEP 1: DATA LOADING")

    from src.data.data_manager import load_raw, RAW_FILE

    print(f"Loading raw data from Kaggle dataset...")
    print(f"Target file: {RAW_FILE}")

    df = load_raw()

    print(f"\n✅ Data loaded successfully!")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def run_preprocessing(df=None):
    """Step 2: Run data preprocessing."""
    print_header("STEP 2: DATA PREPROCESSING")

    from src.data.preprocess import load_and_preprocess, DataPreprocessor
    from src.data.data_manager import load_raw

    if df is None:
        print("Loading raw data first...")
        df = load_raw()

    print(f"Input shape: {df.shape}")

    # Run preprocessing
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.fit_transform(df)

    # Save interim data
    from src.data.data_manager import save_interim
    save_interim(df_clean, "preprocessed_data")

    print(f"\n✅ Preprocessing complete!")
    print(f"   Output shape: {df_clean.shape}")

    return df_clean


def run_feature_engineering(df=None):
    """Step 3: Build features for ML models."""
    print_header("STEP 3: FEATURE ENGINEERING")

    from src.features.build_features import FeatureEngineer, build_features_pipeline
    from src.data.data_manager import load_latest_interim

    if df is None:
        print("Loading preprocessed data...")
        try:
            df = load_latest_interim()
        except FileNotFoundError:
            print("No preprocessed data found. Running preprocessing first...")
            df = run_preprocessing()

    print(f"Input shape: {df.shape}")

    # Build features
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(df)

    print(f"\n✅ Feature engineering complete!")
    print(f"   Feature matrix: {X.shape}")
    print(f"   Target variable: {y.shape if y is not None else 'N/A'}")

    return X, y, df


def run_ml_training(X=None, y=None):
    """Step 4: Train ML classification models."""
    print_header("STEP 4: ML MODEL TRAINING")

    from src.models.train_ml import SupplyChainClassifier
    from src.features.build_features import build_features_pipeline
    from src.data.data_manager import load_latest_interim

    if X is None or y is None:
        print("Preparing features...")
        try:
            df = load_latest_interim()
        except FileNotFoundError:
            print("No preprocessed data found. Running preprocessing first...")
            df = run_preprocessing()
        X, y = build_features_pipeline(df)

    print(f"Feature matrix: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Initialize and train classifier
    classifier = SupplyChainClassifier(random_state=42)
    classifier.initialize_models()

    # Split data
    X_train, X_test, y_train, y_test = classifier.split_data(X, y, test_size=0.2)

    # Train all models
    classifier.train_all_models(X_train, y_train, X_test, y_test)

    # Get feature importance
    classifier.get_feature_importance(X.columns.tolist(), top_n=15)

    # Save models
    classifier.save_models()

    # Generate report
    classifier.generate_report()

    print(f"\n✅ ML training complete!")
    print(f"   Best model: {classifier.best_model_name}")

    return classifier


def run_lstm_training(df=None):
    """Step 5: Train LSTM forecasting model."""
    print_header("STEP 5: LSTM MODEL TRAINING")

    from src.models.train_lstm import DemandForecaster
    from src.data.data_manager import load_latest_interim

    if df is None:
        print("Loading preprocessed data...")
        try:
            df = load_latest_interim()
        except FileNotFoundError:
            print("No preprocessed data found. Running preprocessing first...")
            df = run_preprocessing()

    print(f"Input data shape: {df.shape}")

    # Initialize forecaster
    forecaster = DemandForecaster(
        sequence_length=30,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100,
        random_state=42
    )

    # Prepare data
    X_train, X_test, y_train, y_test = forecaster.prepare_data(df)

    # Train model
    forecaster.train_model(X_train, y_train, X_test, y_test)

    # Evaluate
    results = forecaster.evaluate_model(X_test, y_test)

    # Save model
    forecaster.save_model()

    print(f"\n✅ LSTM training complete!")
    print(f"   RMSE: {results['rmse']:.6f}")
    print(f"   R²: {results['r2']:.4f}")

    return forecaster, results


def run_evaluation():
    """Step 6: Evaluate trained models."""
    print_header("STEP 6: MODEL EVALUATION")

    import joblib
    from pathlib import Path
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from src.features.build_features import build_features_pipeline
    from src.data.data_manager import load_latest_interim
    from sklearn.model_selection import train_test_split

    model_dir = Path(__file__).parent / "models"

    # Load best ML model
    best_model_files = list(model_dir.glob('best_model*.pkl'))
    if not best_model_files:
        print("⚠️  No trained models found. Please run training first.")
        return

    best_model_path = sorted(best_model_files)[-1]
    print(f"Loading model: {best_model_path.name}")
    best_model = joblib.load(best_model_path)

    # Load and prepare data
    try:
        df = load_latest_interim()
    except FileNotFoundError:
        print("⚠️  No preprocessed data found. Please run preprocessing first.")
        return

    X, y = build_features_pipeline(df)

    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluate
    y_pred = best_model.predict(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['On-time', 'Late']))

    print(f"\n✅ Evaluation complete!")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")


def run_full_pipeline():
    """Run the complete pipeline from data loading to model training."""
    print_header("SUPPLY CHAIN ML - FULL PIPELINE")

    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_steps = 5

    # Step 1: Data Loading
    print_step(1, total_steps, "Data Loading")
    df = run_data_loading()

    # Step 2: Preprocessing
    print_step(2, total_steps, "Data Preprocessing")
    df_clean = run_preprocessing(df)

    # Step 3: Feature Engineering
    print_step(3, total_steps, "Feature Engineering")
    X, y, df_features = run_feature_engineering(df_clean)

    # Step 4: ML Training
    print_step(4, total_steps, "ML Model Training")
    classifier = run_ml_training(X, y)

    # Step 5: LSTM Training
    print_step(5, total_steps, "LSTM Model Training")
    forecaster, lstm_results = run_lstm_training(df_clean)

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print_header("PIPELINE COMPLETE")
    print(f"✅ All steps completed successfully!")
    print(f"   Duration: {duration}")
    print(f"   Best ML Model: {classifier.best_model_name}")
    print(f"   LSTM RMSE: {lstm_results['rmse']:.6f}")

    return {
        'classifier': classifier,
        'forecaster': forecaster,
        'lstm_results': lstm_results
    }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Supply Chain ML Project - Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --all              Run complete pipeline
    python main.py --data             Download/load raw data
    python main.py --preprocess       Preprocess data
    python main.py --features         Build features
    python main.py --train-ml         Train ML models
    python main.py --train-lstm       Train LSTM model
    python main.py --train-all        Train all models
    python main.py --evaluate         Evaluate models
        """
    )

    # Add arguments
    parser.add_argument('--all', action='store_true',
                        help='Run complete pipeline (data → preprocessing → features → training)')
    parser.add_argument('--data', action='store_true',
                        help='Download and load raw data')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing')
    parser.add_argument('--features', action='store_true',
                        help='Build features from preprocessed data')
    parser.add_argument('--train-ml', action='store_true',
                        help='Train ML classification models')
    parser.add_argument('--train-lstm', action='store_true',
                        help='Train LSTM forecasting model')
    parser.add_argument('--train-all', action='store_true',
                        help='Train all models (ML + LSTM)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained models')

    args = parser.parse_args()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # Execute based on arguments
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
            if args.train_ml:
                run_ml_training()
            if args.train_lstm:
                run_lstm_training()
            if args.train_all:
                run_ml_training()
                run_lstm_training()
            if args.evaluate:
                run_evaluation()

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

