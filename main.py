#!/usr/bin/env python3
"""
Supply Chain ML Project - Main Entry Point

Run the complete pipeline or individual steps for late delivery classification.

Usage:
    python main.py --all                    # Run complete pipeline
    python main.py --data                   # Download/load raw data
    python main.py --preprocess             # Run data preprocessing
    python main.py --features               # Build features
    python main.py --train-classification   # Train classification models (with ensembles)
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


def run_classification_training(X=None, y=None, parallel: bool = True,
                                tune_hyperparameters: bool = True,
                                optimize_thresholds: bool = True):
    """Step 4: Train classification models with ensemble, hyperparameter tuning, and threshold optimization."""
    print_header("STEP 4: CLASSIFICATION MODEL TRAINING")

    from src.models.classifier import run_training_pipeline

    if X is None or y is None:
        from src.data.preprocess import load_and_preprocess
        from src.features.build_features import build_features_pipeline
        df = load_and_preprocess()
        X, y = build_features_pipeline(df)

    # Run complete pipeline with tuning and threshold optimization
    # Reduced n_trials from 30 to 10 for faster training
    classifier = run_training_pipeline(
        parallel=parallel,
        tune_hyperparameters=tune_hyperparameters,
        optimize_thresholds=optimize_thresholds,
        n_trials=10
    )

    classifier.get_feature_importance(X.columns.tolist(), top_n=15)

    print(f"\n✅ Classification training complete!")
    print(f"   Best model: {classifier.best_model_name}")
    if optimize_thresholds and classifier.optimal_thresholds:
        print(f"   Optimal threshold: {classifier.optimal_thresholds.get(classifier.best_model_name, 0.5):.4f}")
    return classifier




def run_evaluation():
    """Step 5: Evaluate trained models."""
    print_header("STEP 5: MODEL EVALUATION")

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
    """
    Run the complete pipeline with all optimizations enabled.

    This includes:
    - Parallel training (all CPU cores)
    - Hyperparameter tuning with Optuna
    - Threshold optimization for classification
    """
    print_header("SUPPLY CHAIN ML - FULL PIPELINE")
    print("\n🔧 Optimization settings:")
    print("   ✅ Parallel training: ENABLED")
    print("   ✅ Hyperparameter tuning: ENABLED")
    print("   ✅ Threshold optimization: ENABLED")
    print()

    start_time = datetime.now()

    # Step 1-3: Data
    df = run_data_loading()
    df_clean = run_preprocessing(df)
    X, y, _ = run_feature_engineering(df_clean)

    # Step 4: Training (with ALL optimizations enabled)
    classifier = run_classification_training(
        X, y,
        parallel=True,              # Use all CPU cores
        tune_hyperparameters=True,  # Optuna hyperparameter tuning
        optimize_thresholds=True    # Threshold optimization
    )

    # Summary
    duration = datetime.now() - start_time
    print_header("PIPELINE COMPLETE")
    print(f"✅ Duration: {duration}")
    print(f"   Best Classification Model: {classifier.best_model_name}")

    return {'classifier': classifier}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Supply Chain ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --all                    # Full pipeline
    python main.py --train-classification   # Classification only
        """
    )

    parser.add_argument('--all', action='store_true', help='Run complete pipeline with all optimizations (tuning + threshold optimization)')
    parser.add_argument('--data', action='store_true', help='Load raw data')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess data')
    parser.add_argument('--features', action='store_true', help='Build features')
    parser.add_argument('--train-classification', action='store_true', help='Train classification models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel training (use sequential)')
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--no-threshold-opt', action='store_true', help='Skip threshold optimization')

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
                run_classification_training(
                    parallel=not args.no_parallel,
                    tune_hyperparameters=not args.no_tuning,
                    optimize_thresholds=not args.no_threshold_opt
                )
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
