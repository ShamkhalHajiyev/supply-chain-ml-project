"""
Machine Learning Model Training Module
Trains classification models for late delivery prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, accuracy_score
)
import joblib
from datetime import datetime
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class SupplyChainClassifier:
    """
    Train and evaluate classification models for late delivery prediction.

    Models:
    - Logistic Regression (baseline)
    - Random Forest
    - Gradient Boosting
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model = None
        self.best_model_name = None

        # Model directory
        self.model_dir = Path(__file__).parents[2] / "models"
        self.model_dir.mkdir(exist_ok=True)

    def initialize_models(self):
        """Initialize classification models with optimized hyperparameters."""

        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced',
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=self.random_state
            )
        }

        print(" Initialized 3 classification models")

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets with stratification.

        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of test set

        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        print(f"\n Data split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
        print(f"  Test class distribution:  {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
        """Train a single model."""
        print(f"\n{'=' * 60}")
        print(f"Training {model_name}...")
        print(f"{'=' * 60}")

        model = self.models[model_name]
        model.fit(X_train, y_train)

        print(f" {model_name} training complete")

        return model

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None
    ) -> Dict:
        """
        Evaluate model performance.

        Metrics:
        - Accuracy
        - Precision, Recall, F1
        - ROC-AUC
        - Confusion Matrix
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Cross-validation score (if train data provided)
        cv_score = None
        if X_train is not None and y_train is not None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
            cv_score = cv_scores.mean()

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'cv_score': cv_score
        }

        self.results[model_name] = results

        # Print results
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        if cv_score:
            print(f"  CV F1:     {cv_score:.4f}")

        print(f"\nConfusion Matrix:")
        print(cm)

        return results

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        """Train and evaluate all models."""
        print("\n" + "=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)

        for model_name in self.models.keys():
            # Train
            trained_model = self.train_model(model_name, X_train, y_train)

            # Evaluate
            self.evaluate_model(model_name, trained_model, X_test, y_test, X_train, y_train)

            # Update model
            self.models[model_name] = trained_model

        # Select best model
        self.select_best_model()

    def select_best_model(self):
        """Select best model based on F1 score."""
        best_f1 = 0
        best_name = None

        for model_name, results in self.results.items():
            if results['f1_score'] > best_f1:
                best_f1 = results['f1_score']
                best_name = model_name

        self.best_model_name = best_name
        self.best_model = self.models[best_name]

        print("\n" + "=" * 60)
        print(f"<Æ BEST MODEL: {best_name}")
        print(f"   F1 Score: {best_f1:.4f}")
        print("=" * 60)

    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from best model.

        Args:
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.best_model is None:
            raise ValueError("No model trained yet")

        # Check if model has feature_importances_
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            print("  Model does not support feature importance")
            return None

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        print(f"\n= Top {top_n} Important Features ({self.best_model_name}):")
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")

        return importance_df

    def save_models(self):
        """Save all trained models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        for model_name, model in self.models.items():
            filename = f"{model_name.replace(' ', '_').lower()}_{timestamp}.pkl"
            filepath = self.model_dir / filename
            joblib.dump(model, filepath)
            print(f" Saved {model_name} ’ {filepath}")

        # Save best model separately
        best_filename = f"best_model_{timestamp}.pkl"
        best_filepath = self.model_dir / best_filename
        joblib.dump(self.best_model, best_filepath)
        print(f" Saved best model ({self.best_model_name}) ’ {best_filepath}")

        # Save results
        results_filename = f"training_results_{timestamp}.pkl"
        results_filepath = self.model_dir / results_filename
        joblib.dump(self.results, results_filepath)
        print(f" Saved training results ’ {results_filepath}")

    def generate_report(self) -> str:
        """Generate a summary report of model performance."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("=" * 80)

        for model_name, results in self.results.items():
            report.append(f"\n{model_name}:")
            report.append(f"  Accuracy:  {results['accuracy']:.4f}")
            report.append(f"  F1 Score:  {results['f1_score']:.4f}")
            if results['roc_auc']:
                report.append(f"  ROC-AUC:   {results['roc_auc']:.4f}")
            if results['cv_score']:
                report.append(f"  CV F1:     {results['cv_score']:.4f}")

        report.append("\n" + "=" * 80)
        report.append(f"<Æ Best Model: {self.best_model_name}")
        report.append("=" * 80)

        report_text = "\n".join(report)
        print(report_text)

        return report_text


def run_training_pipeline():
    """
    Complete ML training pipeline.

    Steps:
    1. Load preprocessed data
    2. Build features
    3. Split data
    4. Train models
    5. Evaluate models
    6. Save models
    """
    print("\n" + "=" * 80)
    print("SUPPLY CHAIN ML TRAINING PIPELINE")
    print("=" * 80)

    # Import required modules
    from src.data.preprocess import load_and_preprocess
    from src.features.build_features import build_features_pipeline

    # Step 1: Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    df = load_and_preprocess()

    # Step 2: Build features
    print("\n[2/6] Building features...")
    X, y = build_features_pipeline(df)

    # Step 3: Initialize classifier
    print("\n[3/6] Initializing classifier...")
    classifier = SupplyChainClassifier(random_state=42)
    classifier.initialize_models()

    # Step 4: Split data
    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = classifier.split_data(X, y, test_size=0.2)

    # Step 5: Train all models
    print("\n[5/6] Training models...")
    classifier.train_all_models(X_train, y_train, X_test, y_test)

    # Step 6: Feature importance
    print("\n[6/6] Analyzing feature importance...")
    classifier.get_feature_importance(X.columns.tolist(), top_n=15)

    # Save models
    print("\n=æ Saving models...")
    classifier.save_models()

    # Generate report
    classifier.generate_report()

    print("\n" + "=" * 80)
    print(" TRAINING PIPELINE COMPLETE!")
    print("=" * 80)

    return classifier


if __name__ == "__main__":
    classifier = run_training_pipeline()
