"""
Classification Models for Late Delivery Prediction
Trains ensemble and base models with overfitting detection and parallel computing.

Features:
- Parallel training using joblib
- Overfitting/underfitting detection
- Multiple ensemble methods
- SHAP explainability support

Usage:
    from src.models.classifier import SupplyChainClassifier
    classifier = SupplyChainClassifier()
    classifier.initialize_models(include_ensemble=True)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from joblib import Parallel, delayed
import joblib
from datetime import datetime
from typing import Dict, Tuple, Any, List
import warnings
warnings.filterwarnings('ignore')


class SupplyChainClassifier:
    """
    Train and evaluate classification models for late delivery prediction.

    Supports parallel training for faster model development.

    Models:
    - Logistic Regression (baseline)
    - Decision Tree
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - AdaBoost
    - Voting Ensemble
    - Stacking Ensemble
    """

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize classifier.

        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model = None
        self.best_model_name = None
        self.overfitting_analysis: Dict[str, Dict] = {}
        self.model_dir = Path(__file__).parents[2] / "models"
        self.model_dir.mkdir(exist_ok=True)

    def initialize_models(self, include_ensemble: bool = True):
        """Initialize classification models with optimized hyperparameters."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=self.random_state,
                class_weight='balanced', solver='lbfgs', n_jobs=self.n_jobs
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, min_samples_split=10, min_samples_leaf=5,
                random_state=self.random_state, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=4, max_features='sqrt',
                random_state=self.random_state, class_weight='balanced', n_jobs=self.n_jobs
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=4, random_state=self.random_state,
                class_weight='balanced', n_jobs=self.n_jobs
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                min_samples_split=10, min_samples_leaf=4, subsample=0.8,
                random_state=self.random_state
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, learning_rate=0.1, random_state=self.random_state,
                algorithm='SAMME'  # Updated for sklearn compatibility
            )
        }

        if include_ensemble:
            self.models['Voting Ensemble'] = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                                  random_state=self.random_state, n_jobs=self.n_jobs)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                      random_state=self.random_state)),
                    ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                                random_state=self.random_state, n_jobs=self.n_jobs))
                ],
                voting='soft', n_jobs=self.n_jobs
            )
            self.models['Stacking Ensemble'] = StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                                  random_state=self.random_state, n_jobs=self.n_jobs)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                      random_state=self.random_state)),
                    ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                                random_state=self.random_state, n_jobs=self.n_jobs))
                ],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3, n_jobs=self.n_jobs
            )

        print(f"✅ Initialized {len(self.models)} classification models")
        print(f"   Parallel jobs: {self.n_jobs} (-1 = all cores)")
        for name in self.models.keys():
            print(f"   • {name}")

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Split data with stratification."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        print(f"\n📊 Data split:")
        print(f"   Train: {X_train.shape[0]:,} ({100-test_size*100:.0f}%)")
        print(f"   Test:  {X_test.shape[0]:,} ({test_size*100:.0f}%)")
        return X_train, X_test, y_train, y_test

    def _train_single_model(self, model_name: str, model: Any,
                            X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[str, Any]:
        """Train a single model (used for parallel execution)."""
        model.fit(X_train, y_train)
        return model_name, model

    def _evaluate_single_model(self, model_name: str, model: Any,
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate a single model (used for parallel execution)."""
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        train_roc = roc_auc_score(y_train, y_train_proba) if y_train_proba is not None else None
        test_roc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None

        # Overfitting check
        acc_gap = train_acc - test_acc
        if acc_gap > 0.05:
            fit_status = "⚠️ OVERFITTING"
        elif test_acc < 0.6:
            fit_status = "⚠️ UNDERFITTING"
        else:
            fit_status = "✅ GOOD FIT"

        return {
            'model_name': model_name,
            'train_accuracy': train_acc, 'test_accuracy': test_acc,
            'train_f1': train_f1, 'test_f1': test_f1,
            'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
            'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
            'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
            'test_recall': recall_score(y_test, y_test_pred, average='weighted'),
            'train_roc_auc': train_roc, 'test_roc_auc': test_roc,
            'accuracy_gap': acc_gap, 'f1_gap': train_f1 - test_f1,
            'fit_status': fit_status,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }

    def train_all_models_parallel(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series):
        """Train and evaluate all models in parallel."""
        print("\n" + "=" * 60)
        print("PARALLEL TRAINING - ALL MODELS")
        print("=" * 60)

        # Parallel training
        print(f"\n🚀 Training {len(self.models)} models in parallel...")
        trained = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._train_single_model)(name, model, X_train, y_train)
            for name, model in self.models.items()
        )

        # Update models dict with trained models
        for name, model in trained:
            self.models[name] = model

        print(f"\n✅ All models trained!")

        # Parallel evaluation
        print(f"\n📊 Evaluating all models in parallel...")
        evaluations = Parallel(n_jobs=self.n_jobs, verbose=5)(
            delayed(self._evaluate_single_model)(name, model, X_train, y_train, X_test, y_test)
            for name, model in self.models.items()
        )

        # Store results
        for result in evaluations:
            name = result['model_name']
            self.results[name] = result
            self.overfitting_analysis[name] = {
                'train_accuracy': result['train_accuracy'],
                'test_accuracy': result['test_accuracy'],
                'gap': result['accuracy_gap'],
                'status': result['fit_status']
            }

        # Print results
        self._print_results_summary()
        self.select_best_model()
        self.print_overfitting_summary()

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series):
        """Train and evaluate all models (sequential version for debugging)."""
        print("\n" + "=" * 60 + "\nTRAINING ALL MODELS\n" + "=" * 60)

        for model_name, model in self.models.items():
            print(f"\n{'=' * 60}\nTraining {model_name}...\n{'=' * 60}")
            model.fit(X_train, y_train)
            print(f"✅ {model_name} training complete")

            # Evaluate
            result = self._evaluate_single_model(
                model_name, model, X_train, y_train, X_test, y_test
            )
            self.results[model_name] = result
            self.overfitting_analysis[model_name] = {
                'train_accuracy': result['train_accuracy'],
                'test_accuracy': result['test_accuracy'],
                'gap': result['accuracy_gap'],
                'status': result['fit_status']
            }

            # Print individual result
            self._print_model_result(result)

            self.models[model_name] = model

        self.select_best_model()
        self.print_overfitting_summary()

    def _print_model_result(self, result: Dict):
        """Print single model result."""
        print(f"\n{result['model_name']} Performance:")
        print(f"  {'Metric':<15} {'Train':>10} {'Test':>10} {'Gap':>10}")
        print(f"  {'-'*45}")
        print(f"  {'Accuracy':<15} {result['train_accuracy']:>10.4f} {result['test_accuracy']:>10.4f} {result['accuracy_gap']:>10.4f}")
        print(f"  {'F1 Score':<15} {result['train_f1']:>10.4f} {result['test_f1']:>10.4f}")
        print(f"  Status: {result['fit_status']}")

    def _print_results_summary(self):
        """Print summary of all model results."""
        print("\n" + "=" * 60)
        print("MODEL RESULTS SUMMARY")
        print("=" * 60)

        for name, result in self.results.items():
            self._print_model_result(result)

    def select_best_model(self):
        """Select best model based on test F1."""
        best_f1, best_name = 0, None
        for name, res in self.results.items():
            if res['test_f1'] > best_f1:
                best_f1, best_name = res['test_f1'], name
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        print(f"\n{'=' * 60}\n🏆 BEST MODEL: {best_name}\n   Test F1: {best_f1:.4f}\n{'=' * 60}")

    def print_overfitting_summary(self):
        """Print overfitting analysis summary."""
        print(f"\n{'=' * 60}\nOVERFITTING ANALYSIS\n{'=' * 60}")
        print(f"\n{'Model':<25} {'Train':>10} {'Test':>10} {'Gap':>8} {'Status':<15}")
        print("-" * 70)
        for name, a in self.overfitting_analysis.items():
            print(f"{name:<25} {a['train_accuracy']:>10.4f} {a['test_accuracy']:>10.4f} {a['gap']:>8.4f} {a['status']}")

    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from best model."""
        if self.best_model is None:
            raise ValueError("No model trained")
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            print("⚠️ Model does not support feature importance")
            return None
        df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        return df.sort_values('importance', ascending=False).head(top_n)

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get comparison DataFrame of all models."""
        data = [{
            'Model': n, 'Train Accuracy': r['train_accuracy'], 'Test Accuracy': r['test_accuracy'],
            'Train F1': r['train_f1'], 'Test F1': r['test_f1'], 'Test ROC-AUC': r['test_roc_auc'],
            'Accuracy Gap': r['accuracy_gap'], 'Fit Status': r['fit_status']
        } for n, r in self.results.items()]
        return pd.DataFrame(data).sort_values('Test F1', ascending=False)

    def save_models(self):
        """Save all trained models."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        for name, model in self.models.items():
            path = self.model_dir / f"{name.replace(' ', '_').lower()}_{ts}.pkl"
            joblib.dump(model, path)
            print(f"  ✅ Saved {name} → {path}")
        best_path = self.model_dir / f"best_model_{ts}.pkl"
        joblib.dump(self.best_model, best_path)
        print(f"  ✅ Saved best model → {best_path}")
        joblib.dump(self.results, self.model_dir / f"training_results_{ts}.pkl")

    def generate_report(self) -> str:
        """Generate summary report."""
        report = [f"\n{'=' * 80}\nMODEL PERFORMANCE SUMMARY\n{'=' * 80}"]
        for name, r in self.results.items():
            report.append(f"\n{name}: Test F1={r['test_f1']:.4f}, Status={r['fit_status']}")
        report.append(f"\n{'=' * 80}\n🏆 Best: {self.best_model_name}\n{'=' * 80}")
        text = "\n".join(report)
        print(text)
        return text


def run_training_pipeline(parallel: bool = True):
    """
    Complete classification training pipeline.

    Args:
        parallel: If True, use parallel training. If False, train sequentially.
    """
    print("\n" + "=" * 80 + "\nCLASSIFICATION TRAINING PIPELINE\n" + "=" * 80)
    from src.data.preprocess import load_and_preprocess
    from src.features.build_features import build_features_pipeline

    df = load_and_preprocess()
    X, y = build_features_pipeline(df)

    classifier = SupplyChainClassifier(random_state=42, n_jobs=-1)
    classifier.initialize_models(include_ensemble=True)
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)

    if parallel:
        classifier.train_all_models_parallel(X_train, y_train, X_test, y_test)
    else:
        classifier.train_all_models(X_train, y_train, X_test, y_test)

    classifier.save_models()
    classifier.generate_report()

    print("\n✅ CLASSIFICATION PIPELINE COMPLETE!")
    return classifier


if __name__ == "__main__":
    classifier = run_training_pipeline(parallel=True)
