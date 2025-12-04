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
    f1_score, accuracy_score, precision_score, recall_score,
    roc_curve, precision_recall_curve
)
from joblib import Parallel, delayed
import joblib
from datetime import datetime
from typing import Dict, Tuple, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional imports for hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Hyperparameter tuning will be disabled.")


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
        self.optimal_thresholds: Dict[str, float] = {}  # Store optimal thresholds
        self.tuned_models: Dict[str, Any] = {}  # Store hyperparameter-tuned models
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

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                   val_size: float = 0.1) -> Tuple:
        """
        Split data with stratification into train/val/test.

        Args:
            X: Features
            y: Targets
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining after test)

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted,
            random_state=self.random_state, stratify=y_train_val
        )

        print(f"\n📊 Data split:")
        print(f"   Train: {X_train.shape[0]:,} ({100-test_size*100-val_size*100:.0f}%)")
        print(f"   Val:   {X_val.shape[0]:,} ({val_size*100:.0f}%)")
        print(f"   Test:  {X_test.shape[0]:,} ({test_size*100:.0f}%)")
        return X_train, X_val, X_test, y_train, y_val, y_test

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

    def optimize_threshold(self, model_name: str, model: Any, X_val: pd.DataFrame,
                           y_val: pd.Series, metric: str = 'f1') -> float:
        """
        Optimize classification threshold for a model.

        Args:
            model_name: Name of the model
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
            metric: Metric to optimize ('f1', 'youden', 'precision_recall')

        Returns:
            Optimal threshold value
        """
        if not hasattr(model, 'predict_proba'):
            print(f"⚠️ {model_name} does not support probability predictions. Using default threshold 0.5.")
            return 0.5

        y_proba = model.predict_proba(X_val)[:, 1]

        if metric == 'f1':
            # Optimize F1 score
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_threshold = 0.5
            best_f1 = 0

            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred, average='weighted')
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        elif metric == 'youden':
            # Youden's J statistic (maximize TPR - FPR)
            fpr, tpr, thresholds = roc_curve(y_val, y_proba)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_idx]

        elif metric == 'precision_recall':
            # Optimize F1 on precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        else:
            best_threshold = 0.5

        self.optimal_thresholds[model_name] = best_threshold
        return best_threshold

    def optimize_all_thresholds(self, X_val: pd.DataFrame, y_val: pd.Series,
                                 metric: str = 'f1'):
        """
        Optimize thresholds for all models.

        Args:
            X_val: Validation features
            y_val: Validation targets
            metric: Metric to optimize
        """
        print("\n" + "=" * 60)
        print("THRESHOLD OPTIMIZATION")
        print("=" * 60)

        for model_name, model in self.models.items():
            threshold = self.optimize_threshold(model_name, model, X_val, y_val, metric)
            print(f"  {model_name:<25} Optimal threshold: {threshold:.4f}")

        print("=" * 60)

    def evaluate_with_optimal_threshold(self, model_name: str, model: Any,
                                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model using optimized threshold.

        Args:
            model_name: Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with metrics using optimal threshold
        """
        if not hasattr(model, 'predict_proba'):
            y_pred = model.predict(X_test)
        else:
            threshold = self.optimal_thresholds.get(model_name, 0.5)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'threshold': self.optimal_thresholds.get(model_name, 0.5)
        }

    def tune_hyperparameters(self, model_name: str, model_type: str,
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           n_trials: int = 50) -> Any:
        """
        Tune hyperparameters using Optuna.

        Args:
            model_name: Name of the model
            model_type: Type of model ('logistic', 'tree', 'rf', 'et', 'gb', 'ada')
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of Optuna trials

        Returns:
            Best tuned model
        """
        if not OPTUNA_AVAILABLE:
            print(f"⚠️ Optuna not available. Skipping hyperparameter tuning for {model_name}.")
            return None

        print(f"\n🔍 Tuning hyperparameters for {model_name} ({n_trials} trials)...")

        def objective(trial):
            if model_type == 'logistic':
                model = LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    class_weight='balanced',
                    solver=trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
                    C=trial.suggest_float('C', 0.01, 100, log=True),
                    n_jobs=self.n_jobs
                )

            elif model_type == 'tree':
                model = DecisionTreeClassifier(
                    max_depth=trial.suggest_int('max_depth', 5, 30),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=self.random_state,
                    class_weight='balanced'
                )

            elif model_type == 'rf':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 30),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=self.n_jobs
                )

            elif model_type == 'et':
                model = ExtraTreesClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 5, 30),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=self.n_jobs
                )

            elif model_type == 'gb':
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    random_state=self.random_state
                )

            elif model_type == 'ada':
                model = AdaBoostClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
                    random_state=self.random_state,
                    algorithm='SAMME'
                )

            else:
                return None

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            return score

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Train best model on full training set
        best_params = study.best_params

        if model_type == 'logistic':
            best_model = LogisticRegression(
                max_iter=1000, random_state=self.random_state, class_weight='balanced',
                n_jobs=self.n_jobs, **best_params
            )
        elif model_type == 'tree':
            best_model = DecisionTreeClassifier(
                random_state=self.random_state, class_weight='balanced', **best_params
            )
        elif model_type == 'rf':
            best_model = RandomForestClassifier(
                random_state=self.random_state, class_weight='balanced',
                n_jobs=self.n_jobs, **best_params
            )
        elif model_type == 'et':
            best_model = ExtraTreesClassifier(
                random_state=self.random_state, class_weight='balanced',
                n_jobs=self.n_jobs, **best_params
            )
        elif model_type == 'gb':
            best_model = GradientBoostingClassifier(
                random_state=self.random_state, **best_params
            )
        elif model_type == 'ada':
            best_model = AdaBoostClassifier(
                random_state=self.random_state, algorithm='SAMME', **best_params
            )

        best_model.fit(X_train, y_train)
        self.tuned_models[model_name] = best_model

        print(f"  ✅ Best F1: {study.best_value:.4f}")
        print(f"  📊 Best params: {best_params}")

        return best_model

    def tune_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 30):
        """
        Tune hyperparameters for all base models (not ensembles).

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of trials per model
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available. Skipping hyperparameter tuning.")
            return

        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)

        model_configs = {
            'Logistic Regression': 'logistic',
            'Decision Tree': 'tree',
            'Random Forest': 'rf',
            'Extra Trees': 'et',
            'Gradient Boosting': 'gb',
            'AdaBoost': 'ada'
        }

        for model_name, model_type in model_configs.items():
            if model_name in self.models:
                self.tune_hyperparameters(
                    model_name, model_type, X_train, y_train, X_val, y_val, n_trials
                )

        print("=" * 60)
        print("✅ Hyperparameter tuning complete!")

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get comparison DataFrame of all models."""
        data = [{
            'Model': n, 'Train Accuracy': r['train_accuracy'], 'Test Accuracy': r['test_accuracy'],
            'Train F1': r['train_f1'], 'Test F1': r['test_f1'], 'Test ROC-AUC': r['test_roc_auc'],
            'Accuracy Gap': r['accuracy_gap'], 'Fit Status': r['fit_status']
        } for n, r in self.results.items()]
        return pd.DataFrame(data).sort_values('Test F1', ascending=False)

    def save_models(self):
        """Save all trained models and optimal thresholds."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        for name, model in self.models.items():
            path = self.model_dir / f"{name.replace(' ', '_').lower()}_{ts}.pkl"
            joblib.dump(model, path)
            print(f"  ✅ Saved {name} → {path}")
        best_path = self.model_dir / f"best_model_{ts}.pkl"
        joblib.dump(self.best_model, best_path)
        print(f"  ✅ Saved best model → {best_path}")
        joblib.dump(self.results, self.model_dir / f"training_results_{ts}.pkl")

        # Save optimal thresholds
        if self.optimal_thresholds:
            thresholds_path = self.model_dir / f"optimal_thresholds_{ts}.pkl"
            joblib.dump(self.optimal_thresholds, thresholds_path)
            print(f"  ✅ Saved optimal thresholds → {thresholds_path}")

    def generate_report(self) -> str:
        """Generate summary report."""
        report = [f"\n{'=' * 80}\nMODEL PERFORMANCE SUMMARY\n{'=' * 80}"]
        for name, r in self.results.items():
            report.append(f"\n{name}: Test F1={r['test_f1']:.4f}, Status={r['fit_status']}")
        report.append(f"\n{'=' * 80}\n🏆 Best: {self.best_model_name}\n{'=' * 80}")
        text = "\n".join(report)
        print(text)
        return text


def run_training_pipeline(parallel: bool = True, tune_hyperparameters: bool = True,
                          optimize_thresholds: bool = True, n_trials: int = 30):
    """
    Complete classification training pipeline with **threshold optimization first,
    then hyperparameter tuning**, as requested.

    Args:
        parallel: If True, use parallel training. If False, train sequentially.
        tune_hyperparameters: If True, perform hyperparameter tuning with Optuna.
        optimize_thresholds: If True, optimize classification thresholds.
        n_trials: Number of Optuna trials per model (if tuning enabled).
    """
    print("\n" + "=" * 80 + "\nCLASSIFICATION TRAINING PIPELINE\n" + "=" * 80)
    from src.data.preprocess import load_and_preprocess
    from src.features.build_features import build_features_pipeline

    df = load_and_preprocess()
    X, y = build_features_pipeline(df)

    classifier = SupplyChainClassifier(random_state=42, n_jobs=-1)
    classifier.initialize_models(include_ensemble=True)

    # Split into train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.split_data(X, y)

    # STEP 1: Train base models
    print("\n" + "=" * 80)
    print("STEP 1: BASE MODEL TRAINING")
    print("=" * 80)
    if parallel:
        classifier.train_all_models_parallel(X_train, y_train, X_test, y_test)
    else:
        classifier.train_all_models(X_train, y_train, X_test, y_test)

    # STEP 2: Threshold optimization (on base models)
    if optimize_thresholds:
        print("\n" + "=" * 80)
        print("STEP 2: THRESHOLD OPTIMIZATION (BASE MODELS)")
        print("=" * 80)
        classifier.optimize_all_thresholds(X_val, y_val, metric='f1')

        # Evaluate base models with their optimal thresholds
        print("\n📊 Evaluating base models with optimal thresholds...")
        base_threshold_results = {}
        for model_name, model in classifier.models.items():
            if model_name in classifier.optimal_thresholds:
                result = classifier.evaluate_with_optimal_threshold(
                    model_name, model, X_test, y_test
                )
                base_threshold_results[model_name] = result
                print(f"  {model_name:<25} F1: {result['f1']:.4f} (threshold: {result['threshold']:.4f})")

        # Store base threshold metrics (for comparison)
        for name, result in base_threshold_results.items():
            if name in classifier.results:
                classifier.results[name]['base_test_f1_optimized'] = result['f1']
                classifier.results[name]['base_optimal_threshold'] = result['threshold']

    # STEP 3: Hyperparameter tuning (optional, after threshold optimization)
    if tune_hyperparameters and OPTUNA_AVAILABLE:
        print("\n" + "=" * 80)
        print("STEP 3: HYPERPARAMETER TUNING")
        print("=" * 80)
        classifier.tune_all_models(X_train, y_train, X_val, y_val, n_trials=n_trials)

        # Replace base models with tuned models
        for name, tuned_model in classifier.tuned_models.items():
            classifier.models[name] = tuned_model
            print(f"  ✅ Replaced {name} with tuned version")

        # Re-evaluate tuned models (default 0.5 threshold)
        print("\n📊 Re-evaluating tuned models (threshold=0.5)...")
        if parallel:
            evaluations = Parallel(n_jobs=classifier.n_jobs, verbose=5)(
                delayed(classifier._evaluate_single_model)(
                    name, model, X_train, y_train, X_test, y_test
                )
                for name, model in classifier.tuned_models.items()
            )
            for result in evaluations:
                name = result['model_name']
                classifier.results[name] = result
    elif tune_hyperparameters and not OPTUNA_AVAILABLE:
        print("\n⚠️ Hyperparameter tuning requested but Optuna not available.")

    # OPTIONAL: Re-run threshold optimization for tuned models so final models
    # also have optimal thresholds consistent with the new parameters.
    if optimize_thresholds and tune_hyperparameters and OPTUNA_AVAILABLE:
        print("\n" + "=" * 80)
        print("STEP 4: THRESHOLD OPTIMIZATION (TUNED MODELS)")
        print("=" * 80)
        classifier.optimize_all_thresholds(X_val, y_val, metric='f1')

        print("\n📊 Evaluating tuned models with optimal thresholds...")
        tuned_threshold_results = {}
        for model_name, model in classifier.models.items():
            if model_name in classifier.optimal_thresholds:
                result = classifier.evaluate_with_optimal_threshold(
                    model_name, model, X_test, y_test
                )
                tuned_threshold_results[model_name] = result
                print(f"  {model_name:<25} F1: {result['f1']:.4f} (threshold: {result['threshold']:.4f})")

        for name, result in tuned_threshold_results.items():
            if name in classifier.results:
                classifier.results[name]['test_f1_optimized'] = result['f1']
                classifier.results[name]['optimal_threshold'] = result['threshold']

    classifier.save_models()
    classifier.generate_report()

    print("\n✅ CLASSIFICATION PIPELINE COMPLETE!")
    return classifier


if __name__ == "__main__":
    classifier = run_training_pipeline(parallel=True)
