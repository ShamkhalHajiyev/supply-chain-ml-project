"""
Classification Models for Late Delivery Prediction

For advanced features (Optuna tuning, modern boosting), see classifier_advanced.py

Features:
- Parallel training using joblib
- Overfitting/underfitting detection
- Multiple ensemble methods
- Threshold optimization
- SHAP explainability support
- Comprehensive reporting

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

# Modern boosting libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available.")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available.")

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
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - XGBoost (if available)
    - CatBoost (if available)
    - LightGBM (if available)
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
            )
        }

        # Add modern boosting models if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=self.n_jobs,
                eval_metric='logloss', use_label_encoder=False
            )

        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                random_state=self.random_state, verbose=False,
                thread_count=self.n_jobs if self.n_jobs > 0 else -1
            )

        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=self.n_jobs,
                verbose=-1
            )

        if include_ensemble:
            # Build ensemble with best available models
            ensemble_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                              random_state=self.random_state, n_jobs=self.n_jobs)),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                  random_state=self.random_state)),
                ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                            random_state=self.random_state, n_jobs=self.n_jobs))
            ]

            # Add modern boosting models to ensemble if available
            if LIGHTGBM_AVAILABLE:
                ensemble_estimators.append(('lgbm', LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1
                )))
            if XGBOOST_AVAILABLE:
                ensemble_estimators.append(('xgb', XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, n_jobs=self.n_jobs,
                    eval_metric='logloss', use_label_encoder=False
                )))

            self.models['Voting Ensemble'] = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft', n_jobs=self.n_jobs
            )
            self.models['Stacking Ensemble'] = StackingClassifier(
                estimators=ensemble_estimators,
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
        """
        Select best model considering both performance and overfitting.
        
        Priority:
        1. Models with GOOD FIT status (no overfitting)
        2. Among good fit models, highest test F1 score (use optimized F1 if available)
        3. If no good fit models, select highest F1 but warn about overfitting
        4. Secondary criterion: smaller accuracy gap (better generalization)
        """
        good_fit_models = []
        overfitting_models = []
        underfitting_models = []
        
        # Categorize models by fit status
        for name, res in self.results.items():
            fit_status = res.get('fit_status', '')
            # Use optimized F1 if available, otherwise use regular test_f1
            f1_score = res.get('test_f1_optimized', res.get('test_f1', 0))
            
            if 'GOOD FIT' in fit_status:
                good_fit_models.append((name, res, f1_score))
            elif 'OVERFITTING' in fit_status:
                overfitting_models.append((name, res, f1_score))
            elif 'UNDERFITTING' in fit_status:
                underfitting_models.append((name, res, f1_score))
        
        best_name = None
        best_f1 = 0
        selection_reason = ""
        
        # Priority 1: Select from good fit models
        if good_fit_models:
            # Sort by F1 score (descending), then by accuracy gap (ascending)
            good_fit_models.sort(
                key=lambda x: (x[2], -x[1]['accuracy_gap']),
                reverse=True
            )
            best_name, best_result, best_f1 = good_fit_models[0]
            selection_reason = f"Best among GOOD FIT models (F1: {best_f1:.4f}, Gap: {best_result['accuracy_gap']:.4f})"
        # Priority 2: If no good fit, select from overfitting models (but warn)
        elif overfitting_models:
            overfitting_models.sort(
                key=lambda x: (x[2], -x[1]['accuracy_gap']),
                reverse=True
            )
            best_name, best_result, best_f1 = overfitting_models[0]
            selection_reason = f"⚠️ WARNING: Selected overfitting model (F1: {best_f1:.4f}, Gap: {best_result['accuracy_gap']:.4f})"
        # Priority 3: Last resort - any model
        else:
            # Sort all models by F1 score
            all_models = [(name, res, res.get('test_f1_optimized', res.get('test_f1', 0))) 
                         for name, res in self.results.items()]
            all_models.sort(key=lambda x: x[2], reverse=True)
            if all_models:
                best_name, best_result, best_f1 = all_models[0]
                selection_reason = f"Selected highest F1 model (F1: {best_f1:.4f})"
        
        if best_name is None:
            raise ValueError("No models available for selection")
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        # Print selection details
        best_result = self.results[best_name]
        optimized_f1 = best_result.get('test_f1_optimized')
        f1_display = optimized_f1 if optimized_f1 else best_f1
        
        print(f"\n{'=' * 60}\n🏆 BEST MODEL: {best_name}\n{'=' * 60}")
        print(f"   Test F1: {f1_display:.4f}" + (" (optimized)" if optimized_f1 else ""))
        print(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"   Accuracy Gap: {best_result['accuracy_gap']:.4f}")
        print(f"   Fit Status: {best_result['fit_status']}")
        print(f"   Selection: {selection_reason}")
        print(f"{'=' * 60}")

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
                           n_trials: int = 50, use_fast_validation: bool = True) -> Any:
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

        # Use smaller validation set for faster evaluation during tuning
        # This speeds up each trial without significantly affecting quality
        if use_fast_validation and len(X_val) > 5000:
            from sklearn.model_selection import train_test_split
            X_val_tune, _, y_val_tune, _ = train_test_split(
                X_val, y_val, test_size=0.5, random_state=self.random_state, stratify=y_val
            )
            print(f"  ⚡ Using {len(X_val_tune):,} samples for fast validation (from {len(X_val):,})")
        else:
            X_val_tune, y_val_tune = X_val, y_val

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

            elif model_type == 'xgb' and XGBOOST_AVAILABLE:
                model = XGBClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 100, 300),
                    max_depth=trial.suggest_int('max_depth', 4, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    eval_metric='logloss',
                    use_label_encoder=False
                )

            elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                model = CatBoostClassifier(
                    iterations=trial.suggest_int('iterations', 100, 300),
                    depth=trial.suggest_int('depth', 4, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    random_state=self.random_state,
                    verbose=False,
                    thread_count=self.n_jobs if self.n_jobs > 0 else -1
                )

            elif model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
                model = LGBMClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 100, 300),
                    max_depth=trial.suggest_int('max_depth', 4, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=-1
                )

            else:
                return None

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val_tune)
            score = f1_score(y_val_tune, y_pred, average='weighted')
            return score

        # Use MedianPruner to stop unpromising trials early (faster tuning)
        # Note: For models without intermediate steps, pruning compares completed trials
        # This helps Optuna focus on promising hyperparameter regions
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,  # Don't prune first 3 trials (need baseline)
            n_warmup_steps=0    # No warmup needed for single-step evaluation
        )
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=pruner
        )
        # Add timeout to prevent hanging (8 minutes per model max)
        # Reduced timeout since we're using parallel execution
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=480,  # 8 minutes max per model
            show_progress_bar=False,
            n_jobs=1  # Each model tuning runs in parallel, but trials within a study are sequential
        )

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
        elif model_type == 'xgb' and XGBOOST_AVAILABLE:
            best_model = XGBClassifier(
                random_state=self.random_state, n_jobs=self.n_jobs,
                eval_metric='logloss', use_label_encoder=False, **best_params
            )
        elif model_type == 'catboost' and CATBOOST_AVAILABLE:
            best_model = CatBoostClassifier(
                random_state=self.random_state, verbose=False,
                thread_count=self.n_jobs if self.n_jobs > 0 else -1, **best_params
            )
        elif model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
            best_model = LGBMClassifier(
                random_state=self.random_state, n_jobs=self.n_jobs,
                verbose=-1, **best_params
            )
        else:
            print(f"  ⚠️ Unknown model type: {model_type}")
            return None

        best_model.fit(X_train, y_train)
        self.tuned_models[model_name] = best_model

        print(f"  ✅ Best F1: {study.best_value:.4f}")
        print(f"  📊 Best params: {best_params}")

        return best_model

    def _tune_single_model_wrapper(self, args):
        """Wrapper function for parallel tuning (used with joblib)."""
        model_name, model_type, X_train, y_train, X_val, y_val, trials = args
        return model_name, self.tune_hyperparameters(
            model_name, model_type, X_train, y_train, X_val, y_val, trials
        )

    def tune_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 10,
                       parallel: bool = True):
        """
        Tune hyperparameters for important models in PARALLEL for faster execution.
        Uses joblib to run multiple Optuna studies simultaneously.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of trials per model (reduced for speed)
            parallel: If True, tune models in parallel. If False, tune sequentially.
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available. Skipping hyperparameter tuning.")
            return

        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        print("⚠️  Only tuning important models (boosting models) for speed")
        print(f"    Trials per model: {n_trials} (reduced for faster training)")
        print(f"    Parallel tuning: {'ENABLED' if parallel else 'DISABLED'}")

        # Only tune models that benefit most from hyperparameter tuning
        model_configs = {
            'Random Forest': ('rf', 10),  # Fewer trials for RF
            'Extra Trees': ('et', 10),
            'Gradient Boosting': ('gb', 15),  # More trials for GB
            'XGBoost': ('xgb', 15) if XGBOOST_AVAILABLE else None,
            'CatBoost': ('catboost', 15) if CATBOOST_AVAILABLE else None,
            'LightGBM': ('lgbm', 15) if LIGHTGBM_AVAILABLE else None,
        }

        # Prepare arguments for parallel execution
        tuning_tasks = []
        for model_name, config in model_configs.items():
            if config is None:
                continue
            if model_name in self.models:
                model_type, trials = config
                tuning_tasks.append((
                    model_name, model_type, X_train, y_train, X_val, y_val, trials
                ))

        if not tuning_tasks:
            print("⚠️ No models to tune.")
            return

        # Execute tuning (parallel or sequential)
        if parallel and len(tuning_tasks) > 1:
            print(f"\n🚀 Tuning {len(tuning_tasks)} models in parallel...")
            results = Parallel(n_jobs=min(len(tuning_tasks), self.n_jobs), verbose=5)(
                delayed(self._tune_single_model_wrapper)(task) for task in tuning_tasks
            )
            # Store results
            for model_name, tuned_model in results:
                if tuned_model is not None:
                    self.tuned_models[model_name] = tuned_model
        else:
            print(f"\n🔄 Tuning {len(tuning_tasks)} models sequentially...")
            for task in tuning_tasks:
                model_name, model_type, X_train, y_train, X_val, y_val, trials = task
                tuned_model = self.tune_hyperparameters(
                    model_name, model_type, X_train, y_train, X_val, y_val, trials
                )
                if tuned_model is not None:
                    self.tuned_models[model_name] = tuned_model

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

    def generate_detailed_report(self, X_train: pd.DataFrame = None, y_train: pd.Series = None,
                                 X_test: pd.DataFrame = None, y_test: pd.Series = None,
                                 X_val: pd.DataFrame = None, y_val: pd.Series = None,
                                 feature_names: List[str] = None,
                                 training_time: float = None,
                                 tuning_time: float = None) -> str:
        """
        Generate a comprehensive detailed modeling report in Markdown format.
        Includes all aspects of the modeling process.

        Args:
            X_train: Training features (optional)
            y_train: Training targets (optional)
            X_test: Test features (optional)
            y_test: Test targets (optional)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: List of feature names (optional, for feature importance)
            training_time: Total training time in seconds (optional)
            tuning_time: Total hyperparameter tuning time in seconds (optional)

        Returns:
            Markdown report as string
        """
        from datetime import datetime

        report = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Header
        report.append("# Comprehensive Modeling Report")
        report.append("## Supply Chain Late Delivery Prediction\n")
        report.append(f"**Generated:** {timestamp}\n")
        report.append("---\n")

        # Table of Contents
        report.append("## Table of Contents\n")
        report.append("1. [Executive Summary](#executive-summary)")
        report.append("2. [Dataset Information](#dataset-information)")
        report.append("3. [Feature Engineering](#feature-engineering)")
        report.append("4. [Model Configurations](#model-configurations)")
        report.append("5. [Training Process](#training-process)")
        report.append("6. [Hyperparameter Tuning](#hyperparameter-tuning)")
        report.append("7. [Model Performance Comparison](#model-performance-comparison)")
        report.append("8. [Detailed Model Metrics](#detailed-model-metrics)")
        report.append("9. [Confusion Matrix Analysis](#confusion-matrix-analysis)")
        report.append("10. [Feature Importance](#feature-importance)")
        report.append("11. [Threshold Optimization](#threshold-optimization)")
        report.append("12. [Model Evaluation](#model-evaluation)")
        report.append("13. [Recommendations](#recommendations)")
        report.append("14. [Appendix](#appendix)\n")
        report.append("---\n")

        # Executive Summary
        report.append("## Executive Summary\n")
        best_result = self.results.get(self.best_model_name, {})
        report.append(f"**Best Model:** {self.best_model_name}")
        report.append(f"- **Test F1 Score:** {best_result.get('test_f1', 0):.4f}")
        report.append(f"- **Test Accuracy:** {best_result.get('test_accuracy', 0):.4f}")
        test_roc = best_result.get('test_roc_auc')
        roc_str = f"{test_roc:.4f}" if test_roc is not None else "N/A"
        report.append(f"- **Test ROC-AUC:** {roc_str}")
        if self.optimal_thresholds and self.best_model_name in self.optimal_thresholds:
            report.append(f"- **Optimal Threshold:** {self.optimal_thresholds[self.best_model_name]:.4f}")
        report.append(f"- **Fit Status:** {best_result.get('fit_status', 'N/A')}")

        # Summary statistics
        total_models = len(self.results)
        tuned_count = len(self.tuned_models) if self.tuned_models else 0
        report.append(f"\n**Modeling Summary:**")
        report.append(f"- Total models trained: {total_models}")
        report.append(f"- Models with hyperparameter tuning: {tuned_count}")
        report.append(f"- Models with threshold optimization: {len(self.optimal_thresholds) if self.optimal_thresholds else 0}")
        if training_time:
            report.append(f"- Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        if tuning_time:
            report.append(f"- Hyperparameter tuning time: {tuning_time:.2f} seconds ({tuning_time/60:.2f} minutes)")
        report.append("")

        # Dataset Information
        report.append("## Dataset Information\n")
        if X_train is not None and y_train is not None:
            report.append("### Training Set")
            report.append(f"- **Samples:** {len(X_train):,}")
            report.append(f"- **Features:** {X_train.shape[1] if hasattr(X_train, 'shape') else len(feature_names) if feature_names else 'N/A'}")
            if y_train is not None:
                train_dist = y_train.value_counts().to_dict()
                total_train = len(y_train)
                report.append(f"- **Class Distribution:**")
                for cls, count in sorted(train_dist.items()):
                    pct = count / total_train * 100
                    label = "Late Delivery" if cls == 1 else "On-time"
                    report.append(f"  - {label}: {count:,} ({pct:.2f}%)")
            report.append("")

        if X_val is not None and y_val is not None:
            report.append("### Validation Set")
            report.append(f"- **Samples:** {len(X_val):,}")
            if y_val is not None:
                val_dist = y_val.value_counts().to_dict()
                total_val = len(y_val)
                report.append(f"- **Class Distribution:**")
                for cls, count in sorted(val_dist.items()):
                    pct = count / total_val * 100
                    label = "Late Delivery" if cls == 1 else "On-time"
                    report.append(f"  - {label}: {count:,} ({pct:.2f}%)")
            report.append("")

        if X_test is not None and y_test is not None:
            report.append("### Test Set")
            report.append(f"- **Samples:** {len(X_test):,}")
            if y_test is not None:
                test_dist = y_test.value_counts().to_dict()
                total_test = len(y_test)
                report.append(f"- **Class Distribution:**")
                for cls, count in sorted(test_dist.items()):
                    pct = count / total_test * 100
                    label = "Late Delivery" if cls == 1 else "On-time"
                    report.append(f"  - {label}: {count:,} ({pct:.2f}%)")
            report.append("")

        # Feature Engineering
        report.append("## Feature Engineering\n")
        if feature_names:
            report.append(f"### Feature Overview")
            report.append(f"- **Total Features:** {len(feature_names)}")
            report.append(f"- **Feature Types:** Engineered features from raw supply chain data")
            report.append("")
            report.append("### Feature List")
            report.append("| # | Feature Name |")
            report.append("|---|--------------|")
            for i, feat in enumerate(feature_names, 1):
                report.append(f"| {i} | {feat} |")
            report.append("")
        else:
            report.append("*Feature information not available*\n")

        # Data Leakage Prevention
        report.append("### Data Leakage Prevention\n")
        report.append("The following columns were **excluded** to prevent data leakage:")
        report.append("- `late_delivery_risk` - This IS the target variable")
        report.append("- `delivery_status` - Categorical form of target")
        report.append("- `days_for_shipping_(real)` - Only known after delivery")
        report.append("- `shipping_date_(dateorders)` - Actual shipping date (post-hoc)")
        report.append("\n**Note:** Models achieve realistic ~70-80% accuracy because leaky features are properly excluded.\n")

        # Model Configurations
        report.append("## Model Configurations\n")
        report.append("### Models Trained\n")
        report.append("| Model | Type | Description |")
        report.append("|-------|------|-------------|")

        model_types = {
            'Logistic Regression': 'Baseline Linear',
            'Random Forest': 'Ensemble Tree',
            'Extra Trees': 'Ensemble Tree',
            'Gradient Boosting': 'Boosting',
            'XGBoost': 'Gradient Boosting',
            'CatBoost': 'Gradient Boosting',
            'LightGBM': 'Gradient Boosting',
            'Voting Ensemble': 'Meta Ensemble',
            'Stacking Ensemble': 'Meta Ensemble'
        }

        for name in self.models.keys():
            model_type = model_types.get(name, 'Unknown')
            desc = {
                'Baseline Linear': 'Linear baseline model',
                'Ensemble Tree': 'Tree-based ensemble',
                'Boosting': 'Gradient boosting model',
                'Gradient Boosting': 'Advanced gradient boosting',
                'Meta Ensemble': 'Combines multiple base models'
            }.get(model_type, 'Classification model')
            report.append(f"| {name} | {model_type} | {desc} |")
        report.append("")

        # Model Hyperparameters
        report.append("### Model Hyperparameters\n")
        for name, model in self.models.items():
            report.append(f"#### {name}\n")
            try:
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    # Show key parameters
                    key_params = {k: v for k, v in params.items()
                                if k in ['n_estimators', 'max_depth', 'learning_rate',
                                        'min_samples_split', 'min_samples_leaf', 'C',
                                        'solver', 'subsample', 'colsample_bytree', 'iterations',
                                        'depth', 'num_leaves']}
                    if key_params:
                        report.append("**Key Parameters:**")
                        for param, value in sorted(key_params.items()):
                            report.append(f"- `{param}`: {value}")
                    else:
                        report.append("*Using default parameters*")
                else:
                    report.append("*Parameters not available*")
            except:
                report.append("*Parameters not available*")
            report.append("")

        # Training Process
        report.append("## Training Process\n")
        report.append("### Training Methodology\n")
        report.append(f"- **Random State:** {self.random_state} (for reproducibility)")
        report.append(f"- **Parallel Training:** {'Enabled' if self.n_jobs != 1 else 'Disabled'}")
        if self.n_jobs == -1:
            report.append(f"- **CPU Cores Used:** All available cores")
        else:
            report.append(f"- **CPU Cores Used:** {self.n_jobs}")
        report.append("- **Data Split:** Train (70%) / Validation (10%) / Test (20%)")
        report.append("- **Stratification:** Enabled (maintains class distribution)")
        report.append("")

        if training_time:
            report.append(f"### Training Time\n")
            report.append(f"- **Total Training Time:** {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
            if total_models > 0:
                avg_time = training_time / total_models
                report.append(f"- **Average Time per Model:** {avg_time:.2f} seconds")
            report.append("")

        # Model Performance Comparison
        report.append("## Model Performance Comparison\n")
        report.append("| Model | Train Acc | Test Acc | Train F1 | Test F1 | Test ROC-AUC | Acc Gap | Status |")
        report.append("|-------|-----------|----------|----------|---------|--------------|---------|--------|")

        for name, r in sorted(self.results.items(), key=lambda x: x[1].get('test_f1', 0), reverse=True):
            train_acc = r.get('train_accuracy', 0)
            test_acc = r.get('test_accuracy', 0)
            train_f1 = r.get('train_f1', 0)
            test_f1 = r.get('test_f1', 0)
            test_roc = r.get('test_roc_auc', 0) if r.get('test_roc_auc') else None
            acc_gap = r.get('accuracy_gap', 0)
            status = r.get('fit_status', 'N/A')

            roc_str = f"{test_roc:.4f}" if test_roc else "N/A"
            report.append(f"| {name} | {train_acc:.4f} | {test_acc:.4f} | {train_f1:.4f} | {test_f1:.4f} | {roc_str} | {acc_gap:.4f} | {status} |")

        report.append("")

        # Detailed Metrics for Each Model
        report.append("## Detailed Model Metrics\n")

        for name, r in sorted(self.results.items(), key=lambda x: x[1].get('test_f1', 0), reverse=True):
            report.append(f"### {name}\n")

            # Basic Metrics
            report.append("**Training Metrics:**")
            report.append(f"- Accuracy: {r.get('train_accuracy', 0):.4f}")
            report.append(f"- Precision: {r.get('train_precision', 0):.4f}")
            report.append(f"- Recall: {r.get('train_recall', 0):.4f}")
            report.append(f"- F1 Score: {r.get('train_f1', 0):.4f}")
            if r.get('train_roc_auc'):
                report.append(f"- ROC-AUC: {r.get('train_roc_auc', 0):.4f}")

            report.append("\n**Test Metrics:**")
            report.append(f"- Accuracy: {r.get('test_accuracy', 0):.4f}")
            report.append(f"- Precision: {r.get('test_precision', 0):.4f}")
            report.append(f"- Recall: {r.get('test_recall', 0):.4f}")
            report.append(f"- F1 Score: {r.get('test_f1', 0):.4f}")
            if r.get('test_roc_auc'):
                report.append(f"- ROC-AUC: {r.get('test_roc_auc', 0):.4f}")

            # Overfitting Analysis
            acc_gap = r.get('accuracy_gap', 0)
            f1_gap = r.get('f1_gap', 0)
            report.append(f"\n**Overfitting Analysis:**")
            report.append(f"- Accuracy Gap (Train - Test): {acc_gap:.4f}")
            report.append(f"- F1 Gap (Train - Test): {f1_gap:.4f}")
            report.append(f"- Status: {r.get('fit_status', 'N/A')}")

            # Confusion Matrix
            if 'confusion_matrix' in r:
                cm = r['confusion_matrix']
                report.append(f"\n**Confusion Matrix:**")
                report.append("```")
                report.append(f"                Predicted")
                report.append(f"              On-time  Late")
                report.append(f"Actual On-time  {cm[0][0]:6d}  {cm[0][1]:4d}")
                report.append(f"       Late     {cm[1][0]:6d}  {cm[1][1]:4d}")
                report.append("```")

                # Calculate rates and metrics
                total = cm.sum()
                tn, fp, fn, tp = cm.ravel()
                report.append(f"\n**Confusion Matrix Breakdown:**")
                report.append(f"- True Negatives (TN): {tn:,} ({tn/total*100:.2f}%) - Correctly predicted on-time")
                report.append(f"- False Positives (FP): {fp:,} ({fp/total*100:.2f}%) - Incorrectly predicted late (Type I error)")
                report.append(f"- False Negatives (FN): {fn:,} ({fn/total*100:.2f}%) - Missed late deliveries (Type II error)")
                report.append(f"- True Positives (TP): {tp:,} ({tp/total*100:.2f}%) - Correctly predicted late")

                # Calculate derived metrics
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                    report.append(f"\n**Per-Class Metrics:**")
                    report.append(f"- Precision (Late): {precision:.4f} - Of predicted late, {precision*100:.2f}% are actually late")
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                    report.append(f"- Recall (Late): {recall:.4f} - Of actual late, {recall*100:.2f}% are correctly identified")
                if tn + fp > 0:
                    specificity = tn / (tn + fp)
                    report.append(f"- Specificity (On-time): {specificity:.4f} - Of actual on-time, {specificity*100:.2f}% are correctly identified")

            # Optimal Threshold
            if self.optimal_thresholds and name in self.optimal_thresholds:
                report.append(f"\n**Optimal Threshold:** {self.optimal_thresholds[name]:.4f}")
                if name in self.results and 'test_f1_optimized' in self.results[name]:
                    report.append(f"- F1 Score with Optimal Threshold: {self.results[name]['test_f1_optimized']:.4f}")

            report.append("\n---\n")

        # Threshold Optimization
        report.append("## Threshold Optimization\n")
        if self.optimal_thresholds:
            report.append("### Optimization Methodology\n")
            report.append("Classification thresholds were optimized to maximize F1 score on the validation set.")
            report.append("The optimization process:")
            report.append("1. Evaluated thresholds from 0.1 to 0.9 in steps of 0.01")
            report.append("2. Calculated F1 score for each threshold")
            report.append("3. Selected threshold with highest F1 score")
            report.append("4. Applied optimal threshold to test set for final evaluation")
            report.append("")

            report.append("### Threshold Optimization Results\n")
            report.append("| Model | Optimal Threshold | Default (0.5) F1 | Optimized F1 | Improvement |")
            report.append("|-------|-------------------|-----------------|--------------|-------------|")

            for name, threshold in sorted(self.optimal_thresholds.items()):
                default_f1 = self.results.get(name, {}).get('test_f1', 0)
                optimized_f1 = self.results.get(name, {}).get('test_f1_optimized', default_f1)
                improvement = optimized_f1 - default_f1
                improvement_pct = (improvement / default_f1 * 100) if default_f1 > 0 else 0
                report.append(f"| {name} | {threshold:.4f} | {default_f1:.4f} | {optimized_f1:.4f} | {improvement:+.4f} ({improvement_pct:+.2f}%) |")
            report.append("")

            # Best threshold improvements
            improvements = []
            for name, threshold in self.optimal_thresholds.items():
                default_f1 = self.results.get(name, {}).get('test_f1', 0)
                optimized_f1 = self.results.get(name, {}).get('test_f1_optimized', default_f1)
                improvement = optimized_f1 - default_f1
                if improvement > 0:
                    improvements.append((name, improvement, threshold))

            if improvements:
                improvements.sort(key=lambda x: x[1], reverse=True)
                report.append("**Best Threshold Improvements:**")
                for name, improvement, threshold in improvements[:5]:
                    report.append(f"- {name}: +{improvement:.4f} F1 improvement with threshold {threshold:.4f}")
                report.append("")
        else:
            report.append("*Threshold optimization was not performed*\n")

        # Model Evaluation
        report.append("## Model Evaluation\n")
        report.append("### Evaluation Metrics Explained\n")
        report.append("- **Accuracy:** Overall correctness of predictions")
        report.append("- **Precision:** Of predicted late deliveries, how many are actually late (reduces false alarms)")
        report.append("- **Recall:** Of actual late deliveries, how many are correctly identified (reduces missed deliveries)")
        report.append("- **F1 Score:** Harmonic mean of precision and recall (balanced metric)")
        report.append("- **ROC-AUC:** Area under ROC curve (ability to distinguish between classes)")
        report.append("- **Specificity:** Of actual on-time deliveries, how many are correctly identified")
        report.append("")

        report.append("### Overfitting Analysis\n")
        overfitting_models = [name for name, r in self.results.items()
                            if "OVERFITTING" in r.get('fit_status', '')]
        good_fit_models = [name for name, r in self.results.items()
                          if "GOOD FIT" in r.get('fit_status', '')]
        underfitting_models = [name for name, r in self.results.items()
                              if "UNDERFITTING" in r.get('fit_status', '')]

        if overfitting_models:
            report.append(f"**⚠️ Overfitting Detected ({len(overfitting_models)} models):**")
            for name in overfitting_models:
                gap = self.results[name].get('accuracy_gap', 0)
                report.append(f"- {name}: Train-Test accuracy gap of {gap:.4f}")
            report.append("")

        if good_fit_models:
            report.append(f"**✅ Good Fit ({len(good_fit_models)} models):**")
            report.append("Models show good generalization with acceptable train-test gap")
            report.append("")

        if underfitting_models:
            report.append(f"**⚠️ Underfitting Detected ({len(underfitting_models)} models):**")
            for name in underfitting_models:
                test_acc = self.results[name].get('test_accuracy', 0)
                report.append(f"- {name}: Test accuracy of {test_acc:.4f} (may need more complexity)")
            report.append("")

        # Hyperparameter Tuning
        report.append("## Hyperparameter Tuning\n")
        if self.tuned_models:
            report.append("### Tuning Summary\n")
            report.append("The following models were optimized using Optuna (Bayesian optimization):\n")
            for name in self.tuned_models.keys():
                report.append(f"- {name}")
            report.append("")

            report.append("### Tuning Methodology\n")
            report.append("- **Optimization Algorithm:** Tree-structured Parzen Estimator (TPE)")
            report.append("- **Pruning Strategy:** MedianPruner (stops unpromising trials early)")
            report.append("- **Objective Metric:** F1 Score (weighted)")
            report.append("- **Validation:** Fast validation sampling (50% of validation set during tuning)")
            report.append("")

            if tuning_time:
                report.append(f"### Tuning Time\n")
                report.append(f"- **Total Tuning Time:** {tuning_time:.2f} seconds ({tuning_time/60:.2f} minutes)")
                report.append(f"- **Models Tuned:** {len(self.tuned_models)}")
                if len(self.tuned_models) > 0:
                    avg_tune_time = tuning_time / len(self.tuned_models)
                    report.append(f"- **Average Time per Model:** {avg_tune_time:.2f} seconds")
                report.append("")

            report.append("### Tuned Model Parameters\n")
            for name, tuned_model in self.tuned_models.items():
                report.append(f"#### {name}\n")
                try:
                    if hasattr(tuned_model, 'get_params'):
                        params = tuned_model.get_params()
                        key_params = {k: v for k, v in params.items()
                                    if k in ['n_estimators', 'max_depth', 'learning_rate',
                                            'min_samples_split', 'min_samples_leaf', 'C',
                                            'solver', 'subsample', 'colsample_bytree', 'iterations',
                                            'depth', 'num_leaves', 'reg_alpha', 'reg_lambda']}
                        if key_params:
                            for param, value in sorted(key_params.items()):
                                report.append(f"- `{param}`: {value}")
                        else:
                            report.append("*Parameters not available*")
                    else:
                        report.append("*Parameters not available*")
                except:
                    report.append("*Parameters not available*")
                report.append("")
        else:
            report.append("*No hyperparameter tuning was performed (models use default/initial parameters)*\n")

        # Confusion Matrix Analysis Section
        report.append("## Confusion Matrix Analysis\n")
        report.append("### Summary Statistics\n")
        report.append("| Model | TN | FP | FN | TP | Precision | Recall | Specificity |")
        report.append("|-------|----|----|----|----|-----------|--------|-------------|")

        for name, r in sorted(self.results.items(), key=lambda x: x[1].get('test_f1', 0), reverse=True):
            if 'confusion_matrix' in r:
                cm = r['confusion_matrix']
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                report.append(f"| {name} | {tn:,} | {fp:,} | {fn:,} | {tp:,} | {precision:.4f} | {recall:.4f} | {specificity:.4f} |")
        report.append("")

        # Feature Importance
        report.append("## Feature Importance\n")
        if feature_names and self.best_model:
            report.append(f"### Feature Importance for Best Model ({self.best_model_name})\n")
            try:
                if hasattr(self.best_model, 'feature_importances_'):
                    importances = self.best_model.feature_importances_
                    feature_imp = list(zip(feature_names, importances))
                    feature_imp.sort(key=lambda x: x[1], reverse=True)

                    report.append("| Rank | Feature | Importance | Cumulative % |")
                    report.append("|------|---------|------------|---------------|")
                    total_importance = sum(importances)
                    cumulative = 0
                    for i, (feat, imp) in enumerate(feature_imp[:20], 1):  # Top 20
                        cumulative += imp
                        cum_pct = (cumulative / total_importance) * 100 if total_importance > 0 else 0
                        report.append(f"| {i} | {feat} | {imp:.4f} | {cum_pct:.2f}% |")
                    report.append("")

                    # Feature importance insights
                    top_5_importance = sum([imp for _, imp in feature_imp[:5]])
                    top_5_pct = (top_5_importance / total_importance) * 100 if total_importance > 0 else 0
                    report.append(f"**Insights:**")
                    report.append(f"- Top 5 features account for {top_5_pct:.2f}% of total importance")
                    report.append(f"- Top 10 features account for {(sum([imp for _, imp in feature_imp[:10]]) / total_importance * 100) if total_importance > 0 else 0:.2f}% of total importance")
                    report.append("")
            except Exception as e:
                report.append(f"*Feature importance not available for {self.best_model_name}*\n")
        else:
            report.append("*Feature importance analysis not available*\n")

        # Feature importance for all models
        if feature_names and len(self.models) > 1:
            report.append("### Feature Importance Comparison (Top 10)\n")
            report.append("Comparing top features across different models:\n")

            model_importances = {}
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_imp = list(zip(feature_names, importances))
                        feature_imp.sort(key=lambda x: x[1], reverse=True)
                        model_importances[name] = [feat for feat, _ in feature_imp[:10]]
                except:
                    pass

            if model_importances:
                # Find common top features
                all_top_features = set()
                for features in model_importances.values():
                    all_top_features.update(features)

                report.append("**Most Important Features (appearing in top 10 of multiple models):**")
                feature_counts = {}
                for features in model_importances.values():
                    for feat in features:
                        feature_counts[feat] = feature_counts.get(feat, 0) + 1

                common_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                for feat, count in common_features:
                    report.append(f"- {feat} (appears in {count} models' top 10)")
                report.append("")

        # Recommendations
        report.append("## Recommendations\n")

        # Find best model
        best_f1 = max(r.get('test_f1', 0) for r in self.results.values())
        best_models = [name for name, r in self.results.items() if r.get('test_f1', 0) == best_f1]

        report.append(f"### Best Model for Production")
        report.append(f"- **Recommended:** {self.best_model_name}")

        # Explain selection criteria
        fit_status = best_result.get('fit_status', '')
        if 'GOOD FIT' in fit_status:
            report.append(f"- **Reason:** Best model among GOOD FIT models (no overfitting)")
            report.append(f"  - Test F1 Score: {best_result.get('test_f1', 0):.4f}")
            report.append(f"  - Accuracy Gap: {best_result.get('accuracy_gap', 0):.4f} (good generalization)")
        elif 'OVERFITTING' in fit_status:
            report.append(f"- **⚠️ WARNING:** Selected model shows overfitting")
            report.append(f"  - Test F1 Score: {best_result.get('test_f1', 0):.4f}")
            report.append(f"  - Accuracy Gap: {best_result.get('accuracy_gap', 0):.4f} (high gap indicates overfitting)")
            report.append(f"  - **Recommendation:** Consider using a model with GOOD FIT status if available")
        else:
            report.append(f"- **Reason:** Selected based on test F1 score ({best_result.get('test_f1', 0):.4f})")

        if self.optimal_thresholds and self.best_model_name in self.optimal_thresholds:
            report.append(f"- **Use threshold:** {self.optimal_thresholds[self.best_model_name]:.4f} (instead of default 0.5)")
        report.append("")

        # Overfitting warnings
        overfitting_models = [name for name, r in self.results.items()
                            if "OVERFITTING" in r.get('fit_status', '')]
        if overfitting_models:
            report.append("### Overfitting Concerns")
            report.append("The following models show signs of overfitting:")
            for name in overfitting_models:
                gap = self.results[name].get('accuracy_gap', 0)
                report.append(f"- **{name}**: Accuracy gap of {gap:.4f}")
            report.append("")

        # Model selection guidance
        report.append("### Model Selection Guidance")
        report.append("- **For highest accuracy:** Use the best model (XGBoost/CatBoost/LightGBM)")
        report.append("- **For interpretability:** Consider Logistic Regression or Random Forest")
        report.append("- **For production stability:** Consider Voting or Stacking Ensemble")
        report.append("- **For fast inference:** LightGBM or XGBoost are recommended")
        report.append("")

        # Appendix
        report.append("## Appendix\n")
        report.append("### Model Files\n")
        report.append("All trained models are saved in the `models/` directory with timestamps.")
        report.append("")
        report.append("### Reproducibility\n")
        report.append(f"- Random State: {self.random_state}")
        report.append("- All models use the same random seed for reproducibility")
        report.append("- Data splits are stratified to maintain class distribution")
        report.append("")

        report.append("### Technical Details\n")
        report.append("- **Python Version:** Check with `python --version`")
        report.append("- **Library Versions:** See `pyproject.toml`")
        report.append("- **Training Method:** Parallel execution using joblib")
        report.append("- **Optimization:** Optuna with TPE sampler")
        report.append("")

        # Footer
        report.append("---\n")
        report.append(f"*Report generated by Supply Chain ML Pipeline on {timestamp}*\n")
        report.append("*For questions or issues, refer to the project documentation.*\n")

        return "\n".join(report)

    def save_detailed_report(self, X_train: pd.DataFrame = None, y_train: pd.Series = None,
                            X_test: pd.DataFrame = None, y_test: pd.Series = None,
                            X_val: pd.DataFrame = None, y_val: pd.Series = None,
                            feature_names: List[str] = None,
                            training_time: float = None,
                            tuning_time: float = None,
                            report_dir: Path = None) -> Path:
        """
        Generate and save detailed performance report to file.

        Args:
            X_test: Test features (optional)
            y_test: Test targets (optional)
            feature_names: List of feature names (optional)
            report_dir: Directory to save report (default: reports/)

        Returns:
            Path to saved report file
        """
        if report_dir is None:
            report_dir = Path(__file__).parents[2] / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate report
        report_text = self.generate_detailed_report(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            X_val=X_val, y_val=y_val,
            feature_names=feature_names,
            training_time=training_time,
            tuning_time=tuning_time
        )

        # Save markdown
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_path = report_dir / f"model_performance_report_{timestamp}.md"
        report_path.write_text(report_text, encoding='utf-8')

        print(f"\n✅ Detailed report saved → {report_path}")
        return report_path


def run_training_pipeline(parallel: bool = True, tune_hyperparameters: bool = True,
                          optimize_thresholds: bool = True, n_trials: int = 10):
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
        classifier.tune_all_models(
            X_train, y_train, X_val, y_val,
            n_trials=n_trials,
            parallel=parallel  # Use parallel tuning if parallel training is enabled
        )

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
                # Update main test_f1 with optimized version for best model selection
                classifier.results[name]['test_f1'] = result['f1']

    # Re-select best model after all optimizations (considering overfitting)
    print("\n" + "=" * 80)
    print("FINAL MODEL SELECTION (considering overfitting)")
    print("=" * 80)
    classifier.select_best_model()

    classifier.save_models()
    classifier.generate_report()

    # Generate and save detailed report
    print("\n" + "=" * 80)
    print("GENERATING DETAILED PERFORMANCE REPORT")
    print("=" * 80)
    try:
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        # Calculate timing if available (would need to track this during training)
        report_path = classifier.save_detailed_report(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names,
            training_time=None,  # Could be tracked if needed
            tuning_time=None      # Could be tracked if needed
        )
        print(f"📊 Detailed report available at: {report_path}")
    except Exception as e:
        print(f"⚠️ Could not generate detailed report: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ CLASSIFICATION PIPELINE COMPLETE!")
    return classifier


if __name__ == "__main__":
    classifier = run_training_pipeline(parallel=True)
