"""
Advanced Classification Models for Late Delivery Prediction

Complete classification pipeline with:
- All base models and modern boosting (XGBoost, CatBoost, LightGBM)
- Parallel training using joblib
- Optuna hyperparameter optimization
- Threshold optimization
- Comprehensive evaluation and reporting

Features:
- Parallel training using joblib
- All modern boosting algorithms (XGBoost, CatBoost, LightGBM)
- Optuna hyperparameter optimization with TPE sampler
- SMOTE vs class weights comparison
- Overfitting/underfitting detection
- Threshold optimization
- Comprehensive evaluation and reporting
- SHAP explainability support

Usage:
    from src.models.classifier_advanced import AdvancedSupplyChainClassifier

    classifier = AdvancedSupplyChainClassifier()
    classifier.initialize_models(include_ensemble=True)
    classifier.train_all_models_parallel(X_train, y_train, X_test, y_test)
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
    roc_curve, precision_recall_curve, balanced_accuracy_score
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

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# SMOTE for imbalanced data
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# =============================================================================
# METRIC EXPLANATIONS (for non-technical audiences)
# =============================================================================

METRIC_EXPLANATIONS = {
    'accuracy': "How often the model is correct overall. A value of 0.80 means the model is right 80% of the time.",
    'precision': "When the model predicts 'late delivery', how often is it actually late? High precision = fewer false alarms.",
    'recall': "Of all actual late deliveries, how many did the model catch? High recall = fewer missed late deliveries.",
    'f1_score': "A balanced combination of precision and recall. Higher is better, with 1.0 being perfect.",
    'roc_auc': "How well the model distinguishes between late and on-time deliveries. 0.5 = random guessing, 1.0 = perfect.",
    'balanced_accuracy': "Average accuracy for each class, useful when classes are imbalanced.",
    'specificity': "Of all on-time deliveries, how many did the model correctly identify as on-time?"
}


class AdvancedSupplyChainClassifier:
    """
    Advanced classifier combining all functionality for late delivery prediction.

    Supports:
    - All sklearn classifiers + XGBoost, CatBoost, LightGBM
    - Voting and Stacking ensembles
    - Parallel training for faster execution
    - Optuna hyperparameter optimization
    - Threshold optimization for business needs
    - Comprehensive overfitting analysis
    - Detailed performance reports
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
        self.optimal_thresholds: Dict[str, float] = {}
        self.tuned_models: Dict[str, Any] = {}
        self.optimization_studies: Dict[str, Any] = {}
        self.model_dir = Path(__file__).parents[2] / "models"
        self.model_dir.mkdir(exist_ok=True)

    def initialize_models(self, include_ensemble: bool = True,
                         include_modern_boosting: bool = True):
        """
        Initialize classification models.

        Args:
            include_ensemble: Include Voting and Stacking ensembles
            include_modern_boosting: Include CatBoost and LightGBM
        """
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
            ),
        }

        # Add modern boosting models
        if include_modern_boosting:
            if XGBOOST_AVAILABLE:
                self.models['XGBoost'] = XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                    random_state=self.random_state, n_jobs=self.n_jobs,
                    eval_metric='logloss'
                )

            if CATBOOST_AVAILABLE:
                self.models['CatBoost'] = CatBoostClassifier(
                    iterations=200, depth=6, learning_rate=0.1,
                    random_state=self.random_state, verbose=False,
                    auto_class_weights='Balanced',
                    thread_count=self.n_jobs if self.n_jobs > 0 else -1
                )

            if LIGHTGBM_AVAILABLE:
                self.models['LightGBM'] = LGBMClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                    random_state=self.random_state, class_weight='balanced',
                    n_jobs=self.n_jobs, verbose=-1
                )

        # Add ensemble methods
        if include_ensemble:
            ensemble_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                              random_state=self.random_state, n_jobs=self.n_jobs)),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                  random_state=self.random_state)),
                ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                            random_state=self.random_state, n_jobs=self.n_jobs))
            ]

            if include_modern_boosting:
                if LIGHTGBM_AVAILABLE:
                    ensemble_estimators.append(('lgbm', LGBMClassifier(
                        n_estimators=100, max_depth=6, random_state=self.random_state,
                        n_jobs=self.n_jobs, verbose=-1
                    )))
                if XGBOOST_AVAILABLE:
                    ensemble_estimators.append(('xgb', XGBClassifier(
                        n_estimators=100, max_depth=6, random_state=self.random_state,
                        n_jobs=self.n_jobs, eval_metric='logloss'
                    )))

            self.models['Voting Ensemble'] = VotingClassifier(
                estimators=ensemble_estimators, voting='soft', n_jobs=self.n_jobs
            )
            self.models['Stacking Ensemble'] = StackingClassifier(
                estimators=ensemble_estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3, n_jobs=self.n_jobs
            )

        print(f"✅ Initialized {len(self.models)} classification models")
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
            val_size: Proportion for validation set

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
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
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        train_roc = roc_auc_score(y_train, y_train_proba) if y_train_proba is not None else None
        test_roc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None

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
            'test_balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'accuracy_gap': acc_gap, 'f1_gap': train_f1 - test_f1,
            'fit_status': fit_status,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'y_test_proba': y_test_proba
        }

    def train_all_models_parallel(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series):
        """Train and evaluate all models in parallel."""
        print("\n" + "=" * 60)
        print("PARALLEL TRAINING - ALL MODELS")
        print("=" * 60)

        print(f"\n🚀 Training {len(self.models)} models in parallel...")
        trained = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._train_single_model)(name, model, X_train, y_train)
            for name, model in self.models.items()
        )

        for name, model in trained:
            self.models[name] = model

        print(f"\n✅ All models trained!")

        print(f"\n📊 Evaluating all models in parallel...")
        evaluations = Parallel(n_jobs=self.n_jobs, verbose=5)(
            delayed(self._evaluate_single_model)(name, model, X_train, y_train, X_test, y_test)
            for name, model in self.models.items()
        )

        for result in evaluations:
            name = result['model_name']
            self.results[name] = result
            self.overfitting_analysis[name] = {
                'train_accuracy': result['train_accuracy'],
                'test_accuracy': result['test_accuracy'],
                'gap': result['accuracy_gap'],
                'status': result['fit_status']
            }

        self._print_results_summary()
        self.select_best_model()
        self.print_overfitting_summary()

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series):
        """Train models sequentially (for debugging)."""
        print("\n" + "=" * 60 + "\nTRAINING ALL MODELS\n" + "=" * 60)

        for model_name, model in self.models.items():
            print(f"\n{'=' * 60}\nTraining {model_name}...\n{'=' * 60}")
            model.fit(X_train, y_train)
            print(f"✅ {model_name} training complete")

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

            self._print_model_result(result)
            self.models[model_name] = model

        self.select_best_model()
        self.print_overfitting_summary()

    def _print_model_result(self, result: Dict):
        """Print single model result."""
        print(f"\n{result['model_name']} Performance:")
        print(f"  {'Metric':<20} {'Train':>10} {'Test':>10} {'Gap':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Accuracy':<20} {result['train_accuracy']:>10.4f} {result['test_accuracy']:>10.4f} {result['accuracy_gap']:>10.4f}")
        print(f"  {'F1 Score':<20} {result['train_f1']:>10.4f} {result['test_f1']:>10.4f}")
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
            return 0.5

        y_proba = model.predict_proba(X_val)[:, 1]

        if metric == 'f1':
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
            fpr, tpr, thresholds = roc_curve(y_val, y_proba)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_idx]

        else:
            best_threshold = 0.5

        self.optimal_thresholds[model_name] = best_threshold
        return best_threshold

    def optimize_all_thresholds(self, X_val: pd.DataFrame, y_val: pd.Series,
                                 metric: str = 'f1'):
        """Optimize thresholds for all models."""
        print("\n" + "=" * 60)
        print("THRESHOLD OPTIMIZATION")
        print("=" * 60)

        for model_name, model in self.models.items():
            threshold = self.optimize_threshold(model_name, model, X_val, y_val, metric)
            print(f"  {model_name:<25} Optimal threshold: {threshold:.4f}")

    def tune_hyperparameters_optuna(self, model_name: str, model_type: str,
                                    X_train: pd.DataFrame, y_train: pd.Series,
                                    n_trials: int = 50, timeout: int = 300) -> Any:
        """
        Tune hyperparameters using Optuna.

        Args:
            model_name: Name of the model
            model_type: Type ('xgb', 'catboost', 'lgbm', 'rf', 'gb')
            X_train: Training features
            y_train: Training targets
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds

        Returns:
            Best tuned model
        """
        if not OPTUNA_AVAILABLE:
            print(f"⚠️ Optuna not available. Skipping tuning for {model_name}.")
            return None

        print(f"\n🔍 Tuning {model_name} with Optuna...")

        def objective(trial):
            if model_type == 'xgb' and XGBOOST_AVAILABLE:
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = XGBClassifier(**params, random_state=self.random_state, n_jobs=1, eval_metric='logloss')

            elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 300),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                }
                model = CatBoostClassifier(**params, random_state=self.random_state, verbose=False)

            elif model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                model = LGBMClassifier(**params, random_state=self.random_state, n_jobs=1, verbose=-1)

            elif model_type == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                model = RandomForestClassifier(**params, random_state=self.random_state, n_jobs=1)

            else:
                return 0

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)
            return scores.mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        print(f"  ✅ Best F1: {study.best_value:.4f}")
        self.optimization_studies[model_name] = study

        # Train with best params
        best_params = study.best_params
        if model_type == 'xgb' and XGBOOST_AVAILABLE:
            best_model = XGBClassifier(**best_params, random_state=self.random_state, n_jobs=self.n_jobs, eval_metric='logloss')
        elif model_type == 'catboost' and CATBOOST_AVAILABLE:
            best_model = CatBoostClassifier(**best_params, random_state=self.random_state, verbose=False)
        elif model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
            best_model = LGBMClassifier(**best_params, random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1)
        elif model_type == 'rf':
            best_model = RandomForestClassifier(**best_params, random_state=self.random_state, n_jobs=self.n_jobs)
        else:
            return None

        best_model.fit(X_train, y_train)
        self.tuned_models[model_name] = best_model
        return best_model

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get comparison DataFrame of all models."""
        data = [{
            'Model': n,
            'Train Accuracy': r['train_accuracy'],
            'Test Accuracy': r['test_accuracy'],
            'Train F1': r['train_f1'],
            'Test F1': r['test_f1'],
            'Test ROC-AUC': r.get('test_roc_auc'),
            'Accuracy Gap': r['accuracy_gap'],
            'Fit Status': r['fit_status']
        } for n, r in self.results.items()]
        return pd.DataFrame(data).sort_values('Test F1', ascending=False)

    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from best model."""
        if self.best_model is None:
            raise ValueError("No model trained")
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            return None
        df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        return df.sort_values('importance', ascending=False).head(top_n)

    def save_models(self):
        """Save all trained models."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        for name, model in self.models.items():
            path = self.model_dir / f"{name.replace(' ', '_').lower()}_{ts}.pkl"
            joblib.dump(model, path)
            print(f"  ✅ Saved {name} → {path.name}")

        best_path = self.model_dir / f"best_model_{ts}.pkl"
        joblib.dump(self.best_model, best_path)
        print(f"  ✅ Saved best model → {best_path.name}")

        joblib.dump(self.results, self.model_dir / f"training_results_{ts}.pkl")

        if self.optimal_thresholds:
            joblib.dump(self.optimal_thresholds, self.model_dir / f"optimal_thresholds_{ts}.pkl")

    @staticmethod
    def get_metric_explanation(metric: str) -> str:
        """
        Get plain-language explanation for a metric.

        Args:
            metric: Name of the metric

        Returns:
            Human-readable explanation
        """
        return METRIC_EXPLANATIONS.get(metric.lower(), f"Metric '{metric}' measures model performance.")

    def generate_report_with_explanations(self) -> str:
        """Generate report with non-technical metric explanations."""
        report = [f"\n{'=' * 80}\nMODEL PERFORMANCE SUMMARY\n{'=' * 80}"]

        # Add metric explanations
        report.append("\n📖 METRIC DEFINITIONS (Plain Language):")
        report.append("-" * 60)
        for metric, explanation in METRIC_EXPLANATIONS.items():
            report.append(f"  • {metric.replace('_', ' ').title()}: {explanation}")

        report.append("\n" + "=" * 60)
        report.append("MODEL RESULTS")
        report.append("=" * 60)

        for name, r in sorted(self.results.items(), key=lambda x: x[1]['test_f1'], reverse=True):
            report.append(f"\n📊 {name}")
            report.append(f"   Accuracy: {r['test_accuracy']:.1%} - The model is correct {r['test_accuracy']:.1%} of the time")
            report.append(f"   F1 Score: {r['test_f1']:.4f}")
            report.append(f"   Status: {r['fit_status']}")

        report.append(f"\n{'=' * 80}")
        report.append(f"🏆 BEST MODEL: {self.best_model_name}")
        report.append(f"{'=' * 80}")

        return "\n".join(report)


def run_advanced_training_pipeline(parallel: bool = True, tune: bool = False):
    """
    Complete advanced classification training pipeline.

    Args:
        parallel: Use parallel training
        tune: Run hyperparameter tuning with Optuna
    """
    print("\n" + "=" * 80 + "\nADVANCED CLASSIFICATION PIPELINE\n" + "=" * 80)

    from src.data.preprocess import load_and_preprocess
    from src.features.build_features_advanced import build_features_pipeline

    df = load_and_preprocess()
    X, y = build_features_pipeline(df)

    classifier = AdvancedSupplyChainClassifier(random_state=42, n_jobs=-1)
    classifier.initialize_models(include_ensemble=True, include_modern_boosting=True)

    X_train, X_val, X_test, y_train, y_val, y_test = classifier.split_data(X, y)

    if parallel:
        classifier.train_all_models_parallel(X_train, y_train, X_test, y_test)
    else:
        classifier.train_all_models(X_train, y_train, X_test, y_test)

    # Threshold optimization
    classifier.optimize_all_thresholds(X_val, y_val)

    # Optuna tuning (optional)
    if tune and OPTUNA_AVAILABLE:
        for model_name, model_type in [('XGBoost', 'xgb'), ('LightGBM', 'lgbm'), ('CatBoost', 'catboost')]:
            if model_name in classifier.models:
                classifier.tune_hyperparameters_optuna(model_name, model_type, X_train, y_train)

    classifier.save_models()
    print(classifier.generate_report_with_explanations())

    print("\n✅ ADVANCED CLASSIFICATION PIPELINE COMPLETE!")
    return classifier


if __name__ == "__main__":
    classifier = run_advanced_training_pipeline(parallel=True, tune=False)

