"""
Enhanced Classification Models with CatBoost, LightGBM, and Optuna Optimization
Extends the base classifier with modern boosting methods and hyperparameter tuning.

Features:
- CatBoost (native categorical handling)
- LightGBM (fast training, memory efficient)
- Optuna hyperparameter optimization
- SMOTE comparison for imbalanced data
- Comprehensive evaluation plots

Usage:
    from src.models.classifier_optimized import OptimizedSupplyChainClassifier
    classifier = OptimizedSupplyChainClassifier()
    classifier.initialize_models(include_ensemble=True, include_modern_boosting=True)
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
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc,
    f1_score, accuracy_score, precision_score, recall_score,
    precision_recall_curve, balanced_accuracy_score
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
import joblib
from datetime import datetime
from typing import Dict, Tuple, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional: SMOTE for imbalanced data
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Optional: Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class OptimizedSupplyChainClassifier:
    """
    Enhanced classifier with modern boosting methods and optimization.

    New Models:
    - CatBoost: Native categorical handling, excellent performance
    - LightGBM: Fast training, memory efficient
    - Enhanced Stacking: Combines all best performers

    New Features:
    - Optuna hyperparameter optimization
    - SMOTE vs class weights comparison
    - Comprehensive evaluation plots
    """

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize optimized classifier.

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
        self.optimization_studies: Dict[str, Any] = {}
        self.model_dir = Path(__file__).parents[2] / "models"
        self.model_dir.mkdir(exist_ok=True)

    def initialize_models(self, include_ensemble: bool = True,
                         include_modern_boosting: bool = True):
        """
        Initialize classification models with modern boosting algorithms.

        Args:
            include_ensemble: Include Voting and Stacking ensembles
            include_modern_boosting: Include CatBoost and LightGBM
        """
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
                algorithm='SAMME'
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=self.n_jobs
            ),
        }

        # Add modern boosting methods
        if include_modern_boosting:
            self.models['CatBoost'] = CatBoostClassifier(
                iterations=200,
                learning_rate=0.1,
                depth=6,
                random_state=self.random_state,
                verbose=False,
                auto_class_weights='Balanced',
                thread_count=self.n_jobs if self.n_jobs > 0 else -1
            )

            self.models['LightGBM'] = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=self.n_jobs,
                verbose=-1
            )

        # Add ensemble methods
        if include_ensemble:
            # Voting Ensemble with modern boosting
            voting_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                              random_state=self.random_state, n_jobs=self.n_jobs)),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                  random_state=self.random_state)),
                ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                            random_state=self.random_state, n_jobs=self.n_jobs))
            ]

            if include_modern_boosting:
                voting_estimators.append(
                    ('xgb', XGBClassifier(n_estimators=100, max_depth=6,
                                         random_state=self.random_state, n_jobs=self.n_jobs))
                )

            self.models['Voting Ensemble'] = VotingClassifier(
                estimators=voting_estimators,
                voting='soft', n_jobs=self.n_jobs
            )

            # Enhanced Stacking Ensemble with all best models
            stacking_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                              random_state=self.random_state, n_jobs=self.n_jobs)),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                  random_state=self.random_state)),
                ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                            random_state=self.random_state, n_jobs=self.n_jobs)),
            ]

            if include_modern_boosting:
                stacking_estimators.extend([
                    ('xgb', XGBClassifier(n_estimators=100, max_depth=6,
                                         random_state=self.random_state, n_jobs=self.n_jobs)),
                    ('lgbm', LGBMClassifier(n_estimators=100, max_depth=8,
                                           random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1)),
                    ('catboost', CatBoostClassifier(iterations=100, depth=6,
                                                   random_state=self.random_state, verbose=False))
                ])

            self.models['Stacking Ensemble'] = StackingClassifier(
                estimators=stacking_estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3, n_jobs=self.n_jobs
            )

        print(f"✅ Initialized {len(self.models)} classification models")
        print(f"   Parallel jobs: {self.n_jobs} (-1 = all cores)")
        for name in self.models.keys():
            print(f"   • {name}")

    def optimize_model_with_optuna(self, model_name: str, X_train: pd.DataFrame,
                                   y_train: pd.Series, n_trials: int = 100,
                                   timeout: int = 600) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using Optuna.

        Args:
            model_name: Name of model to optimize ('XGBoost', 'CatBoost', 'LightGBM')
            X_train: Training features
            y_train: Training target
            n_trials: Number of optimization trials
            timeout: Timeout in seconds

        Returns:
            Dictionary with best parameters and study
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available. Install with: pip install optuna")
            return {}

        print(f"\n🔍 Optimizing {model_name} with Optuna (TPE Bayesian Search)...")
        print(f"   Trials: {n_trials}, Timeout: {timeout}s")

        def objective(trial):
            if model_name == 'XGBoost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
                model = XGBClassifier(**params, random_state=self.random_state, n_jobs=1)

            elif model_name == 'CatBoost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
                }
                model = CatBoostClassifier(**params, random_state=self.random_state,
                                          verbose=False, thread_count=1)

            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
                model = LGBMClassifier(**params, random_state=self.random_state,
                                      n_jobs=1, verbose=-1)
            else:
                raise ValueError(f"Optimization not supported for {model_name}")

            # Cross-validation score
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='f1_weighted', n_jobs=1)

            return scores.mean()

        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout,
                      show_progress_bar=False, n_jobs=1)

        print(f"\n🏆 Best F1 Score: {study.best_value:.4f}")
        print(f"📊 Best Parameters:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value}")

        self.optimization_studies[model_name] = study

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }

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
        test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

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
            'test_balanced_accuracy': test_balanced_acc,
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

    def _print_model_result(self, result: Dict):
        """Print single model result."""
        print(f"\n{result['model_name']} Performance:")
        print(f"  {'Metric':<20} {'Train':>10} {'Test':>10} {'Gap':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Accuracy':<20} {result['train_accuracy']:>10.4f} {result['test_accuracy']:>10.4f} {result['accuracy_gap']:>10.4f}")
        print(f"  {'F1 Score':<20} {result['train_f1']:>10.4f} {result['test_f1']:>10.4f}")
        print(f"  {'Balanced Accuracy':<20} {'N/A':>10} {result['test_balanced_accuracy']:>10.4f}")
        if result['test_roc_auc']:
            print(f"  {'ROC-AUC':<20} {result['train_roc_auc']:>10.4f} {result['test_roc_auc']:>10.4f}")
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

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get comparison DataFrame of all models."""
        data = [{
            'Model': n,
            'Train Accuracy': r['train_accuracy'],
            'Test Accuracy': r['test_accuracy'],
            'Train F1': r['train_f1'],
            'Test F1': r['test_f1'],
            'Test ROC-AUC': r['test_roc_auc'],
            'Balanced Accuracy': r['test_balanced_accuracy'],
            'Accuracy Gap': r['accuracy_gap'],
            'Fit Status': r['fit_status']
        } for n, r in self.results.items()]
        return pd.DataFrame(data).sort_values('Test F1', ascending=False)

    def save_models(self):
        """Save all trained models."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        for name, model in self.models.items():
            path = self.model_dir / f"{name.replace(' ', '_').lower()}_{ts}.pkl"
            joblib.dump(model, path)
            print(f"  ✅ Saved {name} → {path.name}")

        # Save best model
        best_path = self.model_dir / f"best_model_{ts}.pkl"
        joblib.dump(self.best_model, best_path)
        print(f"  ✅ Saved best model → {best_path.name}")

        # Save results and optimization studies
        joblib.dump(self.results, self.model_dir / f"training_results_{ts}.pkl")
        if self.optimization_studies:
            joblib.dump(self.optimization_studies, self.model_dir / f"optuna_studies_{ts}.pkl")

    def compare_balancing_strategies(self, X_train: pd.DataFrame, y_train: pd.Series,
                                    X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare class weights vs SMOTE vs hybrid approach.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            DataFrame with comparison results
        """
        if not SMOTE_AVAILABLE:
            print("⚠️ SMOTE not available. Install with: pip install imbalanced-learn")
            return pd.DataFrame()

        print("\n📊 COMPARING BALANCING STRATEGIES...")
        print("=" * 60)

        results = {}

        # Strategy 1: Class weights (current approach)
        print("\n1️⃣ Training with CLASS WEIGHTS...")
        model_cw = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced',
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        model_cw.fit(X_train, y_train)
        y_pred_cw = model_cw.predict(X_test)
        results['Class Weights'] = {
            'F1 (weighted)': f1_score(y_test, y_pred_cw, average='weighted'),
            'F1 (macro)': f1_score(y_test, y_pred_cw, average='macro'),
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_cw),
            'Accuracy': accuracy_score(y_test, y_pred_cw)
        }

        # Strategy 2: SMOTE
        print("2️⃣ Training with SMOTE...")
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        model_smote = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        model_smote.fit(X_train_smote, y_train_smote)
        y_pred_smote = model_smote.predict(X_test)
        results['SMOTE'] = {
            'F1 (weighted)': f1_score(y_test, y_pred_smote, average='weighted'),
            'F1 (macro)': f1_score(y_test, y_pred_smote, average='macro'),
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_smote),
            'Accuracy': accuracy_score(y_test, y_pred_smote)
        }

        # Strategy 3: Hybrid (SMOTE + class weights)
        print("3️⃣ Training with SMOTE + CLASS WEIGHTS...")
        model_hybrid = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced',
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        model_hybrid.fit(X_train_smote, y_train_smote)
        y_pred_hybrid = model_hybrid.predict(X_test)
        results['SMOTE + Class Weights'] = {
            'F1 (weighted)': f1_score(y_test, y_pred_hybrid, average='weighted'),
            'F1 (macro)': f1_score(y_test, y_pred_hybrid, average='macro'),
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_hybrid),
            'Accuracy': accuracy_score(y_test, y_pred_hybrid)
        }

        # Create comparison DataFrame
        df_comparison = pd.DataFrame(results).T

        print("\n📊 BALANCING STRATEGY COMPARISON RESULTS:")
        print("=" * 60)
        print(df_comparison.to_string())

        # Determine best strategy
        best_strategy = df_comparison['F1 (weighted)'].idxmax()
        print(f"\n🏆 Best Strategy: {best_strategy}")
        print(f"   F1 (weighted): {df_comparison.loc[best_strategy, 'F1 (weighted)']:.4f}")

        return df_comparison


def run_optimized_training_pipeline(parallel: bool = True, optimize: bool = False,
                                   optimize_n_trials: int = 50):
    """
    Complete optimized classification training pipeline.

    Args:
        parallel: If True, use parallel training
        optimize: If True, optimize hyperparameters with Optuna
        optimize_n_trials: Number of Optuna trials
    """
    print("\n" + "=" * 80 + "\nOPTIMIZED CLASSIFICATION TRAINING PIPELINE\n" + "=" * 80)
    from src.data.preprocess import load_and_preprocess
    from src.features.build_features import build_features_pipeline

    df = load_and_preprocess()
    X, y = build_features_pipeline(df)

    classifier = OptimizedSupplyChainClassifier(random_state=42, n_jobs=-1)
    classifier.initialize_models(include_ensemble=True, include_modern_boosting=True)

    X_train, X_test, y_train, y_test = classifier.split_data(X, y)

    # Optuna optimization (optional)
    if optimize and OPTUNA_AVAILABLE:
        print("\n🔍 HYPERPARAMETER OPTIMIZATION PHASE")
        print("=" * 60)
        for model_name in ['XGBoost', 'CatBoost', 'LightGBM']:
            if model_name in classifier.models:
                result = classifier.optimize_model_with_optuna(
                    model_name, X_train, y_train, n_trials=optimize_n_trials, timeout=300
                )
                # Update model with best parameters
                if result and 'best_params' in result:
                    if model_name == 'XGBoost':
                        classifier.models[model_name] = XGBClassifier(
                            **result['best_params'], random_state=42, n_jobs=-1
                        )
                    elif model_name == 'CatBoost':
                        classifier.models[model_name] = CatBoostClassifier(
                            **result['best_params'], random_state=42, verbose=False
                        )
                    elif model_name == 'LightGBM':
                        classifier.models[model_name] = LGBMClassifier(
                            **result['best_params'], random_state=42, n_jobs=-1, verbose=-1
                        )

    # Train models
    if parallel:
        classifier.train_all_models_parallel(X_train, y_train, X_test, y_test)
    else:
        # Sequential training for debugging
        for name, model in classifier.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            result = classifier._evaluate_single_model(name, model, X_train, y_train, X_test, y_test)
            classifier.results[name] = result
            classifier._print_model_result(result)

    # Compare balancing strategies
    if SMOTE_AVAILABLE:
        print("\n" + "=" * 80)
        classifier.compare_balancing_strategies(X_train, y_train, X_test, y_test)

    # Save models
    classifier.save_models()

    # Print comparison DataFrame
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)
    print(classifier.get_comparison_dataframe().to_string())

    print("\n✅ OPTIMIZED CLASSIFICATION PIPELINE COMPLETE!")
    return classifier


if __name__ == "__main__":
    classifier = run_optimized_training_pipeline(parallel=True, optimize=False)
