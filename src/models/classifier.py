"""
Classification Models for Late Delivery Prediction
Trains ensemble and base models with overfitting detection.

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
    - Decision Tree
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - AdaBoost
    - Voting Ensemble
    - Stacking Ensemble
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
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
                class_weight='balanced', solver='lbfgs'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, min_samples_split=10, min_samples_leaf=5,
                random_state=self.random_state, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=4, max_features='sqrt',
                random_state=self.random_state, class_weight='balanced', n_jobs=-1
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=4, random_state=self.random_state,
                class_weight='balanced', n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                min_samples_split=10, min_samples_leaf=4, subsample=0.8,
                random_state=self.random_state
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, learning_rate=0.1, random_state=self.random_state
            )
        }

        if include_ensemble:
            self.models['Voting Ensemble'] = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                                  random_state=self.random_state, n_jobs=-1)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                      random_state=self.random_state)),
                    ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                                random_state=self.random_state, n_jobs=-1))
                ],
                voting='soft'
            )
            self.models['Stacking Ensemble'] = StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                                  random_state=self.random_state, n_jobs=-1)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                      random_state=self.random_state)),
                    ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                                random_state=self.random_state, n_jobs=-1))
                ],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3
            )

        print(f"✅ Initialized {len(self.models)} classification models")
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

    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
        """Train a single model."""
        print(f"\n{'=' * 60}\nTraining {model_name}...\n{'=' * 60}")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        print(f"✅ {model_name} training complete")
        return model

    def evaluate_model(self, model_name: str, model: Any, X_train: pd.DataFrame,
                       y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model with overfitting detection."""
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

        # CV score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')

        # Overfitting check
        acc_gap = train_acc - test_acc
        fit_status = "⚠️ OVERFITTING" if acc_gap > 0.05 else ("⚠️ UNDERFITTING" if acc_gap < -0.02 else "✅ GOOD FIT")

        results = {
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
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'cv_score': cv_scores.mean(), 'cv_std': cv_scores.std()
        }
        self.results[model_name] = results
        self.overfitting_analysis[model_name] = {
            'train_accuracy': train_acc, 'test_accuracy': test_acc,
            'gap': acc_gap, 'status': fit_status
        }

        # Print
        print(f"\n{model_name} Performance:")
        print(f"  {'Metric':<15} {'Train':>10} {'Test':>10} {'Gap':>10}")
        print(f"  {'-'*45}")
        print(f"  {'Accuracy':<15} {train_acc:>10.4f} {test_acc:>10.4f} {acc_gap:>10.4f}")
        print(f"  {'F1 Score':<15} {train_f1:>10.4f} {test_f1:>10.4f}")
        print(f"  CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  Status: {fit_status}")
        return results

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series):
        """Train and evaluate all models."""
        print("\n" + "=" * 60 + "\nTRAINING ALL MODELS\n" + "=" * 60)
        for model_name in self.models.keys():
            trained_model = self.train_model(model_name, X_train, y_train)
            self.evaluate_model(model_name, trained_model, X_train, y_train, X_test, y_test)
            self.models[model_name] = trained_model
        self.select_best_model()
        self.print_overfitting_summary()

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
            'CV F1': r['cv_score'], 'CV Std': r['cv_std'], 'Accuracy Gap': r['accuracy_gap'],
            'Fit Status': r['fit_status']
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


def run_training_pipeline():
    """Complete classification training pipeline."""
    print("\n" + "=" * 80 + "\nCLASSIFICATION TRAINING PIPELINE\n" + "=" * 80)
    from src.data.preprocess import load_and_preprocess
    from src.features.build_features import build_features_pipeline

    df = load_and_preprocess()
    X, y = build_features_pipeline(df)

    classifier = SupplyChainClassifier(random_state=42)
    classifier.initialize_models(include_ensemble=True)
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)
    classifier.train_all_models(X_train, y_train, X_test, y_test)
    classifier.save_models()
    classifier.generate_report()

    print("\n✅ CLASSIFICATION PIPELINE COMPLETE!")
    return classifier


if __name__ == "__main__":
    classifier = run_training_pipeline()

