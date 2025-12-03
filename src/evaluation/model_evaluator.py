"""
Comprehensive Model Evaluation Module
Advanced evaluation plots and metrics for classification and regression models.

Features:
- ROC curves
- Precision-Recall curves
- Calibration curves
- Confusion matrix heatmaps
- Learning curves
- Feature importance comparison
- Residual plots (regression)

Usage:
    from src.evaluation.model_evaluator import ClassificationEvaluator
    evaluator = ClassificationEvaluator()
    evaluator.comprehensive_evaluation(model, X_test, y_test)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ClassificationEvaluator:
    """
    Comprehensive evaluation for classification models.

    Provides:
    - Multiple metric calculations
    - Confusion matrix heatmap
    - ROC curve
    - Precision-Recall curve
    - Calibration curve
    - Learning curves
    """

    def __init__(self, save_dir: str = 'reports/figures'):
        """
        Initialize evaluator.

        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC-AUC)

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }

        if y_proba is not None:
            from sklearn.metrics import roc_auc_score
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = None

        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: List[str] = None,
                              save_path: Optional[str] = None):
        """
        Plot confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names for labels
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names if class_names else ['0', '1'],
                   yticklabels=class_names if class_names else ['0', '1'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.save_dir / save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      save_path: Optional[str] = None):
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve',
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.save_dir / save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return roc_auc

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    save_path: Optional[str] = None):
        """
        Plot Precision-Recall curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.axhline(y=y_true.mean(), color='navy', linestyle='--',
                   label=f'Baseline (random) = {y_true.mean():.2f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.save_dir / save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return pr_auc

    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                               n_bins: int = 10, save_path: Optional[str] = None):
        """
        Plot calibration curve.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            save_path: Path to save figure
        """
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2,
                label='Model', color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray',
                label='Perfect calibration')
        plt.xlabel('Mean predicted probability', fontsize=12)
        plt.ylabel('Fraction of positives', fontsize=12)
        plt.title('Calibration Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.save_dir / save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_learning_curve(self, model, X: np.ndarray, y: np.ndarray,
                           cv: int = 5, save_path: Optional[str] = None):
        """
        Plot learning curve to diagnose bias/variance.

        Args:
            model: Trained model
            X: Features
            y: Target
            cv: Number of cross-validation folds
            save_path: Path to save figure
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1_weighted'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')

        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.1, color='r')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                         alpha=0.1, color='g')

        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Learning Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.save_dir / save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def comprehensive_evaluation(self, model, X_test: np.ndarray, y_test: np.ndarray,
                                class_names: List[str] = None,
                                save_prefix: str = 'model') -> Dict:
        """
        Perform comprehensive model evaluation with all plots.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            class_names: Class names
            save_prefix: Prefix for saved figures

        Returns:
            Dictionary with all metrics
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)

        print("\n📊 CLASSIFICATION METRICS:")
        print("=" * 60)
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"   {metric_name.replace('_', ' ').title()}: {value:.4f}")

        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=class_names if class_names else ['0', '1'],
                   yticklabels=class_names if class_names else ['0', '1'])
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold', fontsize=12)
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        if y_proba is not None:
            # 2. ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                           label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve', fontweight='bold', fontsize=12)
            axes[0, 1].legend(loc="lower right")
            axes[0, 1].grid(alpha=0.3)

            # 3. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)
            axes[1, 0].plot(recall, precision, color='green', lw=2,
                           label=f'PR curve (AUC = {pr_auc:.2f})')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
            axes[1, 0].legend(loc="lower left")
            axes[1, 0].grid(alpha=0.3)

            # 4. Calibration Curve
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
            axes[1, 1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
            axes[1, 1].plot([0, 1], [0, 1], linestyle='--', color='gray',
                           label='Perfect calibration')
            axes[1, 1].set_xlabel('Mean predicted probability')
            axes[1, 1].set_ylabel('Fraction of positives')
            axes[1, 1].set_title('Calibration Curve', fontweight='bold', fontsize=12)
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_prefix}_comprehensive_evaluation.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ Comprehensive evaluation complete!")
        print(f"   Figures saved to: {self.save_dir}")

        return metrics


class RegressionEvaluator:
    """
    Comprehensive evaluation for regression models.

    Provides:
    - Multiple metric calculations
    - Actual vs Predicted plot
    - Residual plot
    - Residual distribution
    - Q-Q plot
    """

    def __init__(self, save_dir: str = 'reports/figures'):
        """Initialize evaluator."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        }
        return metrics

    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                                save_prefix: str = 'regression') -> Dict:
        """
        Perform comprehensive regression evaluation.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_prefix: Prefix for saved figures

        Returns:
            Dictionary with metrics
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE REGRESSION EVALUATION")
        print("=" * 60)

        metrics = self.calculate_metrics(y_true, y_pred)
        residuals = y_true - y_pred

        print("\n📊 REGRESSION METRICS:")
        print("=" * 60)
        for metric_name, value in metrics.items():
            print(f"   {metric_name.upper()}: {value:.4f}")

        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()],
                       [y_true.min(), y_true.max()],
                       'r--', lw=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. Residual Plot
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot', fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        # 3. Residual Distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / f'{save_prefix}_comprehensive_evaluation.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ Comprehensive evaluation complete!")
        print(f"   Figures saved to: {self.save_dir}")

        return metrics


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    evaluator = ClassificationEvaluator()
    metrics = evaluator.comprehensive_evaluation(model, X_test, y_test,
                                                 class_names=['Class 0', 'Class 1'])

    print(f"\n✅ Evaluation complete! Metrics: {metrics}")
