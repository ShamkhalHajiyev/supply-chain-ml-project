"""
Automated Feature Selection Pipeline with Leakage Prevention

This module implements a comprehensive, automated feature selection pipeline that:
1. Parses feature_description.md for business context and availability
2. Identifies and excludes leaky/post-outcome features
3. Performs statistical and model-based feature selection
4. Generates explicit rationale for each feature decision

Usage:
    from src.features.feature_selector import FeatureSelector
    selector = FeatureSelector()
    X_selected, selection_report = selector.fit_transform(X, y)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureInfo:
    """Information about a single feature from feature_description.md"""
    name: str
    description: str
    data_type: str
    notes: str
    category: str  # Customer, Order, Product, Shipping, Location
    is_leaky: bool = False
    leakage_reason: Optional[str] = None
    is_production_available: bool = True
    availability_note: Optional[str] = None


# Known leaky features - these contain post-outcome information
LEAKY_FEATURES = {
    'late_delivery_risk': 'This IS the target variable',
    'delivery_status': 'Categorical form of target variable',
    'delivery_status_encoded': 'Encoded target variable',
    'days_for_shipping_(real)': 'Only known after delivery completes',
    'delivery_days': 'Calculated from actual delivery date',
    'shipping_date_(dateorders)': 'Actual shipping date - post-hoc information',
}

# Features not available in production at prediction time
PRODUCTION_UNAVAILABLE = {
    'customer_password': 'Sensitive authentication data - should not be used',
    'product_image': 'URL not predictive, storage concern',
    'product_description': 'Free text - requires NLP processing',
}

# ID columns that should be excluded from modeling
ID_COLUMNS = {
    'customer_id', 'order_id', 'product_card_id', 'order_item_id',
    'order_customer_id', 'category_id', 'department_id', 'product_category_id',
    'order_item_cardprod_id'
}


class FeatureSelector:
    """
    Automated feature selection with leakage prevention and explicit rationale.

    This selector:
    1. Identifies and excludes leaky features (post-outcome information)
    2. Excludes production-unavailable features
    3. Applies statistical filters (variance, correlation)
    4. Uses model-based importance ranking
    5. Generates human-readable rationale for each decision
    """

    def __init__(self,
                 feature_desc_path: Optional[str] = None,
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95,
                 missing_threshold: float = 0.5,
                 importance_threshold: float = 0.01,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize feature selector.

        Args:
            feature_desc_path: Path to feature_description.md
            variance_threshold: Minimum variance (drop near-constant features)
            correlation_threshold: Maximum correlation (drop highly correlated pairs)
            missing_threshold: Maximum missing rate (drop if > threshold)
            importance_threshold: Minimum relative importance to keep feature
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.feature_desc_path = feature_desc_path or self._find_feature_desc()
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        self.importance_threshold = importance_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Results storage
        self.feature_info: Dict[str, FeatureInfo] = {}
        self.selection_decisions: Dict[str, Dict] = {}
        self.selected_features: List[str] = []
        self.dropped_features: List[str] = []
        self.importance_scores: Dict[str, float] = {}
        self.stability_scores: Dict[str, float] = {}

    def _find_feature_desc(self) -> str:
        """Find feature_description.md in the project."""
        project_root = Path(__file__).parents[2]
        return str(project_root / 'feature_description.md')

    def parse_feature_descriptions(self) -> Dict[str, FeatureInfo]:
        """
        Parse feature_description.md and extract feature information.

        Returns:
            Dictionary mapping feature name to FeatureInfo
        """
        print("=" * 60)
        print("PARSING FEATURE DESCRIPTIONS")
        print("=" * 60)

        try:
            with open(self.feature_desc_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Warning: {self.feature_desc_path} not found. Using defaults.")
            return {}

        # Parse the markdown table
        lines = content.split('\n')
        in_table = False
        header_found = False
        current_category = 'Unknown'

        for line in lines:
            # Track current category from section headers
            if line.startswith('### '):
                current_category = line.replace('###', '').strip()

            # Parse table rows
            if '|' in line and 'Feature Name' in line:
                in_table = True
                header_found = True
                continue

            if in_table and '|' in line:
                # Skip separator row
                if '---' in line:
                    continue

                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 4:
                    # Clean feature name (remove bold markers)
                    feature_name = parts[0].replace('**', '').strip()
                    if not feature_name:
                        continue

                    # Normalize feature name for matching
                    feature_name_lower = feature_name.lower().replace(' ', '_')

                    # Check if leaky
                    is_leaky = False
                    leakage_reason = None
                    for leaky_name, reason in LEAKY_FEATURES.items():
                        if leaky_name.lower() in feature_name_lower or feature_name_lower in leaky_name.lower():
                            is_leaky = True
                            leakage_reason = reason
                            break

                    # Check if production available
                    is_prod_available = True
                    availability_note = None
                    for unavail_name, note in PRODUCTION_UNAVAILABLE.items():
                        if unavail_name.lower() in feature_name_lower:
                            is_prod_available = False
                            availability_note = note
                            break

                    self.feature_info[feature_name_lower] = FeatureInfo(
                        name=feature_name,
                        description=parts[1] if len(parts) > 1 else '',
                        data_type=parts[2] if len(parts) > 2 else '',
                        notes=parts[3] if len(parts) > 3 else '',
                        category=current_category,
                        is_leaky=is_leaky,
                        leakage_reason=leakage_reason,
                        is_production_available=is_prod_available,
                        availability_note=availability_note
                    )

        print(f"Parsed {len(self.feature_info)} feature descriptions")
        print(f"  Leaky features identified: {sum(1 for f in self.feature_info.values() if f.is_leaky)}")
        print(f"  Production-unavailable: {sum(1 for f in self.feature_info.values() if not f.is_production_available)}")

        return self.feature_info

    def _normalize_column_name(self, col: str) -> str:
        """Normalize column name for matching with feature descriptions."""
        return col.lower().replace(' ', '_').replace('-', '_')

    def _get_feature_info(self, col: str) -> Optional[FeatureInfo]:
        """Get feature info for a column, handling name variations."""
        normalized = self._normalize_column_name(col)

        # Direct match
        if normalized in self.feature_info:
            return self.feature_info[normalized]

        # Partial match
        for key, info in self.feature_info.items():
            if normalized in key or key in normalized:
                return info

        return None

    def identify_leaky_features(self, df: pd.DataFrame) -> List[str]:
        """
        Identify features that would cause data leakage.

        Args:
            df: DataFrame with features

        Returns:
            List of leaky feature names
        """
        leaky = []

        for col in df.columns:
            normalized = self._normalize_column_name(col)

            # Check against known leaky features
            for leaky_name in LEAKY_FEATURES.keys():
                if leaky_name in normalized or normalized in leaky_name:
                    leaky.append(col)
                    self.selection_decisions[col] = {
                        'status': 'dropped',
                        'reason': f'Data leakage: {LEAKY_FEATURES.get(leaky_name, "Post-outcome information")}',
                        'category': 'leakage',
                        'importance': None
                    }
                    break

            # Check feature descriptions
            info = self._get_feature_info(col)
            if info and info.is_leaky and col not in leaky:
                leaky.append(col)
                self.selection_decisions[col] = {
                    'status': 'dropped',
                    'reason': f'Data leakage: {info.leakage_reason}',
                    'category': 'leakage',
                    'importance': None
                }

        return leaky

    def identify_production_unavailable(self, df: pd.DataFrame) -> List[str]:
        """
        Identify features not available at prediction time in production.

        Args:
            df: DataFrame with features

        Returns:
            List of production-unavailable feature names
        """
        unavailable = []

        for col in df.columns:
            normalized = self._normalize_column_name(col)

            # Check against known unavailable features
            for unavail_name, reason in PRODUCTION_UNAVAILABLE.items():
                if unavail_name in normalized:
                    unavailable.append(col)
                    self.selection_decisions[col] = {
                        'status': 'dropped',
                        'reason': f'Not available in production: {reason}',
                        'category': 'production_availability',
                        'importance': None
                    }
                    break

            # Check feature descriptions
            info = self._get_feature_info(col)
            if info and not info.is_production_available and col not in unavailable:
                unavailable.append(col)
                self.selection_decisions[col] = {
                    'status': 'dropped',
                    'reason': f'Not available in production: {info.availability_note}',
                    'category': 'production_availability',
                    'importance': None
                }

        return unavailable

    def identify_id_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify ID columns that should be excluded from modeling.

        Args:
            df: DataFrame with features

        Returns:
            List of ID column names
        """
        id_cols = []

        for col in df.columns:
            normalized = self._normalize_column_name(col)

            # Check if it's a known ID column
            if normalized in ID_COLUMNS or any(id_col in normalized for id_col in ID_COLUMNS):
                id_cols.append(col)
                self.selection_decisions[col] = {
                    'status': 'dropped',
                    'reason': 'Identifier column - not predictive',
                    'category': 'identifier',
                    'importance': None
                }

        return id_cols

    def apply_variance_filter(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove near-constant features based on variance threshold.

        Args:
            X: Feature DataFrame

        Returns:
            Filtered DataFrame and list of dropped columns
        """
        dropped = []

        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                continue  # Skip categorical columns

            variance = X[col].var()
            if variance < self.variance_threshold:
                dropped.append(col)
                self.selection_decisions[col] = {
                    'status': 'dropped',
                    'reason': f'Near-constant feature (variance={variance:.6f} < {self.variance_threshold})',
                    'category': 'variance',
                    'importance': None
                }

        return X.drop(columns=dropped, errors='ignore'), dropped

    def apply_missing_filter(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with excessive missing values.

        Args:
            X: Feature DataFrame

        Returns:
            Filtered DataFrame and list of dropped columns
        """
        dropped = []

        for col in X.columns:
            missing_rate = X[col].isna().mean()
            if missing_rate > self.missing_threshold:
                dropped.append(col)
                self.selection_decisions[col] = {
                    'status': 'dropped',
                    'reason': f'High missing rate ({missing_rate:.1%} > {self.missing_threshold:.0%} threshold)',
                    'category': 'missing',
                    'importance': None
                }

        return X.drop(columns=dropped, errors='ignore'), dropped

    def apply_correlation_filter(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features, keeping the one with higher target correlation.

        Args:
            X: Feature DataFrame

        Returns:
            Filtered DataFrame and list of dropped columns
        """
        dropped = []

        # Only apply to numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return X, dropped

        # Calculate correlation matrix
        corr_matrix = X[numeric_cols].corr().abs()

        # Find highly correlated pairs
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                if col1 in dropped or col2 in dropped:
                    continue

                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    # Drop the second one (can be enhanced with target correlation)
                    dropped.append(col2)
                    self.selection_decisions[col2] = {
                        'status': 'dropped',
                        'reason': f'High correlation ({corr_matrix.iloc[i, j]:.3f}) with {col1}',
                        'category': 'correlation',
                        'importance': None
                    }

        return X.drop(columns=dropped, errors='ignore'), dropped

    def compute_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Compute feature importance using LightGBM.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Dictionary mapping feature name to importance score
        """
        print("\nComputing feature importances with LightGBM...")

        # Handle any remaining issues with X
        X_clean = X.copy()

        # Convert categorical columns to numeric
        for col in X_clean.select_dtypes(include=['object', 'category']).columns:
            X_clean[col] = pd.Categorical(X_clean[col]).codes

        # Fill NaN values
        X_clean = X_clean.fillna(0)

        # Train LightGBM model
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
            force_col_wise=True
        )

        model.fit(X_clean, y)

        # Get feature importances
        importances = model.feature_importances_
        total_importance = importances.sum()

        importance_dict = {}
        for col, imp in zip(X_clean.columns, importances):
            importance_dict[col] = imp / total_importance if total_importance > 0 else 0

        self.importance_scores = importance_dict
        return importance_dict

    def compute_importance_stability(self, X: pd.DataFrame, y: pd.Series,
                                     n_folds: int = 5) -> Dict[str, float]:
        """
        Compute stability of feature importance across CV folds.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_folds: Number of CV folds

        Returns:
            Dictionary mapping feature name to stability score (1 - std/mean)
        """
        print(f"\nComputing importance stability across {n_folds} folds...")

        # Handle any remaining issues with X
        X_clean = X.copy()
        for col in X_clean.select_dtypes(include=['object', 'category']).columns:
            X_clean[col] = pd.Categorical(X_clean[col]).codes
        X_clean = X_clean.fillna(0)

        # Collect importances across folds
        fold_importances = {col: [] for col in X_clean.columns}

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, _ in kfold.split(X_clean, y):
            X_fold = X_clean.iloc[train_idx]
            y_fold = y.iloc[train_idx]

            model = LGBMClassifier(
                n_estimators=50,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
                force_col_wise=True
            )
            model.fit(X_fold, y_fold)

            importances = model.feature_importances_
            total = importances.sum()

            for col, imp in zip(X_clean.columns, importances):
                fold_importances[col].append(imp / total if total > 0 else 0)

        # Calculate stability (1 - coefficient of variation)
        stability_dict = {}
        for col, imps in fold_importances.items():
            mean_imp = np.mean(imps)
            std_imp = np.std(imps)
            cv = std_imp / mean_imp if mean_imp > 0 else 1
            stability_dict[col] = max(0, 1 - cv)  # Stability: 0 to 1, higher is more stable

        self.stability_scores = stability_dict
        return stability_dict

    def apply_importance_filter(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low importance scores.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Filtered DataFrame and list of dropped columns
        """
        if not self.importance_scores:
            self.compute_feature_importance(X, y)

        dropped = []

        for col in X.columns:
            imp = self.importance_scores.get(col, 0)
            stability = self.stability_scores.get(col, 0.5)

            if imp < self.importance_threshold:
                dropped.append(col)
                self.selection_decisions[col] = {
                    'status': 'dropped',
                    'reason': f'Low importance ({imp:.4f} < {self.importance_threshold}), stability={stability:.2f}',
                    'category': 'importance',
                    'importance': imp
                }

        return X.drop(columns=dropped, errors='ignore'), dropped

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete feature selection pipeline.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Selected features DataFrame and selection report DataFrame
        """
        print("\n" + "=" * 60)
        print("AUTOMATED FEATURE SELECTION PIPELINE")
        print("=" * 60)

        # Parse feature descriptions
        self.parse_feature_descriptions()

        original_features = list(X.columns)
        X_filtered = X.copy()

        # Step 1: Remove leaky features
        print("\n[1/6] Checking for data leakage...")
        leaky = self.identify_leaky_features(X_filtered)
        X_filtered = X_filtered.drop(columns=leaky, errors='ignore')
        print(f"      Removed {len(leaky)} leaky features")

        # Step 2: Remove production-unavailable features
        print("[2/6] Checking production availability...")
        unavailable = self.identify_production_unavailable(X_filtered)
        X_filtered = X_filtered.drop(columns=unavailable, errors='ignore')
        print(f"      Removed {len(unavailable)} unavailable features")

        # Step 3: Remove ID columns
        print("[3/6] Removing ID columns...")
        id_cols = self.identify_id_columns(X_filtered)
        X_filtered = X_filtered.drop(columns=id_cols, errors='ignore')
        print(f"      Removed {len(id_cols)} ID columns")

        # Step 4: Variance filter
        print("[4/6] Applying variance filter...")
        X_filtered, variance_dropped = self.apply_variance_filter(X_filtered)
        print(f"      Removed {len(variance_dropped)} near-constant features")

        # Step 5: Missing value filter
        print("[5/6] Applying missing value filter...")
        X_filtered, missing_dropped = self.apply_missing_filter(X_filtered)
        print(f"      Removed {len(missing_dropped)} high-missing features")

        # Step 6: Correlation filter
        print("[6/6] Applying correlation filter...")
        X_filtered, corr_dropped = self.apply_correlation_filter(X_filtered)
        print(f"      Removed {len(corr_dropped)} highly-correlated features")

        # Compute importances for remaining features
        print("\n[BONUS] Computing feature importances...")
        self.compute_feature_importance(X_filtered, y)
        self.compute_importance_stability(X_filtered, y)

        # Mark selected features
        self.selected_features = list(X_filtered.columns)
        for col in self.selected_features:
            if col not in self.selection_decisions:
                info = self._get_feature_info(col)
                imp = self.importance_scores.get(col, 0)
                stability = self.stability_scores.get(col, 0.5)

                # Determine reason for keeping
                if imp > 0.05:
                    reason = f"High importance ({imp:.4f}), stable across folds ({stability:.2f})"
                elif imp > 0.01:
                    reason = f"Moderate importance ({imp:.4f}), contributes to model"
                else:
                    reason = f"Low but non-zero importance ({imp:.4f}), kept for completeness"

                if info:
                    reason += f". Business context: {info.description[:50]}..."

                self.selection_decisions[col] = {
                    'status': 'selected',
                    'reason': reason,
                    'category': 'model_based',
                    'importance': imp
                }

        # Generate report
        report = self.generate_selection_report()

        # Summary
        print("\n" + "=" * 60)
        print("FEATURE SELECTION SUMMARY")
        print("=" * 60)
        print(f"  Original features: {len(original_features)}")
        print(f"  Selected features: {len(self.selected_features)}")
        print(f"  Dropped features: {len(original_features) - len(self.selected_features)}")
        print("=" * 60)

        return X_filtered, report

    def generate_selection_report(self) -> pd.DataFrame:
        """
        Generate a detailed feature selection report.

        Returns:
            DataFrame with feature selection decisions and rationale
        """
        report_data = []

        for col, decision in self.selection_decisions.items():
            info = self._get_feature_info(col)

            report_data.append({
                'feature_name': col,
                'description': info.description if info else 'N/A',
                'selection_status': decision['status'],
                'reason_for_decision': decision['reason'],
                'importance_score': decision.get('importance', None),
                'leakage_risk': 'YES' if decision['category'] == 'leakage' else 'NO',
                'production_available': 'NO' if decision['category'] == 'production_availability' else 'YES',
                'category': info.category if info else 'Unknown'
            })

        return pd.DataFrame(report_data).sort_values(
            ['selection_status', 'importance_score'],
            ascending=[True, False]
        )

    def get_selected_features(self) -> List[str]:
        """Return list of selected feature names."""
        return self.selected_features

    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """Return feature importance ranking as DataFrame."""
        data = []
        for col in self.selected_features:
            data.append({
                'feature': col,
                'importance': self.importance_scores.get(col, 0),
                'stability': self.stability_scores.get(col, 0.5)
            })

        return pd.DataFrame(data).sort_values('importance', ascending=False)


def run_feature_selection(X: pd.DataFrame, y: pd.Series,
                          return_report: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Convenience function to run feature selection pipeline.

    Args:
        X: Feature DataFrame
        y: Target Series
        return_report: If True, return selection report

    Returns:
        Selected features DataFrame (and optionally selection report)
    """
    selector = FeatureSelector()
    X_selected, report = selector.fit_transform(X, y)

    if return_report:
        return X_selected, report
    return X_selected, None


if __name__ == "__main__":
    from src.data.preprocess import load_or_preprocess
    from src.features.build_features import build_features_pipeline

    # Load data
    df = load_or_preprocess()
    X, y = build_features_pipeline(df)

    # Run feature selection
    selector = FeatureSelector()
    X_selected, report = selector.fit_transform(X, y)

    print("\n\nFEATURE SELECTION REPORT:")
    print(report.to_string())

    print("\n\nFEATURE IMPORTANCE RANKING:")
    print(selector.get_feature_importance_ranking().to_string())
