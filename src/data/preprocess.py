"""
Data Preprocessing Module
Handles data cleaning, validation, and transformation for supply chain ML pipeline.

IMPORTANT: This module preserves 'late_delivery_risk' as the target variable
but DOES NOT use post-delivery information for feature creation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Columns that should NOT be used as features (they leak the target)
LEAKY_COLUMNS = [
    'late_delivery_risk',           # This IS the target
    'delivery_status',              # This IS the target (categorical)
    'days_for_shipping_(real)',     # Only known after delivery
]


class DataPreprocessor:
    """
    Preprocesses raw supply chain data for ML pipeline.

    Responsibilities:
    - Handle missing values
    - Encode categorical variables
    - Parse datetime features
    - Remove duplicates and outliers
    - Standardize column names
    - Preserve target variable (late_delivery_risk)
    """

    def __init__(self):
        self.categorical_encodings: Dict[str, Dict] = {}
        self.numeric_stats: Dict[str, Dict] = {}

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names: lowercase, remove spaces."""
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date columns to datetime format."""
        date_columns = [
            'order_date_(dateorders)',
            'shipping_date_(dateorders)',
            'order_date',
            'shipping_date'
        ]

        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with domain-specific strategies.

        Strategy:
        - Numeric: median imputation
        - Categorical: mode imputation or 'Unknown'
        - Dates: forward fill
        """
        # Numeric columns: median imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                self.numeric_stats[col] = {'median': median_val}
                df[col].fillna(median_val, inplace=True)

        # Categorical columns: mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if df[col].mode().shape[0] > 0:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)

        # Datetime columns: forward fill
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            df[col] = df[col].ffill()

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)

        if initial_count > final_count:
            print(f"✅ Removed {initial_count - final_count} duplicate rows")

        return df

    def handle_outliers(self, df: pd.DataFrame, cols: list = None, threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers using IQR method for specified numeric columns.

        Args:
            df: DataFrame
            cols: List of columns to check (default: all numeric)
            threshold: IQR multiplier for outlier detection
        """
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def preserve_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preserve the late_delivery_risk column as our target.
        
        The raw data already has this column as binary (0/1):
        - 1: Late delivery expected/occurred
        - 0: On-time delivery expected/occurred
        
        NOTE: We keep this separate from features to prevent leakage.
        """
        if 'late_delivery_risk' in df.columns:
            # Ensure it's properly typed
            df['late_delivery'] = df['late_delivery_risk'].astype(int)
            print(f"✅ Target variable 'late_delivery' preserved from 'late_delivery_risk'")
            print(f"   Distribution: {df['late_delivery'].value_counts().to_dict()}")
        elif 'delivery_status' in df.columns:
            # Fallback: create from delivery_status if late_delivery_risk missing
            df['late_delivery'] = df['delivery_status'].apply(
                lambda x: 1 if 'late' in str(x).lower() else 0
            )
            print(f"✅ Created binary target 'late_delivery' from 'delivery_status'")
            print(f"   Distribution: {df['late_delivery'].value_counts().to_dict()}")
        
        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic derived features for preprocessing.
        
        IMPORTANT: Only creates features from information available AT ORDER TIME.
        Does NOT use actual shipping date or delivery outcome.
        """
        # Scheduled delivery time (SAFE - this is the promised time, known at order)
        if 'days_for_shipment_(scheduled)' in df.columns:
            df['scheduled_days'] = df['days_for_shipment_(scheduled)']
            
        # Profit margin (known at order time from pricing)
        if 'order_profit_per_order' in df.columns and 'sales' in df.columns:
            df['profit_margin'] = df['order_profit_per_order'] / (df['sales'] + 1e-6)
            df['profit_margin'] = df['profit_margin'].clip(-1, 1)

        # NOTE: We do NOT create 'delivery_days' from shipping_date - order_date
        # because that would be using post-hoc information!
        
        print("✅ Derived features created (leakage-free)")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.

        Steps:
        1. Clean column names
        2. Parse dates
        3. Handle missing values
        4. Remove duplicates
        5. Handle outliers
        6. Preserve target variable
        7. Create derived features (leakage-free)
        """
        print("=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)

        print(f"\nInitial shape: {df.shape}")

        # Step 1: Clean column names
        df = self.clean_column_names(df)
        print("✅ Column names standardized")

        # Step 2: Parse dates
        df = self.parse_dates(df)
        print("✅ Date columns parsed")

        # Step 3: Handle missing values
        df = self.handle_missing_values(df)
        print("✅ Missing values handled")

        # Step 4: Remove duplicates
        df = self.remove_duplicates(df)

        # Step 5: Handle outliers in key numeric columns
        numeric_cols = ['order_item_quantity', 'sales', 'order_profit_per_order']
        df = self.handle_outliers(df, cols=numeric_cols)
        print("✅ Outliers handled")

        # Step 6: Preserve target variable
        df = self.preserve_target_variable(df)

        # Step 7: Create derived features (leakage-free)
        df = self.create_derived_features(df)

        print(f"\nFinal shape: {df.shape}")
        print("=" * 60)
        print("\n⚠️  LEAKAGE WARNING: The following columns should NOT be used as features:")
        for col in LEAKY_COLUMNS:
            if col in df.columns:
                print(f"    - {col}")
        print("=" * 60)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing using fitted parameters."""
        df = self.clean_column_names(df)
        df = self.parse_dates(df)
        df = self.handle_missing_values(df)
        df = self.preserve_target_variable(df)
        df = self.create_derived_features(df)
        return df


def load_and_preprocess(raw_path: str = None) -> pd.DataFrame:
    """
    Convenience function to load and preprocess data.

    Args:
        raw_path: Path to raw CSV file (optional)

    Returns:
        Preprocessed DataFrame
    """
    from .data_manager import load_raw, save_interim

    # Load raw data
    if raw_path:
        df = pd.read_csv(raw_path)
    else:
        df = load_raw()

    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.fit_transform(df)

    # Save interim data
    save_interim(df_clean, "cleaned_data")

    return df_clean


if __name__ == "__main__":
    # Run preprocessing pipeline
    df_cleaned = load_and_preprocess()
    print("\n✅ Preprocessing complete!")
    print(f"Cleaned data shape: {df_cleaned.shape}")
    print(f"\nColumns: {list(df_cleaned.columns[:10])}...")
