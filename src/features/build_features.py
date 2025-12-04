"""
Feature Engineering Module
Transforms preprocessed data into ML-ready features for late delivery classification.

IMPORTANT: This module carefully avoids data leakage by only using features
available at ORDER TIME (before delivery outcome is known).

LEAKY COLUMNS TO AVOID:
- late_delivery_risk (this IS the target)
- delivery_status (this IS the target)
- days_for_shipping_(real) (only known after delivery)
- delivery_days (calculated from actual shipping date)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# Columns that MUST be excluded to prevent data leakage
LEAKY_COLUMNS = {
    'late_delivery_risk',           # Target variable
    'delivery_status',              # Target variable (categorical form)
    'delivery_status_encoded',      # Encoded target
    'days_for_shipping_(real)',     # Only known after delivery
    'delivery_days',                # Calculated from actual shipping date
    'shipping_date_(dateorders)',   # Actual shipping date (post-hoc)
}


class FeatureEngineer:
    """
    Feature engineering for late delivery classification.

    Creates features for late delivery prediction using only pre-delivery features.

    NOTE: All features are derived from information available AT ORDER TIME
    to prevent data leakage.
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from ORDER dates only.

        NOTE: We use order_date, NOT shipping_date (which would be leakage).

        Features:
        - Day of week
        - Month
        - Quarter
        - Is weekend
        - Days since start
        """
        date_col = None
        for col in ['order_date_(dateorders)', 'order_date']:
            if col in df.columns:
                date_col = col
                break

        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            # Day of week (0=Monday, 6=Sunday)
            df['order_day_of_week'] = df[date_col].dt.dayofweek

            # Month
            df['order_month'] = df[date_col].dt.month

            # Quarter
            df['order_quarter'] = df[date_col].dt.quarter

            # Is weekend
            df['is_weekend'] = (df['order_day_of_week'] >= 5).astype(int)

            # Days since start (for trend analysis)
            df['days_since_start'] = (df[date_col] - df[date_col].min()).dt.days

            print("✅ Temporal features created: day_of_week, month, quarter, is_weekend, days_since_start")

        return df

    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-related features (available at order time).

        Features:
        - Customer order frequency
        - Customer lifetime value
        - Customer segment encoding
        """
        # Customer order count
        if 'customer_id' in df.columns:
            customer_counts = df.groupby('customer_id').size()
            df['customer_order_count'] = df['customer_id'].map(customer_counts)

            # Customer total sales
            if 'sales' in df.columns:
                customer_sales = df.groupby('customer_id')['sales'].sum()
                df['customer_lifetime_value'] = df['customer_id'].map(customer_sales)

            print("✅ Customer features created: order_count, lifetime_value")

        return df

    def create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create product-related features (available at order time).

        Features:
        - Product popularity (order frequency)
        - Category-level aggregations
        - Price features
        """
        # Product order frequency
        if 'product_name' in df.columns:
            product_counts = df.groupby('product_name').size()
            df['product_popularity'] = df['product_name'].map(product_counts)

        # Category order frequency
        if 'category_name' in df.columns:
            category_counts = df.groupby('category_name').size()
            df['category_popularity'] = df['category_name'].map(category_counts)

        # Price-based features
        if 'product_price' in df.columns and 'order_item_quantity' in df.columns:
            df['order_value'] = df['product_price'] * df['order_item_quantity']

        # Discount features
        if 'order_item_discount' in df.columns and 'product_price' in df.columns:
            df['discount_rate'] = df['order_item_discount'] / (df['product_price'] + 1e-6)
            df['discount_rate'] = df['discount_rate'].clip(0, 1)

        print("✅ Product features created: popularity, category_popularity, order_value, discount_rate")

        return df

    def create_shipping_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create shipping and logistics features (ONLY pre-delivery info).

        Features:
        - Shipping mode encoding
        - Scheduled shipping days (NOT actual)
        - Geographic distance proxy
        - Order priority

        NOTE: days_for_shipment_(scheduled) IS safe - it's the promised delivery time
        """
        # Shipping urgency (based on shipping mode)
        if 'shipping_mode' in df.columns:
            urgency_map = {
                'Same Day': 4,
                'First Class': 3,
                'Second Class': 2,
                'Standard Class': 1
            }
            df['shipping_urgency'] = df['shipping_mode'].map(urgency_map).fillna(1)

        # Scheduled days for shipping (this is SAFE - it's the promised time)
        if 'days_for_shipment_(scheduled)' in df.columns:
            df['scheduled_shipping_days'] = df['days_for_shipment_(scheduled)']

        # Order priority encoding
        if 'order_region' in df.columns and 'order_country' in df.columns:
            # Create region-country combination as proxy for distance
            df['region_country'] = df['order_region'].astype(str) + '_' + df['order_country'].astype(str)

        print("✅ Shipping features created: shipping_urgency, scheduled_shipping_days, region_country")

        return df

    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial and profitability features.

        Features:
        - Profit margin
        - Revenue per item
        - Discount impact
        """
        # Profit margin
        if 'order_profit_per_order' in df.columns and 'sales' in df.columns:
            df['profit_margin_pct'] = (df['order_profit_per_order'] / (df['sales'] + 1e-6)) * 100
            df['profit_margin_pct'] = df['profit_margin_pct'].clip(-100, 100)

        # Sales per item
        if 'sales' in df.columns and 'order_item_quantity' in df.columns:
            df['sales_per_item'] = df['sales'] / (df['order_item_quantity'] + 1e-6)

        # High value order flag
        if 'sales' in df.columns:
            df['is_high_value'] = (df['sales'] > df['sales'].quantile(0.75)).astype(int)

        print("✅ Financial features created: profit_margin_pct, sales_per_item, is_high_value")

        return df


    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using label encoding.

        IMPORTANT: Excludes 'delivery_status' to prevent data leakage!

        Args:
            df: DataFrame
            fit: If True, fit encoders. If False, use existing encoders.
        """
        # SAFE categorical columns (available at order time, NOT target-related)
        categorical_cols = [
            'type', 'category_name', 'customer_segment',
            'department_name', 'market', 'order_region', 'order_country',
            'order_state', 'order_city', 'shipping_mode'
            # NOTE: 'delivery_status' EXCLUDED - it's the target!
            # NOTE: 'product_name' EXCLUDED - too many categories, use popularity instead
        ]

        self.categorical_columns = [col for col in categorical_cols if col in df.columns]

        for col in self.categorical_columns:
            if fit:
                le = LabelEncoder()
                # Handle unseen categories
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    # Handle unseen categories in test set
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )

        if fit:
            print(f"✅ Encoded {len(self.categorical_columns)} categorical features")
            print(f"   (Excluded 'delivery_status' to prevent leakage)")

        return df

    def select_features_for_classification(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select features for late delivery classification.

        IMPORTANT: Only uses features available at ORDER TIME.
        Excludes all post-delivery information to prevent leakage.

        Returns:
            X: Feature matrix
            y: Target variable (late_delivery)
        """
        # Core features for classification
        feature_cols = []

        # Temporal features (from ORDER date only)
        temporal = ['order_day_of_week', 'order_month', 'order_quarter', 'is_weekend']
        feature_cols.extend([c for c in temporal if c in df.columns])

        # Customer features
        customer = ['customer_order_count', 'customer_lifetime_value']
        feature_cols.extend([c for c in customer if c in df.columns])

        # Product features
        product = ['product_popularity', 'category_popularity', 'order_value', 'discount_rate']
        feature_cols.extend([c for c in product if c in df.columns])

        # Shipping features (ONLY pre-delivery info!)
        # NOTE: 'delivery_days' EXCLUDED - it's calculated from actual delivery date!
        shipping = ['shipping_urgency', 'scheduled_shipping_days']
        feature_cols.extend([c for c in shipping if c in df.columns])

        # Financial features
        financial = ['profit_margin_pct', 'sales_per_item', 'is_high_value', 'order_item_quantity']
        feature_cols.extend([c for c in financial if c in df.columns])

        # Encoded categorical features (excluding delivery_status!)
        encoded = [f'{c}_encoded' for c in self.categorical_columns if f'{c}_encoded' in df.columns]
        feature_cols.extend(encoded)

        # Remove duplicates and VERIFY no leaky columns
        feature_cols = list(set(feature_cols))
        feature_cols = [c for c in feature_cols if c.lower() not in LEAKY_COLUMNS
                        and not c.lower().startswith('delivery')]

        print(f"\n✅ Selected {len(feature_cols)} features for classification")
        print(f"   (Verified: No leaky features included)")

        X = df[feature_cols].copy()

        # Use late_delivery_risk directly if available, otherwise use late_delivery
        # Always convert to int for consistency and robust model training
        if 'late_delivery_risk' in df.columns:
            y = df['late_delivery_risk'].astype(int)
        elif 'late_delivery' in df.columns:
            y = df['late_delivery'].astype(int)
        else:
            y = None

        return X, y


    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete feature engineering pipeline for classification.

        Returns:
            X: Feature matrix
            y: Target variable
        """
        print("=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        print("⚠️  LEAKAGE PREVENTION ACTIVE")
        print(f"    Excluded columns: {', '.join(LEAKY_COLUMNS)}")
        print("=" * 60)

        # Create features
        df = self.create_temporal_features(df)
        df = self.create_customer_features(df)
        df = self.create_product_features(df)
        df = self.create_shipping_features(df)
        df = self.create_financial_features(df)

        # Encode categoricals (excluding delivery_status)
        df = self.encode_categorical_features(df, fit=True)

        # Select features for classification
        X, y = self.select_features_for_classification(df)

        # Handle any remaining NaN values
        X = X.fillna(0)

        print(f"\nFeature matrix shape: {X.shape}")
        if y is not None:
            print(f"Target distribution: {y.value_counts().to_dict()}")
        print("=" * 60)

        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply feature engineering using fitted parameters."""
        df = self.create_temporal_features(df)
        df = self.create_customer_features(df)
        df = self.create_product_features(df)
        df = self.create_shipping_features(df)
        df = self.create_financial_features(df)
        df = self.encode_categorical_features(df, fit=False)
        X, y = self.select_features_for_classification(df)
        X = X.fillna(0)
        return X, y


def build_features_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function for feature engineering.

    Args:
        df: Preprocessed DataFrame

    Returns:
        X: Feature matrix
        y: Target variable
    """
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(df)
    return X, y


if __name__ == "__main__":
    from src.data.preprocess import load_and_preprocess

    # Load preprocessed data
    df = load_and_preprocess()

    # Build features
    X, y = build_features_pipeline(df)

    print("\n✅ Feature engineering complete!")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"\nFeature columns: {list(X.columns)}")