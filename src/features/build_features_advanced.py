"""
Advanced Feature Engineering Module
Transforms preprocessed data into ML-ready features for late delivery classification.

Features include:
- Base features (temporal, customer, product, shipping, financial)
- RFM (Recency, Frequency, Monetary) customer segmentation
- Zone-based aggregations
- Feature interactions

IMPORTANT: This module carefully avoids data leakage by only using features
available at ORDER TIME (before delivery outcome is known).

LEAKY COLUMNS TO AVOID:
- late_delivery_risk (this IS the target)
- delivery_status (this IS the target)
- days_for_shipping_(real) (only known after delivery)
- delivery_days (calculated from actual shipping date)

Features Created:
- Temporal: day_of_week, month, quarter, is_weekend, days_since_start
- Customer: order_count, lifetime_value, RFM scores
- Product: popularity, category_popularity, order_value, discount_rate
- Shipping: urgency, scheduled_days, zone aggregations
- Financial: profit_margin, sales_per_item, is_high_value
- Interaction: region×shipping, urgency×distance, etc.

Usage:
    from src.features.build_features_advanced import build_features_pipeline
    X, y = build_features_pipeline(df)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional: Target encoding
try:
    from category_encoders import TargetEncoder
    TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TARGET_ENCODER_AVAILABLE = False


# =============================================================================
# LEAKY COLUMNS - MUST BE EXCLUDED TO PREVENT DATA LEAKAGE
# =============================================================================
LEAKY_COLUMNS = {
    'late_delivery_risk',           # Target variable
    'delivery_status',              # Target variable (categorical form)
    'delivery_status_encoded',      # Encoded target
    'days_for_shipping_(real)',     # Only known after delivery
    'days_for_shipping_real',       # Alternative naming
    'delivery_days',                # Calculated from actual shipping date
    'shipping_date_(dateorders)',   # Actual shipping date (post-hoc)
    'shipping_date_dateorders',     # Alternative naming
}


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for late delivery classification.

    Combines base features with enhanced features including:
    - RFM (Recency, Frequency, Monetary) customer segmentation
    - Zone-based late delivery statistics
    - Product category risk scores
    - Feature interactions
    - Optional target encoding

    All features are derived from information available AT ORDER TIME
    to prevent data leakage.
    """

    def __init__(self, use_target_encoding: bool = False):
        """
        Initialize the feature engineer.

        Args:
            use_target_encoding: Whether to use target encoding for high-cardinality
                               categorical features (requires target in transform)
        """
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.use_target_encoding = use_target_encoding

        # Enhanced feature storage
        self.rfm_quantiles = {}
        self.zone_stats = {}
        self.category_risk = {}
        self.target_encoder = None

    # =========================================================================
    # BASE FEATURES
    # =========================================================================

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from ORDER dates only.

        NOTE: We use order_date, NOT shipping_date (which would be leakage).

        Features:
        - Day of week (0-6): Patterns in ordering by day
        - Month (1-12): Seasonal patterns
        - Quarter (1-4): Quarterly trends
        - Is weekend: Weekend vs weekday ordering
        - Days since start: Trend/time effects
        """
        date_col = None
        for col in ['order_date_(dateorders)', 'order_date_dateorders', 'order_date']:
            if col in df.columns:
                date_col = col
                break

        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df['order_day_of_week'] = df[date_col].dt.dayofweek
            df['order_month'] = df[date_col].dt.month
            df['order_quarter'] = df[date_col].dt.quarter
            df['is_weekend'] = (df['order_day_of_week'] >= 5).astype(int)
            df['days_since_start'] = (df[date_col] - df[date_col].min()).dt.days
            df['order_hour'] = df[date_col].dt.hour

            print("✅ Temporal features created: day_of_week, month, quarter, is_weekend, days_since_start, hour")

        return df

    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-related features (available at order time).

        Features:
        - Order count: How many orders this customer has made
        - Lifetime value: Total spending by this customer
        """
        if 'customer_id' in df.columns:
            customer_counts = df.groupby('customer_id').size()
            df['customer_order_count'] = df['customer_id'].map(customer_counts)

            if 'sales' in df.columns:
                customer_sales = df.groupby('customer_id')['sales'].sum()
                df['customer_lifetime_value'] = df['customer_id'].map(customer_sales)

            print("✅ Customer features created: order_count, lifetime_value")

        return df

    def create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create product-related features (available at order time).

        Features:
        - Product popularity: How often this product is ordered
        - Category popularity: How often this category is ordered
        - Order value: Price × quantity
        - Discount rate: Discount as percentage of price
        """
        if 'product_name' in df.columns:
            product_counts = df.groupby('product_name').size()
            df['product_popularity'] = df['product_name'].map(product_counts)

        if 'category_name' in df.columns:
            category_counts = df.groupby('category_name').size()
            df['category_popularity'] = df['category_name'].map(category_counts)

        if 'product_price' in df.columns and 'order_item_quantity' in df.columns:
            df['order_value'] = df['product_price'] * df['order_item_quantity']

        if 'order_item_discount' in df.columns and 'product_price' in df.columns:
            df['discount_rate'] = df['order_item_discount'] / (df['product_price'] + 1e-6)
            df['discount_rate'] = df['discount_rate'].clip(0, 1)

        print("✅ Product features created: popularity, category_popularity, order_value, discount_rate")

        return df

    def create_shipping_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create shipping and logistics features (ONLY pre-delivery info).

        Features:
        - Shipping urgency: Encoded shipping mode priority
        - Scheduled shipping days: Promised delivery time (SAFE - known at order)
        - Region-country combo: Geographic proxy

        NOTE: days_for_shipment_(scheduled) IS safe - it's the promised delivery time
        """
        if 'shipping_mode' in df.columns:
            urgency_map = {
                'Same Day': 4,
                'First Class': 3,
                'Second Class': 2,
                'Standard Class': 1
            }
            df['shipping_urgency'] = df['shipping_mode'].map(urgency_map).fillna(1)

        if 'days_for_shipment_(scheduled)' in df.columns:
            df['scheduled_shipping_days'] = df['days_for_shipment_(scheduled)']

        if 'order_region' in df.columns and 'order_country' in df.columns:
            df['region_country'] = df['order_region'].astype(str) + '_' + df['order_country'].astype(str)

        print("✅ Shipping features created: shipping_urgency, scheduled_shipping_days, region_country")

        return df

    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial and profitability features.

        Features:
        - Profit margin: Profit as percentage of sales
        - Sales per item: Average revenue per item in order
        - High value flag: Whether order exceeds 75th percentile
        """
        if 'order_profit_per_order' in df.columns and 'sales' in df.columns:
            df['profit_margin_pct'] = (df['order_profit_per_order'] / (df['sales'] + 1e-6)) * 100
            df['profit_margin_pct'] = df['profit_margin_pct'].clip(-100, 100)

        if 'sales' in df.columns and 'order_item_quantity' in df.columns:
            df['sales_per_item'] = df['sales'] / (df['order_item_quantity'] + 1e-6)

        if 'sales' in df.columns:
            df['is_high_value'] = (df['sales'] > df['sales'].quantile(0.75)).astype(int)

        print("✅ Financial features created: profit_margin_pct, sales_per_item, is_high_value")

        return df

    # =========================================================================
    # ENHANCED FEATURES (RFM, Zone, Category, Interactions)
    # =========================================================================

    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) customer segmentation features.

        RFM Analysis groups customers by:
        - Recency: Days since last order (lower = better)
        - Frequency: Number of orders (higher = better)
        - Monetary: Total spending (higher = better)

        This helps identify valuable customers and at-risk segments.
        """
        print("✅ Creating RFM customer features...")

        if 'customer_id' not in df.columns:
            print("   ⚠️ No customer_id column, skipping RFM features")
            return df

        date_col = next((c for c in df.columns if 'order' in c.lower() and 'date' in c.lower()), None)

        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            reference_date = df[date_col].max()

            # Calculate RFM metrics
            order_col = 'order_id' if 'order_id' in df.columns else date_col
            sales_col = 'sales' if 'sales' in df.columns else date_col

            rfm = df.groupby('customer_id').agg({
                date_col: lambda x: (reference_date - x.max()).days,
                order_col: 'count' if order_col != date_col else 'count',
                sales_col: 'sum' if sales_col != date_col else 'count'
            })

            rfm.columns = ['recency', 'frequency', 'monetary']

            df['rfm_recency'] = df['customer_id'].map(rfm['recency'])
            df['rfm_frequency'] = df['customer_id'].map(rfm['frequency'])
            df['rfm_monetary'] = df['customer_id'].map(rfm['monetary'])

            # Create quintile scores
            try:
                df['rfm_r_score'] = pd.qcut(df['rfm_recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
                df['rfm_f_score'] = pd.qcut(df['rfm_frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                df['rfm_m_score'] = pd.qcut(df['rfm_monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

                df['rfm_score'] = (df['rfm_r_score'].astype(float) +
                                  df['rfm_f_score'].astype(float) +
                                  df['rfm_m_score'].astype(float))

                print(f"   Created RFM features: recency, frequency, monetary, rfm_score")

            except Exception as e:
                print(f"   ⚠️ Could not create RFM quintiles: {str(e)}")

        return df

    def create_zone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create zone-based aggregation features.

        These features capture regional patterns:
        - Zone late rate: Historical late delivery rate by region
        - Zone avg scheduled days: Average promised delivery time
        - Zone complexity: Number of cities (delivery difficulty proxy)
        - Zone order volume: Order volume by region
        """
        print("✅ Creating zone-based aggregation features...")

        if 'order_region' not in df.columns:
            print("   ⚠️ No order_region column, skipping zone features")
            return df

        # Zone-level late delivery rate (if target available)
        if 'late_delivery_risk' in df.columns:
            zone_late_rate = df.groupby('order_region')['late_delivery_risk'].mean()
            df['zone_late_rate'] = df['order_region'].map(zone_late_rate)
            self.zone_stats['late_rate'] = zone_late_rate

        # Zone-level average scheduled shipping days
        sched_col = next((c for c in df.columns if 'scheduled' in c.lower() and 'day' in c.lower()), None)
        if sched_col:
            zone_avg_sched = df.groupby('order_region')[sched_col].mean()
            df['zone_avg_scheduled_days'] = df['order_region'].map(zone_avg_sched)
            self.zone_stats['avg_scheduled_days'] = zone_avg_sched

        # Zone complexity
        if 'order_city' in df.columns:
            zone_complexity = df.groupby('order_region')['order_city'].nunique()
            df['zone_complexity'] = df['order_region'].map(zone_complexity)
            self.zone_stats['complexity'] = zone_complexity

        # Zone order volume
        zone_volume = df.groupby('order_region').size()
        df['zone_order_volume'] = df['order_region'].map(zone_volume)
        self.zone_stats['volume'] = zone_volume

        print(f"   Created zone features: zone_late_rate, zone_avg_scheduled_days, zone_complexity, zone_order_volume")

        return df

    def create_category_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create product category risk indicators.

        Some product categories may have different delivery patterns:
        - Category late rate: Historical late rate by category
        - Category avg ship time: Average shipping time by category
        - Category order volume: Volume by category
        """
        print("✅ Creating category risk features...")

        if 'category_name' not in df.columns:
            print("   ⚠️ No category_name column, skipping category features")
            return df

        if 'late_delivery_risk' in df.columns:
            category_risk = df.groupby('category_name')['late_delivery_risk'].mean()
            df['category_risk_score'] = df['category_name'].map(category_risk)
            self.category_risk['late_rate'] = category_risk

        sched_col = next((c for c in df.columns if 'scheduled' in c.lower() and 'day' in c.lower()), None)
        if sched_col:
            category_ship_time = df.groupby('category_name')[sched_col].mean()
            df['category_avg_ship_time'] = df['category_name'].map(category_ship_time)
            self.category_risk['avg_ship_time'] = category_ship_time

        category_volume = df.groupby('category_name').size()
        df['category_order_volume'] = df['category_name'].map(category_volume)
        self.category_risk['volume'] = category_volume

        print(f"   Created category features: category_risk_score, category_avg_ship_time, category_order_volume")

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.

        Interactions capture combined effects:
        - Region × Shipping Mode: Some modes work better in certain regions
        - High Value × Weekend: Weekend orders of high-value items
        - Urgency × Zone Complexity: Urgent shipments to complex zones
        """
        print("✅ Creating interaction features...")

        if 'order_region' in df.columns and 'shipping_mode' in df.columns:
            df['region_shipping_combo'] = (df['order_region'].astype(str) + '_' +
                                           df['shipping_mode'].astype(str))

        if 'is_high_value' in df.columns and 'is_weekend' in df.columns:
            df['high_value_weekend'] = df['is_high_value'] * df['is_weekend']

        if 'shipping_urgency' in df.columns and 'zone_complexity' in df.columns:
            df['urgency_distance'] = df['shipping_urgency'] * df['zone_complexity']

        if 'customer_segment' in df.columns and 'category_name' in df.columns:
            df['segment_category_combo'] = (df['customer_segment'].astype(str) + '_' +
                                            df['category_name'].astype(str))

        if 'rfm_score' in df.columns and 'order_value' in df.columns:
            df['rfm_order_value'] = df['rfm_score'] * df['order_value']

        print(f"   Created interaction features")

        return df

    # =========================================================================
    # ENCODING
    # =========================================================================

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using label encoding.

        IMPORTANT: Excludes 'delivery_status' to prevent data leakage!
        """
        # SAFE categorical columns (available at order time, NOT target-related)
        categorical_cols = [
            'type', 'category_name', 'customer_segment',
            'department_name', 'market', 'order_region', 'order_country',
            'order_state', 'order_city', 'shipping_mode'
        ]

        self.categorical_columns = [col for col in categorical_cols if col in df.columns]

        for col in self.categorical_columns:
            if fit:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )

        if fit:
            print(f"✅ Encoded {len(self.categorical_columns)} categorical features")
            print(f"   (Excluded 'delivery_status' to prevent leakage)")

        return df

    def apply_target_encoding(self, df: pd.DataFrame, target_col: str = 'late_delivery_risk',
                             fit: bool = True) -> pd.DataFrame:
        """Apply target encoding to high-cardinality categorical features."""
        if not TARGET_ENCODER_AVAILABLE or not self.use_target_encoding:
            return df

        print("✅ Applying target encoding (out-of-fold)...")

        high_card_cols = []
        for col in ['region_shipping_combo', 'segment_category_combo', 'order_city']:
            if col in df.columns and df[col].nunique() > 20:
                high_card_cols.append(col)

        if not high_card_cols or target_col not in df.columns:
            return df

        if fit:
            self.target_encoder = TargetEncoder(cols=high_card_cols, smoothing=10, min_samples_leaf=20)
            df[high_card_cols] = self.target_encoder.fit_transform(df[high_card_cols], df[target_col])
            print(f"   Encoded {len(high_card_cols)} high-cardinality features")
        else:
            if self.target_encoder:
                df[high_card_cols] = self.target_encoder.transform(df[high_card_cols])

        return df

    # =========================================================================
    # FEATURE SELECTION
    # =========================================================================

    def select_features_for_classification(self, df: pd.DataFrame,
                                           include_enhanced: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select features for late delivery classification.

        IMPORTANT: Only uses features available at ORDER TIME.
        Excludes all post-delivery information to prevent leakage.

        Args:
            df: DataFrame with all features
            include_enhanced: Whether to include RFM, zone, interaction features

        Returns:
            X: Feature matrix
            y: Target variable (late_delivery)
        """
        feature_cols = []

        # Temporal features
        temporal = ['order_day_of_week', 'order_month', 'order_quarter', 'is_weekend',
                   'days_since_start', 'order_hour']
        feature_cols.extend([c for c in temporal if c in df.columns])

        # Customer features
        customer = ['customer_order_count', 'customer_lifetime_value']
        if include_enhanced:
            customer.extend(['rfm_recency', 'rfm_frequency', 'rfm_monetary', 'rfm_score'])
        feature_cols.extend([c for c in customer if c in df.columns])

        # Product features
        product = ['product_popularity', 'category_popularity', 'order_value', 'discount_rate']
        if include_enhanced:
            product.extend(['category_risk_score', 'category_avg_ship_time', 'category_order_volume'])
        feature_cols.extend([c for c in product if c in df.columns])

        # Shipping features
        shipping = ['shipping_urgency', 'scheduled_shipping_days']
        if include_enhanced:
            shipping.extend(['zone_late_rate', 'zone_avg_scheduled_days', 'zone_complexity', 'zone_order_volume'])
        feature_cols.extend([c for c in shipping if c in df.columns])

        # Financial features
        financial = ['profit_margin_pct', 'sales_per_item', 'is_high_value', 'order_item_quantity']
        feature_cols.extend([c for c in financial if c in df.columns])

        # Interaction features
        if include_enhanced:
            interaction = ['high_value_weekend', 'urgency_distance', 'rfm_order_value']
            feature_cols.extend([c for c in interaction if c in df.columns])

        # Encoded categorical features
        encoded = [f'{c}_encoded' for c in self.categorical_columns if f'{c}_encoded' in df.columns]
        feature_cols.extend(encoded)

        # Remove duplicates and VERIFY no leaky columns
        feature_cols = list(set(feature_cols))
        feature_cols = [c for c in feature_cols
                       if c.lower() not in [l.lower() for l in LEAKY_COLUMNS]
                       and not c.lower().startswith('delivery')
                       and not c.lower().startswith('late_delivery')]

        print(f"\n✅ Selected {len(feature_cols)} features for classification")
        print(f"   (Verified: No leaky features included)")

        X = df[feature_cols].copy()

        # Target
        if 'late_delivery_risk' in df.columns:
            y = df['late_delivery_risk'].astype(int)
        elif 'late_delivery' in df.columns:
            y = df['late_delivery'].astype(int)
        else:
            y = None

        return X, y

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def fit_transform(self, df: pd.DataFrame, include_enhanced: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete feature engineering pipeline for classification.

        Args:
            df: Preprocessed DataFrame
            include_enhanced: Include RFM, zone, and interaction features

        Returns:
            X: Feature matrix
            y: Target variable
        """
        print("=" * 60)
        print("ADVANCED FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        print("⚠️  LEAKAGE PREVENTION ACTIVE")
        print(f"    Excluded columns: {', '.join(list(LEAKY_COLUMNS)[:4])}...")
        print("=" * 60)

        # Base features
        df = self.create_temporal_features(df)
        df = self.create_customer_features(df)
        df = self.create_product_features(df)
        df = self.create_shipping_features(df)
        df = self.create_financial_features(df)

        # Enhanced features
        if include_enhanced:
            df = self.create_rfm_features(df)
            df = self.create_zone_features(df)
            df = self.create_category_risk_features(df)
            df = self.create_interaction_features(df)

        # Encode categoricals
        df = self.encode_categorical_features(df, fit=True)

        # Target encoding (optional)
        if self.use_target_encoding:
            df = self.apply_target_encoding(df, fit=True)

        # Select features
        X, y = self.select_features_for_classification(df, include_enhanced=include_enhanced)

        # Handle NaN
        X = X.fillna(0)

        print(f"\nFeature matrix shape: {X.shape}")
        if y is not None:
            print(f"Target distribution: {y.value_counts().to_dict()}")
        print("=" * 60)

        return X, y

    def transform(self, df: pd.DataFrame, include_enhanced: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply feature engineering using fitted parameters."""
        df = self.create_temporal_features(df)
        df = self.create_customer_features(df)
        df = self.create_product_features(df)
        df = self.create_shipping_features(df)
        df = self.create_financial_features(df)

        if include_enhanced:
            df = self.create_rfm_features(df)
            df = self.create_zone_features(df)
            df = self.create_category_risk_features(df)
            df = self.create_interaction_features(df)

        df = self.encode_categorical_features(df, fit=False)

        if self.use_target_encoding:
            df = self.apply_target_encoding(df, fit=False)

        X, y = self.select_features_for_classification(df, include_enhanced=include_enhanced)
        X = X.fillna(0)

        return X, y


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_features_pipeline(df: pd.DataFrame, include_enhanced: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function for feature engineering.

    Args:
        df: Preprocessed DataFrame
        include_enhanced: Include RFM, zone, and interaction features

    Returns:
        X: Feature matrix
        y: Target variable
    """
    engineer = AdvancedFeatureEngineer()
    X, y = engineer.fit_transform(df, include_enhanced=include_enhanced)
    return X, y


def build_basic_features_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build only basic features (no RFM, zone, or interaction features)."""
    engineer = AdvancedFeatureEngineer()
    X, y = engineer.fit_transform(df, include_enhanced=False)
    return X, y


if __name__ == "__main__":
    from src.data.preprocess import load_and_preprocess

    df = load_and_preprocess()
    X, y = build_features_pipeline(df)

    print("\n✅ Feature engineering complete!")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"\nFeature columns: {list(X.columns[:15])}...")

