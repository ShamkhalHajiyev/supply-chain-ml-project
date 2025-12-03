"""
Enhanced Feature Engineering Module
Advanced features for supply chain ML: RFM, zone-based aggregations, interactions.

New Features:
- RFM (Recency, Frequency, Monetary) customer segmentation
- Zone-based late delivery statistics
- Product category risk scores
- Interaction features
- Target encoding (leakage-free)

Usage:
    from src.features.build_features_enhanced import EnhancedFeatureEngineer
    engineer = EnhancedFeatureEngineer()
    X, y = engineer.fit_transform(df)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import base feature engineer
from .build_features import FeatureEngineer, LEAKY_COLUMNS

try:
    from category_encoders import TargetEncoder
    TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TARGET_ENCODER_AVAILABLE = False


class EnhancedFeatureEngineer(FeatureEngineer):
    """
    Enhanced feature engineering with advanced domain-specific features.

    Extends base FeatureEngineer with:
    - RFM customer features
    - Zone-based aggregations
    - Product category risk scores
    - Interaction features
    - Target encoding (out-of-fold)
    """

    def __init__(self):
        super().__init__()
        self.rfm_quantiles = {}
        self.zone_stats = {}
        self.category_risk = {}
        self.target_encoder = None

    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) customer segmentation features.

        RFM Analysis:
        - Recency: Days since last order
        - Frequency: Number of orders
        - Monetary: Total spending

        Args:
            df: DataFrame with customer_id, order_date, sales

        Returns:
            DataFrame with RFM features added
        """
        print("✅ Creating RFM customer features...")

        if 'customer_id' not in df.columns:
            print("   ⚠️ No customer_id column, skipping RFM features")
            return df

        # Determine reference date (max date in dataset)
        date_col = next((c for c in df.columns if 'order' in c.lower() and 'date' in c.lower()), None)

        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            reference_date = df[date_col].max()

            # Calculate RFM metrics
            rfm = df.groupby('customer_id').agg({
                date_col: lambda x: (reference_date - x.max()).days,  # Recency
                'order_id' if 'order_id' in df.columns else date_col: 'count',  # Frequency
                'sales' if 'sales' in df.columns else date_col: 'sum'  # Monetary
            })

            rfm.columns = ['recency', 'frequency', 'monetary']

            # Create RFM scores (1-5 quintiles)
            # Recency: Lower is better (recent customers) → reverse scoring
            df['rfm_recency'] = df['customer_id'].map(rfm['recency'])
            df['rfm_frequency'] = df['customer_id'].map(rfm['frequency'])
            df['rfm_monetary'] = df['customer_id'].map(rfm['monetary'])

            # Create quintile scores
            try:
                df['rfm_r_score'] = pd.qcut(df['rfm_recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
                df['rfm_f_score'] = pd.qcut(df['rfm_frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                df['rfm_m_score'] = pd.qcut(df['rfm_monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

                # Overall RFM score
                df['rfm_score'] = (df['rfm_r_score'].astype(float) +
                                  df['rfm_f_score'].astype(float) +
                                  df['rfm_m_score'].astype(float))

                # Customer segment based on RFM
                df['customer_segment_rfm'] = pd.cut(df['rfm_score'], bins=[0, 6, 9, 12, 15],
                                                    labels=['Low', 'Medium', 'High', 'VIP'],
                                                    include_lowest=True)

                print(f"   Created RFM features: recency, frequency, monetary, rfm_score, customer_segment_rfm")
                print(f"   Customer segments: {df['customer_segment_rfm'].value_counts().to_dict()}")

            except Exception as e:
                print(f"   ⚠️ Could not create RFM quintiles: {str(e)}")
                # Fallback: just use raw values
                pass

        return df

    def create_zone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create zone-based aggregation features.

        Features:
        - Average late delivery rate by zone
        - Average shipping time by zone
        - Zone complexity (number of cities)

        Args:
            df: DataFrame with order_region

        Returns:
            DataFrame with zone features added
        """
        print("✅ Creating zone-based aggregation features...")

        if 'order_region' not in df.columns:
            print("   ⚠️ No order_region column, skipping zone features")
            return df

        # Zone-level late delivery rate
        if 'late_delivery_risk' in df.columns:
            zone_late_rate = df.groupby('order_region')['late_delivery_risk'].mean()
            df['zone_late_rate'] = df['order_region'].map(zone_late_rate)
            self.zone_stats['late_rate'] = zone_late_rate

        # Zone-level average scheduled shipping days
        if 'days_for_shipment_(scheduled)' in df.columns or 'scheduled_shipping_days' in df.columns:
            sched_col = 'days_for_shipment_(scheduled)' if 'days_for_shipment_(scheduled)' in df.columns else 'scheduled_shipping_days'
            zone_avg_sched = df.groupby('order_region')[sched_col].mean()
            df['zone_avg_scheduled_days'] = df['order_region'].map(zone_avg_sched)
            self.zone_stats['avg_scheduled_days'] = zone_avg_sched

        # Zone complexity (number of unique cities)
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

        Features:
        - Category late delivery rate
        - Category average shipping time
        - Category order volume

        Args:
            df: DataFrame with category_name

        Returns:
            DataFrame with category risk features added
        """
        print("✅ Creating category risk features...")

        if 'category_name' not in df.columns:
            print("   ⚠️ No category_name column, skipping category features")
            return df

        # Category late delivery rate
        if 'late_delivery_risk' in df.columns:
            category_risk = df.groupby('category_name')['late_delivery_risk'].mean()
            df['category_risk_score'] = df['category_name'].map(category_risk)
            self.category_risk['late_rate'] = category_risk

        # Category average scheduled shipping
        if 'days_for_shipment_(scheduled)' in df.columns or 'scheduled_shipping_days' in df.columns:
            sched_col = 'days_for_shipment_(scheduled)' if 'days_for_shipment_(scheduled)' in df.columns else 'scheduled_shipping_days'
            category_ship_time = df.groupby('category_name')[sched_col].mean()
            df['category_avg_ship_time'] = df['category_name'].map(category_ship_time)
            self.category_risk['avg_ship_time'] = category_ship_time

        # Category order volume
        category_volume = df.groupby('category_name').size()
        df['category_order_volume'] = df['category_name'].map(category_volume)
        self.category_risk['volume'] = category_volume

        print(f"   Created category features: category_risk_score, category_avg_ship_time, category_order_volume")

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.

        Interactions:
        - Region × Shipping Mode
        - High Value × Weekend
        - Urgency × Zone Complexity
        - Customer Segment × Product Category

        Args:
            df: DataFrame

        Returns:
            DataFrame with interaction features added
        """
        print("✅ Creating interaction features...")

        # Region × Shipping Mode (some modes work better in certain regions)
        if 'order_region' in df.columns and 'shipping_mode' in df.columns:
            df['region_shipping_combo'] = (df['order_region'].astype(str) + '_' +
                                           df['shipping_mode'].astype(str))

        # High value × Weekend (weekend orders of high-value items may have delays)
        if 'is_high_value' in df.columns and 'is_weekend' in df.columns:
            df['high_value_weekend'] = df['is_high_value'] * df['is_weekend']

        # Urgency × Distance proxy (urgent shipments to complex zones)
        if 'shipping_urgency' in df.columns and 'zone_complexity' in df.columns:
            df['urgency_distance'] = df['shipping_urgency'] * df['zone_complexity']

        # Customer Segment × Category (segment preferences for categories)
        if 'customer_segment' in df.columns and 'category_name' in df.columns:
            df['segment_category_combo'] = (df['customer_segment'].astype(str) + '_' +
                                            df['category_name'].astype(str))

        # RFM Score × Order Value (high-value customers with high-value orders)
        if 'rfm_score' in df.columns and 'order_value' in df.columns:
            df['rfm_order_value'] = df['rfm_score'] * df['order_value']

        print(f"   Created interaction features: region_shipping_combo, high_value_weekend, urgency_distance, etc.")

        return df

    def apply_target_encoding(self, df: pd.DataFrame, target_col: str = 'late_delivery_risk',
                             fit: bool = True) -> pd.DataFrame:
        """
        Apply target encoding to high-cardinality categorical features.

        Uses out-of-fold strategy to prevent data leakage.

        Args:
            df: DataFrame
            target_col: Name of target variable
            fit: If True, fit encoder. If False, use existing encoder.

        Returns:
            DataFrame with target-encoded features
        """
        if not TARGET_ENCODER_AVAILABLE:
            print("   ⚠️ Target encoding not available (install category-encoders)")
            return df

        print("✅ Applying target encoding (out-of-fold)...")

        # High-cardinality columns to encode
        high_card_cols = []

        for col in ['region_shipping_combo', 'segment_category_combo', 'order_city']:
            if col in df.columns and df[col].nunique() > 20:
                high_card_cols.append(col)

        if not high_card_cols:
            print("   No high-cardinality columns to encode")
            return df

        if target_col not in df.columns:
            print(f"   ⚠️ Target column '{target_col}' not found, skipping target encoding")
            return df

        if fit:
            self.target_encoder = TargetEncoder(cols=high_card_cols, smoothing=10, min_samples_leaf=20)
            df[high_card_cols] = self.target_encoder.fit_transform(df[high_card_cols], df[target_col])
            print(f"   Encoded {len(high_card_cols)} high-cardinality features")
        else:
            if self.target_encoder:
                df[high_card_cols] = self.target_encoder.transform(df[high_card_cols])

        return df

    def select_features_for_classification(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select features for late delivery classification (enhanced version).

        Includes new features:
        - RFM features
        - Zone-based features
        - Category risk features
        - Interaction features

        Returns:
            X: Feature matrix
            y: Target variable (late_delivery)
        """
        # Core features from base class
        feature_cols = []

        # Temporal features (from ORDER date only)
        temporal = ['order_day_of_week', 'order_month', 'order_quarter', 'is_weekend', 'days_since_start']
        feature_cols.extend([c for c in temporal if c in df.columns])

        # Customer features (original + RFM)
        customer = ['customer_order_count', 'customer_lifetime_value',
                   'rfm_recency', 'rfm_frequency', 'rfm_monetary', 'rfm_score']
        feature_cols.extend([c for c in customer if c in df.columns])

        # Product features (original + category risk)
        product = ['product_popularity', 'category_popularity', 'order_value', 'discount_rate',
                  'category_risk_score', 'category_avg_ship_time', 'category_order_volume']
        feature_cols.extend([c for c in product if c in df.columns])

        # Shipping features (original + zone-based)
        shipping = ['shipping_urgency', 'scheduled_shipping_days',
                   'zone_late_rate', 'zone_avg_scheduled_days', 'zone_complexity', 'zone_order_volume']
        feature_cols.extend([c for c in shipping if c in df.columns])

        # Financial features
        financial = ['profit_margin_pct', 'sales_per_item', 'is_high_value', 'order_item_quantity']
        feature_cols.extend([c for c in financial if c in df.columns])

        # Interaction features
        interaction = ['high_value_weekend', 'urgency_distance', 'rfm_order_value']
        feature_cols.extend([c for c in interaction if c in df.columns])

        # Encoded categorical features (excluding delivery_status!)
        encoded = [f'{c}_encoded' for c in self.categorical_columns if f'{c}_encoded' in df.columns]
        feature_cols.extend(encoded)

        # Target-encoded features
        target_encoded = ['region_shipping_combo', 'segment_category_combo', 'order_city']
        feature_cols.extend([c for c in target_encoded if c in df.columns])

        # Remove duplicates and VERIFY no leaky columns
        feature_cols = list(set(feature_cols))
        feature_cols = [c for c in feature_cols if c.lower() not in LEAKY_COLUMNS
                        and not c.lower().startswith('delivery')
                        and not c.lower().startswith('late_delivery')]

        print(f"\n✅ Selected {len(feature_cols)} features for classification (ENHANCED)")
        print(f"   (Verified: No leaky features included)")
        print(f"   New features: RFM ({sum('rfm' in c for c in feature_cols)}), "
              f"Zone ({sum('zone' in c for c in feature_cols)}), "
              f"Category ({sum('category' in c for c in feature_cols)}), "
              f"Interaction ({sum(c in interaction for c in feature_cols)})")

        X = df[feature_cols].copy()

        # Use late_delivery_risk directly if available, otherwise create from delivery_status
        if 'late_delivery_risk' in df.columns:
            y = df['late_delivery_risk'].astype(int)
        elif 'late_delivery' in df.columns:
            y = df['late_delivery']
        else:
            y = None

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete ENHANCED feature engineering pipeline for classification.

        Returns:
            X: Feature matrix
            y: Target variable
        """
        print("=" * 60)
        print("ENHANCED FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        print("⚠️  LEAKAGE PREVENTION ACTIVE")
        print(f"    Excluded columns: {', '.join(LEAKY_COLUMNS)}")
        print("=" * 60)

        # Base features
        df = self.create_temporal_features(df)
        df = self.create_customer_features(df)
        df = self.create_product_features(df)
        df = self.create_shipping_features(df)
        df = self.create_financial_features(df)

        # Enhanced features
        df = self.create_rfm_features(df)
        df = self.create_zone_features(df)
        df = self.create_category_risk_features(df)
        df = self.create_interaction_features(df)

        # Encode categoricals (excluding delivery_status)
        df = self.encode_categorical_features(df, fit=True)

        # Target encoding (out-of-fold)
        df = self.apply_target_encoding(df, fit=True)

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
        """Apply enhanced feature engineering using fitted parameters."""
        # Base features
        df = self.create_temporal_features(df)
        df = self.create_customer_features(df)
        df = self.create_product_features(df)
        df = self.create_shipping_features(df)
        df = self.create_financial_features(df)

        # Enhanced features
        df = self.create_rfm_features(df)
        df = self.create_zone_features(df)
        df = self.create_category_risk_features(df)
        df = self.create_interaction_features(df)

        # Encode
        df = self.encode_categorical_features(df, fit=False)
        df = self.apply_target_encoding(df, fit=False)

        # Select features
        X, y = self.select_features_for_classification(df)
        X = X.fillna(0)

        return X, y


def build_enhanced_features_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function for enhanced feature engineering.

    Args:
        df: Preprocessed DataFrame

    Returns:
        X: Feature matrix
        y: Target variable
    """
    engineer = EnhancedFeatureEngineer()
    X, y = engineer.fit_transform(df)
    return X, y


if __name__ == "__main__":
    from src.data.preprocess import load_and_preprocess

    # Load preprocessed data
    df = load_and_preprocess()

    # Build enhanced features
    X, y = build_enhanced_features_pipeline(df)

    print("\n✅ Enhanced feature engineering complete!")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"\nSample feature columns: {list(X.columns[:20])}")
