"""
Data Preprocessing Script - Sri Lankan Vegetable Prices
========================================================

This script performs all preprocessing steps on the collected dataset:

1. Data Cleaning
   - Fix data types (dates, integers)
   - Handle missing/null values (forward-fill within each vegetable)
   - Treat zero prices as missing (0 LKR is not a valid vegetable price)
   - Remove records with unparseable dates

2. Outlier Treatment
   - Detect outliers using IQR method per vegetable
   - Winsorize (cap) outliers to 1.5*IQR bounds

3. Feature Engineering
   - Time-based features: year, month, week, quarter, season
   - Lag features: price at t-1, t-2, t-4 weeks
   - Rolling statistics: 4-week and 8-week moving averages & std
   - Price change features: week-over-week change, % change
   - Month-sin/cos encoding for cyclical nature of months

4. Encoding
   - Label encoding for vegetable names, categories

5. Normalization
   - Min-Max scaling for price features
   - Standard scaling alternative also saved

Input:  data/sri_lankan_vegetable_prices_long.csv
Output: data/preprocessed_vegetable_prices.csv
        data/preprocessing_summary.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT_FILE = os.path.join(BASE_DATA_DIR, "raw", "sri_lankan_vegetable_prices_long.csv")
OUTPUT_FILE = os.path.join(BASE_DATA_DIR, "processed", "preprocessed_vegetable_prices.csv")
SUMMARY_FILE = os.path.join(BASE_DATA_DIR, "reports", "preprocessing_summary.txt")

# Outlier treatment: IQR multiplier
IQR_MULTIPLIER = 1.5

# Lag periods (in weeks)
LAG_PERIODS = [1, 2, 4]

# Rolling window sizes (in weeks)
ROLLING_WINDOWS = [4, 8]


# ============================================================
# Step 1: Data Cleaning
# ============================================================

def clean_data(df):
    """Clean raw data: fix types, handle missing values."""
    print("[1/5] Cleaning data...")
    summary = []

    original_shape = df.shape
    summary.append(f"Original shape: {original_shape}")

    # 1a. Parse date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    null_dates = df["date"].isna().sum()
    summary.append(f"Records with unparseable dates removed: {null_dates}")
    df = df.dropna(subset=["date"]).copy()

    # 1b. Fix year, month, week_of_year to integers
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # 1c. Treat zero prices as missing (0 LKR is not valid for vegetables)
    zero_count = (df["price_lkr"] == 0).sum()
    summary.append(f"Zero prices treated as missing: {zero_count}")
    df.loc[df["price_lkr"] == 0, "price_lkr"] = np.nan

    # 1d. Report missing prices
    missing_before = df["price_lkr"].isna().sum()
    summary.append(f"Missing prices before imputation: {missing_before} ({missing_before/len(df)*100:.1f}%)")

    # 1e. Forward-fill missing prices within each vegetable (time-series imputation)
    df = df.sort_values(["vegetable", "date"]).reset_index(drop=True)
    df["price_lkr"] = df.groupby("vegetable")["price_lkr"].transform(
        lambda x: x.ffill().bfill()
    )

    missing_after = df["price_lkr"].isna().sum()
    summary.append(f"Missing prices after imputation: {missing_after}")

    # 1f. Drop date_raw column (no longer needed)
    df = df.drop(columns=["date_raw"])

    summary.append(f"Cleaned shape: {df.shape}")
    print(f"  Removed {null_dates} records with bad dates")
    print(f"  Imputed {missing_before - missing_after} missing prices (forward-fill)")
    print(f"  Cleaned shape: {df.shape}")

    return df, summary


# ============================================================
# Step 2: Outlier Treatment
# ============================================================

def treat_outliers(df):
    """Detect and winsorize outliers using IQR method per vegetable."""
    print("\n[2/5] Treating outliers (IQR winsorization)...")
    summary = []

    total_capped = 0
    outlier_details = []

    for veg in df["vegetable"].unique():
        mask = df["vegetable"] == veg
        prices = df.loc[mask, "price_lkr"]

        Q1 = prices.quantile(0.25)
        Q3 = prices.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - IQR_MULTIPLIER * IQR
        upper = Q3 + IQR_MULTIPLIER * IQR

        # Count outliers
        n_outliers = ((prices < lower) | (prices > upper)).sum()
        if n_outliers > 0:
            outlier_details.append(f"  {veg}: {n_outliers} outliers capped to [{lower:.1f}, {upper:.1f}]")
            total_capped += n_outliers

        # Cap (winsorize)
        df.loc[mask, "price_lkr"] = prices.clip(lower=lower, upper=upper)

    summary.append(f"Total outliers winsorized: {total_capped}")
    summary.extend(outlier_details)

    print(f"  Winsorized {total_capped} outlier values across {len(outlier_details)} vegetables")
    return df, summary


# ============================================================
# Step 3: Feature Engineering
# ============================================================

def engineer_features(df):
    """Create time-based, lag, rolling, and derived features."""
    print("\n[3/5] Engineering features...")
    summary = []

    # Sort by vegetable and date for proper time-series feature creation
    df = df.sort_values(["vegetable", "date"]).reset_index(drop=True)

    # 3a. Time-based features
    df["quarter"] = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear

    # Cyclical encoding for month (captures Jan-Dec circular nature)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Season mapping for Sri Lanka (tropical climate)
    # Yala season: May-Sep (SW monsoon), Maha season: Oct-Mar (NE monsoon), Inter-monsoon: Apr, Oct
    def get_season(month):
        if month in [5, 6, 7, 8, 9]:
            return "Yala_SW_Monsoon"
        elif month in [11, 12, 1, 2, 3]:
            return "Maha_NE_Monsoon"
        else:  # April, October
            return "Inter_Monsoon"

    df["season"] = df["month"].apply(get_season)

    summary.append("Time features added: quarter, day_of_year, month_sin, month_cos, season")

    # 3b. Lag features (within each vegetable)
    for lag in LAG_PERIODS:
        col_name = f"price_lag_{lag}w"
        df[col_name] = df.groupby("vegetable")["price_lkr"].shift(lag)
        summary.append(f"Lag feature: {col_name}")

    # 3c. Rolling statistics (within each vegetable)
    # IMPORTANT: Use shift(1) BEFORE rolling to exclude current price (avoid data leakage)
    for window in ROLLING_WINDOWS:
        roll = df.groupby("vegetable")["price_lkr"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"price_rolling_mean_{window}w"] = roll

        roll_std = df.groupby("vegetable")["price_lkr"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        df[f"price_rolling_std_{window}w"] = roll_std

        summary.append(f"Rolling features (shifted): mean_{window}w, std_{window}w")

    # 3d. Price change features (using lagged values to avoid leakage)
    # price_change = lag_1w - lag_2w (change between previous two weeks)
    df["price_change_1w"] = df.groupby("vegetable")["price_lkr"].transform(
        lambda x: x.shift(1) - x.shift(2)
    )
    df["price_pct_change_1w"] = df.groupby("vegetable")["price_lkr"].transform(
        lambda x: (x.shift(1) - x.shift(2)) / x.shift(2) * 100
    )

    summary.append("Price change features (lagged): price_change_1w, price_pct_change_1w")

    # 3e. Drop rows with NaN from lag/rolling (first few weeks per vegetable)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df)

    summary.append(f"Rows dropped due to lag/rolling NaN: {rows_dropped}")
    summary.append(f"Final shape after feature engineering: {df.shape}")

    print(f"  Added lag features: {LAG_PERIODS}")
    print(f"  Added rolling windows: {ROLLING_WINDOWS}")
    print(f"  Added seasonal/cyclical features")
    print(f"  Dropped {rows_dropped} initial rows (lag/rolling warm-up)")
    print(f"  Shape: {df.shape}")

    return df, summary


# ============================================================
# Step 4: Encoding
# ============================================================

def encode_categoricals(df):
    """Label-encode categorical columns."""
    print("\n[4/5] Encoding categorical variables...")
    summary = []

    encoders = {}
    categorical_cols = ["vegetable", "category", "sub_category", "season"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        summary.append(f"Encoded '{col}': {len(le.classes_)} unique values")
        print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return df, summary, encoders


# ============================================================
# Step 5: Normalization
# ============================================================

def normalize_features(df):
    """Apply Min-Max scaling to numerical features."""
    print("\n[5/5] Normalizing numerical features...")
    summary = []

    # Columns to normalize (price-related numerical features)
    price_cols = [
        "price_lkr",
        "price_lag_1w", "price_lag_2w", "price_lag_4w",
        "price_rolling_mean_4w", "price_rolling_mean_8w",
        "price_rolling_std_4w", "price_rolling_std_8w",
        "price_change_1w", "price_pct_change_1w",
    ]

    # Min-Max scaling
    scaler = MinMaxScaler()
    normalized_cols = [f"{col}_normalized" for col in price_cols]
    df[normalized_cols] = scaler.fit_transform(df[price_cols])

    summary.append(f"Min-Max normalized {len(price_cols)} price features")

    # Standard scaling (alternative)
    std_scaler = StandardScaler()
    standardized_cols = [f"{col}_standardized" for col in price_cols]
    df[standardized_cols] = std_scaler.fit_transform(df[price_cols])

    summary.append(f"Standard-scaled {len(price_cols)} price features")

    print(f"  Applied Min-Max normalization to {len(price_cols)} columns")
    print(f"  Applied Standard scaling to {len(price_cols)} columns")

    return df, summary


# ============================================================
# Main Pipeline
# ============================================================

def preprocess():
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load raw data
    print(f"\nLoading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} records, {df.shape[1]} columns")

    all_summaries = []

    # Step 1: Clean
    df, summary = clean_data(df)
    all_summaries.append(("DATA CLEANING", summary))

    # Step 2: Outlier treatment
    df, summary = treat_outliers(df)
    all_summaries.append(("OUTLIER TREATMENT", summary))

    # Step 3: Feature engineering
    df, summary = engineer_features(df)
    all_summaries.append(("FEATURE ENGINEERING", summary))

    # Step 4: Encoding
    df, summary, encoders = encode_categoricals(df)
    all_summaries.append(("ENCODING", summary))

    # Step 5: Normalization
    df, summary = normalize_features(df)
    all_summaries.append(("NORMALIZATION", summary))

    # Save preprocessed data
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Preprocessed data: {OUTPUT_FILE}")
    print(f"  Final shape: {df.shape}")

    # Save summary report
    with open(SUMMARY_FILE, "w") as f:
        f.write("PREPROCESSING SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input:  {INPUT_FILE}\n")
        f.write(f"Output: {OUTPUT_FILE}\n\n")
        for section_name, items in all_summaries:
            f.write(f"\n{section_name}\n")
            f.write("-" * 40 + "\n")
            for item in items:
                f.write(f"  {item}\n")
        f.write(f"\nFinal Dataset Shape: {df.shape}\n")
        f.write(f"Final Columns ({len(df.columns)}):\n")
        for col in df.columns:
            f.write(f"  - {col} ({df[col].dtype})\n")
    print(f"  Summary report: {SUMMARY_FILE}")

    # Print final column overview
    print(f"\n  Final columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    {col:40s} {str(df[col].dtype):>10s}")

    # Quick stats on final data
    print(f"\n  Records: {len(df)}")
    print(f"  Missing values: {df.isna().sum().sum()}")
    print(f"  Vegetables: {df['vegetable'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    return df


if __name__ == "__main__":
    df = preprocess()
