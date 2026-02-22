"""
Model Training Script - XGBoost for Sri Lankan Vegetable Price Prediction
==========================================================================

Algorithm: XGBoost (Extreme Gradient Boosting)
    XGBoost is an ensemble learning method that builds an additive model
    of decision trees in a sequential manner. Unlike standard decision trees,
    XGBoost uses gradient boosting with regularization (L1 and L2) to prevent
    overfitting, and employs second-order Taylor expansion for the loss
    function optimization. It differs from standard models (decision trees,
    logistic regression, k-NN) in that:
      - It sequentially corrects errors from previous trees (boosting)
      - Has built-in L1/L2 regularization to control model complexity
      - Uses histogram-based split finding for efficiency
      - Handles missing values natively
      - Provides feature importance scores for explainability

Pipeline:
    1. Load preprocessed data and select features/target
    2. Chronological train/validation/test split (70/15/15)
    3. Train baseline model (Ridge Regression) for comparison
    4. Train XGBoost with default parameters
    5. Hyperparameter tuning via RandomizedSearchCV + TimeSeriesSplit
    6. Evaluate best model on test set
    7. Save model, plots, and results

Input:  data/preprocessed_vegetable_prices.csv
Output: data/xgboost_model.pkl
        data/training_results.txt
        data/feature_importance.png
        data/actual_vs_predicted.png
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT_FILE = os.path.join(BASE_DATA_DIR, "processed", "preprocessed_vegetable_prices.csv")
MODEL_FILE = os.path.join(BASE_DATA_DIR, "models", "xgboost_model.pkl")
RESULTS_FILE = os.path.join(BASE_DATA_DIR, "reports", "training_results.txt")
FEATURE_IMPORTANCE_PLOT = os.path.join(BASE_DATA_DIR, "plots", "feature_importance.png")
ACTUAL_VS_PREDICTED_PLOT = os.path.join(BASE_DATA_DIR, "plots", "actual_vs_predicted.png")

FEATURE_COLUMNS = [
    "year",
    "month",
    "week_of_year",
    "quarter",
    "day_of_year",
    "month_sin",
    "month_cos",
    "vegetable_encoded",
    "category_encoded",
    "sub_category_encoded",
    "season_encoded",
    "price_lag_1w",
    "price_lag_2w",
    "price_lag_4w",
    "price_rolling_mean_4w",
    "price_rolling_std_4w",
    "price_rolling_mean_8w",
    "price_rolling_std_8w",
    "price_change_1w",
    "price_pct_change_1w",
]

TARGET_COLUMN = "price_lkr"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500, 700, 1000],
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
}

N_SEARCH_ITER = 50
CV_SPLITS = 5
RANDOM_STATE = 42

def load_data():
    """Load preprocessed data and select features/target."""
    print("[1/7] Loading preprocessed data...")
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print(f"  Total samples: {len(df)}")
    print(f"  Features: {len(FEATURE_COLUMNS)}")
    print(f"  Target: {TARGET_COLUMN}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df, X, y

def split_data(df, X, y):
    """Split data chronologically (not randomly) for time-series integrity."""
    print("\n[2/7] Splitting data chronologically (70/15/15)...")

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    dates = df["date"]
    print(f"  Train: {len(X_train)} samples ({dates.iloc[0].date()} to {dates.iloc[train_end-1].date()})")
    print(f"  Val:   {len(X_val)} samples ({dates.iloc[train_end].date()} to {dates.iloc[val_end-1].date()})")
    print(f"  Test:  {len(X_test)} samples ({dates.iloc[val_end].date()} to {dates.iloc[n-1].date()})")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_baseline(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Ridge Regression as baseline for comparison."""
    print("\n[3/7] Training baseline model (Ridge Regression)...")

    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_train, y_train)

    y_val_pred = ridge.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    y_test_pred = ridge.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

    print(f"  Validation - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
    print(f"  Test       - RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}, MAPE: {test_mape:.2f}%")

    baseline_results = {
        "val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2,
        "test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2,
        "test_mape": test_mape,
    }
    return ridge, baseline_results

def train_xgboost_default(X_train, y_train, X_val, y_val):
    """Train XGBoost with default parameters."""
    print("\n[4/7] Training XGBoost with default parameters...")

    xgb_default = XGBRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_default.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_val_pred = xgb_default.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"  Validation - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")

    return xgb_default

def tune_hyperparameters(X_train, y_train):
    """Tune XGBoost hyperparameters using RandomizedSearchCV with TimeSeriesSplit."""
    print(f"\n[5/7] Hyperparameter tuning ({N_SEARCH_ITER} iterations, {CV_SPLITS}-fold TimeSeriesSplit)...")

    xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_SEARCH_ITER,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"\n  Best RMSE (CV): {-search.best_score_:.2f}")
    print(f"  Best parameters:")
    for param, value in search.best_params_.items():
        print(f"    {param}: {value}")

    return search.best_estimator_, search.best_params_, -search.best_score_

def evaluate_model(model, X_test, y_test, df, model_name="XGBoost"):
    """Evaluate model on test set with comprehensive metrics."""
    print(f"\n[6/7] Evaluating {model_name} on test set...")

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"  RMSE:  {rmse:.2f} LKR")
    print(f"  MAE:   {mae:.2f} LKR")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")

    test_idx = X_test.index
    test_df = df.loc[test_idx].copy()
    test_df["predicted"] = y_pred
    test_df["actual"] = y_test.values

    print(f"\n  Per-Vegetable Performance:")
    print(f"  {'Vegetable':<30s} {'RMSE':>8s} {'MAE':>8s} {'R²':>8s} {'MAPE%':>8s}")
    print(f"  {'-'*62}")

    veg_results = {}
    for veg in sorted(test_df["vegetable"].unique()):
        veg_mask = test_df["vegetable"] == veg
        veg_actual = test_df.loc[veg_mask, "actual"]
        veg_pred = test_df.loc[veg_mask, "predicted"]

        v_rmse = np.sqrt(mean_squared_error(veg_actual, veg_pred))
        v_mae = mean_absolute_error(veg_actual, veg_pred)
        v_r2 = r2_score(veg_actual, veg_pred) if len(veg_actual) > 1 else 0
        v_mape = mean_absolute_percentage_error(veg_actual, veg_pred) * 100

        veg_results[veg] = {"rmse": v_rmse, "mae": v_mae, "r2": v_r2, "mape": v_mape}
        print(f"  {veg:<30s} {v_rmse:>8.2f} {v_mae:>8.2f} {v_r2:>8.4f} {v_mape:>7.2f}%")

    overall_metrics = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
    return y_pred, overall_metrics, veg_results

def plot_feature_importance(model, feature_names):
    """Plot and save feature importance chart."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]

    bars = ax.barh(range(len(sorted_names)), sorted_importance[::-1], color="steelblue")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title("XGBoost Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot: {FEATURE_IMPORTANCE_PLOT}")


def plot_actual_vs_predicted(y_test, y_pred, df, test_idx):
    """Plot actual vs predicted prices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    ax1.set_xlabel("Actual Price (LKR)", fontsize=11)
    ax1.set_ylabel("Predicted Price (LKR)", fontsize=11)
    ax1.set_title("Actual vs Predicted Prices", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    test_df = df.loc[test_idx].copy()
    test_df["predicted"] = y_pred

    sample_veg = "Carrot  1kg"
    veg_data = test_df[test_df["vegetable"] == sample_veg].sort_values("date")
    ax2.plot(veg_data["date"], veg_data[TARGET_COLUMN], label="Actual", color="steelblue", linewidth=1.5)
    ax2.plot(veg_data["date"], veg_data["predicted"], label="Predicted", color="orange", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Price (LKR)", fontsize=11)
    ax2.set_title(f"Price Prediction: {sample_veg}", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(ACTUAL_VS_PREDICTED_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Actual vs Predicted plot: {ACTUAL_VS_PREDICTED_PLOT}")


def save_results(baseline_results, xgb_metrics, veg_results, best_params, cv_rmse):
    """Save training results summary."""
    with open(RESULTS_FILE, "w") as f:
        f.write("MODEL TRAINING RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("ALGORITHM: XGBoost (Extreme Gradient Boosting)\n")
        f.write("TARGET: Weekly vegetable price (LKR)\n")
        f.write(f"FEATURES: {len(FEATURE_COLUMNS)} features\n")
        f.write(f"SPLIT: Chronological 70/15/15 (train/val/test)\n\n")

        f.write("BASELINE MODEL (Ridge Regression)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Test RMSE:  {baseline_results['test_rmse']:.2f} LKR\n")
        f.write(f"  Test MAE:   {baseline_results['test_mae']:.2f} LKR\n")
        f.write(f"  Test R²:    {baseline_results['test_r2']:.4f}\n")
        f.write(f"  Test MAPE:  {baseline_results['test_mape']:.2f}%\n\n")

        f.write("XGBOOST MODEL (Tuned)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Cross-Validation RMSE: {cv_rmse:.2f} LKR\n")
        f.write(f"  Test RMSE:  {xgb_metrics['rmse']:.2f} LKR\n")
        f.write(f"  Test MAE:   {xgb_metrics['mae']:.2f} LKR\n")
        f.write(f"  Test R²:    {xgb_metrics['r2']:.4f}\n")
        f.write(f"  Test MAPE:  {xgb_metrics['mape']:.2f}%\n\n")

        improvement_rmse = (baseline_results["test_rmse"] - xgb_metrics["rmse"]) / baseline_results["test_rmse"] * 100
        improvement_r2 = xgb_metrics["r2"] - baseline_results["test_r2"]
        f.write("IMPROVEMENT OVER BASELINE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  RMSE reduction: {improvement_rmse:.1f}%\n")
        f.write(f"  R² improvement: +{improvement_r2:.4f}\n\n")

        f.write("BEST HYPERPARAMETERS\n")
        f.write("-" * 40 + "\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")

        f.write(f"\nPER-VEGETABLE TEST PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Vegetable':<30s} {'RMSE':>8s} {'MAE':>8s} {'R²':>8s} {'MAPE%':>8s}\n")
        f.write("-" * 70 + "\n")
        for veg, metrics in sorted(veg_results.items()):
            f.write(f"{veg:<30s} {metrics['rmse']:>8.2f} {metrics['mae']:>8.2f} "
                    f"{metrics['r2']:>8.4f} {metrics['mape']:>7.2f}%\n")

    print(f"  Results summary: {RESULTS_FILE}")

def train():
    """Run the full model training pipeline."""
    print("=" * 70)
    print("MODEL TRAINING PIPELINE - XGBoost")
    print("=" * 70)

    df, X, y = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, X, y)

    ridge, baseline_results = train_baseline(X_train, y_train, X_val, y_val, X_test, y_test)

    xgb_default = train_xgboost_default(X_train, y_train, X_val, y_val)

    best_xgb, best_params, cv_rmse = tune_hyperparameters(X_train, y_train)

    print("\n  Retraining best model on train + validation data...")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    best_xgb.fit(X_train_full, y_train_full)

    y_pred, xgb_metrics, veg_results = evaluate_model(best_xgb, X_test, y_test, df)

    print(f"\n[7/7] Saving outputs...")
    joblib.dump(best_xgb, MODEL_FILE)
    print(f"  Model saved: {MODEL_FILE}")

    plot_feature_importance(best_xgb, FEATURE_COLUMNS)
    plot_actual_vs_predicted(y_test, y_pred, df, X_test.index)
    save_results(baseline_results, xgb_metrics, veg_results, best_params, cv_rmse)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Baseline (Ridge) Test RMSE: {baseline_results['test_rmse']:.2f} LKR | R²: {baseline_results['test_r2']:.4f}")
    print(f"  XGBoost (Tuned)  Test RMSE: {xgb_metrics['rmse']:.2f} LKR | R²: {xgb_metrics['r2']:.4f}")
    improvement = (baseline_results['test_rmse'] - xgb_metrics['rmse']) / baseline_results['test_rmse'] * 100
    print(f"  Improvement: {improvement:.1f}% RMSE reduction")

    return best_xgb, xgb_metrics


if __name__ == "__main__":
    model, metrics = train()
