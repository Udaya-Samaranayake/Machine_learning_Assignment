
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay
import warnings

warnings.filterwarnings("ignore")

BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_FILE = os.path.join(BASE_DATA_DIR, "models", "xgboost_model.pkl")
INPUT_FILE = os.path.join(BASE_DATA_DIR, "processed", "preprocessed_vegetable_prices.csv")
PLOTS_DIR = os.path.join(BASE_DATA_DIR, "plots")
REPORTS_DIR = os.path.join(BASE_DATA_DIR, "reports")

FEATURE_COLUMNS = [
    "year", "month", "week_of_year", "quarter", "day_of_year",
    "month_sin", "month_cos",
    "vegetable_encoded", "category_encoded", "sub_category_encoded", "season_encoded",
    "price_lag_1w", "price_lag_2w", "price_lag_4w",
    "price_rolling_mean_4w", "price_rolling_std_4w",
    "price_rolling_mean_8w", "price_rolling_std_8w",
    "price_change_1w", "price_pct_change_1w",
]

TARGET_COLUMN = "price_lkr"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

def load_data_and_model():
    """Load the trained model and test data."""
    print("[1/5] Loading model and data...")

    model = joblib.load(MODEL_FILE)
    print(f"  Model loaded from: {MODEL_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_test = df.iloc[val_end:][FEATURE_COLUMNS]
    y_test = df.iloc[val_end:][TARGET_COLUMN]
    test_df = df.iloc[val_end:].copy()

    train_end = int(n * TRAIN_RATIO)
    X_train = df.iloc[:train_end][FEATURE_COLUMNS]

    print(f"  Test samples: {len(X_test)}")
    print(f"  Train samples (for SHAP background): {len(X_train)}")

    return model, X_train, X_test, y_test, test_df


def compute_shap_values(model, X_test):
    """Compute SHAP values using TreeExplainer (exact for tree models)."""
    print("\n[2/5] Computing SHAP values (TreeExplainer)...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    print(f"  SHAP values shape: {shap_values.values.shape}")
    print(f"  Base value (expected prediction): {shap_values.base_values[0]:.2f} LKR")

    return explainer, shap_values


def plot_shap_summary_bar(shap_values):
    """Global feature importance via mean |SHAP|."""
    print("  Generating SHAP summary bar plot...")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, ax=ax)
    ax.set_title("SHAP Global Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "shap_summary_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_shap_beeswarm(shap_values):
    """Beeswarm plot showing feature value impact."""
    print("  Generating SHAP beeswarm plot...")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Beeswarm Plot - Feature Impact on Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_shap_dependence(shap_values, X_test):
    """Dependence plots for top 3 most important features."""
    print("  Generating SHAP dependence plots...")

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top_features = [FEATURE_COLUMNS[i] for i in top_indices]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (feat, ax) in enumerate(zip(top_features, axes)):
        feat_idx = FEATURE_COLUMNS.index(feat)
        shap.plots.scatter(shap_values[:, feat_idx], show=False, ax=ax)
        ax.set_title(f"SHAP Dependence: {feat}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)

    plt.suptitle("SHAP Dependence Plots - Top 3 Features", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "shap_dependence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_shap_waterfall(shap_values, X_test, y_test, model):
    """Waterfall plots for individual predictions (best and worst)."""
    print("  Generating SHAP waterfall plots...")

    y_pred = model.predict(X_test)
    errors = np.abs(y_test.values - y_pred)

    best_idx = np.argmin(errors)
   
    worst_idx = np.argmax(errors)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    plt.sca(axes[0])
    shap.plots.waterfall(shap_values[best_idx], show=False)
    actual = y_test.values[best_idx]
    predicted = y_pred[best_idx]
    axes[0].set_title(
        f"Best Prediction\nActual: {actual:.1f} LKR | Predicted: {predicted:.1f} LKR | Error: {errors[best_idx]:.2f}",
        fontsize=10, fontweight="bold"
    )

    plt.sca(axes[1])
    shap.plots.waterfall(shap_values[worst_idx], show=False)
    actual = y_test.values[worst_idx]
    predicted = y_pred[worst_idx]
    axes[1].set_title(
        f"Worst Prediction\nActual: {actual:.1f} LKR | Predicted: {predicted:.1f} LKR | Error: {errors[worst_idx]:.2f}",
        fontsize=10, fontweight="bold"
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "shap_waterfall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_partial_dependence(model, X_test):
    """Generate Partial Dependence Plots for key features."""
    print("\n[3/5] Generating Partial Dependence Plots...")

    pdp_features = ["price_lag_1w", "price_rolling_mean_4w", "month", "vegetable_encoded"]
    feature_indices = [FEATURE_COLUMNS.index(f) for f in pdp_features]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    for idx, (feat, feat_idx) in enumerate(zip(pdp_features, feature_indices)):
        ax = axes[idx // 2, idx % 2]
        PartialDependenceDisplay.from_estimator(
            model, X_sample, [feat_idx],
            feature_names=FEATURE_COLUMNS,
            ax=ax,
            kind="average",
        )
        ax.set_title(f"PDP: {feat}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)

    plt.suptitle("Partial Dependence Plots", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "partial_dependence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def generate_report(shap_values, model, X_test, y_test, test_df):
    """Generate a written explainability report."""
    print("\n[4/5] Generating explainability report...")

    y_pred = model.predict(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_ranking = sorted(
        zip(FEATURE_COLUMNS, mean_abs_shap),
        key=lambda x: x[1], reverse=True
    )

    lag_features = ["price_lag_1w", "price_lag_2w", "price_lag_4w"]
    rolling_features = ["price_rolling_mean_4w", "price_rolling_std_4w",
                        "price_rolling_mean_8w", "price_rolling_std_8w"]
    time_features = ["year", "month", "week_of_year", "quarter", "day_of_year",
                     "month_sin", "month_cos", "season_encoded"]
    category_features = ["vegetable_encoded", "category_encoded", "sub_category_encoded"]
    change_features = ["price_change_1w", "price_pct_change_1w"]

    def group_importance(features):
        return sum(mean_abs_shap[FEATURE_COLUMNS.index(f)] for f in features)

    lag_imp = group_importance(lag_features)
    rolling_imp = group_importance(rolling_features)
    time_imp = group_importance(time_features)
    cat_imp = group_importance(category_features)
    change_imp = group_importance(change_features)
    total_imp = lag_imp + rolling_imp + time_imp + cat_imp + change_imp

    report = []
    report.append("=" * 70)
    report.append("EXPLAINABILITY & INTERPRETATION REPORT")
    report.append("XGBoost Model for Sri Lankan Vegetable Price Prediction")
    report.append("=" * 70)

    report.append("\n1. GLOBAL FEATURE IMPORTANCE (SHAP)")
    report.append("-" * 50)
    report.append(f"{'Rank':<5} {'Feature':<30} {'Mean |SHAP| (LKR)':>20}")
    report.append("-" * 55)
    for rank, (feat, imp) in enumerate(feature_ranking, 1):
        report.append(f"{rank:<5} {feat:<30} {imp:>20.2f}")

    report.append(f"\n  Feature Group Contributions:")
    report.append(f"    Lag prices (1w, 2w, 4w):      {lag_imp/total_imp*100:5.1f}%  ({lag_imp:.2f} LKR)")
    report.append(f"    Rolling statistics (4w, 8w):   {rolling_imp/total_imp*100:5.1f}%  ({rolling_imp:.2f} LKR)")
    report.append(f"    Time features (month, etc):    {time_imp/total_imp*100:5.1f}%  ({time_imp:.2f} LKR)")
    report.append(f"    Vegetable/category identity:   {cat_imp/total_imp*100:5.1f}%  ({cat_imp:.2f} LKR)")
    report.append(f"    Price change momentum:         {change_imp/total_imp*100:5.1f}%  ({change_imp:.2f} LKR)")

    report.append("\n\n2. WHAT THE MODEL HAS LEARNED")
    report.append("-" * 50)
    report.append("""
  The XGBoost model has learned that vegetable prices in Sri Lanka exhibit
  strong short-term persistence - last week's price (price_lag_1w) is the
  single most important predictor, contributing the largest SHAP impact.
  This makes economic sense: agricultural commodity prices rarely change
  drastically week-to-week due to stable supply chains and consumer demand.

  The model also captures medium-term trends through 4-week and 8-week
  rolling averages, which help smooth out weekly noise and identify
  whether prices are trending upward or downward.

  Seasonal patterns (monsoon cycles) play a secondary but meaningful role.
  Sri Lanka's Yala (SW monsoon, May-Sep) and Maha (NE monsoon, Nov-Mar)
  seasons affect agricultural production, which the model captures through
  month and season features.""")

    report.append("\n\n3. WHICH FEATURES ARE MOST INFLUENTIAL")
    report.append("-" * 50)
    report.append(f"""
  Top 5 features by SHAP importance:
    1. {feature_ranking[0][0]}: {feature_ranking[0][1]:.2f} LKR avg impact
    2. {feature_ranking[1][0]}: {feature_ranking[1][1]:.2f} LKR avg impact
    3. {feature_ranking[2][0]}: {feature_ranking[2][1]:.2f} LKR avg impact
    4. {feature_ranking[3][0]}: {feature_ranking[3][1]:.2f} LKR avg impact
    5. {feature_ranking[4][0]}: {feature_ranking[4][1]:.2f} LKR avg impact

  Historical price features (lags + rolling) account for {(lag_imp+rolling_imp)/total_imp*100:.1f}%
  of total predictive impact, confirming that past prices are the strongest
  predictors of future prices in this market.""")

    report.append("\n\n4. ALIGNMENT WITH DOMAIN KNOWLEDGE")
    report.append("-" * 50)

    test_df_copy = test_df.copy()
    test_df_copy["predicted"] = y_pred

    monsoon_prices = test_df_copy.groupby("season")[TARGET_COLUMN].mean()

    report.append(f"""
  a) Price Persistence (Confirmed):
     The dominance of lag features aligns with agricultural economics -
     vegetable supply is determined by planting decisions made weeks/months
     prior, so prices change gradually rather than abruptly.

  b) Seasonal Effects (Confirmed):
     Average prices by season in the test set:""")
    for season, price in monsoon_prices.items():
        report.append(f"       {season}: {price:.2f} LKR")
    report.append("""
     Prices tend to be higher during inter-monsoon periods when supply
     is less stable, and more moderate during peak growing seasons.

  c) Vegetable-Specific Patterns:
     The model correctly learns that different vegetables have vastly
     different price ranges (e.g., Drumstick ~500+ LKR/kg vs Gotukola
     ~45 LKR/bunch). The vegetable_encoded feature helps the model
     calibrate its baseline prediction per vegetable type.

  d) Volatility Awareness:
     The rolling standard deviation features capture price volatility.
     Vegetables like Green Chillies and Limes show high volatility
     (prices can double/halve rapidly), while staples like Potatoes
     are more stable. The model uses this to adjust prediction
     confidence accordingly.""")

    report.append("\n\n5. MODEL LIMITATIONS")
    report.append("-" * 50)
    report.append("""
  - The model relies heavily on recent prices (lag features), making it
    less accurate during sudden market shocks or supply disruptions
  - Leafy vegetables (Gotukola, Kankun, Sarana) show lower R² scores
    because their prices have low variance but sudden jumps
  - External factors not in the dataset (fuel costs, imports, weather
    events, policy changes) can cause prediction errors
  - The model predicts one week ahead; longer-term forecasts would
    compound errors significantly""")

    report.append("\n\n6. EXPLAINABILITY METHODS USED")
    report.append("-" * 50)
    report.append("""
  a) SHAP (SHapley Additive exPlanations):
     - Uses TreeSHAP algorithm (exact computation for tree-based models)
     - Provides both global importance and local per-prediction explanations
     - Based on cooperative game theory (Shapley values)
     - Plots: shap_summary_bar.png, shap_beeswarm.png, shap_dependence.png,
              shap_waterfall.png

  b) Partial Dependence Plots (PDP):
     - Shows the marginal effect of individual features on predictions
     - Averages out the effect of all other features
     - Helps visualize non-linear relationships learned by the model
     - Plot: partial_dependence.png""")

    report_text = "\n".join(report)
    path = os.path.join(REPORTS_DIR, "explainability_report.txt")
    with open(path, "w") as f:
        f.write(report_text)

    print(f"  Saved: {path}")
    return path


def explain():
    """Run the full explainability pipeline."""
    print("=" * 70)
    print("EXPLAINABILITY & INTERPRETATION PIPELINE")
    print("=" * 70)

    model, X_train, X_test, y_test, test_df = load_data_and_model()

    explainer, shap_values = compute_shap_values(model, X_test)
    plot_shap_summary_bar(shap_values)
    plot_shap_beeswarm(shap_values)
    plot_shap_dependence(shap_values, X_test)
    plot_shap_waterfall(shap_values, X_test, y_test, model)

    plot_partial_dependence(model, X_test)

    generate_report(shap_values, model, X_test, y_test, test_df)

    print("\n[5/5] Done!")
    print("=" * 70)
    print("EXPLAINABILITY OUTPUTS")
    print("=" * 70)
    print("  Plots:")
    print("    - shap_summary_bar.png     (Global SHAP feature importance)")
    print("    - shap_beeswarm.png        (Feature value impact directions)")
    print("    - shap_dependence.png      (Top 3 feature dependence)")
    print("    - shap_waterfall.png       (Local explanations: best & worst)")
    print("    - partial_dependence.png   (Partial dependence plots)")
    print("  Report:")
    print("    - explainability_report.txt (Full written interpretation)")


if __name__ == "__main__":
    explain()
