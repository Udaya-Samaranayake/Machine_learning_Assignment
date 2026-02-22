import os
import sys
import json
import asyncio
import threading
import queue
import traceback
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_FILE = os.path.join(DATA_DIR, "models", "xgboost_model.pkl")
DATA_FILE = os.path.join(DATA_DIR, "processed", "preprocessed_vegetable_prices.csv")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

FEATURE_COLUMNS = [
    "year", "month", "week_of_year", "quarter", "day_of_year",
    "month_sin", "month_cos",
    "vegetable_encoded", "category_encoded", "sub_category_encoded", "season_encoded",
    "price_lag_1w", "price_lag_2w", "price_lag_4w",
    "price_rolling_mean_4w", "price_rolling_std_4w",
    "price_rolling_mean_8w", "price_rolling_std_8w",
    "price_change_1w", "price_pct_change_1w",
]

print("Loading model and data...")
model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

veg_map = df[["vegetable", "vegetable_encoded"]].drop_duplicates().set_index("vegetable")["vegetable_encoded"].to_dict()
cat_map = df[["category", "category_encoded"]].drop_duplicates().set_index("category")["category_encoded"].to_dict()
subcat_map = df[["sub_category", "sub_category_encoded"]].drop_duplicates().set_index("sub_category")["sub_category_encoded"].to_dict()
season_map = df[["season", "season_encoded"]].drop_duplicates().set_index("season")["season_encoded"].to_dict()

veg_info = df[["vegetable", "category", "sub_category"]].drop_duplicates().to_dict("records")
veg_info_map = {v["vegetable"]: v for v in veg_info}

print(f"Loaded {len(df)} records, {len(veg_map)} vegetables")

app = FastAPI(title="Sri Lankan Vegetable Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    vegetable: str
    month: int | None = None
    year: int | None = None
    week: int | None = None


def get_season(month: int) -> str:
    if month in [5, 6, 7, 8, 9]:
        return "Yala_SW_Monsoon"
    elif month in [11, 12, 1, 2, 3]:
        return "Maha_NE_Monsoon"
    else:
        return "Inter_Monsoon"


@app.get("/api/vegetables")
def get_vegetables():
    vegs = []
    for veg_name in sorted(veg_map.keys()):
        info = veg_info_map.get(veg_name, {})
        vegs.append({
            "name": veg_name,
            "category": info.get("category", ""),
            "sub_category": info.get("sub_category", ""),
        })
    return {"vegetables": vegs}


@app.get("/api/defaults/{vegetable}")
def get_defaults(vegetable: str):
    if vegetable not in veg_map:
        return {"error": "Vegetable not found"}

    veg_df = df[df["vegetable"] == vegetable].sort_values("date")
    latest = veg_df.iloc[-1]

    return {
        "vegetable": vegetable,
        "month": int(latest["month"]),
        "year": int(latest["year"]),
        "latest_price": round(float(latest["price_lkr"]), 2),
    }


@app.get("/api/history/{vegetable}")
def get_history(vegetable: str):
    veg_df = df[df["vegetable"] == vegetable].sort_values("date")
    if veg_df.empty:
        return {"error": "Vegetable not found"}

    history = []
    for _, row in veg_df.iterrows():
        history.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "price": round(row["price_lkr"], 2),
        })
    return {"vegetable": vegetable, "history": history}


@app.post("/api/predict")
def predict(req: PredictRequest):
    vegetable = req.vegetable

    if vegetable not in veg_map:
        return {"error": f"Vegetable '{vegetable}' not found"}

    veg_df = df[df["vegetable"] == vegetable].sort_values("date")
    latest = veg_df.iloc[-1]

    month = req.month if req.month is not None else int(latest["month"])
    year = req.year if req.year is not None else int(latest["year"])
    week_in_month = req.week if req.week is not None else 1

    quarter = (month - 1) // 3 + 1
    days_per_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    day_of_year = days_per_month[month - 1] + (week_in_month - 1) * 7 + 1
    week_of_year = day_of_year // 7 + 1
    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))

    info = veg_info_map.get(vegetable, {})
    category = info.get("category", "")
    sub_category = info.get("sub_category", "")
    season = get_season(month)

    lag_1w = float(latest["price_lag_1w"])
    lag_2w = float(latest["price_lag_2w"])
    lag_4w = float(latest["price_lag_4w"])

    features = {
        "year": year,
        "month": month,
        "week_of_year": week_of_year,
        "quarter": quarter,
        "day_of_year": day_of_year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "vegetable_encoded": veg_map[vegetable],
        "category_encoded": cat_map.get(category, 0),
        "sub_category_encoded": subcat_map.get(sub_category, 0),
        "season_encoded": season_map.get(season, 0),
        "price_lag_1w": lag_1w,
        "price_lag_2w": lag_2w,
        "price_lag_4w": lag_4w,
        "price_rolling_mean_4w": float(latest["price_rolling_mean_4w"]),
        "price_rolling_std_4w": float(latest["price_rolling_std_4w"]),
        "price_rolling_mean_8w": float(latest["price_rolling_mean_8w"]),
        "price_rolling_std_8w": float(latest["price_rolling_std_8w"]),
        "price_change_1w": float(latest["price_change_1w"]),
        "price_pct_change_1w": float(latest["price_pct_change_1w"]),
    }

    X = pd.DataFrame([features])
    predicted_price = float(model.predict(X)[0])

    recent = veg_df.tail(8)
    recent_prices = [
        {"date": row["date"].strftime("%Y-%m-%d"), "price": round(row["price_lkr"], 2)}
        for _, row in recent.iterrows()
    ]

    last_price = float(latest["price_lkr"])
    price_change = predicted_price - last_price
    pct_change = (price_change / last_price * 100) if last_price != 0 else 0

    week_labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    return {
        "vegetable": vegetable,
        "predicted_price": round(predicted_price, 2),
        "last_known_price": round(last_price, 2),
        "price_change": round(price_change, 2),
        "pct_change": round(pct_change, 2),
        "prediction_date": f"{week_labels.get(week_in_month, '')} Week, {month_names[month]} {year}",
        "recent_prices": recent_prices,
        "category": category,
        "inputs_used": {
            "month": month,
            "year": year,
            "week": week_in_month,
        },
    }


@app.get("/api/model-info")
def get_model_info():
    results_file = os.path.join(REPORTS_DIR, "training_results.txt")
    results_text = ""
    if os.path.exists(results_file):
        with open(results_file) as f:
            results_text = f.read()

    importance = model.feature_importances_
    feature_importance = sorted(
        [{"feature": f, "importance": round(float(imp), 4)}
         for f, imp in zip(FEATURE_COLUMNS, importance)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    report_file = os.path.join(REPORTS_DIR, "explainability_report.txt")
    report_text = ""
    if os.path.exists(report_file):
        with open(report_file) as f:
            report_text = f.read()

    return {
        "algorithm": "XGBoost (Extreme Gradient Boosting)",
        "features_used": len(FEATURE_COLUMNS),
        "training_samples": int(len(df) * 0.70),
        "test_r2": 0.9483,
        "test_rmse": 58.79,
        "test_mape": 7.09,
        "feature_importance": feature_importance,
        "results_text": results_text,
        "report_text": report_text,
    }


@app.get("/api/plots/{plot_name}")
def get_plot(plot_name: str):
    allowed = [
        "shap_summary_bar.png", "shap_beeswarm.png", "shap_dependence.png",
        "shap_waterfall.png", "partial_dependence.png",
        "feature_importance.png", "actual_vs_predicted.png",
    ]
    if plot_name not in allowed:
        return {"error": "Plot not found"}

    path = os.path.join(PLOTS_DIR, plot_name)
    if not os.path.exists(path):
        return {"error": "Plot file not found"}

    return FileResponse(path, media_type="image/png")


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TARGET_COLUMN = "price_lkr"


def safe_dirname(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def _get_veg_test_data(vegetable: str):
    n = len(df)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_df = df.iloc[val_end:].copy()
    veg_test = test_df[test_df["vegetable"] == vegetable].copy()
    return veg_test


def generate_vegetable_plots(vegetable: str):
    veg_dir = os.path.join(PLOTS_DIR, safe_dirname(vegetable))
    os.makedirs(veg_dir, exist_ok=True)

    expected = [
        "shap_summary_bar.png", "shap_beeswarm.png", "shap_dependence.png",
        "shap_waterfall.png", "partial_dependence.png", "actual_vs_predicted.png",
    ]
    if all(os.path.exists(os.path.join(veg_dir, p)) for p in expected):
        return veg_dir

    veg_test = _get_veg_test_data(vegetable)
    if veg_test.empty:
        return None

    X_test = veg_test[FEATURE_COLUMNS]
    y_test = veg_test[TARGET_COLUMN] if "price_lkr" in veg_test.columns else veg_test["price_lkr"]

    clean_name = re.sub(r'\s+(1kg|1Kg|500g|Bunch)\s*$', '', vegetable, flags=re.IGNORECASE).strip()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, ax=ax)
    ax.set_title(f"SHAP Feature Importance — {clean_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(veg_dir, "shap_summary_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"SHAP Beeswarm — {clean_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(veg_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:3]
    top_feats = [FEATURE_COLUMNS[i] for i in top_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for feat, ax in zip(top_feats, axes):
        fi = FEATURE_COLUMNS.index(feat)
        shap.plots.scatter(shap_values[:, fi], show=False, ax=ax)
        ax.set_title(f"SHAP: {feat}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
    plt.suptitle(f"SHAP Dependence — {clean_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(veg_dir, "shap_dependence.png"), dpi=150, bbox_inches="tight")
    plt.close()

    y_pred = model.predict(X_test)
    errors = np.abs(y_test.values - y_pred)
    best_i = np.argmin(errors)
    worst_i = np.argmax(errors)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plt.sca(axes[0])
    shap.plots.waterfall(shap_values[best_i], show=False)
    axes[0].set_title(
        f"Best — Actual: {y_test.values[best_i]:.1f} | Pred: {y_pred[best_i]:.1f} LKR",
        fontsize=10, fontweight="bold",
    )
    plt.sca(axes[1])
    shap.plots.waterfall(shap_values[worst_i], show=False)
    axes[1].set_title(
        f"Worst — Actual: {y_test.values[worst_i]:.1f} | Pred: {y_pred[worst_i]:.1f} LKR",
        fontsize=10, fontweight="bold",
    )
    plt.suptitle(f"SHAP Waterfall — {clean_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(veg_dir, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
    plt.close()

    pdp_features = ["price_lag_1w", "price_rolling_mean_4w", "month", "week_of_year"]
    feature_indices = [FEATURE_COLUMNS.index(f) for f in pdp_features]
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42) if len(X_test) > sample_size else X_test

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (feat, fi) in enumerate(zip(pdp_features, feature_indices)):
        ax = axes[idx // 2, idx % 2]
        PartialDependenceDisplay.from_estimator(
            model, X_sample, [fi],
            feature_names=FEATURE_COLUMNS, ax=ax, kind="average",
        )
        ax.set_title(f"PDP: {feat}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
    plt.suptitle(f"Partial Dependence — {clean_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(veg_dir, "partial_dependence.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    dates = veg_test["date"]
    ax.plot(dates, y_test.values, label="Actual", color="#10b981", linewidth=1.5)
    ax.plot(dates, y_pred, label="Predicted", color="#f59e0b", linewidth=1.5, linestyle="--")
    ax.fill_between(dates, y_test.values, y_pred, alpha=0.15, color="#3b82f6")
    ax.set_title(f"Actual vs Predicted — {clean_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (LKR)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(veg_dir, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return veg_dir


@app.get("/api/vegetable-plots/{vegetable}/{plot_name}")
def get_vegetable_plot(vegetable: str, plot_name: str):
    allowed = [
        "shap_summary_bar.png", "shap_beeswarm.png", "shap_dependence.png",
        "shap_waterfall.png", "partial_dependence.png", "actual_vs_predicted.png",
    ]
    if plot_name not in allowed:
        return {"error": "Plot not found"}
    if vegetable not in veg_map:
        return {"error": "Vegetable not found"}

    veg_dir = os.path.join(PLOTS_DIR, safe_dirname(vegetable))
    path = os.path.join(veg_dir, plot_name)

    if not os.path.exists(path):
        result = generate_vegetable_plots(vegetable)
        if result is None:
            return {"error": "No test data for this vegetable"}

    return FileResponse(path, media_type="image/png")


sys.path.insert(0, BASE_DIR)

pipeline_status = {"running": False}


def run_pipeline(msg_queue: queue.Queue):
    global model, df, veg_map, cat_map, subcat_map, season_map, veg_info, veg_info_map

    try:
        msg_queue.put({"step": 1, "status": "running", "title": "Fetching Latest Data",
                       "message": "Connecting to statistics.gov.lk..."})
        from dataset_collection import collect_dataset
        try:
            df_wide, df_long = collect_dataset()
            total_weeks = len(df_wide)
            total_records = len(df_long)
            msg_queue.put({"step": 1, "status": "done", "title": "Fetching Latest Data",
                           "message": f"Fetched {total_weeks} weeks, {total_records} records"})
        except Exception as e:
            msg_queue.put({"step": 1, "status": "error", "title": "Fetching Latest Data",
                           "message": f"Fetch failed: {str(e)}"})
            return

        msg_queue.put({"step": 2, "status": "running", "title": "Preprocessing Data",
                       "message": "Cleaning, outlier treatment, feature engineering..."})
        from preprocessing import preprocess
        try:
            df_processed = preprocess()
            msg_queue.put({"step": 2, "status": "done", "title": "Preprocessing Data",
                           "message": f"Preprocessed {len(df_processed)} records, {df_processed.shape[1]} features"})
        except Exception as e:
            msg_queue.put({"step": 2, "status": "error", "title": "Preprocessing Data",
                           "message": f"Preprocessing failed: {str(e)}"})
            return

        msg_queue.put({"step": 3, "status": "running", "title": "Retraining Model",
                       "message": "Training XGBoost with hyperparameter tuning (this may take a few minutes)..."})
        from model_training import train
        try:
            _, metrics = train()
            msg_queue.put({"step": 3, "status": "done", "title": "Retraining Model",
                           "message": f"R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%"})
        except Exception as e:
            msg_queue.put({"step": 3, "status": "error", "title": "Retraining Model",
                           "message": f"Training failed: {str(e)}"})
            return

        msg_queue.put({"step": 4, "status": "running", "title": "Reloading Model",
                       "message": "Loading updated model and data into server..."})
        try:
            model = joblib.load(MODEL_FILE)
            df = pd.read_csv(DATA_FILE)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            veg_map.clear()
            veg_map.update(df[["vegetable", "vegetable_encoded"]].drop_duplicates()
                           .set_index("vegetable")["vegetable_encoded"].to_dict())
            cat_map.clear()
            cat_map.update(df[["category", "category_encoded"]].drop_duplicates()
                           .set_index("category")["category_encoded"].to_dict())
            subcat_map.clear()
            subcat_map.update(df[["sub_category", "sub_category_encoded"]].drop_duplicates()
                              .set_index("sub_category")["sub_category_encoded"].to_dict())
            season_map.clear()
            season_map.update(df[["season", "season_encoded"]].drop_duplicates()
                              .set_index("season")["season_encoded"].to_dict())

            veg_info = df[["vegetable", "category", "sub_category"]].drop_duplicates().to_dict("records")
            veg_info_map.clear()
            veg_info_map.update({v["vegetable"]: v for v in veg_info})

            date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            msg_queue.put({"step": 4, "status": "done", "title": "Reloading Model",
                           "message": f"Loaded {len(df)} records, {len(veg_map)} vegetables. Data: {date_range}"})
        except Exception as e:
            msg_queue.put({"step": 4, "status": "error", "title": "Reloading Model",
                           "message": f"Reload failed: {str(e)}"})
            return

        msg_queue.put({"step": 0, "status": "complete",
                       "title": "Pipeline Complete",
                       "message": f"Model updated with latest data up to {df['date'].max().strftime('%Y-%m-%d')}"})

    except Exception as e:
        msg_queue.put({"step": 0, "status": "error", "title": "Pipeline Error",
                       "message": f"Unexpected error: {traceback.format_exc()}"})
    finally:
        pipeline_status["running"] = False
        msg_queue.put(None)


@app.get("/api/pipeline-status")
def get_pipeline_status():
    latest_date = df["date"].max().strftime("%Y-%m-%d")
    return {"running": pipeline_status["running"], "data_up_to": latest_date,
            "total_records": len(df), "total_vegetables": len(veg_map)}


@app.get("/api/update-pipeline")
async def update_pipeline():
    if pipeline_status["running"]:
        async def already_running():
            yield f"data: {json.dumps({'step': 0, 'status': 'error', 'title': 'Already Running', 'message': 'Pipeline is already running. Please wait.'})}\n\n"
        return StreamingResponse(already_running(), media_type="text/event-stream")

    pipeline_status["running"] = True
    msg_queue = queue.Queue()

    thread = threading.Thread(target=run_pipeline, args=(msg_queue,), daemon=True)
    thread.start()

    async def event_stream():
        while True:
            try:
                msg = msg_queue.get(timeout=0.5)
            except queue.Empty:
                yield f"data: {json.dumps({'step': 0, 'status': 'keepalive', 'title': '', 'message': ''})}\n\n"
                continue

            if msg is None:
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


FRONTEND_DIR = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/")
    def serve_root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/{path:path}")
    def serve_frontend(path: str):
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
