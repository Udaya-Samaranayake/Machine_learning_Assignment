
import os
import sys
import json
import asyncio
import threading
import queue
import traceback
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
    """Map month to Sri Lankan season."""
    if month in [5, 6, 7, 8, 9]:
        return "Yala_SW_Monsoon"
    elif month in [11, 12, 1, 2, 3]:
        return "Maha_NE_Monsoon"
    else:
        return "Inter_Monsoon"

@app.get("/api/vegetables")
def get_vegetables():
    """List all vegetables with their categories."""
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
    """Get default input values for a vegetable (from latest data)."""
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
    """Get price history for a specific vegetable."""
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
    """Predict price for a specific month/week using latest data as features."""
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
    """Get model performance metrics and feature importance."""
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
    """Serve generated plot images."""
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

sys.path.insert(0, BASE_DIR)

pipeline_status = {"running": False}


def run_pipeline(msg_queue: queue.Queue):
    """Run the full pipeline: fetch → preprocess → retrain in a thread."""
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
            best_model, metrics = train()
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
    """Check if pipeline is currently running."""
    latest_date = df["date"].max().strftime("%Y-%m-%d")
    return {"running": pipeline_status["running"], "data_up_to": latest_date,
            "total_records": len(df), "total_vegetables": len(veg_map)}


@app.get("/api/update-pipeline")
async def update_pipeline():
    """Run the full update pipeline and stream progress via SSE."""
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
        """Serve React app root."""
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/{path:path}")
    def serve_frontend(path: str):
        """Serve React app for all non-API routes."""
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
