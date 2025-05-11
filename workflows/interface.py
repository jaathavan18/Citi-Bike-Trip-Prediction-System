import os
import pandas as pd
from datetime import timedelta
from pathlib import Path
import joblib
import hopsworks


# Login to Hopsworks and initialize feature store
def get_hopsworks_project():
    return hopsworks.login(
        project="s3akash",
        api_key_value="LVmrhMHM87zqUPpc.KSnbzXbEPo0sGiqmKTuKbWtM6dNDJAGRCLURFm8tiJF75xz1ye4kNy6d3zP8mQjR"
    )

FEATURE_GROUP_NAME = "time_series_six_hourly_feature_group_bike"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "time_series_six_hourly_feature_view_bike"
FEATURE_VIEW_VERSION = 1
project = get_hopsworks_project()
feature_store = project.get_feature_store()

# Get or create feature group
feature_group = feature_store.get_or_create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description="Time-series Data for Bike at six hour frequency",
    primary_key=["location_id", "pickup_hour"],
    event_time="pickup_hour"
)

# Create or retrieve feature view
try:
    feature_view = feature_store.create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        query=feature_group.select_all(),
    )
    print(f"Feature view '{os.getenv('FEATURE_VIEW_NAME')}' (version {os.getenv('FEATURE_VIEW_VERSION')}) created successfully.")
except Exception as e:
    print(f"Feature view creation failed, attempting to retrieve: {e}")
    try:
        feature_view = feature_store.get_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
        )
        print(f"Feature view '{FEATURE_VIEW_NAME}' (version {FEATURE_VIEW_VERSION}) retrieved successfully.")
    except Exception as e:
        print(f"Error retrieving feature view: {e}")
        exit(1)

# Download the latest model from the model registry
model_registry = project.get_model_registry()
models = model_registry.get_models(name='Bike_demand_predictor_next_hour')
model = max(models, key=lambda model: model.version)
model_dir = model.download()
model = joblib.load(Path(model_dir) / "lightgbm_bikeride_model.joblib")

# Load feature view data
ts_data, _ = feature_view.training_data(
    description="time_series_six_hourly_bike_ride"
)

# Preprocess data
ts_data["location_id"] = ts_data["location_id"].astype(str).str.replace('.', '', regex=False)
ts_data["pickup_hour"] = pd.to_datetime(ts_data["pickup_hour"]).dt.tz_localize(None)  # Remove timezone to avoid comparison issues
valid_ids = {"614005", "590514", "532903"}
ts_data = ts_data[ts_data["location_id"].isin(valid_ids)]

# Setup for prediction
full_df = ts_data.copy()
predictions = []
future_dates = pd.date_range("2025-01-01 00:00:00", "2025-12-31 18:00:00", freq="6H")
location_ids = sorted(valid_ids)

# Define features expected by the LightGBM model
reg_features = [f"target_lag_{i+1}" for i in range(112)] + ["hour", "day_of_week", "month", "is_weekend", "location_id"]

print("🔮 Generating predictions for 2025...")

# Rolling prediction loop
for ts in future_dates:
    for loc in location_ids:
        # Get latest 112 lag entries for this station
        hist = full_df[full_df["location_id"] == loc].sort_values("pickup_hour").tail(112)
        if len(hist) < 112:
            continue

        # Create lag features
        feature_row = {
            f"target_lag_{i+1}": hist.iloc[-(i+1)]["target"] for i in range(112)
        }

        # Add time-based features
        feature_row["hour"] = ts.hour
        feature_row["day_of_week"] = ts.dayofweek
        feature_row["month"] = ts.month
        feature_row["is_weekend"] = int(ts.dayofweek in [5, 6])
        feature_row["pickup_hour"] = ts
        feature_row["location_id"] = loc

        # Prepare DataFrame for prediction
        X_pred = pd.DataFrame([feature_row])[reg_features]
        X_pred["location_id"] = X_pred["location_id"].astype(float)  # Ensure numeric for LightGBM

        # Check for feature mismatch
        expected_features = model.feature_name_
        actual_features = X_pred.columns.tolist()
        missing = set(expected_features) - set(actual_features)
        extra = set(actual_features) - set(expected_features)
        if missing or extra:
            print(f"❌ Missing features: {missing}")
            print(f"🔁 Extra features: {extra}")
            continue

        # Predict
        pred = model.predict(X_pred)[0]

        # Store prediction
        predictions.append({
            "pickup_hour": ts,
            "location_id": loc,
            "predicted_rides": round(pred)
        })

        # Append predicted row to history for future lags
        full_df = pd.concat([
            full_df,
            pd.DataFrame([{
                **feature_row,
                "target": pred
            }])
        ], ignore_index=True)

print("✅ 2025 predictions complete.")

# Save predictions
pred_df = pd.DataFrame(predictions)
pred_df.to_csv("bike_predictions_2025_6hr.csv", index=False)
print("📁 Saved as bike_predictions_2025_6hr.csv")

# Optional: Display sorted data for verification
sorted_data = ts_data.sort_values(["location_id", "pickup_hour"]).reset_index(drop=True)
print("Sorted data preview:")
print(sorted_data.head())