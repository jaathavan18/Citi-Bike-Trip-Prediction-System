import logging
import os
import sys
from datetime import datetime
import requests
import zipfile
import pandas as pd
from dotenv import load_dotenv
import hopsworks
from pathlib import Path
from collections import Counter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Step 1: Download raw data for April 2025
def download_citibike_data(year, month):
    month_str = f"{month:02d}"
    url = f"https://s3.amazonaws.com/tripdata/{year}{month_str}-citibike-tripdata.zip"
    zip_file_path = os.path.join("raw_data", "flow", f"{year}{month_str}-citibike-tripdata.csv.zip")
    
    os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)
    if os.path.exists(zip_file_path):
        logger.info(f"File for {year}-{month_str} already exists")
        return zip_file_path
    
    logger.info(f"Downloading data for {year}-{month_str}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(zip_file_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Downloaded {zip_file_path}")
    return zip_file_path

logger.info("📥 Fetching raw data for April 2025...")
year, month = 2025, 4
zip_path = download_citibike_data(year, month)

# Step 2: Extract and clean data
logger.info("🧹 Processing raw data...")
folder_path = "raw_data/flow"
output_folder = "filter_data_2025/flow"
os.makedirs(output_folder, exist_ok=True)

file_name = f"{year}{month:02d}-citibike-tripdata.csv.zip"
file_path = os.path.join(folder_path, file_name)
monthly_df_list = []
column_set = None
null_counts = {}
row_counts = {}

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv") and "__MACOSX" not in f]
    if not csv_files:
        logger.error(f"No CSVs in {file_name}")
        sys.exit(1)

    for csv_name in csv_files:
        with zip_ref.open(csv_name) as csv_file:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                
                # Drop index or duplicate columns
                if "Unnamed: 0" in df.columns:
                    df.drop(columns=["Unnamed: 0"], inplace=True)
                dup_cols = [col for col in df.columns if "duplicate" in col.lower()]
                if dup_cols:
                    logger.warning(f"Dropping duplicate columns in {csv_name}: {dup_cols}")
                    df.drop(columns=dup_cols, inplace=True)
                
                # Check column consistency
                if column_set is None:
                    column_set = set(df.columns)
                elif set(df.columns) != column_set:
                    logger.warning(f"{csv_name} has different columns, skipping")
                    continue
                
                # Save stats
                row_counts[csv_name] = len(df)
                null_counts[csv_name] = df.isnull().sum().to_dict()
                monthly_df_list.append(df)
            
            except Exception as e:
                logger.error(f"Failed to process {csv_name}: {e}")

# Combine and save cleaned data
if monthly_df_list:
    combined_df = pd.concat(monthly_df_list, ignore_index=True)
    cleaned_df = combined_df.dropna()
    output_csv_name = f"202504_filtered.csv"
    output_path = os.path.join(output_folder, output_csv_name)
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved: {output_csv_name} with {len(cleaned_df)} cleaned rows")
else:
    logger.error("No valid data to process")
    sys.exit(1)

# Step 3: Validate filtered data
logger.info("🔍 Validating filtered data...")
file_path = os.path.join(output_folder, output_csv_name)
try:
    df = pd.read_csv(file_path)
    null_counts[output_csv_name] = df.isnull().sum().to_dict()
    row_counts[output_csv_name] = len(df)
    logger.info(f"✅ Validated: {output_csv_name} with {len(df)} rows")
except Exception as e:
    logger.error(f"Error reading {output_csv_name}: {e}")
    sys.exit(1)

# Step 4: Find top 3 stations
logger.info("📊 Finding top 3 stations...")
df = pd.read_csv(file_path, usecols=["start_station_id"])
df = df.dropna(subset=["start_station_id"])
df["start_station_id"] = df["start_station_id"].astype(str)

top3 = df["start_station_id"].value_counts().head(3)
top_station_ids = set(top3.index)
logger.info(f"Top 3 stations: {top3.to_dict()}")

# Step 5: Filter data for top 3 stations and save as parquet
logger.info("📦 Filtering and saving top 3 stations data...")
df = pd.read_csv(file_path)
for col in ["ride_id", "start_station_id", "end_station_id"]:
    df[col] = df[col].astype(str)
df_filtered = df[df["start_station_id"].isin(top_station_ids)]
output_parquet = "BikeRide202504Top3Locationflow.parquet"
df_filtered.to_parquet(output_parquet, index=False)
logger.info(f"Saved as {output_parquet}")

# Step 6: Preprocess data for feature engineering
def load_and_preprocess_data(file_path):
    logger.info("📥 Loading dataset...")
    df = pd.read_parquet(file_path)
    
    logger.info("🕒 Converting datetime columns...")
    df['started_at'] = pd.to_datetime(df['started_at'], format='mixed')
    df['ended_at'] = pd.to_datetime(df['ended_at'], format='mixed')
    df['pickup_hour'] = df['started_at'].dt.floor('6H')
    df['location_id'] = df['start_station_id'].astype(str)
    
    logger.info("⏱️ Calculating ride duration...")
    df['duration_minutes'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60.0
    
    logger.info("📊 Aggregating target (trip counts)...")
    ride_counts = df.groupby(['pickup_hour', 'location_id']).size().reset_index(name='target')
    
    logger.info("🔁 Creating 112 lag features (28 days × 4 bins/day)...")
    lagged_data = []
    for loc in ride_counts['location_id'].unique():
        loc_df = ride_counts[ride_counts['location_id'] == loc].sort_values('pickup_hour')
        for lag in range(1, 113):
            loc_df[f'target_lag_{lag}'] = loc_df['target'].shift(lag)
        lagged_data.append(loc_df)
    
    df_lagged = pd.concat(lagged_data)
    
    logger.info("📅 Extracting time-based features...")
    df_lagged['hour'] = df_lagged['pickup_hour'].dt.hour
    df_lagged['day_of_week'] = df_lagged['pickup_hour'].dt.dayofweek
    df_lagged['month'] = df_lagged['pickup_hour'].dt.month
    df_lagged['is_weekend'] = df_lagged['day_of_week'].isin([5, 6]).astype(int)
    
    logger.info("🧹 Dropping missing values...")
    df_lagged = df_lagged.dropna()
    
    return df_lagged

logger.info("🔄 Preprocessing data...")
df_transformed = load_and_preprocess_data(output_parquet)
transformed_parquet = "transformeddata202504flow.parquet"
df_transformed.to_parquet(transformed_parquet, index=False)
logger.info(f"Saved transformed data as {transformed_parquet}")

# Step 7: Upload to Hopsworks
logger.info("☁️ Uploading to Hopsworks...")
project = hopsworks.login(
    project="s3akash",
    api_key_value="LVmrhMHM87zqUPpc.KSnbzXbEPo0sGiqmKTuKbWtM6dNDJAGRCLURFm8tiJF75xz1ye4kNy6d3zP8mQjR"
)
feature_store = project.get_feature_store()
feature_group = feature_store.get_or_create_feature_group(
    name="bike_prediction_from_flow_202504",
    version=1,
    description="Time-series Data for Bike at six hour frequency for April 2025",
    primary_key=["location_id", "pickup_hour"],
    event_time="pickup_hour"
)
df_transformed["is_weekend"] = df_transformed["is_weekend"].astype(int)
logger.info(df_transformed)
feature_group.insert(df_transformed, write_options={"wait_for_job": False})
logger.info("✅ Data uploaded to Hopsworks")