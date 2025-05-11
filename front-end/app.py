import streamlit as st
import pandas as pd
import os
import hopsworks
import altair as alt

st.set_page_config(page_title="Bike Demand Predictions", layout="wide")
st.title("🚴‍♂️ 6-Hourly Bike Ride Demand: 2024 vs 2025")

from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(
    project="citi_bike",
    api_key_value="xVDMP3X2iA6nyZGr.WK8dropGOARtVcvuSDt5funiBBeKki27I3crjUfpHFRN9EJLbUr9c91KmDUhfnGA"
)
print(os.getenv("HOPSWORKS_PROJECT_NAME"))
fs = project.get_feature_store()

# Load environment variables
FEATURE_GROUP_NAME = "time_series_six_hourly_feature_group_bike"
FEATURE_GROUP_VERSION = 1

# Load actual 2024 data
feature_group = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

actual_df = feature_group.read()
actual_df["pickup_hour"] = pd.to_datetime(actual_df["pickup_hour"], utc=True)

# Group 2024 actuals to get 'target' count
actual_df_grouped = actual_df.groupby(["pickup_hour", "location_id"])["target"].sum().reset_index()

# Load 2025 predictions
fg_pred = fs.get_feature_group(name="bike_demand_predictions", version=1)
pred_df = fg_pred.read()
pred_df["pickup_hour"] = pd.to_datetime(pred_df["pickup_hour"], utc=True)

# Get unique location IDs
location_ids = sorted(set(actual_df_grouped["location_id"]).union(set(pred_df["location_id"])))

# Sidebar filter
selected_loc = st.selectbox("Select Location ID", location_ids)

# Filter both datasets
actual_filtered = actual_df_grouped[actual_df_grouped["location_id"] == selected_loc]
pred_filtered = pred_df[pred_df["location_id"] == selected_loc]

# Rename for chart merging
actual_filtered = actual_filtered.rename(columns={"target": "ride_count"})
actual_filtered["year"] = "2024"
pred_filtered = pred_filtered.rename(columns={"predicted_rides": "ride_count"})
pred_filtered["year"] = "2025"

# Combine both
combined = pd.concat([
    actual_filtered[["pickup_hour", "ride_count", "year"]],
    pred_filtered[["pickup_hour", "ride_count", "year"]]
])

# Plot
chart = alt.Chart(combined).mark_line().encode(
    x=alt.X("pickup_hour:T", title="Time"),
    y=alt.Y("ride_count:Q", title="Ride Count"),
    color="year:N"
).properties(
    title=f"🚲 Ride Demand for Location ID: {selected_loc}",
    width=1000,
    height=400
).interactive()

st.altair_chart(chart, use_container_width=True)
