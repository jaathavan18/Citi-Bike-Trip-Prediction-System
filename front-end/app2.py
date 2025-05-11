import os
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

MLFLOW_TRACKING_URI2="https://dagshub.com/jaathavan18/citi_bike_pred.mlflow" 
MLFLOW_TRACKING_USERNAME="jaathavan18"

# Load .env vars
load_dotenv()
MLFLOW_TRACKING_URI =   MLFLOW_TRACKING_URI2

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

st.title("🚀 MLflow Experiment Dashboard")

# Sidebar: Choose experiment
experiments = client.search_experiments()

experiment_names = [exp.name for exp in experiments]
selected_exp_name = st.sidebar.selectbox("Select Experiment", experiment_names)

# Get experiment ID
experiment = client.get_experiment_by_name(selected_exp_name)
if not experiment:
    st.error("Experiment not found.")
    st.stop()

# Get runs
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"])

if not runs:
    st.info("No runs found for this experiment.")
    st.stop()

# Table of metrics
st.subheader(f"📊 Runs for '{selected_exp_name}'")
run_table = []

for run in runs:
    metrics = run.data.metrics
    run_table.append({
        "Run ID": run.info.run_id,
        "MAE": metrics.get("mean_absolute_error"),
        "MAPE": metrics.get("mape"),
        "RMSE": metrics.get("rmse"),
        "R²": metrics.get("r2"),
    })

st.dataframe(run_table)

# Optionally visualize one run
st.subheader("🔍 Inspect a Run")
selected_run_id = st.selectbox("Select Run ID", [r["Run ID"] for r in run_table])

if selected_run_id:
    run = client.get_run(selected_run_id)
    st.write("📐 Metrics", run.data.metrics)