import os
# import config as config
from pathlib import Path
import hopsworks
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
feature_store = project.get_feature_store()
feature_group=feature_store.get_or_create_feature_group(
    name=os.getenv("FEATURE_GROUP_NAME"),
    version=os.getenv("FEATURE_GROUP_VERSION"),
    description= "Time-series Data for Bike at six hour frequency",
    primary_key=["location_id","pickup_hour"],
    event_time="pickup_hour"
)
import hopsworks

api_key = os.getenv('HOPSWORKS_API_KEY')  
project_name = os.getenv('HOPSWORKS_PROJECT_NAME')  

# pip install confluent-kafka
# Initialize connection to Hopsworks  
project = hopsworks.login(  
    api_key_value=api_key,  
    project=project_name  
)  
print(f"Successfully connected to Hopsworks project: {project_name}")
feature_store = project.get_feature_store()
FEATURE_GROUP_NAME = "time_series_six_hourly_feature_group_bike"
FEATURE_GROUP_VERSION = 1
feature_group=feature_store.get_or_create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description= "Time-series Data for Bike at six hour frequency",
    primary_key=["location_id","pickup_hour"],
    event_time="pickup_hour"
)

import pandas as pd
df = pd.read_parquet("../transformeddata2024.parquet")
feature_group.insert(df,write_options={"wait_for_job":False})