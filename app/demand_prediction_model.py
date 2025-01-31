import boto3
import sagemaker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sagemaker import get_execution_role

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()
# Load dataset
file_path = "/Users/parasagarwal/downloads/portals.housingdotcom_ts_output_report_cleaned.csv"
df = pd.read_csv(file_path)

# Selecting only relevant columns
selected_features = ["city", "locality", "builtuparea_value", 
                     "bedrooms", "Possession Status", "isActiveProperty", "Possession Starts"]
target = "priceperSqft"  # Define this in your dataset as a target variable

df = df[selected_features + [target]].dropna()

# Encode categorical variables
df = pd.get_dummies(df, columns=["city", "locality", "Possession Status", "isActiveProperty"])

# Train-Test Split
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert data to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# Define model parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100
}

# Train model
model = xgb.train(params, dtrain, num_boost_round=100)

# Save model locally
model.save_model("property_demand_model.json")


s3 = boto3.client("s3")
bucket_name = "paras-hackathon-sagemaker-bucket"
s3.upload_file("property_demand_model.json", bucket_name, "models/property_demand_model.json")

# Model S3 path
model_data = f"s3://{bucket_name}/models/property_demand_model.json"
xgb_model = sagemaker.model.Model(
    model_data=model_data,
    role=role,
    image_uri=sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, "1.2-1"),
)

predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

print("Model Deployed! Endpoint Name:", predictor.endpoint_name)
import json

# Initialize SageMaker runtime
runtime = boto3.client("sagemaker-runtime")

def predict_property_demand(data):
    response = runtime.invoke_endpoint(
        EndpointName=predictor.endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"instances": [data]})
    )
    result = json.loads(response["Body"].read().decode())
    return result["predictions"]

# Example input for demand prediction
sample_input = {
    "city_NewYork": 1,
    "locality_Downtown": 1,
    "builtuparea_value": 120,
    "bedrooms": 3,
    "Possession Status_Ready to Move": 1,
    "Possession Status_Under Construction": 0,
    "isActiveProperty_Yes": 1,
    "isActiveProperty_No": 0,
    "Possession Starts": 2025  # Year when possession starts
}

# Get prediction
predicted_demand = predict_property_demand(sample_input)
print("Predicted Property Demand Score:", predicted_demand)
