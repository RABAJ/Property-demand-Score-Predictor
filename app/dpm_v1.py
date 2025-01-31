import boto3
import sagemaker
import pandas as pd
import numpy as np
import json
import tarfile
import re
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sagemaker import get_execution_role
# from datetime import datetime
from time import gmtime, strftime

# 🔹 Initialize SageMaker session
sagemaker_session = sagemaker.Session()
# role = get_execution_role()
role = "arn:aws:iam::388132971956:role/paras-sagemaker-fullaccess"

# 🔹 Load dataset
file_path = "/Users/parasagarwal/downloads/portals.housingdotcom_ts_output_report_cleaned.csv"
df = pd.read_csv(file_path, low_memory=False)

# 🔹 Select relevant features
# selected_features = [
#     "city", "locality", "builtuparea_value", "bedrooms",
#     "Possession Status", "isActiveProperty", "Possession Starts"
# ]
selected_features = [
    "builtuparea_value", "bedrooms"
]
target = "priceperSqft"  # Target variable

# 🔹 Convert 'priceperSqft' to numeric
def convert_price_to_number(price):
    if isinstance(price, str):
        price = price.lower().replace("/sq.ft", "").replace("/sq.yd", "").replace("/sq.mt", "").strip()  # Remove unit
        price = price.replace(",", "")  # Remove commas (e.g., "1,200" → "1200")

        if "k" in price:
            return float(price.replace("k", "")) * 1000  # Convert K to actual number
        elif "m" in price:
            return float(price.replace("m", "")) * 1000000  # Convert M to actual number
        elif "price on request" in price:
            return np.nan  # Set as NaN for missing values
        elif price.replace(".", "", 1).isdigit():  # Check if it's a valid number
            return float(price)
    
    return np.nan  # If conversion fails, set as NaN

df[target] = df[target].apply(convert_price_to_number)

# 🔹 Handle missing price values
df[target].fillna(df[target].median(), inplace=True)

# 🔹 Convert 'Possession Starts' to numerical year format
# df["Possession Starts"] = pd.to_datetime(df["Possession Starts"], format="%b, %Y", errors="coerce").dt.year
# df["Possession Starts"].fillna(df["Possession Starts"].median(), inplace=True)

# 🔹 Handle missing values
df.fillna({
    "builtuparea_value": df["builtuparea_value"].median(),
    "bedrooms": df["bedrooms"].mode()[0],
}, inplace=True)

# 🔹 Drop rows where target value is missing
df = df[selected_features + [target]].dropna()

# 🔹 Encode categorical variables
# df = pd.get_dummies(df, columns=["city", "locality", "Possession Status", "isActiveProperty"])

# 🔹 Train-Test Split
X = df.drop(columns=[target])
y = df[target]

if X.shape[0] == 0:
    raise ValueError("No valid samples available for training. Check your dataset.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Convert data to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 🔹 Define model parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100
}

# 🔹 Train XGBoost model
model = xgb.train(params, dtrain, num_boost_round=100)

# 🔹 Save trained model locally
model_file = "property-demand-model-v3.bin"
model.save_model(model_file)

# 🔹 Compress model into `.tar.gz` format (required for SageMaker)
tar_model_path = "property-demand-model-v3.tar.gz"
with tarfile.open(tar_model_path, "w:gz") as tar:
    tar.add(model_file, arcname="xgboost-model")

print(f"Model saved as {tar_model_path}")

# 🔹 Upload model to S3
s3 = boto3.client("s3", region_name="us-east-1")
bucket_name = "paras-hackathon-sagemaker-bucket-1"
s3_path = f"models/{tar_model_path}"
s3.upload_file(tar_model_path, bucket_name, s3_path)

# 🔹 Model S3 path
model_data = f"s3://{bucket_name}/{s3_path}"
print(f"Uploaded model to: {model_data}")

# 🔹 Deploy model to SageMaker
xgb_model = sagemaker.model.Model(
    model_data=model_data,
    role=role,
    image_uri=sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, "1.2-1"),
)

custom_end_point = "Custom-xgboost" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=custom_end_point
)

# predictor.wait()
# print("predictor: ", predictor)
# print("✅ Model Deployed! Endpoint Name:", predictor.endpoint_name)

# 🔹 Initialize SageMaker runtime for predictions
# runtime = boto3.client("sagemaker-runtime")

# def predict_property_demand(data):
#     """Invoke SageMaker endpoint for property demand prediction"""
#     response = runtime.invoke_endpoint(
#         EndpointName=predictor.endpoint_name,
#         ContentType="application/json",
#         Body=json.dumps({"instances": [data]})
#     )
#     result = json.loads(response["Body"].read().decode())
#     return result["predictions"]

# # 🔹 Example input for demand prediction
# sample_input = {
#     "city_Agra": 1,
#     "locality_Mayapura": 1,
#     "builtuparea_value": 120,
#     "bedrooms": 3,
#     "Possession Status_Ready to Move": 1,
#     "Possession Status_Under Construction": 0,
#     "isActiveProperty_True": 1,
#     "Possession Starts": 2025
# }

# # 🔹 Get prediction
# predicted_demand = predict_property_demand(sample_input)
# print("📌 Predicted Property Demand Score:", predicted_demand)
