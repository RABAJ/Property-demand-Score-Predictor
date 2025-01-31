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
selected_features = [
    "city", "builtuparea_value", "bedrooms",
    "Possession Status", "isActiveProperty"
]
# selected_features = [
#     "builtuparea_value", "bedrooms"
# ]
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
df = pd.get_dummies(df, columns=["city", "Possession Status", "isActiveProperty"])

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
model_file = "property-demand-model-v5.bin"
model.save_model(model_file)
model_file = "property-demand-model-v5.bin"
model = xgb.Booster()
model.load_model(model_file)
sample_input = {
    "builtuparea_value": 120,  # Example built-up area
    "bedrooms": 3,  # Example number of bedrooms
    "city_Agra": 1,  # Binary encoding for city Agra
    "city_Mysore": 0,  # Binary encoding for city Mysore
    "city_Nagpur": 0,  # Binary encoding for city Nagpur
    "city_Visakhapatnam": 0,  # Binary encoding for city Visakhapatnam
    "Possession Status_Possession Started": 0,  # Binary encoding for Possession Started
    "Possession Status_Ready to Move": 1,  # Binary encoding for Ready to Move status
    "Possession Status_Under Construction": 0,  # Binary encoding for Under Construction status
    "isActiveProperty_True": 1  # Binary encoding for active property
}

# Convert the input data to a pandas DataFrame
input_df = pd.DataFrame([sample_input])

# Ensure that the data is in the same format as the training data (e.g., handle missing values, dummies, etc.)
# If any preprocessing like encoding is required, do it here (e.g., pd.get_dummies, scaling, etc.)

# Step 3: Convert the data into XGBoost's DMatrix format
dinput = xgb.DMatrix(input_df)

# Step 4: Make the prediction
predictions = model.predict(dinput)

# Step 5: Print the predictions
print("📌 Predicted Property demanded price/unit Score:", predictions[0])