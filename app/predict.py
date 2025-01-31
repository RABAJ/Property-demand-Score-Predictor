import boto3
import pandas as pd
import csv
import io
import json
import sagemaker
sagemaker_session = sagemaker.Session()
# xgb_model = sagemaker.model.Model(
#     model_data="s3://paras-hackathon-sagemaker-bucket-1/models/model.tar.gz",
#     role="arn:aws:iam::388132971956:role/paras-sagemaker-fullaccess",
#     image_uri=sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, "1.2-1"),
# )

# predictor = xgb_model.deploy(
#     initial_instance_count=1,
#     instance_type="ml.m5.large"
# )


# Initialize SageMaker runtime client
runtime = boto3.client("sagemaker-runtime")

# Function to convert input data to CSV (values only, no headers)
def dict_to_csv(input_dict):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(input_dict.values())  # Write only values, no headers
    # return output.getvalue()
    output.seek(0)
    return output

# def dict_to_csv(input_dict):
#     df = pd.DataFrame(input_dict)
#     df.to_csv

# Function to invoke SageMaker endpoint for property demand prediction
def predict_property_demand(data):
    """Invoke SageMaker endpoint for property demand prediction"""
    csv_data = dict_to_csv(data)
    print("csv_data", csv_data)
    response = runtime.invoke_endpoint(
        EndpointName="Custom-xgboost2025-01-31-17-31-08",  # Replace with your endpoint name
        ContentType="text/csv",  # Use CSV content type
        Body="120,3"
    )
    result = json.loads(response["Body"].read().decode())
    return result["predictions"]

# 🔹 Example input for demand prediction
sample_input = {
    "builtuparea_value": 120,
    "bedrooms": 3
}

# 🔹 Get prediction
predicted_demand = predict_property_demand(sample_input)
print("📌 Predicted Property Demand Score:", predicted_demand)
