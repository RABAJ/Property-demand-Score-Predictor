import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client("sagemaker-runtime")

# Function to invoke SageMaker endpoint for property demand prediction
def predict_property_demand(data):
    """Invoke SageMaker endpoint for property demand prediction"""
    # Prepare the JSON data payload
    json_data = json.dumps({"instances": [data]})
    
    response = runtime.invoke_endpoint(
        EndpointName="Custom-xgboost2025-01-31-17-31-08",  # Replace with your endpoint name
        ContentType="application/json",  # Set content type to JSON
        Body=json_data
    )
    
    # Get and return the prediction from the response
    result = json.loads(response["Body"].read().decode())
    return result["predictions"]

# 🔹 Example input for demand prediction (direct JSON input)
# sample_input = {
#     "builtuparea_value": 120,
#     "bedrooms": 3
# }
sample_input = {"data": "1,1"}

# 🔹 Get prediction
predicted_demand = predict_property_demand(sample_input)
print("📌 Predicted Property Demand Score:", predicted_demand)