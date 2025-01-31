import boto3
import json
from config import AWS_REGION, BEDROCK_MODEL_ID

# Initialize AWS Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def generate_response(prompt):
    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"prompt": prompt})
    )
    response_data = json.loads(response["body"].read())
    return response_data["completion"]

# Example usage
if __name__ == "__main__":
    user_query = "What is AWS Lambda?"
    response = generate_response(f"User Query: {user_query}\nAnswer:")
    print("AI Response:", response)
