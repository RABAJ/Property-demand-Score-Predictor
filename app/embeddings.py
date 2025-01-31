from sentence_transformers import SentenceTransformer
import requests
import json
from config import OPENSEARCH_URL, OPENSEARCH_INDEX, OPENSEARCH_AUTH

model = SentenceTransformer("all-MiniLM-L6-v2")

def store_document(doc_id, text):
    embedding = model.encode(text).tolist()
    
    document = {
        "id": doc_id,
        "text": text,
        "embedding": embedding
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{OPENSEARCH_URL}/{OPENSEARCH_INDEX}/_doc/{doc_id}",
                             data=json.dumps(document),
                             headers=headers,
                             auth=OPENSEARCH_AUTH)
    return response.json()

# Example: Storing a document
if __name__ == "__main__":
    response = store_document("1", "AWS Lambda is a serverless compute service.")
    print(response)
