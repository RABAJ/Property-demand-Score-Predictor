import requests
import json
from sentence_transformers import SentenceTransformer
from config import OPENSEARCH_URL, OPENSEARCH_INDEX, OPENSEARCH_AUTH

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_documents(query, k=3):
    query_vector = model.encode(query).tolist()
    
    search_payload = {
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k
                }
            }
        }
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.get(f"{OPENSEARCH_URL}/{OPENSEARCH_INDEX}/_search",
                            data=json.dumps(search_payload),
                            headers=headers,
                            auth=OPENSEARCH_AUTH)
    return response.json()

# Example Usage
if __name__ == "__main__":
    query = "What is AWS Lambda?"
    docs = retrieve_documents(query)
    print(json.dumps(docs, indent=2))
