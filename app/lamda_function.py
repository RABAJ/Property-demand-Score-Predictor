import json
from retrieval import retrieve_documents
from generation import generate_response

def lambda_handler(event, context):
    # Parse user input
    body = json.loads(event["body"])
    query = body.get("query", "")

    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, k=3)
    context_text = " ".join([doc["_source"]["text"] for doc in retrieved_docs["hits"]["hits"]])

    # Generate AI response
    final_prompt = f"Context: {context_text}\nUser Question: {query}\nAnswer:"
    ai_response = generate_response(final_prompt)

    return {
        "statusCode": 200,
        "body": json.dumps({"response": ai_response})
    }
