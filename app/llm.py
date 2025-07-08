
import faiss
import json
import numpy as np
import os
import requests
import time
from sentence_transformers import SentenceTransformer
from .utils import extract_json_from_response

def load_index(index_dir="data/faiss_index"):
    index = faiss.read_index(f"{index_dir}/index.faiss")
    with open(f"{index_dir}/labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    with open("data/extracted_docs.json", encoding="utf-8") as f:
        docs = json.load(f)
    return index, labels, docs

def search_similar_docs(query_text, k=3):
    index, labels, docs = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode([query_text])
    distances, indices = index.search(np.array(embedding).astype("float32"), k)

    results = []
    for i in indices[0]:
        results.append({
            "text": docs[i]["text"],
            "label": labels[i]
        })
    return results

def build_prompt(context_docs, query_text):
    prompt = """
    You are an expert document classification and information extraction assistant.

    Your task is to analyze the following text extracted from a scanned document via OCR. Based on this content, do the following:

    1. Determine the type of document (e.g., Invoice, Receipt, Contract, etc).
    2. Provide a confidence score between 0 and 1 (e.g., 0.85 for 85%).
    3. Extract relevant entities from the document, depending on its type.

    Your output must strictly follow the JSON format below:

    
    ❌ DO NOT do this:
    <think>
    Analyzing the text...
    </think>

    ✅ INSTEAD, only do this:
    {
    "document_type": "Invoice",
    "confidence": 0.92,
    "entities": {
        "key1": "value1",
        "key2": "value2",
    }
    }

    The "entities" field should contain only meaningful information relevant to the document type.
    Do not add explanations or additional text outside the JSON structure.
    Here are some classified documents to help you
    """

    for i, doc in enumerate(context_docs):
        prompt += f"Document {i+1}:\n{doc['text']}\nClass: {doc['label']}\n\n"
    
    prompt += f"Here is the OCR text: \n{query_text}\n"
    return prompt

def generate_with_qwen(prompt, model="qwen3:4b"):
    """
    Access the Qwen API to generate a response based on the provided prompt.

    Args:
        prompt (str): The input text to generate a response for.
        model (str): The model to use for generation, default is "qwen3:4b".
    
    Returns:
        str: The generated response from the model.
    """

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{OLLAMA_BASE_URL}/api/generate"


    response = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def classify_text(text: str) -> str:
    start_time = time.time()

    context = search_similar_docs(text, k=3)
    prompt = build_prompt(context, text)

    answer = generate_with_qwen(prompt)
    json_answer = extract_json_from_response(answer)

    end_time = time.time()
    processing_time = round(end_time - start_time, 3)

    json_answer['processing_time'] = processing_time

    return json_answer