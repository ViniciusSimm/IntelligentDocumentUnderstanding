
import faiss
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

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
    prompt = "You are an assistent that classifies documents using its content. Here are some documents and their classes:\n\n"
    for i, doc in enumerate(context_docs):
        prompt += f"Document {i+1}:\n{doc['text']}\nClass: {doc['label']}\n\n"
    
    prompt += f"Now, classify this document and answer only with its class:\n{query_text}\n"
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

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def classify_text(text: str) -> str:
    context = search_similar_docs(text, k=3)
    prompt = build_prompt(context, text)
    return generate_with_qwen(prompt)