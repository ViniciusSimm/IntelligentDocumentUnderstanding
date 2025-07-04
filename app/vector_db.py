import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

def build_faiss_index(json_path="data/extracted_docs.json", index_dir="data/faiss_index"):
    """
    Creates the FAISS index from the specified JSON file.

    Args:
        json_path (str): Path to the JSON file containing documents.
        index_dir (str): Directory where the FAISS index will be saved.
    
    Returns:
        index (faiss.Index): The FAISS index object.
        labels (list): List of labels corresponding to the documents.
    """

    with open(json_path, encoding="utf-8") as f:
        docs = json.load(f)

    texts = [doc["text"] for doc in docs]
    labels = [doc["label"] for doc in docs]

    print("[INFO] Carregando modelo de embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[INFO] Gerando embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f"{index_dir}/index.faiss")

    with open(f"{index_dir}/labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)

    return index, labels
