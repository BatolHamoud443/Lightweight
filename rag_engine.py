import os
import json
import numpy as np
import faiss
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

# Load .env variables first
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_FILE = "faiss.index"
DATA_FILE = "chunks.json"

# Load your text chunks and embeddings from disk
def load_data():
    if not os.path.exists(DATA_FILE):
        return [], None
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = None
    return chunks, index

# Save index and chunks
def save_data(chunks, index):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, INDEX_FILE)

# Generate embedding vector from OpenAI API
def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

# Build the FAISS index from a list of chunks (text strings)
def build_index(chunks: List[str]):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    save_data(chunks, index)
    return chunks, index

# Search for similar chunks given a query
def find_similar_chunks(query: str, k: int = 3) -> List[str]:
    chunks, index = load_data()
    if not chunks or not index:
        print("❌ No data or index found, please build the index first.")
        return []
    q_emb = np.array(get_embedding(query)).astype('float32').reshape(1, -1)
    distances, indices = index.search(q_emb, k)
    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

# Example function to initialize data (call once with your docs)
def initialize_data(documents: List[str]):
    print("Building FAISS index with", len(documents), "chunks...")
    build_index(documents)
    print("Index built and saved.")

