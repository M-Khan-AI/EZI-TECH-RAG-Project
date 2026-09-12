import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

CSV_PATH = "data/customer_support_tickets.csv"
INDEX_PATH = "data/tickets.index"
CHUNKS_PATH = "data/ticket_chunks.pkl"

CHUNK_SIZE = 120
OVERLAP = 30


# -----------------------------
# Load dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} tickets.")


# -----------------------------
# Create text from ticket
# -----------------------------

def create_ticket_text(row):
    subject = str(row.get("subject", ""))
    body = str(row.get("body", ""))
    answer = str(row.get("answer", ""))

    return f"""
Subject: {subject}

Customer:
{body}

Support Answer:
{answer}
""".strip()


# -----------------------------
# Chunk text
# -----------------------------

def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# Build chunks
# -----------------------------

print("Creating chunks...")

all_chunks = []

for index, row in df.iterrows():

    text = create_ticket_text(row)

    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    for chunk_number, chunk in enumerate(chunks):

        all_chunks.append({
            "ticket_id": int(index),
            "chunk_id": chunk_number,
            "text": chunk
        })


print(f"Created {len(all_chunks)} chunks.")


# -----------------------------
# Load embedding model
# -----------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# Create embeddings
# -----------------------------

print("Creating embeddings...")

texts = [item["text"] for item in all_chunks]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)


# -----------------------------
# Normalize embeddings
# -----------------------------

embeddings = embeddings.astype("float32")

faiss.normalize_L2(embeddings)


# -----------------------------
# Create FAISS index
# -----------------------------

print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# -----------------------------
# Save FAISS index
# -----------------------------

faiss.write_index(
    index,
    INDEX_PATH
)


# -----------------------------
# Save chunk information
# -----------------------------

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(all_chunks, f)


print()
print("===================================")
print("RAG INDEX BUILD COMPLETE")
print("===================================")
print(f"Tickets: {len(df)}")
print(f"Chunks: {len(all_chunks)}")
print(f"Embedding dimension: {dimension}")
print(f"FAISS index: {INDEX_PATH}")
print(f"Chunks file: {CHUNKS_PATH}")
print("===================================")