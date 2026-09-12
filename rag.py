import faiss
import pickle
from sentence_transformers import SentenceTransformer


# ==========================================
# Configuration
# ==========================================

INDEX_PATH = "data/tickets.index"
CHUNKS_PATH = "data/ticket_chunks.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3


# ==========================================
# Load resources
# ==========================================

print("Loading RAG resources...")

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer(MODEL_NAME)

print("RAG resources loaded successfully.")


# ==========================================
# Retrieve relevant chunks
# ==========================================

def retrieve(question, top_k=TOP_K):

    # Convert question into an embedding
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype("float32")

    # Normalize
    faiss.normalize_L2(question_embedding)

    # Search FAISS
    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        result = chunks[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ==========================================
# Simple grounded answer
# ==========================================

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return "I could not find relevant information in the support tickets."

    # Use the highest-ranked retrieved chunk
    best_chunk = retrieved_chunks[0]["text"]

    # Try to extract the support answer
    if "Support Answer:" in best_chunk:

        answer = best_chunk.split(
            "Support Answer:",
            1
        )[1].strip()

        if answer:
            return answer

    # If there is no explicit support answer,
    # return the retrieved information itself.
    return (
        "The retrieved support information says:\n\n"
        + best_chunk
    )


# ==========================================
# Test Execution
# ==========================================

if __name__ == "__main__":

    test_questions = [
        "How can I reset my password?",
        "What should I do if I cannot login?",
        "How can I get help with my invoice?"
    ]

    for question in test_questions:

        print("\n")
        print("=" * 80)
        print("QUESTION")
        print("=" * 80)

        print(question)

        retrieved = retrieve(question)

        answer = generate_answer(
            question,
            retrieved
        )

        print("\n")
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)

        print(answer)

        print("\n")
        print("=" * 80)
        print("RETRIEVED CHUNKS")
        print("=" * 80)

        for number, result in enumerate(
            retrieved,
            start=1
        ):

            print(
                f"\n--- Chunk {number} ---"
            )

            print(
                f"Similarity: {result['score']:.4f}"
            )

            print(
                f"Ticket ID: {result['ticket_id']}"
            )

            print(
                f"Chunk ID: {result['chunk_id']}"
            )

            print("\nEXACT CHUNK:")

            print(result["text"])