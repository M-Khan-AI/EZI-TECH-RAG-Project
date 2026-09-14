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

    # Normalize embedding for cosine similarity
    faiss.normalize_L2(question_embedding)

    # Search FAISS index
    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        # Make sure the FAISS index is valid
        if idx < 0 or idx >= len(chunks):
            continue

        result = chunks[idx].copy()
        result["score"] = float(score)

        results.append(result)

    return results


# ==========================================
# Generate grounded answer
# ==========================================

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return (
            "I could not find relevant information "
            "in the support tickets."
        )

    # Collect support answers from all retrieved chunks
    support_answers = []

    for chunk in retrieved_chunks:

        text = chunk["text"]

        if "Support Answer:" in text:

            answer = text.split(
                "Support Answer:",
                1
            )[1].strip()

            if answer:
                support_answers.append(answer)

    # Remove duplicate answers while preserving order
    unique_answers = []

    for answer in support_answers:

        if answer not in unique_answers:
            unique_answers.append(answer)

    # Return information from all retrieved chunks
    if unique_answers:

        return "\n\n".join(
            unique_answers
        )

    # Fallback if no explicit support answer exists
    combined_information = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        combined_information.append(
            f"Chunk {number}:\n{chunk['text']}"
        )

    return (
        "The retrieved support information says:\n\n"
        + "\n\n".join(combined_information)
    )


# ==========================================
# Display retrieved chunks
# ==========================================

def display_retrieved_chunks(retrieved_chunks):

    print("\n")
    print("=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    if not retrieved_chunks:

        print("\nNo relevant chunks found.")
        return

    for number, result in enumerate(
        retrieved_chunks,
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

        # Retrieve top 3 FAISS chunks
        retrieved = retrieve(question)

        # Generate answer using retrieved information
        answer = generate_answer(
            question,
            retrieved
        )

        print("\n")
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)

        print(answer)

        # Display all 3 retrieved chunks
        display_retrieved_chunks(
            retrieved
        )