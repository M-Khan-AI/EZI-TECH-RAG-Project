# EZITECH Customer Support RAG

A retrieval-augmented question-answering system built over a real customer-support ticket dataset. Given a question, the system retrieves the most relevant ticket chunks with FAISS + sentence embeddings, then answers **only using what was retrieved** — no free-form generation from a language model.

---

## How it works

```
customer_support_tickets.csv
        │
        ▼
  build_index.py  ──► chunks tickets, embeds them, builds a FAISS index
        │
        ▼
   data/tickets.index        (FAISS vector index)
   data/ticket_chunks.pkl    (chunk text + metadata)
        │
        ▼
      rag.py       ──► retrieve(question) + generate_answer(question, chunks)
        │
        ▼
      app.py       ──► Streamlit UI: ask a question, see the answer
                        AND the exact chunks that produced it
```

There is no LLM call in the answer step. `generate_answer()` pulls the `Support Answer:` section straight out of the top-ranked retrieved chunk. This keeps the system fully grounded and auditable — every word in the answer can be traced back to a specific ticket and chunk.

---

## Chunking strategy

**Approach: per-ticket text, split into overlapping fixed-size word windows (120 words, 30-word overlap).**

Each ticket is first flattened into a single text block:

```
Subject: <subject>

Customer:
<body>

Support Answer:
<answer>
```

That block is then split into chunks of **120 words**, sliding forward by **90 words** each time (a 30-word / 25% overlap). This was chosen deliberately over the two extremes:

- **One giant chunk per ticket** was rejected because tickets vary a lot in length. Long tickets would dominate the embedding with unrelated context (e.g., a rambling customer message diluting the actual support answer), which hurts retrieval precision — the embedding for the whole ticket stops looking like the embedding for the specific question being asked.
- **One sentence per chunk** was rejected because a single sentence pulled out of a ticket often loses the context needed to answer a question (e.g., "Please try that." means nothing without the preceding sentence). Supporting answers in this dataset are frequently 2–4 sentences of connected instructions, so sentence-level splitting would fragment the exact information a question is trying to retrieve.

**120 words** was chosen because it's roughly the length of a short paragraph — long enough to keep a subject, a customer's issue, and a support answer together in most tickets, short enough that a chunk stays topically focused and doesn't blur multiple issues together. The **30-word overlap** protects against the failure case where the useful sentence lands right at a chunk boundary and would otherwise be split across two chunks and lose meaning in both.

This is implemented in `build_index.py`:

```python
CHUNK_SIZE = 120
OVERLAP = 30
```

Chunks are embedded with `all-MiniLM-L6-v2` (via `sentence-transformers`), L2-normalized, and indexed with `faiss.IndexFlatIP` so that inner product search is equivalent to cosine similarity.

---

## Project structure

| File | Purpose |
|---|---|
| `inspect_data.py` | Quick look at the raw CSV — row count, columns, sample rows |
| `build_index.py` | Chunks all tickets, embeds them, builds and saves the FAISS index |
| `rag.py` | Core retrieval (`retrieve`) and answer extraction (`generate_answer`) logic; also runnable as a standalone test script |
| `app.py` | Streamlit front-end: ask a question, see the answer and the retrieved evidence |
| `data/customer_support_tickets.csv` | Source dataset (not committed — see `.gitignore`) |
| `data/tickets.index` | Generated FAISS index (build artifact) |
| `data/ticket_chunks.pkl` | Generated chunk text + metadata (build artifact) |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` should include:

```
streamlit
pandas
numpy
faiss-cpu
sentence-transformers
```

Place your dataset at `data/customer_support_tickets.csv`. It's expected to have (at minimum) `subject`, `body`, and `answer` columns — check with:

```bash
python inspect_data.py
```

---

## Build the index

```bash
python build_index.py
```

This reads the CSV, creates the word-window chunks described above, embeds them, and writes `data/tickets.index` and `data/ticket_chunks.pkl`. Re-run this any time the dataset changes.

---

## Verify retrieval (sample questions + exact chunks)

`rag.py` doubles as a retrieval sanity check. Running it directly retrieves the top chunks for three sample questions and prints the **exact chunk text** used, so retrieval quality can be checked independently of the final answer:

```bash
python rag.py
```

### Sample results

> Run `python rag.py` after building the index and paste the console output below. Each entry should show the question, the similarity score, the ticket/chunk ID, and the exact retrieved chunk text — this is what proves the retrieval step is working, not just the final answer.

**Question 1: "How can I reset my password?"**
- Similarity score: `<fill in>`
- Ticket ID / Chunk ID: `<fill in>`
- Exact retrieved chunk:
  ```
  <paste exact chunk text here>
  ```
- Answer produced: `<paste answer here>`

**Question 2: "What should I do if I cannot login?"**
- Similarity score: `<fill in>`
- Ticket ID / Chunk ID: `<fill in>`
- Exact retrieved chunk:
  ```
  <paste exact chunk text here>
  ```
- Answer produced: `<paste answer here>`

**Question 3: "How can I get help with my invoice?"**
- Similarity score: `<fill in>`
- Ticket ID / Chunk ID: `<fill in>`
- Exact retrieved chunk:
  ```
  <paste exact chunk text here>
  ```
- Answer produced: `<paste answer here>`

---

## Run the app

```bash
streamlit run app.py
```

Enter a question, hit **Search**, and the UI shows:
1. **Answer** — the grounded answer extracted from the top-ranked chunk.
2. **Retrieved Evidence** — every retrieved chunk in an expander, with its similarity score, ticket ID, chunk ID, and the exact chunk text, so any answer can be checked against its source.

---

## Design notes / limitations

- Answers are extracted verbatim from the top chunk's `Support Answer:` section — there's no paraphrasing or synthesis across multiple chunks, which keeps answers fully traceable but means the system can't currently combine information from more than one ticket.
- Retrieval uses cosine similarity (via normalized inner product) over `all-MiniLM-L6-v2` embeddings — a small, fast model well-suited to short support-ticket text, at some cost to nuance versus larger embedding models.
- `top_k=3` is used both in `app.py` and as the default in `retrieve()`; increase this if you want more candidate evidence per query.
