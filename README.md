# EZITECH Customer Support RAG

A retrieval-augmented question-answering system built over a customer-support ticket dataset.
Given a user's question, the system retrieves the **top 3 most relevant ticket chunks using FAISS and sentence embeddings**, then extracts a grounded support answer from the retrieved information.
The system does **not** use a large language model to generate the final answer. This keeps the responses grounded in the original support-ticket data and makes the results easier to inspect and audit.

---

## How It Works

```text
customer_support_tickets.csv
            │
            ▼
     inspect_data.py
            │
            ▼
      build_index.py
            │
            ├── Split tickets into chunks
            ├── Generate embeddings
            └── Build FAISS index
            │
            ▼
    data/tickets.index
    data/ticket_chunks.pkl
            │
            ▼
          rag.py
            │
            ├── Convert question to embedding
            ├── Search FAISS
            ├── Retrieve top 3 chunks
            └── Extract support answer
            │
            ▼
          app.py
            │
            ▼
      Streamlit Web Interface
```

The complete retrieval pipeline is:

```text
User Question
      ↓
Sentence Transformer
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Top 3 Relevant Chunks
      ↓
Highest-Ranked Chunk
      ↓
Support Answer Extraction
      ↓
Grounded Answer
```

---

## Main Features

* Customer-support ticket retrieval
* Text chunking with overlapping word windows
* Sentence-transformer embeddings
* FAISS vector similarity search
* Top 3 relevant chunk retrieval
* Grounded answer extraction
* Exact retrieved evidence display
* Ticket and chunk metadata
* Streamlit web interface
* No LLM required for answer generation

---

## Chunking Strategy

### Approach

Each ticket is first converted into a single text block:

```text
Subject: <subject>

Customer:
<body>

Support Answer:
<answer>
```

The ticket text is then divided into overlapping fixed-size word windows.

The current configuration is:

```python
CHUNK_SIZE = 120
OVERLAP = 30
```

This means:

* Each chunk contains up to 120 words.
* Consecutive chunks overlap by 30 words.
* The chunk window moves forward by 90 words.

### Why 120 Words?

A complete ticket can sometimes contain a large amount of text. Using one entire ticket as a single embedding can introduce unrelated information and reduce retrieval precision.

On the other hand, splitting every sentence into its own chunk can remove important context.

For example:

```text
Please try that.
```

has very little meaning without the surrounding conversation.

A 120-word window provides a balance between:

* Context preservation
* Retrieval precision
* Embedding efficiency
* Topical focus

The 30-word overlap helps prevent important information from being lost when it falls near a chunk boundary.

---

## Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

through the `sentence-transformers` library.

The model converts both:

* Customer-support chunks
* User questions

into numerical vector representations.

These vectors allow the system to compare the semantic similarity between a question and the stored support information.

---

## FAISS Retrieval

The project uses FAISS for vector similarity search.

The embeddings are L2-normalized before searching the index.

The FAISS index uses:

```python
faiss.IndexFlatIP
```

Because the vectors are normalized, inner-product similarity corresponds to cosine similarity.

For every question, the system retrieves:

```text
TOP_K = 3
```

Therefore, each query returns the **3 highest-ranked chunks** from the FAISS index.

Each retrieved chunk contains:

* Similarity score
* Ticket ID
* Chunk ID
* Exact chunk text

---

## Answer Generation

The current system does not call an LLM to generate an answer.

Instead, `rag.py` extracts the support answer directly from the highest-ranked retrieved chunk.

A chunk can contain information such as:

```text
Subject: Password Reset

Customer:
I forgot my password and cannot access my account.

Support Answer:
Please use the password reset option on the login page.
Follow the instructions sent to your registered email address.
```

The system looks for:

```text
Support Answer:
```

and extracts the text that follows it.

This makes the answer directly traceable to the original support-ticket data.

---

## Retrieved Evidence

Although the system retrieves 3 chunks, the current answer-generation process uses the **highest-ranked chunk** to produce the final answer.

All 3 retrieved chunks are still displayed so that retrieval quality can be inspected.

For example:

```text
--- Chunk 1 ---
Similarity: 0.8234
Ticket ID: 123
Chunk ID: 123_1

EXACT CHUNK:
...

--- Chunk 2 ---
Similarity: 0.7912
Ticket ID: 456
Chunk ID: 456_1

EXACT CHUNK:
...

--- Chunk 3 ---
Similarity: 0.7543
Ticket ID: 789
Chunk ID: 789_1

EXACT CHUNK:
...
```

The exact scores and retrieved chunks depend on the dataset and the generated FAISS index.

---

## Project Structure

```text
customer_support_rag/
│
├── data/
│   ├── customer_support_tickets.csv
│   ├── tickets.index
│   └── ticket_chunks.pkl
│
├── inspect_data.py
├── build_index.py
├── rag.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Descriptions

| File                                | Purpose                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------ |
| `inspect_data.py`                   | Inspects the CSV by displaying the number of tickets, column names, and first five rows    |
| `build_index.py`                    | Chunks tickets, generates embeddings, builds the FAISS index, and saves the retrieval data |
| `rag.py`                            | Performs question embedding, FAISS retrieval, answer extraction, and retrieval testing     |
| `app.py`                            | Provides the Streamlit user interface                                                      |
| `requirements.txt`                  | Contains the Python dependencies                                                           |
| `README.md`                         | Contains project documentation                                                             |
| `data/customer_support_tickets.csv` | Source customer-support dataset                                                            |
| `data/tickets.index`                | Generated FAISS vector index                                                               |
| `data/ticket_chunks.pkl`            | Generated chunks and metadata                                                              |

---

## Requirements

The project uses the following Python packages:

```text
streamlit
pandas
numpy
faiss-cpu
sentence-transformers
```

These dependencies are listed in:

```text
requirements.txt
```

---

## Setup

### 1. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Linux/macOS:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

Place the customer-support dataset at:

```text
data/customer_support_tickets.csv
```

The dataset should contain, at minimum, the following columns:

```text
subject
body
answer
```

---

## Inspect the Dataset

Before building the vector index, run:

```bash
python inspect_data.py
```

The script displays:

```text
Number of tickets: ...

Column names:
[...]

First 5 tickets:
...
```

This provides a quick way to verify that the CSV file has been loaded correctly.

---

## Build the FAISS Index

After verifying the dataset, run:

```bash
python build_index.py
```

This performs the following steps:

```text
CSV Dataset
     ↓
Ticket Text Preparation
     ↓
120-Word Chunking
     ↓
30-Word Overlap
     ↓
Sentence Embeddings
     ↓
L2 Normalization
     ↓
FAISS Index
```

The generated files are:

```text
data/tickets.index
data/ticket_chunks.pkl
```

Run `build_index.py` again whenever the source dataset changes.

---

## Test the RAG Retrieval

The `rag.py` file can be executed directly to test the retrieval system:

```bash
python rag.py
```

The script tests sample questions such as:

```text
How can I reset my password?

What should I do if I cannot login?

How can I get help with my invoice?
```

For every question, the program displays:

1. The question
2. The grounded answer
3. The top 3 retrieved chunks
4. Similarity scores
5. Ticket IDs
6. Chunk IDs
7. Exact retrieved chunk text

A typical output structure is:

```text
================================================================================
QUESTION
================================================================================

How can I reset my password?

================================================================================
ANSWER
================================================================================

[Grounded support answer]

================================================================================
RETRIEVED CHUNKS
================================================================================

--- Chunk 1 ---
Similarity: 0.xxxx
Ticket ID: ...
Chunk ID: ...

EXACT CHUNK:
...

--- Chunk 2 ---
Similarity: 0.xxxx
Ticket ID: ...
Chunk ID: ...

EXACT CHUNK:
...

--- Chunk 3 ---
Similarity: 0.xxxx
Ticket ID: ...
Chunk ID: ...

EXACT CHUNK:
...
```

---

## Run the Streamlit Application

Start the web application with:

```bash
streamlit run app.py
```

The Streamlit interface allows the user to enter a customer-support question.

After submitting a question, the application displays:

### Answer

The grounded support answer extracted from the highest-ranked retrieved chunk.

### Retrieved Evidence

The application also displays the retrieved chunks with:

* Similarity score
* Ticket ID
* Chunk ID
* Exact chunk text

This makes it possible to inspect the evidence behind the answer.

---

## Retrieval Configuration

The number of retrieved chunks is controlled in `rag.py`:

```python
TOP_K = 3
```

The default behavior is therefore:

```text
Question
   ↓
FAISS Search
   ↓
Top 3 Chunks
```

Increasing this value retrieves more candidate chunks.

For example:

```python
TOP_K = 5
```

would retrieve five chunks instead of three.

---

## Design Decisions

### Why FAISS?

FAISS provides efficient vector similarity search and is well suited for retrieving semantically related text from an embedding index.

### Why Sentence Transformers?

`all-MiniLM-L6-v2` is a lightweight sentence-embedding model that provides a practical balance between speed and semantic retrieval quality.

### Why Use Chunking?

Chunking prevents long tickets from becoming a single overly broad embedding.

Smaller chunks allow FAISS to identify more focused pieces of information that are relevant to a particular quest
