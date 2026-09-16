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


## Chunking Strategy 
 
### Approaches Considered 
## Chunking Strategy

### Approaches Considered

There are several possible ways to divide customer-support tickets into chunks. This project considered the following approaches:

#### 1. One Giant Chunk

The entire ticket can be stored as a single chunk.

For example:

```text
Complete Ticket
      ↓

 Subject + Customer + Answer = One Chunk           
```

This approach keeps all of the ticket's information together. However, a long ticket may contain different topics or pieces of information. Embedding the entire ticket as one chunk can make retrieval less focused and may reduce retrieval precision.

#### 2. One Sentence Per Chunk

Another approach is to split the ticket into individual sentences, with each sentence becoming a separate chunk.

For example:

```text
Sentence 1 → Chunk 1
Sentence 2 → Chunk 2
Sentence 3 → Chunk 3
```

This approach creates small and focused chunks, but individual sentences may not contain enough context to understand the customer's issue or the support response.

For example:

```text
Please try that.
```

This sentence has very little meaning without the surrounding conversation.

#### 3. Fixed-Size Overlapping Word Chunks — Used in This Project

This project uses fixed-size overlapping word windows. Each ticket is first converted into a single text block:

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

For example:

```text
Chunk 1 → Words 1–30
Chunk 2 → Words 31–60
Chunk 3 → Words 61-90
```

### Why This Approach Was Selected

The fixed-size overlapping approach provides a balance between the two alternative approaches.

* Unlike one giant chunk, it divides long tickets into smaller and more focused pieces.
* Unlike one sentence per chunk, it preserves more surrounding context.
* The 30-word overlap helps reduce the chance of important information being lost at chunk boundaries.
* The 120-word size provides enough context for meaningful semantic retrieval while keeping chunks reasonably focused.

Therefore, the project uses **120-word chunks with a 30-word overlap** for the FAISS retrieval system.


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
Subject: how can i reset my password?
Exact Chunk 1:be helpful if you could provide us with details on the security protocols and the process for the password resets that have been initiated. Please let's discuss this as soon as possible, and you can call <tel_num> at your earliest convenience.

Chunk 2: policies and, if necessary, contact us at <tel_num> for guidance to promptly address this issue. Support Answer: Dear <name>, we confirm receipt of your email regarding the unauthorized access to confidential medical data and appreciate the prompt actions taken to reset passwords and improve security policies. We recommend enhancing the strength of your password security policies and implementing supplementary security measures. Please carry out a thorough review of your current security policies and, if needed, contact us at <tel_num> for advice to resolve this matter promptly. Thank you.

Chunk 3: of the initial security scans. Please provide us with the reports from the initial security scans and the password resets. We will schedule a call at <tel_num> to discuss the next steps to resolve this issue as soon as possible. Referencing your account number <acc_num>.


Subject: What should I do if I cannot login?

Exact Chunks

Chunk 1: matter and look forward to your assistance. Support Answer: We will investigate the login failure issue. Please provide your account number and a good time to call at <tel_num> for further discussion. Our technical team will work to resolve the issue as soon as possible.

Chunk 2: this matter and look forward to your response. Support Answer: We will investigate the login failure issue. Please provide your account number and a good time to call at <tel_num> for further discussion. Our technical team will work on resolving the issue as soon as possible.

Chunk 3: Subject: Reported Login Problems Today Customer: The platform encountered occasional login difficulties. Support Answer: <name>, we apologize for the inconvenience caused by the occasional login failures on the platform. Our technical team is currently investigating the issue and is working to resolve it as soon as possible. Please provide any details about the error message you received; this information may help us identify the root cause of the problem. If it's convenient, we can schedule a call to discuss the issue further: <tel_num>. Please let us know a suitable time, and we are happy to assist you regarding <acc_num>.

Subject: How can I get help with my invoice?

Exact Chunks

Chunk 1: to proceed? I appreciate your prompt assistance. Looking forward to resolving this issue. Support Answer: Dear [Name], I have taken note of the concern regarding the monthly invoice issue. Please provide me with additional details to assist in resolving this, such as the date and amount in question for the upcoming bill, as well as your account number <acc_num>. Once I have the necessary information, I will investigate and find a solution. If further details are needed, please call the number <tel_num>. Thank you, and I look forward to resolving this issue.

Chunk 2: review the account statements and invoices. Please provide the account number and the specific dates of the invoice in question. For further discussion, I would prefer to call at your convenience. Please let me know a suitable time to reach you at <tel_num>.

Chunk 3: Subject: Assistance with Unexpected Invoice Customer: Received an unexpected invoice with additional charges. This might have occurred due to a billing error or miscommunication. So far, we have reviewed the contract and requested clarification from the billing department. We would appreciate it if you could look into the matter and provide a revised invoice with an explanation of the additional charges. Support Answer: Dear [name], we are writing this email to acknowledge your concern regarding the unexpected invoice with additional charges. We understand your concern and would be happy to assist in resolving this matter. To better understand the issue, could you please provide your account number, invoice number, and any specific questions you have? We will review your account

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

Smaller chunks allow FAISS to identify more focused pieces of information that are relevant to a particular question