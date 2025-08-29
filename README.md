# RAG-Based Retrieval System (Qwen + Sentence-Transformers)

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using sentence-transformer embeddings and a local LLM (`Qwen2.5-3B-Instruct`). It allows you to:

- **Build a vector index** from `.txt`, `.html`, and `.md` files
- **Query the indexed documents** using natural language and receive generated responses grounded in source content

---

## When to Use RAG Instead of Training a Model

RAG is ideal when you want to answer questions based on specific documents **without retraining a language model**.

**Why choose RAG?**
- You have custom data like text files, documentation, or notes
- You don’t want to (or can't) fine-tune a large model on your data
- RAG allows the model to "look up" relevant information during inference

> In simple terms: **RAG helps the model "read" your documents at runtime instead of "memorizing" them through training.**

---

## Files Overview

### `rag_build_index.py`
Builds a vector index from `.txt`, `.html`, and `.md` files using:
- `sentence-transformers/all-MiniLM-L6-v2` for embedding
- FAISS for similarity search
- Generates:
  - FAISS index at `data/rag/faiss.index`
  - Metadata at `data/rag/segments.jsonl`

Scans files under the `data/rag/raw/` directory (recursively) with the extensions `.txt`, `.md`, and `.html`.

### `rag_qa.py`
Queries the FAISS index using natural language with:
- MiniLM embedding for the query
- `Qwen2.5-3B-Instruct` to generate grounded answers
- Prints both the answer and supporting content

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/arojit/Retrieval-Augmented-Generation.git
cd Retrieval-Augmented-Generation
```

### 2. Create and Activate Virtual Environment (Recommended)

```bash
# For Linux or macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Build the Index

```bash
python src/rag_build_index.py
```

- Scans all `.txt`, `.md`, and `.html` files recursively under `data/rag/raw/`
- This will generate:
  - `data/rag/faiss.index`
  - `data/rag/segments.jsonl`

### Step 2: Ask Questions

```bash
uvicorn src.rag_qa:app --host 0.0.0.0 --port 8080 --reload
```

- Invoke the below API.
- The model will return an answer using relevant document segments.
```
curl --location 'http://127.0.0.1:8080/ask' \
--header 'Content-Type: application/json' \
--data '{
    "query":"What is our refund policy?"
}'
```
Response
```
{
    "answer": "Our refund policy allows full refunds within 30 days for unused licenses [1]. For pro-rated subscriptions, partial refunds are available. [1]",
    "sources": [
        "[1] data/rag/raw/policy/refund-policy.md",
        "[2] data/rag/raw/sherlock/adventure-of-sherlock-holmes.pdf",
        "[3] data/rag/raw/sherlock/adventure-of-sherlock-holmes.pdf",
        "[4] data/rag/raw/sherlock/adventure-of-sherlock-holmes.pdf",
        "[5] data/rag/raw/sherlock/adventure-of-sherlock-holmes.pdf"
    ]
}
```

---

## Questions you can ask
- What is our refund policy?
- How do I change my email
- When can I get full refund?
- Address of Sherlock Holmes

## Features

- Supports `.txt`, `.md`, `.html` documents
- Semantic search via sentence-transformers
- Local answer generation with Qwen2.5-3B-Instruct
- Outputs include the source content for transparency
- No need to fine-tune large models

---

## Directory Structure

```
.
├── data/
│   └── rag/
│       ├── faiss.index
│       ├── segments.jsonl
│       └── raw/
│           ├── faq/
│           │   └── account-faq.txt
│           ├── howto/
│           │   └── setup.html
│           └── policy/
│               └── refund-policy.md
├── src/
│   ├── rag_build_index.py
│   └── rag_qa.py
├── README.md
└── requirements.txt
```