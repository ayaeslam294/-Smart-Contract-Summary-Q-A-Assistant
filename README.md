
# Smart Contract Summary & Q&A Assistant

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/langchain-LCEL-informational)
![FAISS](https://img.shields.io/badge/vectorstore-FAISS-success)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![Gradio](https://img.shields.io/badge/frontend-Gradio-orange)
![Groq](https://img.shields.io/badge/LLM-Groq-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

Smart Contract Summary & Q&A Assistant is a **Retrieval-Augmented Generation (RAG)** application for analyzing smart contracts and legal documents.

You can upload **PDF or DOCX contracts**, ask natural-language questions, generate summaries, and automatically **evaluate answer quality** using an LLM-as-a-Judge approach.  
All answers are **grounded in the source documents**, minimizing hallucinations.

---

## Features

- Upload and process PDF / DOCX contracts  
- Semantic search with FAISS vector store  
- Context-aware Q&A using RAG  
- Contract-level summarization  
- Streaming responses  
- Automated evaluation with LLM-as-a-Judge  
- Support for multiple documents in one knowledge base  

---

## Tech Stack

| Component | Technology |
|--------|------------|
| Language Model | Groq (OpenAI-compatible API) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Store | FAISS |
| RAG Framework | LangChain (LCEL) |
| Backend | FastAPI + LangServe |
| Frontend | Gradio |

---

## Project Structure

```

smart-contract-assistant/
├── config.py          # Environment config, LLM & embedder factories
├── ingest.py          # Document ingestion & vector store management
├── rag_chain.py       # Retrieval, answering, summarization logic
├── app.py             # Gradio UI
├── server.py          # FastAPI backend & LangServe routes
├── evaluation.py      # LLM-as-a-Judge evaluation pipeline
├── requirements.txt   # Python dependencies
├── .env               # API keys 
├── vectorstore/       # FAISS index files
└── uploads/           # Uploaded documents

````

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd smart-contract-assistant
````

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

Other settings (models, chunk size, etc.) have sensible defaults.

---

## Running the Application

### Start the backend server

```bash
python server.py
```

Runs at: `http://localhost:9012`

---

### Start the Gradio UI

Open a second terminal, activate the venv, then:

```bash
python app.py
```

Open in browser: `http://localhost:7860`

---

## How It Works

1. Documents are loaded (PDF/DOCX) and split into chunks
2. Chunks are embedded and stored in FAISS
3. User queries retrieve the most relevant chunks
4. The LLM generates answers grounded in retrieved context
5. Optional evaluation compares answers to ground truth using an LLM judge

---

## Evaluation: LLM-as-a-Judge

The evaluation pipeline automatically tests RAG quality:

1. Generate synthetic questions from document chunks
2. Answer them using the RAG pipeline
3. Judge correctness using a separate LLM call

Scoring:

* **1** — Incorrect or incomplete
* **2** — Correct
* **3** — Correct with added useful detail

The final report includes accuracy and per-question breakdowns.

---

## API Endpoints

| Endpoint     | Method | Description                  |
| ------------ | ------ | ---------------------------- |
| `/upload`    | POST   | Upload and process documents |
| `/qa_stream` | POST   | Streaming Q&A                |
| `/summarize` | POST   | Summarize all documents      |
| `/evaluate`  | POST   | Run evaluation               |
| `/qa`        | POST   | LangServe Q&A route          |
| `/retriever` | POST   | LangServe retriever route    |

---

## Use Cases

* Smart contract analysis
* Legal document review
* Compliance & policy checking
* RAG experimentation & evaluation

## Contact

For questions or suggestions, please open an issue or contact:
**Aya Eslam Elsawy**
[LinkedIn](https://www.linkedin.com/in/aya-eslam-1b8792349) | [GitHub](https://github.com/ayaeslam294)


