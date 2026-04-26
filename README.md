# 📚 RAG Document Question Answering Bot

## 🔍 Project Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to ask questions over a collection of documents. It retrieves relevant document chunks using vector similarity search and generates answers grounded strictly in those documents. The system also displays source references for transparency and trust.

---

## 🧰 Tech Stack

* Python 3.13
* LangChain (latest modular version)
* langchain-community
* langchain-text-splitters
* FAISS (vector database)
* sentence-transformers (for embeddings)
* Streamlit (UI)
* python-dotenv

---

## 🧠 Architecture Overview

The system follows a standard RAG pipeline:

1. **Ingestion**
   Documents (TXT/PDF) are loaded from the `data/` folder.

2. **Chunking**
   Documents are split into smaller chunks for better retrieval.

3. **Embedding**
   Each chunk is converted into vector embeddings using a HuggingFace model.

4. **Vector Storage**
   Embeddings are stored in a FAISS vector database.

5. **Retrieval**
   User query is converted into embedding → top-k similar chunks are retrieved.

6. **Generation**
   The system returns answers based on retrieved context (extractive approach).

---

## ✂️ Chunking Strategy

* Method: Recursive Character Text Splitting
* Chunk Size: 500 characters
* Overlap: 100 characters

### Why?

This approach ensures:

* Context continuity across chunks
* Better semantic retrieval
* Reduced information loss at boundaries

---

## 🔗 Embedding Model & Vector Database

### Embedding Model:

* `all-MiniLM-L6-v2` (from sentence-transformers)

### Why?

* Lightweight and fast
* Works locally (no API cost)
* Good semantic similarity performance

### Vector Database:

* FAISS (Facebook AI Similarity Search)

### Why?

* Extremely fast similarity search
* Runs locally (no cloud dependency)
* Easy integration with LangChain

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd rag_doc_qa_bot
```

### 2. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 3. Add documents

Place 4–5 `.txt` or `.pdf` files inside the `data/` folder.

### 4. Run ingestion (build vector database)

```bash
py ingest_documents.py
```

### 5. Run chatbot (CLI version)

```bash
py run_chatbot.py
```

### 6. Run UI (Streamlit)

```bash
py -m streamlit run app.py
```

---

## 🔐 Environment Variables

This project can run **without API keys** (free local setup).

Optional (if using OpenAI in future):

```env
OPENAI_API_KEY=your_api_key_here
```

⚠️ Never commit `.env` file to GitHub.

---

## 💬 Example Queries

* What is artificial intelligence?
* Explain types of machine learning
* What are cloud service models?
* What is a Python function?
* How do businesses use AI?

### Expected Answer Themes:

* Definitions
* Comparisons
* Explanations from documents
* Context-based summaries

---

## ⚠️ Known Limitations

* **No generative reasoning**
  Currently uses extractive answers (no LLM), so responses are limited to document text.

* **Chunk boundary issues**
  Important context may split across chunks.

* **No conversation memory**
  Each query is processed independently (unless UI chat memory is enabled).

* **Limited to provided documents**
  Cannot answer outside the dataset.

* **Basic answer formatting**
  Responses are not refined or summarized deeply.

---

## 🎯 Key Feature

The system avoids hallucination by strictly grounding answers in retrieved document content and showing sources.

---

## 📂 Project Structure

```
rag_doc_qa_bot/
│
├── data/                # Input documents
├── vector_store/        # Generated embeddings
├── ingest_documents.py  # Indexing script
├── run_chatbot.py       # CLI chatbot
├── app.py               # Streamlit UI
├── requirements.txt
├── .env
└── README.md
```

---

## 🚀 Future Improvements

* Add local LLM for better answer generation
* Improve UI with chat memory and formatting
* Support more file formats (DOCX, HTML)
* Add semantic ranking and re-ranking

---
