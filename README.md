# RAG Chatbot - Document Q&A with AI

A full-stack **RAG (Retrieval Augmented Generation)** chatbot that lets you upload documents (PDF, TXT, or scanned images) and ask questions about them. The AI answers based on **your documents**, not its general knowledge.

![Tech Stack](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat&logo=next.js&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white)

---

## What is RAG?

**RAG (Retrieval Augmented Generation)** is a technique that gives LLMs access to your private data:

1. **Retrieve** - Search your documents for relevant chunks using vector similarity
2. **Augment** - Inject those chunks into the LLM prompt as context
3. **Generate** - LLM generates an accurate answer based on your actual data

This prevents **hallucination** (LLM making up facts) by grounding answers in real documents.

---

## Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| **FastAPI** | REST API framework |
| **ChromaDB** | Vector database for storing embeddings |
| **Sentence-Transformers** | Converting text to embeddings (all-MiniLM-L6-v2) |
| **Google Gemini** | LLM for generating answers (gemini-2.0-flash) |
| **PyPDF2** | PDF text extraction |

### Frontend
| Technology | Purpose |
|---|---|
| **Next.js 16** | React framework |
| **TypeScript** | Type safety |
| **Tailwind CSS** | Styling |
| **Axios** | API calls |
| **Lucide React** | Icons |

---

## Project Structure

```
rag-chatbot/
├── app/                          # FastAPI Backend
│   ├── main.py                   # App entry point + CORS config
│   ├── api/
│   │   └── routes.py             # API endpoints (upload, ask, health)
│   ├── core/
│   │   ├── config.py             # Settings from .env
│   │   └── rag_engine.py         # RAG pipeline orchestrator
│   ├── services/
│   │   ├── document_loader.py    # PDF/TXT parsing + text chunking
│   │   ├── vector_store.py       # ChromaDB operations + embeddings
│   │   └── llm_service.py        # Google Gemini integration
│   └── models/
│       └── schemas.py            # Pydantic request/response models
├── frontend/                     # Next.js Frontend
│   └── src/app/
│       ├── page.tsx              # Chat UI with file upload
│       ├── layout.tsx            # Root layout
│       └── globals.css           # Global styles
├── data/                         # Uploaded documents (auto-created)
├── .env                          # API keys and config
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+ (via Conda)
- Node.js 18+
- Google Gemini API key (free from [Google AI Studio](https://aistudio.google.com/apikey))

### 1. Clone & Setup Backend

```bash
# Create conda environment
conda create -n rag-chatbot python=3.10 -y
conda activate rag-chatbot

# Install dependencies
cd rag-chatbot
pip install -r requirements.txt
```

### 2. Configure Environment

Edit the `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./data
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 3. Setup Frontend

```bash
cd frontend
npm install
```

### 4. Run the Application

**Terminal 1 - Backend:**
```bash
conda activate rag-chatbot
cd rag-chatbot
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd rag-chatbot/frontend
npm run dev
```

### 5. Open the App

- **Frontend UI**: http://localhost:3000
- **API Docs (Swagger)**: http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a PDF or TXT document |
| `POST` | `/api/ask` | Ask a question about uploaded documents |
| `GET` | `/api/health` | Health check |

### Upload a Document
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "top_k": 3}'
```

---

## How It Works (RAG Pipeline)

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Upload PDF  │────>│ Extract Text  │────>│ Split Chunks │
└──────────────┘     └───────────────┘     └──────┬───────┘
                                                   │
                                                   v
                                           ┌──────────────┐
                                           │  Embeddings  │
                                           │ (sentence-   │
                                           │ transformers)│
                                           └──────┬───────┘
                                                   │
                                                   v
                                           ┌──────────────┐
                                           │   ChromaDB   │
                                           │ (Vector DB)  │
                                           └──────┬───────┘
                                                   │
┌──────────────┐     ┌───────────────┐             │
│  User Query  │────>│  Find Similar │<────────────┘
└──────────────┘     │    Chunks     │
                     └───────┬───────┘
                             │
                             v
                     ┌───────────────┐     ┌──────────────┐
                     │ Build Prompt  │────>│   Gemini     │
                     │ (chunks +     │     │   LLM        │
                     │  question)    │     └──────┬───────┘
                     └───────────────┘            │
                                                  v
                                          ┌──────────────┐
                                          │   Answer +   │
                                          │   Sources    │
                                          └──────────────┘
```

---

## AI/ML Concepts Used

This project covers the following AI concepts (detailed in `ai_concepts.pdf`):

1. **RAG (Retrieval Augmented Generation)** - The core technique
2. **Embeddings** - Converting text to numerical vectors
3. **FAISS vs BM25** - Semantic vs keyword search methods
4. **Vector Database** - Why ChromaDB, not PostgreSQL
5. **Text Chunking** - Splitting documents with overlap
6. **Prompt Engineering** - Structuring LLM inputs for best results
7. **Temperature & Tokens** - LLM generation parameters
8. **Hallucination** - What it is and how RAG reduces it
9. **Metadata** - Data about data in vector stores

---

## Key Configuration

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between consecutive chunks |
| `top_k` | 3 | Number of similar chunks to retrieve |
| `temperature` | 0.2 | LLM creativity (low = factual) |
| `max_output_tokens` | 1024 | Max answer length |

---

## Built With

- **Backend**: FastAPI + ChromaDB + Sentence-Transformers + Google Gemini
- **Frontend**: Next.js + TypeScript + Tailwind CSS
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **LLM**: Gemini 2.0 Flash
- **Environment**: Conda (rag-chatbot)
