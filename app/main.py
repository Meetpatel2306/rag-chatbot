"""
RAG Chatbot - Main Application Entry Point

This is where the FastAPI app is created and configured.
Run this file to start the server:
    conda activate rag-chatbot
    cd rag-chatbot
    uvicorn app.main:app --reload

Then open http://localhost:8000/docs for the Swagger UI.

WHAT IS THIS PROJECT?
=====================
A RAG (Retrieval Augmented Generation) Chatbot API that:
1. Accepts document uploads (PDF/TXT)
2. Processes and stores them in a vector database (ChromaDB)
3. Answers questions based on those documents using Google Gemini LLM

The key AI concepts used:
- Embeddings (text -> vectors using sentence-transformers)
- Vector Database (ChromaDB for storing and searching embeddings)
- Prompt Engineering (structured prompts for accurate answers)
- RAG Pipeline (Retrieve -> Augment -> Generate)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# Create the FastAPI application
# title and description appear in the Swagger docs at /docs
app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "A Document Q&A Chatbot powered by RAG (Retrieval Augmented Generation). "
        "Upload documents and ask questions - the AI answers based on YOUR documents, "
        "not its general knowledge. Built with FastAPI, ChromaDB, and Google Gemini."
    ),
    version="1.0.0",
)

# Allow frontend (Next.js on port 3000) to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our API routes
# All endpoints from routes.py are now available
app.include_router(router)


@app.get("/")
async def root():
    """
    Root endpoint - provides basic info about the API.
    Visit /docs for the interactive Swagger documentation.
    """
    return {
        "app": "RAG Chatbot API",
        "version": "1.0.0",
        "docs": "Visit /docs for interactive API documentation",
        "endpoints": {
            "upload": "POST /api/upload - Upload a document",
            "ask": "POST /api/ask - Ask a question",
            "health": "GET /api/health - Health check",
        },
    }
