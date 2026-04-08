"""
RAG Engine - The Brain of our Chatbot

This module connects ALL the pieces together into one RAG pipeline:

THE FULL RAG FLOW:
=================
1. USER UPLOADS DOCUMENT
   document_loader.py: PDF/TXT -> extract text -> split into chunks

2. CHUNKS GET STORED
   vector_store.py: chunks -> embeddings -> stored in ChromaDB with metadata

3. USER ASKS A QUESTION
   vector_store.py: question -> embedding -> find similar chunks (RETRIEVAL)
   llm_service.py: chunks + question -> Groq LLM -> answer (GENERATION)

NEW: DOCUMENT CACHING
=====================
If the user uploads the same file again, we skip re-processing and return
the cached chunk count from ChromaDB directly. This avoids duplicate IDs
and saves time. Pass force_reprocess=True to override cache.

This file orchestrates steps 1-3 with two simple functions:
- process_document(): handles upload flow (steps 1-2) with cache check
- ask_question(): handles Q&A flow (step 3)
- get_all_documents(): lists all indexed documents (used by frontend on load)
"""

import os
import shutil
from fastapi import UploadFile
from app.core.config import settings
from app.services.document_loader import load_document
from app.services.vector_store import (
    add_chunks_to_store,
    search_similar_chunks,
    list_documents,
    document_exists,
)
from app.services.llm_service import generate_answer


async def process_document(file: UploadFile, force_reprocess: bool = False) -> dict:
    """
    Process an uploaded document through the RAG pipeline.

    CACHE BEHAVIOUR:
    If the document (matched by filename) is already indexed in ChromaDB,
    we skip extraction and embedding - just return the cached chunk count.
    Pass force_reprocess=True (via ?force=true query param) to re-index.

    Flow (when not cached):
    1. Save the uploaded file to disk (data/ folder)
    2. Extract text and split into chunks (document_loader)
    3. Store chunks + embeddings in ChromaDB (vector_store)

    Args:
        file: The uploaded file from FastAPI (PDF or TXT)
        force_reprocess: If True, re-index even if already in ChromaDB

    Returns:
        Dict with filename, number of chunks, status message, and already_indexed flag
    """
    source_name = file.filename

    # --- CACHE CHECK ---
    # If the document is already indexed and force=False, skip re-processing.
    # We still need to "consume" the file bytes so FastAPI doesn't hang.
    if not force_reprocess and document_exists(source_name):
        await file.read()  # consume file bytes without saving
        docs = list_documents()
        chunk_count = next(
            (d["chunk_count"] for d in docs if d["source"] == source_name), 0
        )
        return {
            "filename": source_name,
            "chunks": chunk_count,
            "message": f'"{source_name}" is already indexed with {chunk_count} chunks. No re-processing needed.',
            "already_indexed": True,
        }

    # Step 1: Save uploaded file to disk
    # We need it on disk so PyPDF2 can read it
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, file.filename)

    # Save the file using shutil.copyfileobj (efficient for large files)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Step 2: Extract text and split into chunks
    # This calls document_loader.load_document() which:
    # - Detects file type (PDF/TXT)
    # - Extracts all text
    # - Splits into overlapping chunks
    chunks = load_document(file_path)

    # Step 3: Store chunks in ChromaDB
    # add_chunks_to_store() handles duplicate IDs by deleting old chunks first
    # then: converts each chunk to an embedding and stores with metadata
    num_chunks = add_chunks_to_store(chunks, source=file.filename)

    return {
        "filename": file.filename,
        "chunks": num_chunks,
        "message": f'Successfully processed "{file.filename}" into {num_chunks} chunks.',
        "already_indexed": False,
    }


def ask_question(question: str, top_k: int = 5) -> dict:
    """
    Answer a question using the RAG pipeline.

    THIS IS WHERE RAG HAPPENS!

    Flow:
    1. RETRIEVE: Search ChromaDB for chunks similar to the question
    2. AUGMENT: Combine chunks into context for the LLM prompt
    3. GENERATE: Send context + question to Groq LLM, get answer

    top_k is set to 5 (up from the old default of 3) so broad questions
    like "who is this person?" get enough context chunks to synthesize
    a complete answer.

    Args:
        question: The user's question about uploaded documents
        top_k: Number of similar chunks to retrieve (default: 5)

    Returns:
        Dict with the answer and source chunks used
    """
    # Step 1: RETRIEVE - Find relevant chunks
    # Converts question to embedding, finds nearest chunks in ChromaDB
    results = search_similar_chunks(query=question, top_k=top_k)

    # Extract the documents and metadata from results
    # results["documents"][0] = list of chunk texts
    # results["metadatas"][0] = list of metadata dicts
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    # If no documents found, return early with a helpful message
    if not documents:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF or TXT document first.",
            "sources": [],
        }

    # Steps 2 & 3: AUGMENT + GENERATE
    # generate_answer() builds the prompt with context and sends to Groq LLM
    answer = generate_answer(
        question=question,
        context_chunks=documents,
        sources=metadatas,
    )

    # Prepare source information for the response
    # This tells the user WHICH chunks were used (transparency!)
    sources = [
        f"[{meta.get('source', 'unknown')}] {doc[:120]}..."
        for doc, meta in zip(documents, metadatas)
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


def get_all_documents() -> list[dict]:
    """
    Return all unique documents currently indexed in ChromaDB.

    Used by the frontend on startup to populate the document list
    without requiring the user to re-upload files.

    Returns:
        List of dicts with 'source' (filename) and 'chunk_count' keys
    """
    return list_documents()
