"""
API Routes for RAG Chatbot

These are the FastAPI endpoints that users interact with.
Since you know FastAPI, this should feel familiar!

ENDPOINTS:
1. POST /api/upload      - Upload a document (PDF/TXT) to be processed
                           Skips re-processing if already indexed (caching).
                           Add ?force=true to force re-indexing.
2. POST /api/ask         - Ask a question about uploaded documents (full RAG pipeline)
3. GET  /api/documents   - List all indexed documents (used by frontend on startup)
4. GET  /api/health      - Health check endpoint
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.models.schemas import (
    QuestionRequest,
    AnswerResponse,
    UploadResponse,
    DocumentListResponse,
)
from app.core.rag_engine import process_document, ask_question, get_all_documents


# Create a router - groups related endpoints together
# prefix="/api" means all routes start with /api
router = APIRouter(prefix="/api")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    force: bool = Query(False, description="Force re-processing even if already indexed"),
):
    """
    Upload a document to be processed for RAG.

    CACHE BEHAVIOUR:
    If the same filename has already been indexed in ChromaDB, this endpoint
    skips re-processing and returns the cached chunk count immediately.
    This means users only need to upload a document ONCE across sessions.
    Pass ?force=true to override the cache and re-index the document.

    What happens on a new upload:
    1. File is saved to disk (data/ folder)
    2. Text is extracted (PyPDF2 for PDF, plain read for TXT)
    3. Text is split into chunks (with overlap)
    4. Chunks are embedded and stored in ChromaDB

    After this, you can ask questions about this document!

    Args:
        file: PDF or TXT file to upload
        force: If True, re-process even if already indexed

    Returns:
        UploadResponse with filename, chunk count, status, and already_indexed flag
    """
    # Validate file type - accept PDF, TXT, and images (for OCR)
    allowed_exts = (".pdf", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(
            status_code=400,
            detail="Supported file types: PDF, TXT, PNG, JPG, JPEG, BMP, WEBP",
        )

    result = await process_document(file, force_reprocess=force)
    return UploadResponse(**result)


@router.post("/ask", response_model=AnswerResponse)
async def ask_question_endpoint(request: QuestionRequest):
    """
    Ask a question about uploaded documents.

    THE FULL RAG PIPELINE RUNS HERE:
    1. RETRIEVE: Your question is converted to an embedding,
       ChromaDB finds the most similar document chunks (top_k=5 default)
    2. AUGMENT: Those chunks are injected into a prompt
    3. GENERATE: Groq LLM generates an answer based on the chunks

    Works well for both specific questions ("What is Meet's email?")
    and broad identity/summary questions ("Who is this person?",
    "What is this document about?").

    Args:
        request: QuestionRequest with question and optional top_k

    Returns:
        AnswerResponse with the answer and source chunks used
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    result = ask_question(
        question=request.question,
        top_k=request.top_k,
    )
    return AnswerResponse(**result)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all documents currently indexed in the vector store.

    The frontend calls this endpoint on startup to populate the
    document sidebar with previously uploaded files - so users
    don't need to re-upload documents after refreshing the page.

    Returns:
        DocumentListResponse with list of {source, chunk_count} dicts
    """
    docs = get_all_documents()
    return DocumentListResponse(documents=docs)


@router.get("/health")
async def health_check():
    """
    Simple health check endpoint.

    Use this to verify the API is running.
    Returns {\"status\": \"healthy\"} if everything is OK.
    """
    return {"status": "healthy", "message": "RAG Chatbot API is running!"}
