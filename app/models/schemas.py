"""
Pydantic Schemas for RAG Chatbot API

These are the request/response models that define what data
our API accepts and returns. FastAPI uses these for:
- Input validation (reject bad requests automatically)
- API documentation (Swagger UI at /docs)
- Serialization (convert Python objects to JSON)
"""

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """
    What the user sends when asking a question.

    Fields:
        question: The user's question about the uploaded documents.
        top_k: How many relevant chunks to retrieve from vector DB.
               Default is 5 (raised from 3) - giving the LLM more context
               for broad/identity questions like "who is this person?".
               More chunks = more context but slower and may dilute relevance.
    """
    question: str
    top_k: int = 5


class AnswerResponse(BaseModel):
    """
    What our API returns after processing the question.

    Fields:
        answer: The LLM-generated answer based on document context.
        sources: List of text chunk snippets that were used as context.
                 This helps the user verify the answer (reduces hallucination trust).
    """
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    """
    Response after a document is uploaded and processed.

    Fields:
        filename: Name of the uploaded file.
        chunks: How many text chunks were created (or are cached) for the document.
        message: Success/status message describing what happened.
        already_indexed: True if the document was already in ChromaDB and
                         was NOT re-processed (cached result returned).
    """
    filename: str
    chunks: int
    message: str
    already_indexed: bool = False  # NEW: tells frontend whether this was a cache hit


class DocumentInfo(BaseModel):
    """
    Information about a single indexed document.

    Fields:
        source: The original filename of the document.
        chunk_count: Total number of chunks stored in ChromaDB for this document.
    """
    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """
    Response for the GET /api/documents endpoint.

    Used by the frontend on startup to populate the document sidebar
    with all previously indexed files - no re-upload required.

    Fields:
        documents: List of DocumentInfo objects (one per unique indexed file).
    """
    documents: list[DocumentInfo]
