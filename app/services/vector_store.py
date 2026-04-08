"""
Vector Store Service (ChromaDB)

This service handles all interactions with ChromaDB - our vector database.

WHAT THIS DOES:
1. Creates/connects to a ChromaDB collection
2. Converts text chunks into embeddings using sentence-transformers
3. Stores chunks + embeddings + metadata in ChromaDB
4. Searches for similar chunks when user asks a question

WHY ChromaDB?
- Lightweight - runs locally, no server setup needed
- Persists data to disk - survives app restarts
- Built-in embedding support
- Free and open source
- Perfect for learning and small-medium projects
"""

import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings


# Global variable to cache the embedding model
# Loading this takes 1-2 seconds, so we ONLY do it once!
_EMBEDDING_FN_CACHE = None


# Sentence Transformer model for creating embeddings
# "all-MiniLM-L6-v2" is a small but effective model:
# - Fast inference (good for development)
# - 384-dimensional embeddings
# - Good balance of speed vs quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create and return a ChromaDB client.

    PersistentClient means data is saved to disk at chroma_persist_dir.
    Even if the app restarts, all stored embeddings remain.

    Returns:
        A ChromaDB PersistentClient instance
    """
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client


def get_embedding_function():
    """
    Get the embedding function that converts text -> vectors.

    This function uses a GLOBAL CACHE to ensure the model is
    only loaded once in memory, rather than on every request.
    """
    global _EMBEDDING_FN_CACHE
    if _EMBEDDING_FN_CACHE is None:
        _EMBEDDING_FN_CACHE = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _EMBEDDING_FN_CACHE


def get_or_create_collection(collection_name: str = "documents"):
    """
    Get an existing collection or create a new one.

    A COLLECTION in ChromaDB is like a TABLE in a normal database.
    It holds all the chunks, their embeddings, and metadata for
    a group of documents.

    Args:
        collection_name: Name of the collection (default: "documents")

    Returns:
        A ChromaDB Collection object
    """
    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    # get_or_create_collection: if it exists, returns it; if not, creates it
    # We pass our embedding function so ChromaDB automatically
    # converts text to embeddings when we add documents
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )
    return collection


def add_chunks_to_store(chunks: list[str], source: str) -> int:
    """
    Store text chunks in ChromaDB with their metadata.
    Uses batch processing for high-speed indexing of large documents.
    """
    collection = get_or_create_collection()

    # Step 1: Clean up old chunks for this file
    try:
        # Avoid loading 1M IDs into memory - just delete by source
        collection.delete(where={"source": source})
    except Exception:
        pass

    # Step 2: Add in batches (e.g. 500 chunks at a time)
    # This prevents memory spikes and ensures the server stays responsive.
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = [f"{source}_chunk{j}" for j in range(i, i + len(batch_chunks))]
        batch_metadatas = [
            {"source": source, "chunk_index": j}
            for j in range(i, i + len(batch_chunks))
        ]

        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )

    return len(chunks)


def list_documents() -> list[dict]:
    """
    Return a list of all unique documents stored in ChromaDB.

    Returns:
        List of dicts with 'source' and 'chunk_count' keys
    """
    collection = get_or_create_collection()
    try:
        results = collection.get(include=["metadatas"])
        if not results or not results["metadatas"]:
            return []
        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunk_count": cnt} for src, cnt in counts.items()]
    except Exception:
        return []


def document_exists(source: str) -> bool:
    """
    Check whether a document has already been indexed in ChromaDB.

    Args:
        source: The filename to check

    Returns:
        True if the document already has chunks stored
    """
    collection = get_or_create_collection()
    try:
        existing = collection.get(where={"source": source})
        return bool(existing and existing["ids"])
    except Exception:
        return False


def search_similar_chunks(query: str, top_k: int = 3) -> dict:
    """
    Search for the most similar chunks to the user's question.

    THIS IS THE "RETRIEVAL" STEP IN RAG!

    How it works:
    1. ChromaDB converts the query to an embedding (same model)
    2. Finds the top_k closest chunk embeddings (cosine similarity)
    3. Returns those chunks along with their metadata

    Args:
        query: The user's question
        top_k: How many similar chunks to return (default: 3)

    Returns:
        Dict with 'documents' (chunk texts) and 'metadatas' (source info)
    """
    collection = get_or_create_collection()

    # query() does the similarity search
    # It converts our question to an embedding, then finds
    # the nearest chunk embeddings in the collection
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    return results
