"""
Document Loader Service (with HYBRID OCR support)

This service handles:
1. Reading PDF and text files
2. Extracting text content from them (with OCR fallback for scanned PDFs)
3. Splitting text into chunks (with overlap)

WHY CHUNKING?
- LLMs have token limits - can't send 100 pages at once
- Smaller chunks = better embeddings = better search results
- Chunk overlap ensures no information is lost at boundaries

WHY HYBRID OCR?
- Text-based PDFs: PyMuPDF extracts text in milliseconds (FAST)
- Scanned PDFs: PyMuPDF returns empty -> we fall back to EasyOCR (SLOW but works)
- This gives us the best of both worlds: fast for normal PDFs, OCR only when needed
"""

import fitz  # PyMuPDF - MUCH faster than PyPDF2
import numpy as np
from app.core.config import settings


# ─── Lazy-loaded EasyOCR reader ──────────────────────────────────────────────
# EasyOCR loads ML models that take ~10-15 seconds to initialize.
# We use a global cache so the model is only loaded ONCE per process,
# not on every PDF upload. The first scanned PDF will be slow; subsequent
# scanned PDFs will be fast because the model is already in memory.
_OCR_READER_CACHE = None


def get_ocr_reader():
    """
    Get a cached EasyOCR Reader instance.

    EasyOCR's Reader loads PyTorch models which is slow (~10-15s).
    We cache it globally so it's only loaded once per process.

    Returns:
        easyocr.Reader instance configured for English
    """
    global _OCR_READER_CACHE
    if _OCR_READER_CACHE is None:
        # Import here so the app starts fast even if OCR isn't used
        import easyocr
        # ['en'] = English only. Add more languages like ['en', 'hi'] if needed.
        # gpu=False because most users don't have CUDA setup
        _OCR_READER_CACHE = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _OCR_READER_CACHE


def ocr_page_image(page) -> str:
    """
    Run OCR on a PDF page that has no extractable text (scanned page).

    Steps:
    1. Render the page as a high-resolution image (300 DPI for good OCR accuracy)
    2. Convert the pixmap into a numpy array (EasyOCR's input format)
    3. Run EasyOCR on the image to get text
    4. Join all detected text pieces into a single string

    Args:
        page: A PyMuPDF page object

    Returns:
        The OCR-extracted text from the page
    """
    # Step 1: Render page as image at 300 DPI
    # Higher DPI = better OCR accuracy but slower
    pix = page.get_pixmap(dpi=300)

    # Step 2: Convert pixmap to numpy array
    # EasyOCR works directly with numpy arrays (H x W x channels)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    # If image has alpha channel (RGBA), drop it -> EasyOCR wants RGB
    if pix.n == 4:
        img_array = img_array[:, :, :3]

    # Step 3: Run OCR
    reader = get_ocr_reader()
    # readtext returns list of (bbox, text, confidence) tuples
    # We only need the text part
    results = reader.readtext(img_array, detail=0)

    # Step 4: Join all detected text into one string
    return " ".join(results)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file using HYBRID approach:
    - First tries fast PyMuPDF text extraction
    - Falls back to OCR for pages where text extraction fails (scanned pages)

    PyMuPDF is significantly faster than PyPDF2 (often 10x-50x faster)
    and handles complex layouts better.

    OCR FALLBACK:
    If page.get_text() returns empty/whitespace, the page is likely scanned
    (just an image of text). We then render the page as an image and run
    EasyOCR on it to extract the text.

    Returns:
        Combined text from all pages (using direct extraction OR OCR per page)
    """
    doc = fitz.open(file_path)
    text_parts = []
    ocr_pages = 0  # Count how many pages needed OCR (for logging)

    for page_num, page in enumerate(doc):
        # Try fast text extraction first
        page_text = page.get_text()

        # If no text found, the page is likely scanned -> use OCR
        if not page_text or not page_text.strip():
            try:
                page_text = ocr_page_image(page)
                ocr_pages += 1
                print(f"[OCR] Page {page_num + 1}: extracted {len(page_text)} chars via OCR")
            except Exception as e:
                print(f"[OCR] Page {page_num + 1}: OCR failed - {e}")
                page_text = ""

        if page_text:
            text_parts.append(page_text)

    doc.close()

    if ocr_pages > 0:
        print(f"[OCR] Total pages processed with OCR: {ocr_pages}")

    return "\n".join(text_parts)


def extract_text_from_txt(file_path: str) -> str:
    """
    Read a plain text file and return its content.

    Args:
        file_path: Path to the .txt file

    Returns:
        The full text content of the file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_image(file_path: str) -> str:
    """
    Extract text from an image file (PNG, JPG, JPEG) using EasyOCR.

    This is for direct image uploads (e.g., a photo of a document,
    a screenshot, a receipt). Unlike PDFs, images go straight to OCR
    since there's no embedded text to extract first.

    Args:
        file_path: Path to the image file

    Returns:
        The OCR-extracted text from the image
    """
    reader = get_ocr_reader()
    # readtext can take a file path directly for images
    # detail=0 means we only get the text strings (no bounding boxes / confidence)
    results = reader.readtext(file_path, detail=0)
    text = " ".join(results)
    print(f"[OCR] Image '{file_path}': extracted {len(text)} chars")
    return text


def split_text_into_chunks(text: str) -> list[str]:
    """
    Split a large text into smaller overlapping chunks.

    WHY OVERLAP?
    Imagine a sentence at position 490-510 in the text.
    Without overlap (chunk_size=500): Chunk 1 gets words 1-500, Chunk 2 gets 501-1000.
    The sentence is CUT IN HALF! Neither chunk has the full meaning.
    With overlap (chunk_size=500, overlap=50): Chunk 2 starts at 450, so it captures
    the full sentence. Overlap = safety net against lost context.

    Args:
        text: The full document text to split

    Returns:
        List of text chunks, each approximately chunk_size characters
        with chunk_overlap characters of overlap between consecutive chunks
    """
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    # If text is smaller than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Get a chunk of text starting from 'start' position
        end = start + chunk_size
        chunk = text[start:end]

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk.strip())

        # Move start forward by (chunk_size - overlap)
        # This creates the overlap between consecutive chunks
        # Example: chunk_size=500, overlap=50
        # Chunk 1: 0-500, Chunk 2: 450-950, Chunk 3: 900-1400
        start += chunk_size - chunk_overlap

    return chunks


def load_document(file_path: str) -> list[str]:
    """
    Main function - Load a document and return its text chunks.

    This is the entry point that other parts of the app will call.
    It detects the file type, extracts text, and splits into chunks.

    Args:
        file_path: Path to the document (PDF or TXT)

    Returns:
        List of text chunks ready to be embedded and stored

    Raises:
        ValueError: If the file type is not supported (not PDF or TXT)
    """
    # Detect file type by extension
    lower_path = file_path.lower()
    if lower_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif lower_path.endswith(".txt"):
        text = extract_text_from_txt(file_path)
    elif lower_path.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        text = extract_text_from_image(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {file_path}. "
            f"Supported: PDF, TXT, PNG, JPG, JPEG, BMP, WEBP"
        )

    # Split the extracted text into chunks
    chunks = split_text_into_chunks(text)
    return chunks
