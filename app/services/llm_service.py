"""
LLM Service (Groq)

This service handles communication with the Groq LLM provider.

WHAT THIS DOES:
- Takes the user's question + retrieved context chunks
- Constructs a well-engineered prompt (Prompt Engineering!)
- Sends it to Groq's Llama model and returns the generated answer

KEY IMPROVEMENTS OVER V1:
- Handles broad/identity questions: "who is this?", "what is this document?"
  The system prompt explicitly instructs the LLM to synthesize across ALL
  provided chunks rather than just looking for a single specific answer.
- max_tokens raised to 1500 for richer answers
- temperature slightly raised to 0.3 for more natural language

Set GROQ_API_KEY in .env to authenticate.
"""

from groq import Groq
from app.core.config import settings


# Create Groq client - connects to Groq's API
# Groq is free and extremely fast (uses custom LPU hardware)
client = Groq(api_key=settings.groq_api_key)


def generate_answer(question: str, context_chunks: list[str], sources: list[dict]) -> str:
    """
    Generate an answer using the LLM with RAG context.

    THIS IS THE "GENERATION" STEP IN RAG!

    How it works:
    1. We combine all retrieved chunks into a single context string
    2. We build a prompt that tells the LLM:
       - What role it plays (intelligent document assistant)
       - What context documents it has (with numbered chunks)
       - What question to answer
       - Special rules for identity/summary questions
    3. We send this with temperature=0.3 for mostly-factual,
       still readable answers
    4. Return the generated answer

    WHY THE IMPROVED SYSTEM PROMPT?
    The original "Answer ONLY from context" prompt caused the bot to fail
    on broad questions like "who is this PDF?" because it would look for
    a single matching chunk instead of synthesizing information across all
    retrieved chunks. The new prompt explicitly handles this case.

    Args:
        question: The user's question
        context_chunks: List of relevant text chunks from vector DB
        sources: Metadata about each chunk (source file, chunk index)

    Returns:
        The LLM-generated answer as a string
    """
    # =========================================================
    # STEP 1: BUILD THE CONTEXT BLOCK
    # Combine all chunks into one context block.
    # We number them so the LLM can reference specific chunks.
    # =========================================================
    context = ""
    for i, chunk in enumerate(context_chunks):
        source_info = sources[i].get("source", "unknown") if i < len(sources) else "unknown"
        context += f"\n--- Chunk {i + 1} (from: {source_info}) ---\n"
        context += chunk + "\n"

    # =========================================================
    # STEP 2: SYSTEM MESSAGE (Prompt Engineering!)
    # Sets the LLM's role, rules, and special case handling.
    # =========================================================
    system_message = """You are an intelligent document assistant. Your role is to answer questions \
based on the provided document context.

GUIDELINES:
1. Answer based on the information found in the context chunks.
2. If the user asks about the document itself (e.g. "what is this document?", \
"who is this resume about?", "summarize this document", "who is this person?"), \
provide a helpful overview by synthesizing details from ALL available chunks.
3. If the answer is clearly not present anywhere in the context, say: \
"I could not find this information in the uploaded documents."
4. Be conversational, helpful, and concise.
5. When relevant, mention the source document name.
6. Do NOT fabricate or assume information not present in the context.
7. For identity/summary questions, synthesize key facts (name, role, skills, \
experience, contact info) from ALL chunks to give a complete picture."""

    # =========================================================
    # STEP 3: USER MESSAGE
    # Contains the actual context + question sent to the model.
    # =========================================================
    user_message = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""

    # Generate response using Groq API
    # Uses Llama 3.3 70B - a powerful open-source model
    # temperature=0.3 means: mostly factual, still readable/natural
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Free, fast, powerful model on Groq
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,   # Slightly higher than 0.2 for more natural language
        max_tokens=1500,   # Raised from 1024 to allow fuller answers
    )

    return response.choices[0].message.content
