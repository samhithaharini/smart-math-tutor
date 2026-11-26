"""
rag_engine.py — Local RAG (Retrieval-Augmented Generation) Pipeline
===================================================================

CONCEPT — RAG (Retrieval-Augmented Generation):
LLMs only know what they saw during training—they do NOT know your private
notes, company docs, or made-up facts like "Vishal's Constant = 42".
RAG solves this in 3 steps:
  1. Load & Split: Read `formulas.txt` and split it into small chunks.
  2. Embed & Store: Convert each chunk into a vector (numbers capturing meaning)
     using a local HuggingFace model, and store them in a Chroma vector database.
  3. Retrieve: When a question arrives, find the most similar chunks and hand
     them to the LLM as context so it answers from your data, not guesswork.

CONCEPT — LangChain Tool:
Wrapping our retriever in `@tool` turns it into a standardized tool with a
name, description, and input schema that a LangChain Agent can decide to call.
"""

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

FORMULAS_FILE = Path(__file__).parent / "formulas.txt"


@lru_cache(maxsize=1)
def _get_vector_store() -> Chroma:
    """
    Build and cache an in-memory Chroma vector store from `formulas.txt`.

    WHY in-memory + `@lru_cache(maxsize=1)`?
    - `@lru_cache` loads the embedding model and indexes `formulas.txt` once per
      process instead of on every query.
    - Keeping Chroma in-memory (no persist_directory) means if you edit
      `formulas.txt` and restart the app, your changes take effect immediately.
    """
    raw_text = FORMULAS_FILE.read_text(encoding="utf-8")

    # Each formula block in `formulas.txt` is separated by a blank line (`\n\n`).
    # Splitting on `\n\n` keeps each formula intact as its own searchable chunk.
    chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
    documents = [
        Document(page_content=chunk, metadata={"source": FORMULAS_FILE.name, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]

    # Local embeddings run completely on your machine (no API key or cost needed).
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="smart_math_formulas",
    )


@tool
def search_formulas(query: str) -> str:
    """
    Search the local `formulas.txt` knowledge base (via RAG) for math/physics
    formulas and custom constants such as Vishal's Constant, Kinetic Energy,
    Ohm's Law, Compound Interest, and the Quadratic Formula.
    Always use this tool when the user asks about a formula or a custom constant.
    """
    vector_store = _get_vector_store()
    # Retrieve the top 2 most semantically relevant formula chunks.
    results = vector_store.similarity_search(query, k=2)

    if not results:
        return "No matching formulas found in formulas.txt."

    return "\n\n---\n\n".join(doc.page_content for doc in results)
