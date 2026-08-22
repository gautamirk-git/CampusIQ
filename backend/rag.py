"""Core RAG pipeline: chunking, embedding, retrieval, and grounded generation."""

import os
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from anthropic import Anthropic

CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "university_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# claude-haiku-4-5 for cheap dev/testing (~1/5th the cost of Opus 5 per
# token). Swap to "claude-opus-5" for production-quality answers.
CLAUDE_MODEL = "claude-haiku-4-5"

# Models where an explicit thinking={"type": "disabled"} is a recognized
# request shape (thinking is on by default for these, so it must be turned
# off explicitly to avoid spending output tokens on it). Older models like
# Haiku 4.5 have no thinking unless explicitly enabled, so the param is
# simply omitted for them further down.
_THINKING_TOGGLEABLE_PREFIXES = ("claude-opus-5", "claude-sonnet-5")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3
MAX_ANSWER_TOKENS = 400

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a \
university using ONLY the excerpts provided below as context. These excerpts \
were retrieved from the university's official information document.

Rules:
- Answer using only information found in the provided context.
- If the context does not contain the answer, say plainly: "I don't know — \
that information isn't in the document I have access to." Do not guess or \
use outside knowledge.
- When you do answer, be specific and cite concrete details (numbers, dates, \
names) exactly as they appear in the context.
- Keep answers concise and direct.
- Do not include internal or system XML tags in your response."""


def _embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
    )


def load_text(path: str) -> str:
    """Load raw text from a .txt or .pdf source file."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking on paragraph/sentence
    boundaries where possible so chunks stay semantically coherent."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                # Paragraph itself is too long — split on sentences.
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= chunk_size:
                        current = f"{current} {sent}".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail} {chunks[i]}")
        return overlapped

    return chunks


def ingest_document(path: str, source_name: str | None = None) -> int:
    """Chunk a document and (re)store its embeddings in Chroma. Returns the
    number of chunks stored."""
    source_name = source_name or Path(path).name
    text = load_text(path)
    chunks = chunk_text(text)

    collection = get_collection()

    # Remove any prior chunks from this same source before re-ingesting.
    existing = collection.get(where={"source": source_name})
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    ids = [f"{source_name}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


# Filename -> friendly source name, for the curated Georgia Tech pages in
# data/. Used to (re)seed the vector store on a fresh deploy where local
# Chroma persistence doesn't carry over (e.g. a free-tier host with no
# persistent disk), without depending on someone having run ingest.py by hand
# on the server.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CURATED_SOURCES = {
    "gt_about_overview.txt": "Georgia Tech About / Overview",
    "gt_admissions_requirements.txt": "Georgia Tech First-Year Admission Requirements",
    "gt_application_deadlines.txt": "Georgia Tech Application Deadlines",
    "gt_tuition_cost_of_attendance.txt": "Georgia Tech Tuition and Cost of Attendance",
    "gt_scholarships_financial_aid.txt": "Georgia Tech Scholarships and Financial Aid",
    "gt_housing_first_year.txt": "Georgia Tech First-Year Housing",
    "gt_major_computer_science.txt": "Georgia Tech B.S. Computer Science",
    "gt_major_mechanical_engineering.txt": "Georgia Tech B.S. Mechanical Engineering",
    "gt_major_electrical_computer_engineering.txt": "Georgia Tech B.S. Electrical/Computer Engineering",
    "gt_major_industrial_systems_engineering.txt": "Georgia Tech B.S. Industrial and Systems Engineering",
    "gt_major_aerospace_engineering.txt": "Georgia Tech B.S. Aerospace Engineering",
    "gt_major_biomedical_engineering.txt": "Georgia Tech B.S. Biomedical Engineering",
    "gt_major_business_administration.txt": "Georgia Tech B.S. Business Administration (Scheller)",
}


def ingest_all_curated() -> int:
    """(Re)ingest every curated Georgia Tech source file in data/. Returns the
    total number of chunks stored. Safe to call repeatedly — ingest_document
    replaces a source's prior chunks rather than duplicating them."""
    total = 0
    for filename, source_name in CURATED_SOURCES.items():
        path = DATA_DIR / filename
        if path.exists():
            total += ingest_document(str(path), source_name=source_name)
    return total


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    hits = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "source": meta.get("source"), "distance": distance})
    return hits


def _anthropic_client() -> Anthropic:
    return Anthropic()  # picks up ANTHROPIC_API_KEY from the environment


def answer_question(question: str, k: int = TOP_K) -> dict:
    """Retrieve relevant chunks and generate a grounded answer via Claude."""
    hits = retrieve(question, k=k)

    if not hits:
        return {
            "answer": "I don't know — no source document has been ingested yet.",
            "sources": [],
        }

    context_block = "\n\n".join(
        f"[Excerpt {i + 1} — source: {h['source']}]\n{h['text']}"
        for i, h in enumerate(hits)
    )

    client = _anthropic_client()
    kwargs = dict(
        model=CLAUDE_MODEL,
        max_tokens=MAX_ANSWER_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context excerpts:\n\n{context_block}\n\nQuestion: {question}",
            }
        ],
    )
    if CLAUDE_MODEL.startswith(_THINKING_TOGGLEABLE_PREFIXES):
        kwargs["thinking"] = {"type": "disabled"}

    response = client.messages.create(**kwargs)

    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    return {
        "answer": answer_text,
        "sources": [
            {"source": h["source"], "excerpt": h["text"][:200], "distance": h["distance"]}
            for h in hits
        ],
    }
