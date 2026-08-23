"""FastAPI backend exposing the RAG pipeline as a simple chat API."""

import os

from dotenv import load_dotenv

load_dotenv()  # must run before rag.py constructs the Anthropic client

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag import answer_question, get_collection, ingest_all_curated

# Caps requests per client IP to the paid /ask endpoint so a public demo link
# can't run up API costs. Override via env, e.g. "5/minute" or "50/day".
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/hour")

# Only this origin may call the API from a browser. Defaults to the deployed
# Streamlit app; override via env if the frontend URL changes (e.g. a custom
# domain), or set to "http://localhost:8501" for local-only testing.
FRONTEND_URL = os.environ.get(
    "FRONTEND_URL", "https://gautamirk-git-campusiq-frontendapp-ax4jfn.streamlit.app"
)

app = FastAPI(title="CampusIQ")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    excerpt: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.on_event("startup")
def seed_index_if_empty():
    # Hosts without persistent disk (e.g. a free-tier deploy) start with an
    # empty Chroma store on every boot — rebuild it from the curated GT
    # source files rather than requiring a manual ingest step on the server.
    if get_collection().count() == 0:
        ingest_all_curated()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(RATE_LIMIT)
def ask(req: AskRequest, request: Request):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        result = answer_question(question)
    except anthropic.APIStatusError as exc:
        # Pass through the Anthropic API's own status code and message
        # (e.g. 400 credit balance too low, 429 rate limited) instead of
        # flattening every failure into an opaque 500.
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:  # anything else really is an unexpected server error
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result
