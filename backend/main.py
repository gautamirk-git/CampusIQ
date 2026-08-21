"""FastAPI backend exposing the RAG pipeline as a simple chat API."""

from dotenv import load_dotenv

load_dotenv()  # must run before rag.py constructs the Anthropic client

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import answer_question

app = FastAPI(title="CampusIQ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only — tighten before deploying publicly
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
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
