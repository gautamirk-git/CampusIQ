# CampusIQ

CampusIQ is an AI-powered assistant that helps parents, high-school students,
and university researchers ask natural-language questions and get concise,
grounded answers with source citations. Answers are grounded in ingested
official documents — if the answer isn't there, the assistant says so instead
of guessing.

This POC currently ships with placeholder sample data (a fictional
"Riverbend University") to prove the pipeline end-to-end; swapping in real
Georgia Tech content is the next step.

## Stack

- **Backend:** Python + FastAPI (`backend/main.py`)
- **Retrieval:** ChromaDB (local, persisted to `chroma_db/`) with local
  `sentence-transformers` embeddings (`all-MiniLM-L6-v2`) — no embedding API
  key needed
- **Generation:** Claude API (`claude-haiku-4-5` by default; see below)
- **Frontend:** Streamlit chat UI (`frontend/app.py`)

## Setup

1. Create a virtual environment and install dependencies:

   ```
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r backend\requirements.txt -r frontend\requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your Anthropic API key:

   ```
   copy .env.example .env
   ```

   Edit `.env` and set `ANTHROPIC_API_KEY`.

3. Ingest a source document. A sample document (`data/sample_university.txt`)
   is included so you can test immediately. Swap in real Georgia Tech source
   documents when ready:

   ```
   cd backend
   python ingest.py ..\data\sample_university.txt
   ```

   Re-run `ingest.py` any time you want to replace or add a document —
   re-ingesting a file with the same `--source` name replaces its old chunks.

4. Start the backend (from the `backend/` folder):

   ```
   uvicorn main:app --reload --port 8000
   ```

5. In a second terminal, start the frontend (from the `frontend/` folder):

   ```
   streamlit run app.py
   ```

   Streamlit opens at http://localhost:8501 and talks to the backend at
   http://localhost:8000.

## How it works

1. `backend/ingest.py` splits the source document into ~800-character
   overlapping chunks (breaking on paragraph/sentence boundaries) and stores
   each chunk's embedding in a local Chroma collection.
2. On each question, `backend/rag.py` embeds the question, retrieves the top
   3 most similar chunks, and sends them to Claude as context with a system
   prompt that instructs it to answer only from that context and say "I
   don't know" otherwise.
3. The FastAPI `/ask` endpoint wraps this and returns the answer plus the
   source excerpts used, which the Streamlit UI shows in a collapsible
   "Sources" panel — so you can verify every answer is actually grounded.

## Testing it

Try questions the sample document *does* cover:
- "What's the tuition and total cost of attendance?"
- "What are the admissions deadlines?"
- "What GPA do I need for the Trustee Scholarship?"

Then try questions it deliberately does *not* cover (the sample doc calls
these out explicitly):
- "What are the parking permit rules?"
- "Which schools does Riverbend have study-abroad partnerships with?"

The assistant should say it doesn't know for the second set, not guess.

## Next steps (Georgia Tech POC)

- Curate ~10-15 real, official Georgia Tech pages (majors, admissions,
  deadlines, tuition, housing, scholarships) and ingest them in place of the
  sample document.
- Add a `url` field to chunk metadata so citations link to real official GT
  pages instead of internal filenames.
- Add a topic-keyword → official GT URL fallback map so "I don't know"
  answers still point somewhere useful.
- Add short session-only conversation history (last 2-3 turns) so follow-up
  questions like "tell me more about Computer Science" work.
- Update the system prompt to return the structured Program/College/Overview
  /Source answer format.
- Add Georgia Tech branding (logo, colors, campus image) to the Streamlit UI.
- Currently defaults to `claude-haiku-4-5` for cheap dev/testing (~$0.002 per
  question). Switch `CLAUDE_MODEL` in `backend/rag.py` to `claude-opus-5` for
  production-quality answers if needed.
- Other cost knobs in `backend/rag.py`: `TOP_K` (chunks retrieved per
  question) and `MAX_ANSWER_TOKENS` (output cap).
- Deploy: the backend is a standard FastAPI app (deployable to Render,
  Fly.io, Railway, etc.); Streamlit apps deploy for free on Streamlit
  Community Cloud. Chroma's local persistence works for a single-instance
  deployment; move to a hosted vector DB only if multiple backend replicas
  are ever needed.
