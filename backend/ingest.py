"""CLI: chunk + embed a university document into the local Chroma store.

Usage:
    python ingest.py path/to/document.pdf
    python ingest.py path/to/document.txt --source "Riverbend Handbook 2026"
"""

import argparse
import sys

from rag import ingest_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document into the RAG store.")
    parser.add_argument("path", help="Path to a .txt or .pdf source document")
    parser.add_argument(
        "--source", default=None, help="Friendly name for this source (defaults to filename)"
    )
    args = parser.parse_args()

    try:
        count = ingest_document(args.path, source_name=args.source)
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingested {count} chunks from {args.path}")


if __name__ == "__main__":
    main()
