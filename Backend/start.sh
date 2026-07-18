#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/uploads data/extracted_text data/chroma_db
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
