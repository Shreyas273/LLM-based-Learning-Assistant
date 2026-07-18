#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/uploads data/extracted_text data/chroma_db
# Keep memory footprint low on free-tier hosts
export SKIP_LOCAL_EMBEDDINGS="${SKIP_LOCAL_EMBEDDINGS:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
