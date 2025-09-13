Table of contents

Features

Quickstart (dev)

Environment variables

Run locally (without Docker)

Usage examples

Tokenizers & chunking

Qdrant collection & vector size

Celery workers & tuning

Metrics & monitoring

Testing

Security & secrets

Development notes

Contributing & license

Features

Token-aware chunking with deterministic chunk IDs and char_start/char_end offsets for accurate citations.

Async embedding + chat client (DeepInfra) using httpx.

Qdrant vector store with explicit DEEPINFRA_VECTOR_SIZE environment variable to create the collection safely.

Celery worker to process ingestion asynchronously.

Redis for job/status, Postgres for document metadata.

Prometheus metrics (/metrics).

Configurable tokenizer via TOKENIZER_NAME (supports HF fast tokenizers and tiktoken encodings).

/upload endpoint for PDFs, /status/{doc_id} to check status, /qa/{doc_id} for retrieval-augmented QA.

Quickstart (dev)
Requirements

Python 3.10+

Docker & Docker Compose (recommended for quick local dev)

git

Start everything with Docker Compose

Drop a working .env at repo root (see next section). Then:

# build & start services
docker compose up -d --build

# watch logs for app
docker compose logs -f app


This will bring up:

app (FastAPI)

worker (Celery)

postgres (Postgres)

redis (Redis)

qdrant (Qdrant)

minio (optional object storage)

If you run into port conflicts, check docker-compose.yml.

Environment variables

Create .env (copy .env.example) and set required variables. Minimal important ones:

# DeepInfra
DEEPINFRA_BASE=https://api.deepinfra.com/v1/openai
DEEPINFRA_TOKEN=your_token_here
DEEPINFRA_VECTOR_SIZE=1536         # REQUIRED — embedding vector size (INT)

# Tokenizer
TOKENIZER_NAME=BAAI/bge-large-en-v1.5
# or for tiktoken/OpenAI-style:
# TOKENIZER_NAME=tiktoken:cl100k_base

# DB / Redis / Qdrant
DATABASE_URL=postgresql+asyncpg://tender:tenderpass@postgres:5432/tenderdb
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333
COLLECTION=tenders

# Celery
CELERY_QUEUE=ingest_queue

# Uploads and limits
UPLOAD_DIR=.data
MAX_UPLOAD_SIZE=52428800
TOKENIZER_NAME=gpt2

# MinIO (optional)
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=tender-uploads


DEEPINFRA_VECTOR_SIZE is mandatory. If you set the wrong value, Qdrant will reject embeddings or store them inconsistently. Don’t be that person.

Run locally (without Docker)

Install dependencies:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Start Postgres/Redis/Qdrant/MinIO locally (or use Docker), then:

# run the app
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# in another terminal start a celery worker
celery -A app.celery_app.celery_app worker --loglevel=info -Q ${CELERY_QUEUE:-ingest_queue}

Usage examples (curl)

Upload a PDF (returns doc_id):

curl -F "file=@/path/to/tender.pdf" http://localhost:8000/upload
# => {"doc_id":"...", "filename":"tender.pdf", "status":"processing"}


Check status:

curl http://localhost:8000/status/<doc_id>
# => {"status":"processing", ...}


Ask a question:

curl -X POST http://localhost:8000/qa/<doc_id> \
  -H "Content-Type: application/json" \
  -d '{"q":"What is the delivery timeline?", "limit":6}'
# => {"answer": "...", "evidence":[...]}


Prometheus metrics: http://localhost:8000/metrics.

Tokenizers & chunking

TOKENIZER_NAME configures tokenizer. Use a tokenizer matching your model:

HuggingFace: BAAI/bge-large-en-v1.5 or gpt2 (fast tokenizer required for offsets).

tiktoken: tiktoken:cl100k_base (install tiktoken and use as TOKENIZER_NAME=tiktoken:cl100k_base).

Token-aware chunking stores char_start and char_end in chunk payloads for precise provenance.

Defaults: chunk size ~800 tokens, overlap ~150 tokens (tune for your LLM context window).

If you need absolute parity with provider tokenization, set TOKENIZER_NAME accordingly.

Qdrant collection & vector size

DEEPINFRA_VECTOR_SIZE must equal the embedding vector length from your model. The app attempts to create the Qdrant collection at client init using this size. If creation fails, check app logs — it will warn and instruct manual creation.

To verify:

curl -sS http://localhost:6333/collections | jq
# check your collection and vectors.size

Celery workers & tuning

Celery is configured with safe defaults:

task_acks_late=True

worker_prefetch_multiplier=1

task_soft_time_limit (configurable)

Adjust Celery worker count to match CPU/IO and embedding throughput. Use separate worker queues for heavy tasks if desired.

Metrics & monitoring

/metrics exposes Prometheus counters: tender_uploads_total, tender_ingests_total, tender_ingests_failed, tender_qa_requests_total, tender_embed_errors_total.

Add Prometheus to scrape http://app:8000/metrics.

Testing

Unit test example for chunking included at tests/test_token_chunking.py. Run:

pip install -r dev-requirements.txt   # includes pytest, etc.
pytest -q


Add regression tests for QA pairs (create a small dataset and verify answers / citations). Track retrieval precision@k over time.

Security & secrets

Never commit .env or secret files. .gitignore already contains .env, .data/, etc.

If you accidentally push a secret: rotate it immediately (DeepInfra token, DB passwords, MinIO keys). Then use git filter-repo or BFG to purge history and force-push.

Use secret stores for production: Vault, GitHub Actions secrets, or your cloud provider’s secret manager.

Development notes & tips

The ingestion pipeline uses HF fast tokenizers for offsets. If tiktoken is configured but does not provide offsets, a fallback to HF gpt2 offsets occurs with a warning.

Keep DEEPINFRA_VECTOR_SIZE consistent with your embedding model.

To avoid large disk usage, enable MinIO and set ingestion to upload then delete local files after success.

For higher retrieval quality: add a reranker (cross-encoder), MMR for diversity, and token-based reranking.

For production, enable TLS (HTTPS), lock CORS_ORIGINS, and use migrations (Alembic) for DB schema.

Typical workflow

git checkout -b feat/whatever

Implement changes, add tests.

git add -A && git commit -m "feat: descriptive message"

Push branch and open PR; preserve CI that runs tests. Don’t push secrets. Rotate them if leaked.

If you need help producing a PR patch with the recent changes (tokenizer wiring, async DeepInfra, Postgres wiring), say so and I’ll make a tidy diff you can apply — begrudgingly helpful, as always.

Contributing

Fork → branch → PR.

Keep commits focused and tests green.

Document breaking changes in the PR body.

License

Suggested: MIT (change if legal demands it). Add LICENSE in repo root.

Contact & support

If something explodes:

Check app logs: docker compose logs -f app

Check worker logs: docker compose logs -f worker

Check Qdrant/Redis/Postgres logs in compose logs.

If you still can’t figure it out, open an issue with logs attached (do not attach .env or secrets).
