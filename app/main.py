# app/main.py
import os
import uuid
import logging
from typing import List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles
import asyncio

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter

from app.config import settings
from app.deepinfra import embed_batch, chat_completion
from app.qdrant_client import get_qdrant_client
from app.tasks import process_pdf_task
from app.db import init_models, get_async_session
import redis.asyncio as aioredis
import json
from app.models import Document

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Tender QA", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = settings.upload_dir
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Prometheus counters
uploads_total = Counter("tender_uploads_total", "Total uploads")
ingests_total = Counter("tender_ingests_total", "Total ingests started")
ingests_failed = Counter("tender_ingests_failed", "Ingest failures")
qa_requests_total = Counter("tender_qa_requests_total", "QA requests")
embed_errors_total = Counter("tender_embed_errors_total", "Embedding errors")

# Redis helper
_redis = None
def get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis

# DB init on startup
@app.on_event("startup")
async def startup():
    await init_models()
    # ensure minio bucket if configured (light touch)
    # any other startup tasks

class QARequest(BaseModel):
    q: str
    limit: int = 6

@app.post("/upload", status_code=202)
async def upload(file: UploadFile = File(...)):
    uploads_total.inc()
    filename = Path(file.filename or "uploaded").name
    ext = Path(filename).suffix.lower()
    if settings.allowed_extensions and ext not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file extension")

    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")

    max_size = settings.max_upload_size
    written = 0
    try:
        async with aiofiles.open(path, "wb") as out_file:
            while True:
                chunk = await file.read(1024 * 64)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_size:
                    await out_file.close()
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail="Payload too large")
                await out_file.write(chunk)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to write uploaded file")
        raise HTTPException(status_code=500, detail="Failed to save file")

    # Set Redis status & DB record
    r = get_redis()
    await r.hset(f"doc:{file_id}", mapping={"status": "queued", "filename": filename, "uploaded_bytes": written})
    # Optionally also persist to Postgres metadata
    try:
        async with get_async_session() as session:
            doc = Document(id=file_id, filename=filename, status="queued")
            session.add(doc)
            await session.commit()
    except Exception:
        # don't fail upload if DB write fails; just log
        logger.exception("Failed to write document metadata to DB")

    process_pdf_task.apply_async(args=[path, file_id], queue=settings.celery_queue)
    return {"doc_id": file_id, "filename": filename, "status": "processing"}

@app.get("/status/{doc_id}")
async def status(doc_id: str):
    r = get_redis()
    info = await r.hgetall(f"doc:{doc_id}")
    if not info:
        raise HTTPException(status_code=404, detail="doc_id not found")
    return info

@app.post("/qa/{doc_id}")
async def qa(doc_id: str, request: QARequest):
    qa_requests_total.inc()
    q = request.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    # Get client and embeddings (async)
    client = get_qdrant_client()
    try:
        qvecs = await embed_batch([q], batch_size=settings.embed_batch)
        qvec = qvecs[0]
    except Exception:
        embed_errors_total.inc()
        logger.exception("Embedding failed")
        raise HTTPException(status_code=502, detail="Embedding service error")

    try:
        results = client.search(collection_name=settings.collection, query_vector=qvec, limit=request.limit)
    except Exception:
        logger.exception("Qdrant search failed")
        raise HTTPException(status_code=502, detail="Vector DB error")

    # normalize results
    evidence = []
    for i, r in enumerate(results):
        payload = getattr(r, "payload", {}) or {}
        text = payload.get("text", "") or ""
        ev_text = text[:400].replace("\n", " ")
        evidence.append({
            "idx": i + 1,
            "page": payload.get("page"),
            "chunk_id": payload.get("chunk_id"),
            "text": ev_text,
            "score": getattr(r, "score", None),
        })

    # Limit evidence to top N and truncate to safe size for LLM
    MAX_EVIDENCE = 8
    MAX_SLICE_CHARS = 1200
    evidence = evidence[:MAX_EVIDENCE]
    user_prompt = "EVIDENCE:\n" + "\n".join(
        f"[{e['idx']}] (page {e['page']}) {e['text'][:MAX_SLICE_CHARS]}" for e in evidence
    ) + f"\nQUESTION: {q}"

    messages = [
        {"role": "system", "content": (
            "You are a procurement assistant. Use ONLY the provided evidence slices to answer. "
            "Each answer must include a short direct response, the evidence citations in format (page X, chunk Y), "
            "and a confidence score 0-100. If the answer cannot be found in the evidence, reply \"Not stated in document.\" "
            "Do not hallucinate."
        )},
        {"role": "user", "content": user_prompt}
    ]

    try:
        ans = await chat_completion(messages)
    except Exception:
        logger.exception("Chat completion failed")
        raise HTTPException(status_code=502, detail="LLM service error")

    return {"answer": ans, "evidence": evidence}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
