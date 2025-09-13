# app/main.py
import os
import uuid
import logging
from typing import List, Dict, Any
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

import httpx
import numpy as np

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


# --------------- Reranker / MMR config (tune these) ----------------
DEEPINFRA_KEY = os.getenv("DEEPINFRA_TOKEN", "")
DEEPINFRA_URL = os.getenv("DEEPINFRA_URL", "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B")
INITIAL_K = 50          # how many to fetch from Qdrant initially
FINAL_K = 8             # how many to return to LLM
BATCH_SIZE = 8          # deepinfra batch size (tune to your infra)
MMR_LAMBDA = 0.6        # 0..1 (higher => favor relevance)
MAX_SLICE_CHARS = 1200
MAX_EVIDENCE = FINAL_K
# --------------------------------------------------------------------

if not DEEPINFRA_KEY:
    logger.warning("DEEPINFRA_KEY is not set; reranker calls will fail until set in env.")

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def _parse_deepinfra_output_item(item) -> float:
    if item is None:
        return 0.0
    if isinstance(item, dict):
        for f in ("score", "logit", "value", "relevance"):
            if f in item and item[f] is not None:
                try:
                    return float(item[f])
                except Exception:
                    pass
        txt = item.get("text") or item.get("output") or ""
        try:
            return float(txt.strip())
        except Exception:
            # Conservatively convert to a small positive score (not ideal but safe)
            return float(min(len(txt.split()) / 100.0, 1.0))
    try:
        return float(item)
    except Exception:
        return 0.0

async def _call_deepinfra_batch(pairs: List[Dict[str, str]]) -> List[float]:
    """
    Call DeepInfra Qwen3-Reranker-8B using the documented HTTP API:
      POST https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B
      Body: {"queries": [...], "documents": [...]}
    Returns: list of float scores (one per document).
    On error, returns [0.0]*len(pairs) so caller can gracefully continue.
    """
    if not pairs:
        return []

    if not DEEPINFRA_KEY:
        logger.warning("DEEPINFRA_KEY missing in _call_deepinfra_batch — returning zero scores")
        return [0.0] * len(pairs)

    # Build queries/documents arrays: repeat the single query for each passage
    queries = [p["question"] for p in pairs]         # could be same question repeated
    documents = [p["passage"] for p in pairs]

    headers = {
        # DeepInfra examples use lowercase 'bearer' but 'Bearer' is OK too; include the key.
        "Authorization": f"Bearer {DEEPINFRA_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"queries": queries, "documents": documents}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(DEEPINFRA_URL, headers=headers, json=payload)
        except Exception as e:
            logger.exception("Network error calling DeepInfra reranker: %s", e)
            return [0.0] * len(pairs)

    # log non-2xx (include body to help debug)
    if resp.status_code < 200 or resp.status_code >= 300:
        body_text = None
        try:
            body_text = resp.json()
        except Exception:
            body_text = resp.text
        logger.warning(
            "DeepInfra reranker returned status=%s body=%s",
            resp.status_code, repr(body_text)[:2000]
        )
        return [0.0] * len(pairs)

    # parse expected successful shape:
    # {
    #   "scores": [0.1, 0.2, ...],
    #   "input_tokens": ...,
    #   ...
    # }
    try:
        j = resp.json()
    except Exception:
        logger.exception("DeepInfra returned non-JSON response; returning zero scores")
        return [0.0] * len(pairs)

    # Primary: scores field
    scores = j.get("scores")
    if isinstance(scores, list) and len(scores) >= len(pairs):
        # take only as many as pairs
        out = []
        for s in scores[: len(pairs)]:
            try:
                out.append(float(s))
            except Exception:
                out.append(0.0)
        return out

    # Fallback: try to parse outputs/results etc.
    outputs = j.get("outputs") or j.get("results") or j.get("data")
    if isinstance(outputs, list) and len(outputs) >= len(pairs):
        out = []
        for o in outputs[: len(pairs)]:
            try:
                out.append(float(o))
            except Exception:
                out.append(_parse_deepinfra_output_item(o))
        return out

    # If nothing matched, log full response for debugging then return zeros
    logger.warning("DeepInfra reranker: unexpected response shape: %s", repr(j)[:2000])
    return [0.0] * len(pairs)

def mmr_select(question_emb: np.ndarray, candidate_embs: np.ndarray, candidates: List[Dict], top_k: int = FINAL_K, lambda_param: float = MMR_LAMBDA) -> List[Dict]:
    n = candidate_embs.shape[0]
    if n == 0:
        return []
    sims_q = [_cosine(question_emb, candidate_embs[i]) for i in range(n)]
    selected_idx = []
    rerank_scores = [c.get("rerank_score") for c in candidates]
    if any(s is not None for s in rerank_scores):
        seed = int(max(range(n), key=lambda i: (rerank_scores[i] if rerank_scores[i] is not None else -1e9)))
    else:
        seed = int(np.argmax(sims_q))
    selected_idx.append(seed)
    selected = [candidates[seed]]
    while len(selected) < min(top_k, n):
        mmr_scores = []
        for i in range(n):
            if i in selected_idx:
                mmr_scores.append(-1e9)
                continue
            relevance = sims_q[i]
            diversity = max(_cosine(candidate_embs[i], candidate_embs[j]) for j in selected_idx)
            mmr_scores.append(lambda_param * relevance - (1.0 - lambda_param) * diversity)
        next_i = int(np.argmax(mmr_scores))
        selected_idx.append(next_i)
        selected.append(candidates[next_i])
    return selected

# ---------- Modified /qa handler using reranker + MMR ----------
@app.post("/qa/{doc_id}")
async def qa(doc_id: str, request: QARequest):
    qa_requests_total.inc()
    q = request.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    client = get_qdrant_client()

    # 1) get question embedding (async)
    try:
        qvecs = await embed_batch([q], batch_size=settings.embed_batch)
        if not qvecs or not isinstance(qvecs, list):
            raise ValueError("embed_batch returned unexpected shape")
        qvec = qvecs[0]
    except Exception:
        embed_errors_total.inc()
        logger.exception("Embedding failed")
        raise HTTPException(status_code=502, detail="Embedding service error")

    # 2) initial Qdrant search (pull enough candidates)
    initial_k = max(request.limit or 0, INITIAL_K)
    try:
        # wrap in lambda to avoid unexpected named-arg issues in to_thread
        hits = await asyncio.to_thread(lambda: client.search(
            collection_name=settings.collection,
            query_vector=qvec,
            limit=initial_k,
            with_payload=True,
        ))
    except Exception:
        logger.exception("Qdrant search failed")
        raise HTTPException(status_code=502, detail="Vector DB error")

    # normalize basic candidates
    candidates: List[Dict[str, Any]] = []
    for h in hits:
        payload = getattr(h, "payload", {}) or {}
        txt = payload.get("text") or payload.get("content") or ""
        candidates.append({
            "id": getattr(h, "id", None),
            "text": txt,
            "payload": payload,
            "original_score": float(getattr(h, "score", 0.0) or 0.0),
        })

    # If no candidates, short-circuit
    if not candidates:
        return {"answer": "Not stated in document.", "evidence": []}

    # 3) rerank via DeepInfra (batched async). If DeepInfra fails, fallback to original ordering.
    try:
        pairs = [{"question": q, "passage": c["text"]} for c in candidates]
        rerank_scores: List[float] = []
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            batch_scores = await _call_deepinfra_batch(batch)
            rerank_scores.extend(batch_scores)
            # yield control to event loop
            await asyncio.sleep(0.0)
        # attach
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)
        # sort by rerank desc
        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception:
        logger.exception("Reranker failed; falling back to original Qdrant ordering")
        # leave candidates as-is (original qdrant order)

    # 4) Prepare for MMR selection: embed candidate texts in batches using embed_batch
    try:
        texts = [c["text"] for c in candidates]
        candidate_embs_list = await embed_batch(texts, batch_size=settings.embed_batch)
        candidate_embs = np.array(candidate_embs_list)
        q_emb_for_mmr = np.array(qvec)
        # run MMR select
        final_selected = mmr_select(q_emb_for_mmr, candidate_embs, candidates, top_k=FINAL_K, lambda_param=MMR_LAMBDA)
    except Exception:
        logger.exception("MMR/embedding of candidates failed; falling back to top-K by rerank/original")
        final_selected = candidates[:FINAL_K]

    # 5) normalize evidence objects for the LLM prompt (truncate, safe fields)
    evidence: List[Dict[str, Any]] = []
    for i, c in enumerate(final_selected):
        payload = c.get("payload", {}) or {}
        ev_text = (c.get("text") or "")[:400].replace("\n", " ")
        evidence.append({
            "idx": i + 1,
            "page": payload.get("page"),
            "chunk_id": payload.get("chunk_id"),
            "text": ev_text,
            "score": c.get("rerank_score") if c.get("rerank_score") is not None else c.get("original_score"),
        })

    # build user prompt for the LLM
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
