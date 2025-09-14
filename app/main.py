# app/main.py
import os
import uuid
import logging
from typing import List, Dict, Any,Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter

from app.config import settings
from app.deepinfra import embed_batch, chat_completion
from app.qdrant_client import get_qdrant_client
from app.tasks import process_pdf_task
from app.db import init_models, get_async_session
from app.db import init_models, close_engine

import redis.asyncio as aioredis
import json
from app.models import Document
import time
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
_httpx_client: Optional[httpx.AsyncClient] = None

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

@app.on_event("startup")
async def startup():
    global _httpx_client
    # init DB models first (if you need)
    await init_models()
    # create a reusable httpx client; if settings has timeout use it
    timeout = getattr(settings, "reranker_timeout", 20.0)
    # only set if not already set
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=timeout)
class QARequest(BaseModel):
    q: str
    limit: int = 6


# --------------- Reranker / MMR config (tune these) ----------------
DEEPINFRA_KEY = os.getenv("DEEPINFRA_TOKEN", "")
DEEPINFRA_URL = os.getenv("DEEPINFRA_URL", "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B")
INITIAL_K = getattr(settings, 'initial_k', 50)          # how many to fetch from Qdrant initially
FINAL_K = getattr(settings, 'final_k', 8)              # how many to return to LLM
BATCH_SIZE = getattr(settings, 'reranker_batch_size', 8)            # deepinfra batch size (tune to your infra)
MMR_LAMBDA = getattr(settings, 'mmr_lambda', 0.6)        # 0..1 (higher => favor relevance)
MAX_SLICE_CHARS = getattr(settings, 'max_slice_chars', 1200)
MAX_EVIDENCE = FINAL_K
PREFER_STORED_VECTORS = True
EMBED_FALLBACK_TOPK = 12
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
            return float(min(len(txt.split()) / 100.0, 1.0))
    try:
        return float(item)
    except Exception:
        return 0.0


async def _single_deepinfra_call(batch_pairs: List[Dict[str, str]]) -> List[float]:
    """
    Call DeepInfra for a single batch using global _httpx_client when available.
    If the global client isn't initialized, create a temporary client for this call
    (less efficient but safe). Returns zero-scores on failure.
    """
    global _httpx_client

    # Quick config guard
    if not DEEPINFRA_KEY:
        logger.debug("DEEPINFRA_KEY missing; skipping reranker for batch len=%d", len(batch_pairs))
        return [0.0] * len(batch_pairs)

    # Choose client: prefer global, otherwise create a local one
    client_to_use = _httpx_client
    created_local_client = False
    if client_to_use is None:
        created_local_client = True
        client_to_use = httpx.AsyncClient(timeout=getattr(settings, "reranker_timeout", 20.0))
        logger.debug("Global httpx client not initialized; using temporary client for this batch")

    headers = {
        "Authorization": f"Bearer {DEEPINFRA_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"queries": [p["question"] for p in batch_pairs], "documents": [p["passage"] for p in batch_pairs]}

    try:
        resp = await client_to_use.post(DEEPINFRA_URL, headers=headers, json=payload)
    except Exception as e:
        logger.exception("Network error while calling DeepInfra for batch len=%d: %s", len(batch_pairs), repr(e))
        if created_local_client:
            try: await client_to_use.aclose()
            except Exception: pass
        return [0.0] * len(batch_pairs)

    # read body defensively
    resp_text = "<unable to read>"
    try:
        resp_text = resp.text
    except Exception:
        pass

    if resp.status_code < 200 or resp.status_code >= 300:
        logger.error("DeepInfra non-2xx (status=%d) for batch len=%d. Body snippet: %.800s",
                     resp.status_code, len(batch_pairs), resp_text)
        if created_local_client:
            try: await client_to_use.aclose()
            except Exception: pass
        return [0.0] * len(batch_pairs)

    try:
        j = resp.json()
    except Exception as e:
        logger.error("DeepInfra returned non-JSON response for batch len=%d; error=%s; body snippet: %.800s",
                     len(batch_pairs), repr(e), resp_text)
        if created_local_client:
            try: await client_to_use.aclose()
            except Exception: pass
        return [0.0] * len(batch_pairs)

    # parse scores (same as before)
    scores = j.get("scores")
    if isinstance(scores, list) and len(scores) >= len(batch_pairs):
        out = []
        for s in scores[: len(batch_pairs)]:
            try:
                out.append(float(s))
            except Exception:
                out.append(0.0)
        if created_local_client:
            try: await client_to_use.aclose()
            except Exception: pass
        return out

    outputs = j.get("outputs") or j.get("results") or j.get("data")
    if isinstance(outputs, list) and len(outputs) >= len(batch_pairs):
        out = []
        for o in outputs[: len(batch_pairs)]:
            try:
                out.append(float(o))
            except Exception:
                out.append(_parse_deepinfra_output_item(o))
        if created_local_client:
            try: await client_to_use.aclose()
            except Exception: pass
        return out

    logger.warning("DeepInfra reranker: unexpected response shape for batch len=%d: %s", len(batch_pairs), repr(j)[:2000])
    logger.debug("Full DeepInfra body for unexpected shape: %.2000s", resp_text)
    if created_local_client:
        try: await client_to_use.aclose()
        except Exception: pass
    return [0.0] * len(batch_pairs)


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

@app.get("/healthz")
async def healthz():
    # quick checks: redis and qdrant client ping (non-blocking)
    r = get_redis()
    ok = {"redis": False, "qdrant": False}
    try:
        await r.ping()
        ok["redis"] = True
    except Exception:
        logger.exception("Redis ping failed")
    try:
        client = get_qdrant_client()
        # if your client has health/collections call it, else a no-op
        await asyncio.wait_for(asyncio.to_thread(lambda: client.get_collections()), timeout=2.0)
        ok["qdrant"] = True
    except Exception:
        logger.exception("Qdrant health check failed")
    status = 200 if all(ok.values()) else 503
    return JSONResponse(ok, status_code=status)

@app.on_event("shutdown")
async def shutdown():
    global _httpx_client, _redis
    try:
        if _httpx_client is not None:
            await _httpx_client.aclose()
    except Exception:
        logger.exception("Failed to close httpx client on shutdown")
    if _redis is not None:
        try:
            await _redis.close()
        except Exception:
            logger.exception("Failed to close redis on shutdown")

# ---------- Modified /qa handler using reranker + MMR (optimized) ----------
@app.post("/qa/{doc_id}")
async def qa(doc_id: str, body: QARequest, request: Request,session: AsyncSession = Depends(get_async_session),):
    qa_requests_total.inc()
    q = body.q.strip()
    doc = await session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="doc_id not found")
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")
    client = get_qdrant_client()

    t0 = time.perf_counter()
    if not await allow_request(f"qa:{request.client.host}", limit=20, period=60):
        raise HTTPException(status_code=429, detail="Too many requests")
    # 1) get question embedding (async)
    try:
        qvecs = await asyncio.to_thread(lambda: embed_batch([q], batch_size=settings.embed_batch))
        if not qvecs or not isinstance(qvecs, list):
            raise ValueError("embed_batch returned unexpected shape")
        qvec = qvecs[0]
    except Exception:
        embed_errors_total.inc()
        logger.exception("Embedding failed")
        raise HTTPException(status_code=502, detail="Embedding service error")
    t1 = time.perf_counter()

    # 2) initial Qdrant search (pull enough candidates)
    initial_k = min(max(request.limit, INITIAL_K), getattr(settings, "max_initial_k", 200))

    try:
        hits = await asyncio.to_thread(lambda: client.search(
            collection_name=settings.collection,
            query_vector=qvec,
            limit=initial_k,
            with_payload=True,
        ))
    except Exception:
        logger.exception("Qdrant search failed")
        raise HTTPException(status_code=502, detail="Vector DB error")
    t2 = time.perf_counter()

    # normalize basic candidates
    candidates: List[Dict[str, Any]] = []
    for h in hits:
        payload = getattr(h, "payload", {}) or {}
        txt = payload.get("text") or payload.get("content") or ""
        # Attempt to pull stored vector - depends on qdrant client and how you upserted
        stored_vec = None
        if hasattr(h, 'vector'):
            stored_vec = getattr(h, 'vector')
        elif payload.get('vector'):
            stored_vec = payload.get('vector')
        candidates.append({
            "id": getattr(h, "id", None),
            "text": txt,
            "payload": payload,
            "original_score": float(getattr(h, "score", 0.0) or 0.0),
            "stored_vector": np.array(stored_vec) if stored_vec is not None else None,
        })

    # If no candidates, short-circuit
    if not candidates:
        return {"answer": "Not stated in document.", "evidence": []}

    # 3) rerank via DeepInfra (batched async). Run batches in parallel to reduce wall time
    try:
        # build pairs
        pairs = [{"question": q, "passage": c["text"]} for c in candidates]
        tasks = []
        batch_sizes = []
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            batch_sizes.append(len(batch))
            tasks.append(_single_deepinfra_call(batch))

        # execute concurrently and inspect results carefully
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        rerank_scores: List[float] = []
        for idx, br in enumerate(batch_results):
            expected_len = batch_sizes[idx]
            if isinstance(br, Exception):
                # real exception object returned by gather
                import traceback
                tb = "".join(traceback.format_exception(type(br), br, br.__traceback__))
                logger.error("deepinfra batch %d raised exception (expected_len=%d): %s", idx, expected_len, repr(br))
                logger.debug("deepinfra batch %d traceback:\n%s", idx, tb)
                rerank_scores.extend([0.0] * expected_len)
                continue

            if br is None:
                logger.warning("deepinfra batch %d returned None; padding zeros (expected_len=%d)", idx, expected_len)
                rerank_scores.extend([0.0] * expected_len)
                continue

            # normal path: ensure it's a list-like of floats/nums
            if not isinstance(br, (list, tuple)):
                logger.warning("deepinfra batch %d returned unexpected type %s; padding zeros", idx, type(br))
                rerank_scores.extend([0.0] * expected_len)
                continue

            scores = list(br)
            if len(scores) < expected_len:
                scores.extend([0.0] * (expected_len - len(scores)))
            rerank_scores.extend(scores[:expected_len])

        # Trim/assign
        rerank_scores = rerank_scores[: len(candidates)]
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)
        # prefer rerank when present
        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception:
        logger.exception("Reranker failed; falling back to original Qdrant ordering")
        # leave candidates as-is (original qdrant order)
    t3 = time.perf_counter()

    # 4) Prepare candidate embeddings for MMR selection
    try:
        # Prefer stored vectors (fast). If not available, compute embeddings for only top-N candidates.
        stored_vecs = [v for v in (c.get("stored_vector") for c in candidates) if v is not None]
        if stored_vecs and all(getattr(v, "shape", None) == stored_vecs[0].shape for v in stored_vecs):
            candidate_embs = np.vstack(stored_vecs)
            # filter candidates accordingly (only those with stored_vector)
            filtered_candidates = [c for c in candidates if c.get("stored_vector") is not None]
            final_selected = mmr_select(q_emb_for_mmr, candidate_embs, filtered_candidates, top_k=FINAL_K, lambda_param=MMR_LAMBDA)
        else:
            # fallback: embed only top-EMBED_FALLBACK_TOPK candidates to save time
            to_embed_candidates = candidates[: EMBED_FALLBACK_TOPK]
            texts = [c["text"] for c in to_embed_candidates]
            candidate_embs_list = await embed_batch(texts, batch_size=settings.embed_batch)
            candidate_embs = np.array(candidate_embs_list)
            q_emb_for_mmr = np.array(qvec)
            final_subset = mmr_select(q_emb_for_mmr, candidate_embs, to_embed_candidates, top_k=min(FINAL_K, len(to_embed_candidates)), lambda_param=MMR_LAMBDA)
            # if we selected fewer than FINAL_K from embedded set, pad with next best original candidates
            selected_ids = {c["id"] for c in final_subset}
            final_selected = list(final_subset)
            for c in candidates:
                if len(final_selected) >= FINAL_K:
                    break
                if c["id"] not in selected_ids:
                    final_selected.append(c)
    except Exception:
        logger.exception("MMR/embedding of candidates failed; falling back to top-K by rerank/original")
        final_selected = candidates[:FINAL_K]
    t4 = time.perf_counter()

    # 5) normalize evidence objects for the LLM prompt (truncate, safe fields)
    evidence: List[Dict[str, Any]] = []
    for i, c in enumerate(final_selected):
        payload = c.get("payload", {}) or {}
        TRUNCATE_CHARS = min(400, MAX_SLICE_CHARS)
        ev_text = (c.get("text") or "")[:TRUNCATE_CHARS].replace("\n"," ")
        evidence.append({
            "idx": i + 1,
            "page": payload.get("page"),
            "chunk_id": payload.get("chunk_id"),
            "text": ev_text,
            "score": c.get("rerank_score") if c.get("rerank_score") is not None else c.get("original_score"),
        })

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
    t5 = time.perf_counter()

    logger.info("timings embed_q=%.3fs qdrant=%.3fs rerank=%.3fs embed_cands=%.3fs llm=%.3fs total=%.3fs",
                t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t5-t0)

    return {"answer": ans, "evidence": evidence}


async def allow_request(key: str, limit: int, period: int = 60) -> bool:
    r = get_redis()
    now = int(time.time())
    bucket_key = f"rate:{key}:{now // period}"
    val = await r.incr(bucket_key)
    if val == 1:
        await r.expire(bucket_key, period + 1)
    return val <= limit

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
