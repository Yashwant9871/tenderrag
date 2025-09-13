# app/deepinfra.py
import asyncio
import httpx
from typing import List, Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)

_default_client: Optional[httpx.AsyncClient] = None

def _get_client():
    global _default_client
    if _default_client is None:
        _default_client = httpx.AsyncClient(timeout=30.0)
    return _default_client

async def _post_with_retry(url, json, headers, timeout=30, retries=3, backoff=1.0):
    client = _get_client()
    delay = backoff
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = await client.post(url, json=json, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt == retries:
                logger.exception("Request failed after %s attempts to %s", retries, url)
                raise
            logger.warning("Request attempt %s failed, retrying in %.1fs: %s", attempt, delay, e)
            await asyncio.sleep(delay)
            delay *= 2
    raise last_exc

async def embed_batch(texts: List[str], model: str = "BAAI/bge-large-en-v1.5", timeout=30, batch_size: Optional[int] = None):
    url = f"{settings.deepinfra_base}/embeddings"
    headers = {"Authorization": f"Bearer {settings.deepinfra_token}", "Content-Type": "application/json"}
    batch_size = batch_size or max(1, settings.embed_batch or 64)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        payload = {"model": model, "input": batch}
        data = await _post_with_retry(url, payload, headers, timeout=timeout)
        for item in data.get("data", []):
            embeddings.append(item["embedding"])
    return embeddings

async def chat_completion(messages, model: str = "deepseek-ai/DeepSeek-V3.1", max_tokens: int = 600, timeout=60):
    url = f"{settings.deepinfra_base}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.deepinfra_token}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    resp = await _post_with_retry(url, payload, headers, timeout=timeout)
    choices = resp.get("choices")
    if not choices or not choices[0].get("message"):
        raise RuntimeError("Invalid LLM response")
    return choices[0]["message"]["content"]
