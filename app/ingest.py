# app/ingest.py
"""
Ingestion pipeline:
 - parse a PDF (or fallback to plain text if you extend this)
 - chunk text with overlap (sliding window)
 - embed chunks using deepinfra.embed_batch
 - upsert points into Qdrant in batches

Design goals:
 - keep worker imports lazy (this file is safe to import from Celery workers)
 - robust to transient network errors (retries with backoff)
 - predictable chunking so evidence citations map easily back to pages/chunk ids
 - minimal external dependencies (pypdf, requests/qdrant-client already in requirements)
"""
import uuid

from typing import List, Dict, Any, Iterable, Tuple
import os
import math
import time
import logging
import hashlib
from dataclasses import dataclass, asdict
# add transformers tokenizer
from transformers import AutoTokenizer
from typing import Dict, Optional, List, Tuple, Any
import asyncio

from app.config import settings
from app.deepinfra import embed_batch
from app.qdrant_client import get_qdrant_client

# Local imports (lazy if used inside a task)
try:
    from pypdf import PdfReader
except Exception:
    # If pypdf not installed, user will see ImportError when worker runs the task — that's fine.
    raise


# Optional import for tiktoken support
_TIKTOKEN_AVAILABLE = True
try:
    import tiktoken
except Exception:
    _TIKTOKEN_AVAILABLE = False
    # we'll warn if user sets tiktoken but it's not installed

logger = logging.getLogger(__name__)
_tokenizer_cache: Dict[str, AutoTokenizer] = {}

class HFTokenizerWrapper:
    """
    Wraps a HF fast tokenizer to provide an interface with offsets.
    """
    def __init__(self, hf_tokenizer):
        self.tok = hf_tokenizer

    def encode_with_offsets(self, text: str):
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return {"input_ids": enc["input_ids"], "offsets": enc["offset_mapping"]}
    

class TiktokenWrapper:
    """
    Wraps a tiktoken encoding to give offsets approximately.
    tiktoken doesn't provide offsets directly; but we can obtain tokenization
    and then re-run a simple mapping. This is best-effort.
    """
    def __init__(self, encoding_name: str):
        if not _TIKTOKEN_AVAILABLE:
            raise RuntimeError("tiktoken not installed; cannot use tiktoken tokenizer")
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode_with_offsets(self, text: str):
        # tiktoken gives token ids but not offsets; we approximate offsets by progressively matching tokens
        token_bytes = [self.enc.encode_ordinary(tok) if isinstance(tok, str) else self.enc.encode(tok) for tok in []]  # unused
        toks = self.enc.encode(text, disallowed_special=())
        # fallback: we cannot get exact offsets easily — but we can return token ids and empty offsets (code will handle)
        # Better approach: prefer HF tokenizer offsets for char-accurate mapping. Use tiktoken only if user doesn't need exact offsets.
        return {"input_ids": toks, "offsets": []}

def get_tokenizer_wrapper(tokenizer_name: str):
    """
    Supports:
      - 'tiktoken:<encoding>' -> uses tiktoken
      - '<hf-model-name>' -> AutoTokenizer.from_pretrained(..., use_fast=True)
    """
    tn = tokenizer_name.strip()
    if tn in _tokenizer_cache:
        return _tokenizer_cache[tn]

    if tn.startswith("tiktoken:"):
        enc_name = tn.split(":", 1)[1]
        if not _TIKTOKEN_AVAILABLE:
            logger.warning("TOKENIZER_NAME requested tiktoken:%s but tiktoken is not installed. Falling back to gpt2.", enc_name)
            wrapper = get_tokenizer_wrapper("gpt2")
            _tokenizer_cache[tn] = wrapper
            return wrapper
        try:
            wrapper = TiktokenWrapper(enc_name)
            _tokenizer_cache[tn] = wrapper
            return wrapper
        except Exception as e:
            logger.exception("Failed to init tiktoken encoding %s: %s", enc_name, e)
            return get_tokenizer_wrapper("gpt2")

    # Otherwise treat as HF tokenizer name
    try:
        hf_tok = AutoTokenizer.from_pretrained(tn, use_fast=True)
        wrapper = HFTokenizerWrapper(hf_tok)
        _tokenizer_cache[tn] = wrapper
        return wrapper
    except Exception as e:
        logger.warning("Failed to load HF tokenizer '%s': %s. Falling back to gpt2.", tn, e)
        if tn != "gpt2":
            return get_tokenizer_wrapper("gpt2")
        raise

def get_tokenizer(name: str = "gpt2"):
    """
    Returns a HF fast tokenizer with offset mapping enabled.
    Default is 'gpt2' which has a fast tokenizer with offsets.
    Choose a tokenizer appropriate to your model family if you care about exact tokenization parity.
    """
    global _tokenizer_cache
    if name not in _tokenizer_cache:
        # use_fast=True to ensure offsets are available
        _tokenizer_cache[name] = AutoTokenizer.from_pretrained(name, use_fast=True)
    return _tokenizer_cache[name]
# Tunable params (you can move these to settings later)
MAX_CHUNK_TOKENS = 800    # approximate size target per      in tokens (we'll treat char ~ token approx)
CHUNK_OVERLAP = 150       # overlap in characters (not perfect but works)
CHARS_PER_TOKEN = 4       # very rough approximation for chunking by characters
DEFAULT_CHUNK_SIZE = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
DEFAULT_OVERLAP = CHUNK_OVERLAP

# Qdrant upsert batch size (keep reasonable)
QDRANT_UPSERT_BATCH = max(32, settings.upsert_batch or 128)
EMBED_BATCH = max(1, settings.embed_batch or 64)

# Helpers / small dataclass to keep track of chunks
@dataclass
class Chunk:
    doc_id: str
    page: int
    text: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    tokens: Optional[int] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None   # added

    def id(self) -> str:
        """
        Return a deterministic UUID string for this chunk.
        If chunk_id (explicit) is set and is a valid UUID, return it.
        Otherwise create a uuid5 using the doc UUID (if valid) as namespace
        and a name composed of page+index+text-hash to remain deterministic.
        """
        # 1) if explicit chunk_id and valid UUID, use it
        if self.chunk_id:
            try:
                _ = uuid.UUID(str(self.chunk_id))
                return str(self.chunk_id)
            except Exception:
                # fall through to generate deterministic uuid
                pass

        # 2) compute stable name
        # prefer chunk_index if present (keeps name short), otherwise fallback to text hash
        name_parts = []
        if self.page is not None:
            name_parts.append(str(self.page))
        if self.chunk_index is not None:
            name_parts.append(str(self.chunk_index))
        # short stable hash of text so repeated changes to other metadata won't change id unnecessarily
        text_hash = hashlib.sha1((self.text or "").encode("utf-8")).hexdigest()[:12]
        name_parts.append(text_hash)
        name = "-".join(name_parts)

        # if doc_id is a valid UUID, use it as namespace to generate deterministic UUIDv5
        try:
            namespace = uuid.UUID(self.doc_id)
        except Exception:
            namespace = uuid.NAMESPACE_DNS

        generated = uuid.uuid5(namespace, name)
        return str(generated)

    def payload(self) -> Dict[str, Any]:
        base = {
            "doc_id": self.doc_id,
            "page": self.page,
            "chunk_id": self.id(),
            "text": self.text,
        }
        if self.char_start is not None:
            base["char_start"] = int(self.char_start)
        if self.char_end is not None:
            base["char_end"] = int(self.char_end)
        if self.chunk_index is not None:
            base["chunk_index"] = int(self.chunk_index)
        return base
    
def chunk_page_text_tokenized(
    page_text: str,
    page_num: int,
    doc_id: str,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    tokenizer=None,
    tokenizer_name: Optional[str] = None,
) -> List[Chunk]:
    """
    Produce token-aware chunks for a single page.
    Always returns a list of Chunk objects (no dicts).
    """
    text = (page_text or "").strip()
    if not text:
        return []

    # resolve tokenizer wrapper if name provided
    wrapper = None
    if tokenizer is None and tokenizer_name:
        try:
            wrapper = get_tokenizer_wrapper(tokenizer_name)
        except Exception as e:
            logger.warning("Failed to get tokenizer wrapper '%s': %s. Falling back to char-split.", tokenizer_name, e)
            wrapper = None
    elif tokenizer is not None:
        wrapper = tokenizer

    chunks: List[Chunk] = []

    # Try tokenization offsets via wrapper API
    token_ids = None
    token_offsets = None
    try:
        if wrapper is not None:
            enc = wrapper.encode_with_offsets(text)
            token_ids = enc.get("input_ids") or enc.get("ids")
            token_offsets = enc.get("offsets") or enc.get("offset_mapping") or enc.get("offsets", [])
            # Ensure token_offsets is list or None
            if token_offsets == []:
                token_offsets = None
    except Exception as e:
        logger.debug("Tokenizer wrapper failed on page %s of %s: %s. Falling back to char-split.", page_num, doc_id, e)
        token_ids = None
        token_offsets = None

    # Token-offset based chunking (preferred if we have offsets)
    if token_ids and token_offsets:
        start_token = 0
        total_tokens = len(token_ids)
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be > 0")
        step = max(chunk_tokens - overlap_tokens, 1)
        chunk_index = 0
        while start_token < total_tokens:
            end_token = min(start_token + chunk_tokens, total_tokens)
            char_start = token_offsets[start_token][0] if start_token < len(token_offsets) else 0
            char_end = token_offsets[end_token - 1][1] if (end_token - 1) < len(token_offsets) else len(text)

            chunk_text = text[char_start:char_end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    doc_id=doc_id,
                    page=page_num,
                    text=chunk_text,
                    char_start=int(char_start),
                    char_end=int(char_end),
                    tokens=(end_token - start_token),
                    chunk_id=None,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

            if end_token == total_tokens:
                break
            start_token += step

        return chunks

    # Fallback: character-based chunking approximating token sizes
    est_chars_per_token = CHARS_PER_TOKEN
    chunk_chars = max(100, chunk_tokens * est_chars_per_token)
    overlap_chars = int(overlap_tokens * est_chars_per_token)
    step_chars = max(1, chunk_chars - overlap_chars)

    start = 0
    text_len = len(text)
    index = 0
    while start < text_len:
        end = min(start + chunk_chars, text_len)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                doc_id=doc_id,
                page=page_num,
                text=chunk_text,
                char_start=int(start),
                char_end=int(end),
                tokens=None,
                chunk_id=None,
                chunk_index=index,
            ))
            index += 1
        if end == text_len:
            break
        start += step_chars

    return chunks
def chunk_document_tokenized(
        pages: List[Tuple[int, str]],
        doc_id: str,
        chunk_tokens: int = 800,
        overlap_tokens: int = 150,
        tokenizer_name: str = "gpt2"
    ) -> List[Chunk]:
    """
    Chunk all pages token-aware and return a flat list of Chunk objects.
    pages: List of (page_num, page_text)
    """
    all_chunks: List[Chunk] = []
    for page_num, page_text in pages:
        page_chunks = chunk_page_text_tokenized(
            page_text=page_text,
            page_num=page_num,
            doc_id=doc_id,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer_name=tokenizer_name
        )
        all_chunks.extend(page_chunks)
    return all_chunks

def ensure_collection_with_vector_size(client, collection, vector_size):
    try:
        # may raise if collection not exist
        client.get_collection(collection_name=collection)
    except Exception:
        try:
            client.recreate_collection(collection_name=collection, vectors_config={"size": vector_size, "distance": "Cosine"})
            logger.info("Created qdrant collection %s with vector size %s", collection, vector_size)
        except Exception as e:
            logger.warning("Could not create collection automatically: %s", e)

def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    """
    Extract text by page from a PDF. Returns list of tuples (page_number (1-based), text).
    Keeps pages empty-string if extraction fails for that page to preserve page numbering.
    """
    reader = PdfReader(path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Failed to extract text from page %s of %s: %s", i + 1, path, e)
            text = ""
        pages_text.append((i + 1, text))
    return pages_text


def chunk_page_text(page_text: str, doc_id: str, page_num: int,
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    overlap: int = DEFAULT_OVERLAP) -> List[Chunk]:
    """
    Chunk a single page's text using a sliding window over characters.
    Returns list of Chunk objects for the page.
    """
    if not page_text:
        return []

    text = page_text.strip()
    if len(text) <= chunk_size:
        return [Chunk(doc_id=doc_id, page=page_num, chunk_index=0, text=text)]

    chunks: List[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(doc_id=doc_id, page=page_num, chunk_index=index, text=chunk_text))
        index += 1
        start += step
    return chunks


def chunk_document(pages: List[Tuple[int, str]], doc_id: str,
                   chunk_size: int = DEFAULT_CHUNK_SIZE,
                   overlap: int = DEFAULT_OVERLAP) -> List[Chunk]:
    """
    Chunk all pages and return a flat list of chunks.
    """
    all_chunks: List[Chunk] = []
    for page_num, page_text in pages:
        page_chunks = chunk_page_text(page_text, doc_id, page_num, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(page_chunks)
    return all_chunks


def batch_iterable(iterable: Iterable, batch_size: int):
    """
    Yield lists of items of size up to batch_size from iterable.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _retry(fn, tries=3, initial_delay=1.0, backoff=2.0, exceptions=(Exception,), log_prefix=""):
    """
    Simple retry helper for network calls.
    """
    delay = initial_delay
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except exceptions as e:
            if attempt == tries:
                logger.exception("%sOperation failed after %s attempts: %s", log_prefix, tries, e)
                raise
            logger.warning("%sOperation attempt %s failed, retrying in %.1fs: %s", log_prefix, attempt, delay, e)
            time.sleep(delay)
            delay *= backoff


def upsert_chunks_to_qdrant(client, collection: str, chunks: List[Chunk], embeddings: List[List[float]]):
    """
    Upsert points into Qdrant. Expect chunks and embeddings lists to be same length and aligned.
    Uses payload structure with doc/page/chunk id and text.
    """
    assert len(chunks) == len(embeddings)
    points = []
    for chunk, emb in zip(chunks, embeddings):
        pid = chunk.id()
        payload = chunk.payload()
        points.append({"id": pid, "vector": emb, "payload": payload})

    # send in a few smaller batches to avoid timeouts; but caller already provides batches
    def _do_upsert():
        # The Qdrant client accepts 'collection_name' and 'points'
        client.upsert(collection_name=collection, points=points)

    _retry(_do_upsert, tries=4, initial_delay=1.0, backoff=2.0, log_prefix="[qdrant.upsert] ")


def process_pdf(path: str, doc_id: str, remove_after: bool = False):
    """
    Main entrypoint for the ingestion.
    - path: local path to the uploaded file
    - doc_id: canonical doc id (from upload API)
    - remove_after: if True, remove the file on success to save disk
    """
    logger.info("Started ingest for doc_id=%s path=%s", doc_id, path)
    client = get_qdrant_client()
    collection = settings.collection

    # 1) Extract pages
    pages = extract_text_from_pdf(path)
    if not pages:
        logger.warning("No pages extracted from %s", path)

    TOK_CHUNK_SIZE = 800
    TOK_OVERLAP = 150
    TOKENIZER_NAME = "gpt2"  # or another HF tokenizer
    # 2) Chunk document
    chunks = chunk_document_tokenized(pages, doc_id, chunk_tokens=TOK_CHUNK_SIZE, overlap_tokens=TOK_OVERLAP, tokenizer_name=TOKENIZER_NAME)

    logger.info("Document %s produced %d chunks", doc_id, len(chunks))
    if not chunks:
        # nothing to upsert; still mark success
        if remove_after:
            try:
                os.remove(path)
            except Exception:
                logger.debug("Failed to remove file after ingest: %s", path)
        return

    # 3) Batch embed and upsert
    total = len(chunks)
    processed = 0
    for chunk_batch in batch_iterable(chunks, EMBED_BATCH):
        texts = [c.text if hasattr(c, "text") else c.get("text") for c in chunk_batch]

        # embed_batch is async -> run it synchronously in this worker
        try:
            embeddings = asyncio.run(embed_batch(texts, batch_size=EMBED_BATCH))
        except Exception as e:
            # fallback if embed_batch signature doesn't accept batch_size param
            logger.debug("embed_batch call with batch_size failed: %s. Retrying without batch_size.", e)
            embeddings = asyncio.run(embed_batch(texts))

        if not hasattr(embeddings, "__len__"):
            raise TypeError(f"embed_batch returned non-sequence {type(embeddings)}. Did you forget to await?")

        if len(chunk_batch) != len(embeddings):
            raise ValueError(f"chunk/embedding mismatch: {len(chunk_batch)} vs {len(embeddings)}")

        upsert_chunks_to_qdrant(client, collection, chunk_batch, embeddings)

        processed += len(chunk_batch)
        logger.info("Upserted %d/%d chunks for doc_id=%s", processed, total, doc_id)

    logger.info("Completed ingest for doc_id=%s (chunks=%d)", doc_id, total)

    # optional cleanup
    if remove_after:
        try:
            os.remove(path)
            logger.debug("Removed source file %s after ingest", path)
        except Exception as e:
            logger.warning("Failed to remove source file %s: %s", path, e)
