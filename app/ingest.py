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
    chunk_index: int
    text: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None


    def id(self) -> str:
        # stable deterministic id for chunk: docid-page-chunkindex-hash
        h = hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:10]
        return f"{self.doc_id}::p{self.page}::c{self.chunk_index}::{h}"

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
        return base
    

def chunk_page_text_tokenized(
        page_text: str,
        doc_id: str,
        page_num: int,
        chunk_tokens: int = 800,
        overlap_tokens: int = 150,
        tokenizer_name: Optional[str] = None
    ):
    if not page_text:
        return []
    tokenizer_name = tokenizer_name or settings.tokenizer_name
    tok_wrapper = get_tokenizer_wrapper(tokenizer_name)

    # Get tokenization with offsets
    enc = tok_wrapper.encode_with_offsets(page_text)
    input_ids = enc.get("input_ids", [])
    offsets = enc.get("offsets", [])

    # If offsets are empty (e.g., tiktoken fallback), try to also use HF gpt2 to get offsets
    if not offsets:
        if tokenizer_name.startswith("tiktoken:"):
            try:
                # try gpt2 for offsets as fallback, only for offset mapping
                fallback = get_tokenizer_wrapper("gpt2")
                enc2 = fallback.encode_with_offsets(page_text)
                offsets = enc2.get("offsets", [])
                logger.warning("Using gpt2 offsets as fallback for tiktoken encoding; offsets may not match tiktoken tokenization exactly.")
            except Exception:
                offsets = []

    n_tokens = len(input_ids)
    if n_tokens == 0 and not offsets:
        return []

    # If offsets exist, use them. If not, create coarse char-based chunks
    chunks = []
    if offsets:
        step = max(1, chunk_tokens - overlap_tokens)
        index = 0
        for start_token in range(0, n_tokens, step):
            end_token = min(start_token + chunk_tokens, n_tokens)
            char_start = offsets[start_token][0]
            char_end = offsets[end_token - 1][1]
            chunk_text = page_text[char_start:char_end].strip()
            if chunk_text:
                from dataclasses import dataclass
                import hashlib
                cid_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:10]
                chunk = {
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk_index": index,
                    "text": chunk_text,
                    "char_start": int(char_start),
                    "char_end": int(char_end),
                    "chunk_id": f"{doc_id}::p{page_num}::c{index}::{cid_hash}"
                }
                chunks.append(chunk)
            index += 1
            if end_token >= n_tokens:
                break
        return chunks

    # final fallback: char sliding window
    logger.warning("No token offsets available for tokenizer '%s'. Falling back to char-based chunking.", tokenizer_name)
    CHUNK_SIZE = chunk_tokens * 4
    OVERLAP = overlap_tokens * 4
    start = 0
    index = 0
    while start < len(page_text):
        end = start + CHUNK_SIZE
        chunk_text = page_text[start:end].strip()
        if chunk_text:
            import hashlib
            cid_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:10]
            chunks.append({
                "doc_id": doc_id,
                "page": page_num,
                "chunk_index": index,
                "text": chunk_text,
                "char_start": start,
                "char_end": min(len(page_text), end),
                "chunk_id": f"{doc_id}::p{page_num}::c{index}::{cid_hash}"
            })
        index += 1
        start += (CHUNK_SIZE - OVERLAP)
    return chunks

def chunk_document_tokenized(
        pages: List[Tuple[int, str]],
        doc_id: str,
        chunk_tokens: int = 800,
        overlap_tokens: int = 150,
        tokenizer_name: str = "gpt2"
    ) -> List[Chunk]:
    """
    Chunk all pages token-aware and return a flat list of chunks.
    """
    all_chunks: List[Chunk] = []
    for page_num, page_text in pages:
        page_chunks = chunk_page_text_tokenized(
            page_text, doc_id, page_num,
            chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
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
        texts = [c.text for c in chunk_batch]

        # embedding with retries
        def _embed_call():
            return embed_batch(texts, batch_size=EMBED_BATCH)

        embeddings = _retry(_embed_call, tries=3, initial_delay=1.0, backoff=2.0, log_prefix="[embed] ")

        # upsert to qdrant
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
