# app/celery_app.py (update)
from celery import Celery
from app.config import settings

celery_app = Celery("tender_ingest", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_routes = {
    "app.tasks.process_pdf_task": {"queue": settings.celery_queue}
}

# Tuning
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_soft_time_limit=60*30,  # 30 minutes soft limit
)

# app/tasks.py
from app.celery_app import celery_app
import logging
from app.db import get_async_session
from app.models import Document
from app.ingest import process_pdf
from app.config import settings
from app.qdrant_client import get_qdrant_client
import redis
import traceback

logger = logging.getLogger(__name__)
_redis_sync = None
def get_redis_sync():
    global _redis_sync
    if _redis_sync is None:
        import redis as _r
        _redis_sync = _r.from_url(settings.redis_url, decode_responses=True)
    return _redis_sync

@celery_app.task(bind=True, name="app.tasks.process_pdf_task")
def process_pdf_task(self, path: str, doc_id: str):
    redis_sync = get_redis_sync()
    try:
        # mark processing
        redis_sync.hset(f"doc:{doc_id}", mapping={"status": "processing"})
        # run ingestion (blocking ok in worker)
        process_pdf(path, doc_id)
        # success
        redis_sync.hset(f"doc:{doc_id}", mapping={"status": "completed"})
        return {"status": "completed", "doc_id": doc_id}
    except Exception:
        tb = traceback.format_exc()
        logger.exception("process_pdf failed for %s", doc_id)
        try:
            redis_sync.hset(f"doc:{doc_id}", mapping={"status": "failed", "error": str(tb)})
        except Exception:
            logger.exception("Failed to set redis status for %s", doc_id)
        raise
