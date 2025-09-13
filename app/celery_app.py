from celery import Celery
from app.config import settings

celery_app = Celery(
    "tender_ingest",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],   # <<-- ensure tasks module is imported on worker start
)

celery_app.conf.task_routes = {
    "app.tasks.process_pdf_task": {"queue": settings.celery_queue}
}

celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_soft_time_limit=60 * 30,
)
