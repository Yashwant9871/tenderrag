# app/celery_app.py
from celery import Celery
from app.config import settings

celery_app = Celery("tender_ingest", broker=settings.redis_url, backend=settings.redis_url)
# Optional: put config here (prefetch, task_routes, etc.)
celery_app.conf.task_routes = {
    "app.tasks.process_pdf_task": {"queue": settings.celery_queue}
}
