# app/qdrant_client.py
import threading
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models  # best-effort import if available
from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client = None

def _create_collection_safe(client: QdrantClient, name: str, vector_size: int, distance: str = "Cosine"):
    """
    Try to create the collection with the specified vector size.
    This will try multiple client APIs depending on qdrant-client version.
    """
    try:
        # 1) If client has get_collection and it doesn't raise, use it to check existence
        try:
            col = client.get_collection(collection_name=name)
            logger.info("Qdrant collection '%s' already exists.", name)
            return
        except Exception:
            # Not found or client doesn't support get_collection in this environment
            logger.info("Qdrant collection '%s' not found; attempting to create.", name)

        # 2) Try modern API: create_collection with rest_models.VectorParams (qdrant-client >= 1.4 style)
        try:
            vec_params = rest_models.VectorParams(size=vector_size, distance=rest_models.Distance(distance.upper()))
            client.create_collection(collection_name=name, vectors_config=vec_params)
            logger.info("Created qdrant collection '%s' with vector size %s via rest_models.VectorParams", name, vector_size)
            return
        except Exception as e:
            logger.debug("create_collection with rest_models failed: %s", e)

        # 3) Try simple dict-style call (some versions accept dicts)
        try:
            client.create_collection(collection_name=name, vectors={"size": vector_size, "distance": distance})
            logger.info("Created qdrant collection '%s' with vector size %s using dict-style call", name, vector_size)
            return
        except Exception as e:
            logger.debug("dict-style create_collection failed: %s", e)

        # 4) As last resort try recreate_collection if available
        try:
            client.recreate_collection(collection_name=name, vectors_config={"size": vector_size, "distance": distance})
            logger.info("Recreated qdrant collection '%s' with vector size %s", name, vector_size)
            return
        except Exception as e:
            logger.debug("recreate_collection failed: %s", e)

        # If we reach here, creation failed
        logger.warning("Unable to create qdrant collection '%s' automatically. Please create it manually with vector_size=%s.", name, vector_size)
    except Exception as outer:
        logger.exception("Unexpected error while ensuring qdrant collection: %s", outer)


def get_qdrant_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = QdrantClient(url=settings.qdrant_url, timeout=30)
                # Ensure collection exists or attempt to create it with explicit vector size
                try:
                    _create_collection_safe(_client, settings.collection, int(settings.deepinfra_vector_size), distance="Cosine")
                except Exception:
                    logger.exception("Error while ensuring qdrant collection existence.")
    return _client
