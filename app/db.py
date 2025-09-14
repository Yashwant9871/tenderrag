# app/db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models import Base
import asyncio
import logging

logger = logging.getLogger(__name__)

engine = create_async_engine(settings.database_url, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_models():
    # create tables if not exist (simple approach; for prod use migrations)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/checked")

# helper for startup
def get_async_session():
    return AsyncSessionLocal()


class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), nullable=False)   # user id who uploaded
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)           # local path or minio object name
    storage_backend = Column(String, nullable=False, default="local")  # "local" or "minio"
    status = Column(String, nullable=False, default="queued")
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    pages = Column(Integer, default=0)
    chunks = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    size = Column(Integer, default=0)  # bytes
    removed = Column(Boolean, default=False)