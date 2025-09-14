# app/models.py (update)
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
import uuid
from sqlalchemy.sql import func

Base = declarative_base()

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
