# app/models.py
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
import uuid
from sqlalchemy.sql import func

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    pages = Column(Integer, default=0)
    chunks = Column(Integer, default=0)
    error = Column(Text, nullable=True)
