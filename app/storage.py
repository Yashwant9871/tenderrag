# app/storage.py
import os
from app.config import settings
from uuid import uuid4

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_file_local(upload_file, dest_dir=None):
    dest_dir = dest_dir or settings.upload_dir
    ensure_dir(dest_dir)
    filename = upload_file.filename
    # sanitize filename if you want (strip path parts, check chars)
    uid = str(uuid4())
    saved_name = f"{uid}__{filename}"
    path = os.path.join(dest_dir, saved_name)
    with open(path, "wb") as f:
        while True:
            chunk = upload_file.file.read(1024*64)
            if not chunk:
                break
            f.write(chunk)
    size = os.path.getsize(path)
    return path, filename, size

# Optional MinIO (boto3-compatible via minio-py or boto3)
from minio import Minio
def save_file_minio(upload_file):
    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=False,  # or True depending on setup
    )
    bucket = "uploads"
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    object_name = f"{uuid4()}__{upload_file.filename}"
    # upload_file.file is a file-like — use put_object with length if known
    upload_file.file.seek(0)
    data = upload_file.file.read()
    client.put_object(bucket, object_name, io.BytesIO(data), length=len(data), content_type="application/octet-stream")
    return f"{bucket}/{object_name}", upload_file.filename, len(data)
