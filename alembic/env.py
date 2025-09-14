# alembic/env.py
from __future__ import with_statement
import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from sqlalchemy import create_engine
from alembic import context

# make sure project root is on sys.path so `app` imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# attempt to import your app settings and metadata
try:
    # adjust if your app package name differs; this matches your repo structure where app/ exists
    from app.config import settings as app_settings  # type: ignore
except Exception:
    app_settings = None

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import your SQLAlchemy models' Base metadata
# Adjust this import if your models are in a different module
try:
    from app.models import Base  # type: ignore
    target_metadata = Base.metadata
except Exception:
    target_metadata = None
    # If metadata cannot be imported, autogenerate will be limited.
    # Keep going anyway; user will see an error when generating migrations.

def _get_db_url_from_settings_or_env():
    """
    Try several common names on app.settings, then environment variables.
    Returns a URL string or None.
    """
    candidates = []

    # common attribute names used in different codebases
    if app_settings is not None:
        for attr in ("DATABASE_URL", "database_url", "SQLALCHEMY_DATABASE_URI", "database_uri", "DB_URL", "db_url"):
            try:
                val = getattr(app_settings, attr)
            except Exception:
                val = None
            if val:
                candidates.append(str(val))

    # environment variables fallback
    for env_name in ("DATABASE_URL", "DATABASE_URI", "SQLALCHEMY_DATABASE_URI", "DATABASE"):
        v = os.environ.get(env_name)
        if v:
            candidates.append(v)

    # return the first candidate or None
    return candidates[0] if candidates else None

def _to_sync_url(async_url: str) -> str:
    """
    Convert async driver scheme to sync driver for Alembic (e.g. asyncpg -> psycopg2).
    Only does simple replacement; adjust if you use something else.
    """
    if async_url is None:
        return None
    # common case: postgresql+asyncpg -> postgresql+psycopg2
    return async_url.replace("asyncpg", "psycopg2")

# choose DB URL
db_url = _get_db_url_from_settings_or_env()

if not db_url:
    raise RuntimeError(
        "No database URL found. Set env var DATABASE_URL or ensure app.config.settings exposes DATABASE_URL/database_url."
    )

# Convert async DB URL to sync DB URL for alembic ops
sync_db_url = _to_sync_url(db_url)

# Optionally print which URL was chosen (masked lightly)
print(f"[alembic/env.py] Using DB URL (masked): {sync_db_url[:40]}...")

# If alembic.ini has sqlalchemy.url, prefer it unless we have env override
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", sync_db_url)

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = create_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
