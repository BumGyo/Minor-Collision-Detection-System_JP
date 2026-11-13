"""SQLAlchemy エンジン / セッション / Base / get_db 依存関係。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.settings import settings


def normalize_database_url(url: str) -> str:
    """SQLAlchemyがSupabase PostgreSQL URLをpsycopgで接続できるように正規化する。"""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def create_database_engine(url: str):
    """Create an engine that does not keep unusable cloud DB connections."""
    database_url = normalize_database_url(url)
    connect_args = {"connect_timeout": 5}
    if database_url.startswith("postgresql+"):
        # A simple list query must not occupy a worker forever when the
        # managed PostgreSQL connection is unhealthy.
        connect_args["options"] = "-c statement_timeout=5000"
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_timeout=5,
        connect_args=connect_args,
    )


engine = create_database_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI 依存関係: リクエスト単位のDBセッション。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
