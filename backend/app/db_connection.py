"""SQLAlchemy エンジン / セッション / Base / get_db 依存関係。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.settings import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI 依存関係: リクエスト単位のDBセッション。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
