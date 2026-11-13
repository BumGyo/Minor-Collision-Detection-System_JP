"""FastAPI エントリーポイント — CORS、ルーター登録、起動時テーブル作成、保存期間経過動画のクリーンアップ。"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db_connection import engine, Base
from app import db_models  # noqa: F401  (テーブル登録のためにインポート)
from app.settings import settings
from app.routers import auth, videos, analysis


async def _retention_loop():
    """保存期間 (VIDEO_RETENTION_DAYS) を経過した動画をサーバー起動時に1回、その後1時間ごとに削除する。"""
    from app.retention import CHECK_INTERVAL_SEC, delete_expired_videos
    while True:
        try:
            deleted = await asyncio.to_thread(delete_expired_videos)
            if deleted:
                print(f"[保存期間] {settings.VIDEO_RETENTION_DAYS}日を経過した動画 {len(deleted)}件を削除しました: {deleted}", flush=True)
        except Exception as exc:  # noqa: BLE001 — クリーンアップ失敗時もサーバーは継続稼働(次回再試行)
            print(f"[保存期間] クリーンアップ失敗 (次回再試行): {exc}", flush=True)
        await asyncio.sleep(CHECK_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(_retention_loop())
    yield
    task.cancel()


app = FastAPI(title="当て逃げ・軽衝突自動検知API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _wait_for_db(timeout_sec=60):
    """DB (MySQL) が接続可能になるまで待機する。

    DBコンテナの起動直後はMySQLが準備中であるため初回の接続が切断され、以前はその際に
    バックエンドが即時終了してUIのログインが502となっていた。即時終了せず短時間リトライを行う
    (どのようなOS・実行環境であっても共通)。
    """
    import time
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError as exc:
            if time.monotonic() > deadline:
                raise
            print(f"[backend] DB接続待機中 (最大 {timeout_sec}秒): {getattr(exc, 'orig', exc)}", flush=True)
            time.sleep(2)


# サーバー起動時にテーブルが存在しなければ自動作成 — DB準備完了を待機後
_wait_for_db()
Base.metadata.create_all(bind=engine)

def _ensure_column(table, column_ddl):
    """後から追加されたカラムを既存テーブルに追加 (create_all は既存テーブルを変更しないため)。
    既に存在する場合はMySQLがエラーを返すためスキップ。"""
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column_ddl};"))
            conn.commit()
        except Exception:
            pass


_ensure_column("videos", "detected_vehicles TEXT NULL")
_ensure_column("videos", "title VARCHAR(100) NULL")             # アップロード時に入力されたタイトル
_ensure_column("analysis_tasks", "finished_at DATETIME NULL")   # 分析履歴: 完了日時
_ensure_column("analysis_tasks", "event_count INT NULL")        # 分析履歴: 検出クリップ数
_ensure_column("crash_events", "clip_start_sec FLOAT NULL")     # CAMクリップが元動画の何秒から開始するか


def _drop_column(table, column):
    """不要となったカラムを既存テーブルから削除。
    既に存在しない場合はMySQLがエラーを返すためスキップ。"""
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column};"))
            conn.commit()
        except Exception:
            pass


# 録画開始時刻 — 未入力時は常にデフォルト値(20:30)だった。時間帯別統計と共に削除(2026-10-18)
_drop_column("videos", "recording_start_time")

app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(analysis.router)



@app.get("/")
def read_root():
    return {"message": "当て逃げ・軽衝突自動検知API 稼働中"}
