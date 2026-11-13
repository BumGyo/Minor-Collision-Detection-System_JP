"""動画の保存期間 — アップロード後 VIDEO_RETENTION_DAYS (デフォルト90日) を経過した動画と
関連データ (事故クリップ・元クリップ・サムネイル・分析ログ・アップロード元ファイル) をすべて削除する。

Webサーバー (main.py) の起動時に1回、その後 CHECK_INTERVAL_SEC ごとに定期実行する。
分析が待機中・進行中の動画はスキップし、完了後の次回クリーンアップ時に削除する。
"""
from datetime import datetime, timedelta

from app import db_models
from app.db_connection import SessionLocal
from app.routers.analysis import ACTIVE_STATUSES
from app.routers.videos import purge_video
from app.settings import settings

CHECK_INTERVAL_SEC = 60 * 60   # 1時間ごと


def expired_videos(db):
    """保存期間を経過した動画 (アップロード日時基準)。保存期間設定が無効の場合は空リスト。"""
    if settings.VIDEO_RETENTION_DAYS <= 0:
        return []
    cutoff = datetime.utcnow() - timedelta(days=settings.VIDEO_RETENTION_DAYS)
    return (db.query(db_models.Video)
            .filter(db_models.Video.created_at < cutoff)
            .order_by(db_models.Video.id).all())


def delete_expired_videos() -> list[int]:
    """期限切れの動画を削除し、削除された動画IDのリストを返す。1件ずつcommit — 途中で失敗してもそれ以前の変更は保持。"""
    db = SessionLocal()
    deleted = []
    try:
        for video in expired_videos(db):
            if any(t.status in ACTIVE_STATUSES for t in video.analysis_tasks):
                continue   # 分析完了後の次回クリーンアップ時に処理
            video_id = video.id
            purge_video(video, db)
            db.commit()
            deleted.append(video_id)
    finally:
        db.close()
    return deleted
