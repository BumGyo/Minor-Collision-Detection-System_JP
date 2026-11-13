"""Celery インスタンス — Redis ブローカー/バックエンド。

ワーカー実行: cd backend && celery -A app.worker worker --loglevel=info
(ワーカー環境には torch/opencv などの ML 依存ライブラリがインストールされている必要があります。)
"""
from celery import Celery
from celery.signals import worker_ready

from app.settings import settings

celery_app = Celery(
    "hitandrun",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

# タスク登録 (import 時にデコレーターが celery_app にバインド)
from app import prediction_job  # noqa: E402,F401


@worker_ready.connect
def _fail_interrupted_tasks(**_):
    """ワーカーの(再)起動時に、以前のワーカーで処理中(PROCESSING)だったタスクを失敗として処理する。

    処理中のメッセージは起動時点で既に確認(ack)されており再配送されないため、そのまま放置すると
    DB に PROCESSING が残り続け、画面側で無限待機状態となる。失敗ステータスに変更しておくことで、
    画面側に「再実行」ボタンを表示できるようにする。
    ⚠️ ワーカーが単一プロセスである前提。複数ワーカー運用の場合は他のワーカーが実際に処理中のタスクまで
       失敗化してしまうため、このクリーンアップ方式を見直す必要がある。
    """
    from app.db_connection import SessionLocal
    from app import db_models
    db = SessionLocal()
    try:
        n = (db.query(db_models.AnalysisTask)
             .filter(db_models.AnalysisTask.status == "PROCESSING")
             .update({"status": "FAILURE",
                      "error_message": "分析サーバーが再起動したため分析が中断されました。再実行してください。"},
                     synchronize_session=False))
        db.commit()
        if n:
            print(f"[worker] 再起動により中断された分析 {n}件を失敗として処理")
    finally:
        db.close()
