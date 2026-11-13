"""Celery インスタンス — Redis ブローカー/バックエンド。

ワーカー実行: cd backend && celery -A app.worker worker --loglevel=info
(ワーカー環境には torch/opencv などの ML 依存ライブラリがインストールされている必要があります。)
"""
import os
from datetime import datetime

from celery import Celery
from celery.signals import worker_ready

from app.settings import settings


def build_task_routes(gpu_queue: str) -> dict[str, dict[str, str]]:
    """ML事故分析タスクを指定キューにのみルーティングする。

    デフォルトキュー(``celery``)を使用すると既存CPUワーカーと完全互換。GPU VMを
    準備した後にのみ ``GPU_TASK_QUEUE=gpu`` を指定する。
    """
    queue = gpu_queue.strip() or "celery"
    return {
        "app.prediction_job.run_prediction_task": {"queue": queue},
        "app.vehicle_job.run_vehicle_detection_task": {"queue": queue},
    }

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
    task_routes=build_task_routes(os.getenv("GPU_TASK_QUEUE", "celery")),
)

# タスク登録 (import 時にデコレーターが celery_app にバインド)
from app import prediction_job, vehicle_job  # noqa: E402,F401


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
        detection_n = (db.query(db_models.VehicleDetectionTask)
                       .filter(db_models.VehicleDetectionTask.status == "PROCESSING")
                       .update({"status": "FAILURE",
                                "error_message": "検知サーバーの再起動によりタスクが中断されました。再度実行してください。",
                                "finished_at": datetime.utcnow()},
                               synchronize_session=False))
        db.commit()
        if detection_n:
            print(f"[worker] 再起動により中断された車両検知 {detection_n}件を失敗としてクリーンアップ")
    finally:
        db.close()
