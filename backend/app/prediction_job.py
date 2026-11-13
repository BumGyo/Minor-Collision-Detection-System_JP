"""Celery タスク — AI 推論 + 事故区間 CAM クリップ生成。

torch/opencv/モデルのインポートはすべて**タスク内部で遅延ロード**。
→ FastAPI Web プロセスは ML 依存関係なしでもこのモジュールをインポート (.delay 呼び出し) 可能。
"""
from datetime import datetime
from pathlib import Path

from app.worker import celery_app
from app.settings import settings
from app.db_connection import SessionLocal
from app import db_models

# ワーカープロセスごとにモデルを1回ロードして再利用
_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    import sys
    sys.path.insert(0, str(settings.MODEL_DIR))

    import torch
    import config as model_config
    from hitandrun_model import HitAndRun3DCNN
    from device_utils import get_device, is_channels_last_3d_supported

    device = get_device(model_config.INFER_DEVICE_TYPE)
    model = HitAndRun3DCNN(num_classes=model_config.MODEL_NUM_CLASSES).to(device)
    if is_channels_last_3d_supported(device) and model_config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last_3d)

    from hitandrun_model import check_weights
    check_weights(model_config.SERVICE_WEIGHTS_PATH)
    state_dict = torch.load(
        str(model_config.SERVICE_WEIGHTS_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[worker] モデルロード完了 — {getattr(model_config, 'MODEL_NAME', '?')} "
          f"(device={device}, weights={model_config.SERVICE_WEIGHTS_PATH.name})")

    try:
        from infer_runtime import get_runner
        get_runner(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] 推論実行方式の選定失敗 — PyTorch で継続: {exc}")
    _model = model
    return _model


@celery_app.task(bind=True)
def run_prediction_task(self, task_id: int):
    """AnalysisTask を受信して推論 → 事故区間クリップ生成 → CrashEvent 保存。"""
    db = SessionLocal()
    try:
        task = db.get(db_models.AnalysisTask, task_id)
        if task is None:
            return {"error": f"task {task_id} not found"}

        if task.status != "PENDING":
            return {"task_id": task_id, "skipped": task.status}

        task.status = "PROCESSING"
        db.commit()
        self.update_state(state="PROGRESS", meta={"percent": 0, "stage": "loading"})

        video = db.get(db_models.Video, task.video_id)
        model = _get_model()

        try:
            from inference import predict_events_and_clips
        except ImportError:
            from predict_cam import predict_events_and_clips

        import time as _time
        _last = {"pct": -1, "t": 0.0}

        def _on_progress(frac, eta=None, eta_min=False, stage=None, count=None, index=None):
            pct = round(min(99.0, frac * 100), 1)
            now = _time.time()
            key = (stage, count, index)
            if (key != _last.get("key")
                    or (pct != _last["pct"] and now - _last["t"] >= 0.4)
                    or now - _last["t"] >= 1.5):
                _last.update(pct=pct, t=now, key=key)
                self.update_state(state="PROGRESS", meta={
                    "percent": pct, "eta": None if eta is None else int(round(eta)),
                    "eta_min": bool(eta_min), "stage": stage, "count": count, "index": index})

        results = predict_events_and_clips(
            model,
            video_path=settings.abs_path(video.video_path),
            bbox=(task.bbox_xmin, task.bbox_ymin, task.bbox_xmax, task.bbox_ymax),
            output_dir=settings.CLIP_DIR,
            progress_callback=_on_progress,
            clip_name=f"{Path(video.video_path).stem}_t{task.id}",
        )

        db.refresh(task)
        if task.status == "CANCELLED":
            for r in results:
                for key in ("clip_path", "raw_clip_path"):
                    if r.get(key):
                        Path(r[key]).unlink(missing_ok=True)
            return {"task_id": task_id, "skipped": "CANCELLED"}

        for r in results:
            db.add(db_models.CrashEvent(
                task_id=task.id,
                video_id=video.id,
                timestamp_sec=r["start_sec"],
                frame_number=r["start_frame"],
                end_timestamp_sec=r["end_sec"],
                end_frame_number=r["end_frame"],
                crash_prob=r["crash_prob"],
                clip_path=settings.rel_path(r["clip_path"]),
                clip_start_sec=r.get("clip_start_sec"),
            ))

        task.status = "SUCCESS"
        task.finished_at = datetime.utcnow()
        task.event_count = len(results)
        db.commit()
        return {"task_id": task_id, "events": len(results)}

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        task = db.get(db_models.AnalysisTask, task_id)
        if task is not None:
            task.status = "FAILURE"
            task.error_message = str(exc)[:2000]
            task.finished_at = datetime.utcnow()
            db.commit()
        raise
    finally:
        db.close()
