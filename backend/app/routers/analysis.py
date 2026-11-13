"""分析リクエスト / タスクステータス / イベント・クリップ取得ルーター。"""
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import get_current_user, get_media_user
from app.settings import settings
from app.routers.videos import to_event_out, _get_owned_video, raw_clip_key
from app.object_storage import LocalObjectStorage, get_storage

router = APIRouter(prefix="/api", tags=["analysis"])


# 待機・実行中とみなすステータス
ACTIVE_STATUSES = ("PENDING", "PROCESSING")


def _get_owned_task(task_id: int, db: Session, user: db_models.User):
    """他ユーザーのタスクは参照・キャンセル不可 (404 Not Found)。"""
    task = db.get(db_models.AnalysisTask, task_id)
    if task is None or task.video is None or task.video.user_id != user.id:
        raise HTTPException(status_code=404, detail="タスクが見つかりません。")
    return task


def _active_task_for_video(video_id: int, db: Session):
    return (db.query(db_models.AnalysisTask)
            .filter(db_models.AnalysisTask.video_id == video_id,
                    db_models.AnalysisTask.status.in_(ACTIVE_STATUSES))
            .order_by(db_models.AnalysisTask.id).first())


def _progress(task: db_models.AnalysisTask) -> dict:
    """処理中の場合、Celeryの詳細進捗状態を取得する。"""
    empty = {"progress": None, "eta_sec": None, "eta_is_min": False,
             "stage": None, "stage_count": None, "stage_index": None}
    if task.status != "PROCESSING" or not task.celery_task_id:
        return empty
    from app.worker import celery_app
    res = celery_app.AsyncResult(task.celery_task_id)
    if res.state == "PROGRESS" and isinstance(res.info, dict):
        info = res.info
        return {"progress": info.get("percent"), "eta_sec": info.get("eta"),
                "eta_is_min": bool(info.get("eta_min")), "stage": info.get("stage"),
                "stage_count": info.get("count"), "stage_index": info.get("index")}
    return empty


def _enqueue(video_id: int, bbox: dict, db: Session) -> api_schemas.AnalyzeResponse:
    """分析タスクを作成しCeleryキューに登録する。同一動画に対して待機・実行中の分析が
    存在する場合は新規作成せず既存タスクを返却する(同一動画の重複分析防止 —
    ワーカーが単一のため重複リクエストは先行タスクの完了まで待機するのみとなる)。"""
    existing = _active_task_for_video(video_id, db)
    if existing is not None:
        return api_schemas.AnalyzeResponse(
            task_id=existing.id, celery_task_id=existing.celery_task_id,
            status=existing.status, already_running=True)

    task = db_models.AnalysisTask(video_id=video_id, status="PENDING", **bbox)
    db.add(task)
    db.commit()
    db.refresh(task)

    # Celeryキューに投入 (torch未インストールのWebプロセスでも .delay は動作可能)
    from app.prediction_job import run_prediction_task
    async_result = run_prediction_task.delay(task.id)
    task.celery_task_id = async_result.id
    db.commit()

    return api_schemas.AnalyzeResponse(
        task_id=task.id, celery_task_id=task.celery_task_id, status=task.status)


@router.post("/videos/{video_id}/analyze",
             response_model=api_schemas.AnalyzeResponse)
def analyze_video(
    video_id: int,
    req: api_schemas.AnalyzeRequest,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    video = _get_owned_video(video_id, db, user)
    return _enqueue(video.id, req.model_dump(), db)


# ⚠️ /tasks/{task_id} より前に登録必須('active' が task_id として捕捉されるのを防止)
@router.get("/tasks/active", response_model=list[api_schemas.ActiveTaskOut])
def list_active_tasks(
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """ログインユーザーの待機・実行中の分析タスク一覧。画面リロード時の進捗復元に使用。"""
    tasks = (db.query(db_models.AnalysisTask)
             .join(db_models.Video, db_models.AnalysisTask.video_id == db_models.Video.id)
             .filter(db_models.Video.user_id == user.id,
                     db_models.AnalysisTask.status.in_(ACTIVE_STATUSES))
             .order_by(db_models.AnalysisTask.id).all())
    out = []
    for t in tasks:
        out.append(api_schemas.ActiveTaskOut(
            task_id=t.id, video_id=t.video_id, video_name=t.video.video_name,
            recording_date=t.video.recording_date,
            camera_location=t.video.camera_location or "駐車場",
            status=t.status, created_at=t.created_at, **_progress(t)))
    return out


@router.get("/tasks/{task_id}", response_model=api_schemas.TaskStatusOut)
def get_task_status(
    task_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    task = _get_owned_task(task_id, db, user)
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.task_id == task.id).all()
    return api_schemas.TaskStatusOut(
        task_id=task.id,
        status=task.status,
        **_progress(task),
        error_message=task.error_message,
        events=[to_event_out(e) for e in events],
    )


@router.post("/tasks/{task_id}/cancel", response_model=api_schemas.TaskStatusOut)
def cancel_task(
    task_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    task = _get_owned_task(task_id, db, user)
    if task.status in ACTIVE_STATUSES:
        task.status = "CANCELLED"
        task.error_message = "ユーザーによりタスクがキャンセルされました"
        task.finished_at = datetime.utcnow()
        if task.celery_task_id:
            try:
                from app.worker import celery_app
                celery_app.control.revoke(task.celery_task_id, terminate=True)
            except Exception:
                pass
        db.commit()
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.task_id == task.id).all()
    return api_schemas.TaskStatusOut(
        task_id=task.id,
        status=task.status,
        progress=None,
        error_message=task.error_message,
        events=[to_event_out(e) for e in events],
    )


@router.post("/tasks/{task_id}/retry", response_model=api_schemas.AnalyzeResponse)
def retry_task(
    task_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """失敗・キャンセルされた分析を同一動画・同一車両ボックスで再実行。"""
    task = _get_owned_task(task_id, db, user)
    if task.status in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="既に待機中または実行中のタスクです。")
    bbox = dict(bbox_xmin=task.bbox_xmin, bbox_ymin=task.bbox_ymin,
                bbox_xmax=task.bbox_xmax, bbox_ymax=task.bbox_ymax)
    return _enqueue(task.video_id, bbox, db)



@router.get("/videos/{video_id}/events",
            response_model=list[api_schemas.EventOut])
def list_events(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    _get_owned_video(video_id, db, user)
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.video_id == video_id).order_by(
        db_models.CrashEvent.timestamp_sec).all()
    return [to_event_out(e) for e in events]


def _get_owned_event(event_id: int, db: Session, user: db_models.User):
    """他ユーザーの事故区間は存在しないように隠蔽する。"""
    event = db.get(db_models.CrashEvent, event_id)
    if event is None or event.video is None or event.video.user_id != user.id:
        raise HTTPException(status_code=404, detail="クリップが見つかりません。")
    return event


@router.get("/events/{event_id}/clip/raw")
def download_raw_clip(event_id: int, db: Session = Depends(get_db),
                      user: db_models.User = Depends(get_media_user)):
    event = _get_owned_event(event_id, db, user)
    key = raw_clip_key(event)
    storage = get_storage()
    if key is None or not storage.exists(key):
        raise HTTPException(status_code=404,
                            detail="元クリップがありません。動画を再分析すると生成されます。")
    t = event.timestamp_sec or 0.0
    stem = Path(event.video.video_name).stem if event.video else "video"
    name = f"{stem}_{int(t // 60):02d}分{int(t % 60):02d}秒_元事故クリップ.mp4"
    if not isinstance(storage, LocalObjectStorage):
        disposition = f"attachment; filename=clip.mp4; filename*=UTF-8''{quote(name, safe='')}"
        return RedirectResponse(storage.presigned_url(
            key,
            settings.S3_PRESIGNED_URL_EXPIRES,
            response_content_disposition=disposition,
        ))
    return FileResponse(str(storage.local_path(key)), media_type="video/mp4", filename=name)


@router.get("/events/{event_id}/clip")
def get_event_clip(event_id: int, db: Session = Depends(get_db),
                   user: db_models.User = Depends(get_media_user)):
    event = _get_owned_event(event_id, db, user)
    if not event.clip_path:
        raise HTTPException(status_code=404, detail="クリップが見つかりません。")
    storage = get_storage()
    if not storage.exists(event.clip_path):
        raise HTTPException(status_code=404, detail="クリップファイルが見つかりません。")
    if not isinstance(storage, LocalObjectStorage):
        return RedirectResponse(storage.presigned_url(
            event.clip_path, settings.S3_PRESIGNED_URL_EXPIRES))
    path = storage.local_path(event.clip_path)
    media_type = "video/webm" if str(path).endswith(".webm") else "video/mp4"
    return FileResponse(str(path), media_type=media_type)
