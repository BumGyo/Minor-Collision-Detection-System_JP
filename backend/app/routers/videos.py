"""動画アップロード / 一覧 / 詳細 / ストリーミングルーター。"""
import math
import shutil
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4
from pathlib import Path

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, UploadFile, Query,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import get_current_user, get_media_user
from app.settings import settings

router = APIRouter(prefix="/api/videos", tags=["videos"])


# ---------- シリアライズヘルパー ----------
def raw_clip_path(ev: db_models.CrashEvent):
    """オーバーレイなしの元事故クリップパス — 分析時にクリップと同じディレクトリに '<名前>_raw.mp4' として生成。"""
    if not ev.clip_path:
        return None
    p = settings.abs_path(ev.clip_path)
    return p.with_name(f"{p.stem}_raw{p.suffix}")


def _unlink_event_clips(events, db: Session):
    """削除対象イベントの表示クリップおよび元クリップファイルを削除。
    同一ファイルを他のイベントが参照している場合はファイルを保持する。"""
    ids = {ev.id for ev in events}
    for path in {ev.clip_path for ev in events if ev.clip_path}:
        still_used = db.query(db_models.CrashEvent).filter(
            db_models.CrashEvent.clip_path == path,
            db_models.CrashEvent.id.notin_(ids)).count()
        if still_used:
            continue
        p = settings.abs_path(path)
        p.unlink(missing_ok=True)
        p.with_name(f"{p.stem}_raw{p.suffix}").unlink(missing_ok=True)


def analysis_no(ev: db_models.CrashEvent):
    """このクリップを生成した分析がその動画の何回目の分析か (1開始)。"""
    ids = sorted(t.id for t in ev.video.analysis_tasks)
    return ids.index(ev.task_id) + 1 if ev.task_id in ids else None


def to_event_out(ev: db_models.CrashEvent) -> api_schemas.EventOut:
    return api_schemas.EventOut(
        id=ev.id,
        timestamp_sec=ev.timestamp_sec,
        frame_number=ev.frame_number,
        end_timestamp_sec=ev.end_timestamp_sec,
        end_frame_number=ev.end_frame_number,
        crash_prob=ev.crash_prob,
        has_clip=bool(ev.clip_path),
        has_raw_clip=bool(ev.clip_path) and raw_clip_path(ev).exists(),
        analyzed_at=_utc(ev.created_at),
        analysis_no=analysis_no(ev),
        clip_start_sec=ev.clip_start_sec,
    )


def _utc(dt):
    """DBはUTC時刻をタイムゾーン情報なしで保持しているため、UTCタイムゾーンを付与してクライアントに送信。"""
    return dt.replace(tzinfo=timezone.utc) if dt else None


def analysis_history(v: db_models.Video) -> list[api_schemas.AnalysisHistoryOut]:
    """動画の分析履歴 (リクエスト順)。"""
    first_clip_at = {}
    for ev in v.crash_events:
        if ev.created_at and (ev.task_id not in first_clip_at or ev.created_at < first_clip_at[ev.task_id]):
            first_clip_at[ev.task_id] = ev.created_at
    out = []
    for i, t in enumerate(sorted(v.analysis_tasks, key=lambda x: x.id)):
        out.append(api_schemas.AnalysisHistoryOut(
            analysis_no=i + 1,
            task_id=t.id,
            status=t.status,
            error_message=t.error_message,
            requested_at=_utc(t.created_at),
            finished_at=_utc(t.finished_at or first_clip_at.get(t.id)),
            event_count=t.event_count,
        ))
    return out


def video_expires_at(v: db_models.Video):
    """保存期間満了日時 (UTC)。保存期間無効時は None。"""
    if settings.VIDEO_RETENTION_DAYS <= 0 or v.created_at is None:
        return None
    return v.created_at + timedelta(days=settings.VIDEO_RETENTION_DAYS)


def to_video_out(v: db_models.Video, with_history: bool = False) -> api_schemas.VideoOut:
    duration = v.total_frames / v.fps if v.fps else 0.0
    expires_at = video_expires_at(v)
    expires_in_days = (math.ceil((expires_at - datetime.utcnow()).total_seconds() / 86400)
                       if expires_at else None)
    return api_schemas.VideoOut(
        id=v.id,
        video_name=v.video_name,
        title=v.title,
        recording_date=v.recording_date,
        camera_location=v.camera_location or "駐車場",
        width=v.width,
        height=v.height,
        fps=v.fps,
        total_frames=v.total_frames,
        duration_sec=duration,
        detected_vehicles=v.detected_vehicles,
        created_at=_utc(v.created_at),
        expires_at=_utc(expires_at),
        expires_in_days=expires_in_days,
        events=[to_event_out(e) for e in v.crash_events],
        last_task=_last_task_out(v),
        analysis_done=any(t.status == "SUCCESS" for t in v.analysis_tasks),
        analysis_history=analysis_history(v) if with_history else [],
    )


def _last_task_out(v: db_models.Video):
    """最新の分析タスク (存在しない場合は None)。"""
    if not v.analysis_tasks:
        return None
    t = max(v.analysis_tasks, key=lambda x: x.id)
    return api_schemas.LastTaskOut(task_id=t.id, status=t.status,
                                   error_message=t.error_message, created_at=t.created_at)


def _extract_metadata(path: Path):
    """cv2で動画メタデータ (幅/高さ/fps/総フレーム数) を抽出。"""
    import cv2
    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, total


@router.post("", response_model=api_schemas.VideoOut)
def upload_video(
    file: UploadFile = File(...),
    recording_date: str | None = Form(None),
    title: str | None = Form(None),   # 任意 — 空欄の場合は画面が「録画日 録画データ #N」と表示
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    title = (title or "").strip()
    if len(title) > 100:
        raise HTTPException(status_code=400, detail="タイトルは100文字以内で入力してください。")

    ext = Path(file.filename).suffix or ".mp4"
    stored_name = f"{uuid4().hex}{ext}"
    dest = settings.UPLOAD_DIR / stored_name
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    width, height, fps, total_frames = _extract_metadata(dest)
    if total_frames <= 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="動画を読み込めません。")

    rec_date: date | None = None
    if recording_date:
        try:
            rec_date = date.fromisoformat(recording_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="録画日の形式が正しくありません。")

    video = db_models.Video(
        user_id=user.id,
        video_name=file.filename,
        title=title or None,
        video_path=settings.rel_path(dest),
        recording_date=rec_date,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return to_video_out(video)


@router.get("", response_model=list[api_schemas.VideoOut])
def list_videos(
    days: int | None = Query(None),
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    q = db.query(db_models.Video).filter(db_models.Video.user_id == user.id)
    if days and days < 9999:
        cutoff = date.today() - timedelta(days=days)
        q = q.filter(db_models.Video.recording_date >= cutoff)
    videos = q.order_by(db_models.Video.created_at.desc()).all()
    return [to_video_out(v) for v in videos]


def _get_owned_video(video_id: int, db: Session, user: db_models.User):
    video = db.get(db_models.Video, video_id)
    if video is None or video.user_id != user.id:
        raise HTTPException(status_code=404, detail="動画が見つかりません。")
    return video


@router.get("/{video_id}", response_model=api_schemas.VideoOut)
def get_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    return to_video_out(_get_owned_video(video_id, db, user), with_history=True)


@router.delete("/{video_id}")
def delete_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """動画削除 — 関連ファイル (元動画・サムネイル・CAMクリップ) および DB (イベント・タスク・動画レコード) をすべて削除。"""
    video = _get_owned_video(video_id, db, user)
    purge_video(video, db)
    db.commit()
    return {"deleted": video_id}


def purge_video(video: db_models.Video, db: Session) -> None:
    """動画と関連データをすべて削除 — ユーザー削除時および保存期間経過時 (app/retention.py) に共通使用。"""
    # 1) 関連CAMクリップ・元クリップファイル削除 + crash_events 行削除
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.video_id == video.id).all()
    _unlink_event_clips(events, db)
    for ev in events:
        db.delete(ev)

    # 2) analysis_tasks 行削除
    for task in db.query(db_models.AnalysisTask).filter(
            db_models.AnalysisTask.video_id == video.id).all():
        db.delete(task)

    # 3) 元動画 + サムネイルファイル削除
    settings.abs_path(video.video_path).unlink(missing_ok=True)
    (settings.THUMBNAIL_DIR / f"{Path(video.video_path).stem}.jpg").unlink(missing_ok=True)

    # 4) 動画レコード削除
    db.delete(video)


@router.delete("/{video_id}/events/{event_id}")
def delete_crash_event(
    video_id: int,
    event_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """特定の事故検出イベント1件を削除。"""
    video = _get_owned_video(video_id, db, user)
    event = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.id == event_id,
        db_models.CrashEvent.video_id == video.id,
    ).first()
    if event is None:
        raise HTTPException(status_code=404, detail="イベントが見つかりません。")

    _unlink_event_clips([event], db)

    db.delete(event)
    db.commit()
    return {"deleted_event_id": event_id}


@router.delete("/{video_id}/events")
def clear_all_crash_events(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """該当動画のすべての事故検出イベントを削除。"""
    video = _get_owned_video(video_id, db, user)
    events = db.query(db_models.CrashEvent).filter(
        db_models.CrashEvent.video_id == video.id
    ).all()
    count = len(events)
    _unlink_event_clips(events, db)
    for ev in events:
        db.delete(ev)
    db.commit()
    return {"deleted_count": count}


@router.get("/{video_id}/stream")
def stream_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_media_user),
):
    video = _get_owned_video(video_id, db, user)
    path = settings.abs_path(video.video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="動画ファイルが見つかりません。")
    return FileResponse(str(path), media_type="video/mp4")


@router.get("/{video_id}/thumbnail")
def get_video_thumbnail(
    video_id: int,
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_media_user),
):
    video = _get_owned_video(video_id, db, user)

    thumb = settings.THUMBNAIL_DIR / f"{Path(video.video_path).stem}.jpg"
    if not thumb.exists():
        src = settings.abs_path(video.video_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail="動画ファイルが見つかりません。")
        import cv2
        cap = cv2.VideoCapture(str(src))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(status_code=404, detail="サムネイルを生成できません。")
        cv2.imwrite(str(thumb), frame)

    return FileResponse(str(thumb), media_type="image/jpeg")


@router.post("/{video_id}/detect-vehicles", response_model=api_schemas.VehicleDetectionResponse)
def detect_vehicles_in_video(
    video_id: int,
    timestamp_sec: float | None = Query(0.0),
    db: Session = Depends(get_db),
    user: db_models.User = Depends(get_current_user),
):
    """YOLOモデルで指定時刻のフレームから車両を検出し、BBOX JSONをDBに保存して返します。"""
    import json

    video = _get_owned_video(video_id, db, user)
    src_path = settings.abs_path(video.video_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="動画ファイルが見つかりません。")

    try:
        from app.vehicle_detector import get_detector, read_source_frame
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"車両検出モジュールのロードに失敗しました: {err}")

    frame_index = int((timestamp_sec or 0.0) * (video.fps or 30.0))
    frame_index = max(0, min(frame_index, max(0, video.total_frames - 1)))

    try:
        frame = read_source_frame(src_path, frame_index=frame_index)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"フレームの抽出に失敗しました: {err}")

    model_path = str(settings.BASE_DIR / "yolo11x.pt")
    if not Path(model_path).exists():
        model_path = "yolo11x.pt"

    try:
        detector = get_detector(
            model_path=model_path, conf=0.15, imgsz=1536,
            enhance_night=True, dynamic_imgsz=True, imgsz_min=640)
        detections = detector.detect_frame(frame)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"YOLO車両検出処理エラー: {err}")

    detected_list = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        detected_list.append({
            "id": idx,
            "class_name": det.class_name,
            "confidence": round(det.confidence, 4),
            "bbox": [x1, y1, x2, y2],
        })

    if settings.DETECT_DEBUG_IMAGES:
        try:
            import cv2
            dbg = frame.copy()
            for item in detected_list:
                x1, y1, x2, y2 = item["bbox"]
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'#{item["id"] + 1} {item["class_name"]} {round(item["confidence"] * 100)}%'
                cv2.putText(dbg, label, (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            out_dir = settings.BASE_DIR / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            base = Path(video.video_name).stem
            out_img = out_dir / f"detect_{base}_f{frame_index}.jpg"
            cv2.imwrite(str(out_img), dbg)
            print(f"[detect] デバッグ画像保存: {out_img}  (検出 {len(detected_list)}台, フレーム {frame.shape[1]}x{frame.shape[0]})")
        except Exception as _dbg_err:
            print(f"[detect] デバッグ画像保存失敗 (無視): {_dbg_err}")

    json_str = json.dumps(detected_list, ensure_ascii=False)
    video.detected_vehicles = json_str
    db.commit()

    return api_schemas.VehicleDetectionResponse(
        video_id=video.id,
        total_detected=len(detected_list),
        detected_vehicles=[
            api_schemas.DetectedVehicleBox(**item) for item in detected_list
        ],
    )
