"""Celery タスク — AI 推論 + 事故区間 CAM クリップ生成。

torch/opencv/モデルのインポートはすべて**タスク内部で遅延ロード**。
→ FastAPI Web プロセスは ML 依存関係なしでもこのモジュールをインポート (.delay 呼び出し) 可能。
"""
import tempfile
from datetime import datetime
from pathlib import Path

from app.worker import celery_app
from app.settings import settings
from app.db_connection import SessionLocal
from app import db_models
from app.object_storage import get_storage, materialize

# ワーカープロセスごとにモデルを1回ロードして再利用
_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    import sys
    # 選択されたバックボーンフォルダをpathに追加 — 内部絶対インポート(import config 等)に対応。
    # フォルダは settings.MODEL_VARIANT(環境変数)により決定。
    sys.path.insert(0, str(settings.MODEL_DIR))

    import torch
    import config as model_config
    from hitandrun_model import HitAndRun3DCNN
    from device_utils import get_device, is_channels_last_3d_supported

    # 推論デバイス(config.INFER_DEVICE_TYPE="cuda") — GPUが存在しない場合は get_device が
    # 警告を出力してCPUへ自動切り替え。実際のデバイスは後続の「モデルロード完了」ログで確認。
    device = get_device(model_config.INFER_DEVICE_TYPE)
    model = HitAndRun3DCNN(num_classes=model_config.MODEL_NUM_CLASSES).to(device)
    if is_channels_last_3d_supported(device) and model_config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last_3d)
    # 本番重みファイルのパスは model/config.py で管理 (SERVICE_WEIGHTS_PATH)
    # 224以外の入力サイズ(_s256)・avg以外のヘッド時間プーリング(_top3 等)で学習した重みの場合はサーバー .env の INPUT_SIZE・
    # HEAD_TEMPORAL_POOL を適合させる必要がある — 不一致時はここで停止
    from hitandrun_model import check_weights
    check_weights(model_config.SERVICE_WEIGHTS_PATH)
    state_dict = torch.load(
        str(model_config.SERVICE_WEIGHTS_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[worker] モデルロード完了 — {getattr(model_config, 'MODEL_NAME', '?')} "
          f"(device={device}, weights={model_config.SERVICE_WEIGHTS_PATH.name})")
    # 推論実行ランナーをモデルロード時に確定(本PCで初回時は自己診断、以降は保存済み結果)。
    # 失敗時も分析はPyTorchで実行する必要があるためここで例外を捕捉。
    try:
        from infer_runtime import get_runner
        get_runner(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] 推論実行ランナーの選択に失敗 — PyTorchで続行: {exc}")
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
        # キャンセル表示はワーカーメモリ(revoke)のみに存在するためワーカー再起動時に消失し、
        # 待機中だったメッセージが再配信される可能性がある → DB状態により二重チェック。
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
            # 過去のバックボーンフォルダ("s3d model" 等、MODEL_VARIANT)は旧ファイル名のまま維持。
            from predict_cam import predict_events_and_clips

        # 進捗率(%)と残り時間(秒)をCelery状態(Redis)に記録 → /tasks が読み取りUIへ伝達。
        # 進捗率は 0〜99 — 100%はタスク完了(SUCCESS)でのみ表示。残り時間は分析関数が
        # 実測した速度で算出し、進捗率が同一であっても1.5秒間隔で更新する。
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
                    "eta_min": bool(eta_min), "stage": stage,
                    "count": count, "index": index})

        storage = get_storage()
        with materialize(storage, video.video_path, suffix=Path(video.video_path).suffix) as source_path, \
                tempfile.TemporaryDirectory(prefix="collision-clips-") as output_dir:
            results = predict_events_and_clips(
                model,
                video_path=source_path,
                bbox=(task.bbox_xmin, task.bbox_ymin, task.bbox_xmax, task.bbox_ymax),
                output_dir=Path(output_dir),
                progress_callback=_on_progress,
                # 分析ごとに一意な名前を使用し、前回の分析クリップを上書きしないようにする。
                clip_name=f"{Path(video.video_path).stem}_t{task.id}",
            )

            db.refresh(task)
            if task.status == "CANCELLED":
                return {"task_id": task_id, "skipped": "CANCELLED"}

            for r in results:
                clip_path = Path(r["clip_path"])
                clip_key = f"clips/{clip_path.name}"
                storage.upload_path(clip_path, clip_key, content_type="video/mp4")
                raw_path = r.get("raw_clip_path")
                if raw_path:
                    raw_path = Path(raw_path)
                    storage.upload_path(raw_path, f"clips/{raw_path.name}", content_type="video/mp4")
                db.add(db_models.CrashEvent(
                    task_id=task.id,
                    video_id=video.id,
                    timestamp_sec=r["start_sec"],
                    frame_number=r["start_frame"],
                    end_timestamp_sec=r["end_sec"],
                    end_frame_number=r["end_frame"],
                    crash_prob=r["crash_prob"],
                    clip_path=clip_key,
                    clip_start_sec=r.get("clip_start_sec"),
                ))

        # 分析途中でキャンセルされたにもかかわらず(revoke失敗等)最後まで実行された場合は結果を破棄
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
