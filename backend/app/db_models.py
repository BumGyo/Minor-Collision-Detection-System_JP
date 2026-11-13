"""SQLAlchemy DB モデル。

テーブル関連: users → videos → analysis_tasks → crash_events
(crash_events は高速クエリ用に video_id も非正規化して保持)
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, ForeignKey, Text,
)
from sqlalchemy.orm import relationship

from app.db_connection import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=True)      # 会員登録: 氏名
    email = Column(String(255), unique=True, nullable=True)
    phone = Column(String(50), nullable=True)      # 連絡先
    role = Column(String(50), default="USER")
    created_at = Column(DateTime, default=datetime.utcnow)

    videos = relationship("Video", back_populates="owner")


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    video_name = Column(String(255), nullable=False)   # 元ファイル名
    title = Column(String(100), nullable=True)         # アップロード時に入力されたタイトル (未入力時は「録画日 録画データ #N」)
    video_path = Column(String(500), nullable=False)   # storage 基準の相対パス
    recording_date = Column(Date, nullable=True)       # アップロード時に指定された録画日
    camera_location = Column(String(255), nullable=False, default="駐車場")   # 防犯カメラ設置場所

    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    fps = Column(Float, nullable=False)
    total_frames = Column(Integer, nullable=False)
    detected_vehicles = Column(Text, nullable=True)  # YOLO検出バウンディングボックスJSONマッピング情報
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="videos")
    analysis_tasks = relationship("AnalysisTask", back_populates="video")
    crash_events = relationship("CrashEvent", back_populates="video")


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    celery_task_id = Column(String(255), unique=True, index=True, nullable=True)

    # ユーザーが指定した対象車両バウンディングボックス (元解像度ピクセル座標)
    bbox_xmin = Column(Integer, nullable=False)
    bbox_ymin = Column(Integer, nullable=False)
    bbox_xmax = Column(Integer, nullable=False)
    bbox_ymax = Column(Integer, nullable=False)

    status = Column(String(50), default="PENDING")  # PENDING/PROCESSING/SUCCESS/FAILURE/CANCELLED
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)   # 分析リクエスト日時 (UTC)
    # 分析履歴用 — カラム追加前の分析データは NULL
    finished_at = Column(DateTime, nullable=True)   # 完了日時 (UTC): 成功・失敗・キャンセル
    event_count = Column(Integer, nullable=True)    # 検出された事故疑いクリップ数 (成功時)

    video = relationship("Video", back_populates="analysis_tasks")
    crash_events = relationship("CrashEvent", back_populates="task")


class CrashEvent(Base):
    """事故疑い区間 (イベント) 1件。区間ごとのCAMクリップ1件。"""
    __tablename__ = "crash_events"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("analysis_tasks.id"), nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)

    # 区間開始
    timestamp_sec = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=False)
    # 区間終了 (クリップ長・区間表示用)
    end_timestamp_sec = Column(Float, nullable=True)
    end_frame_number = Column(Integer, nullable=True)

    crash_prob = Column(Float, nullable=True)          # 区間代表確率 (0〜1)
    # 事故区間クリップパス (storage 相対パス)。既存スキーマ維持のためカラム名は cam_heatmap_path のまま
    clip_path = Column("cam_heatmap_path", String(500), nullable=True)
    # クリップ(マージン含む)が元動画の何秒から開始するか — 再生画面でのCAMオーバーレイ同期に使用
    clip_start_sec = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("AnalysisTask", back_populates="crash_events")
    video = relationship("Video", back_populates="crash_events")
