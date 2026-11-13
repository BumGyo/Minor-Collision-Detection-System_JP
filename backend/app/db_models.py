"""SQLAlchemy DB モデル。

テーブル関連: users → videos → analysis_tasks → crash_events
(crash_events は高速クエリ用に video_id も非正規化して保持)
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, ForeignKey, Index, Text,
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
    __table_args__ = (Index("idx_videos_user_id_fk", "user_id"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    video_name = Column(String(255), nullable=False)   # 元ファイル名
    title = Column(String(100), nullable=True)         # アップロード時に指定された表示タイトル
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
    vehicle_detection_tasks = relationship("VehicleDetectionTask", back_populates="video")
    crash_events = relationship("CrashEvent", back_populates="video")


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"
    __table_args__ = (Index("idx_analysis_tasks_video_id_fk", "video_id"),)

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
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    event_count = Column(Integer, nullable=True)

    video = relationship("Video", back_populates="analysis_tasks")
    crash_events = relationship("CrashEvent", back_populates="task")


class VehicleDetectionTask(Base):
    """GPUワーカーが実行するフレーム別車両検知タスク。"""
    __tablename__ = "vehicle_detection_tasks"
    __table_args__ = (Index("idx_vehicle_detection_tasks_video_id_fk", "video_id"),)

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    celery_task_id = Column(String(255), unique=True, index=True, nullable=True)
    timestamp_sec = Column(Float, nullable=False, default=0.0)
    status = Column(String(50), nullable=False, default="PENDING")
    error_message = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)

    video = relationship("Video", back_populates="vehicle_detection_tasks")


class CrashEvent(Base):
    """事故疑い区間 (イベント) 1件。区間ごとのCAMクリップ1件。"""
    __tablename__ = "crash_events"
    __table_args__ = (
        Index("idx_crash_events_task_id_fk", "task_id"),
        Index("idx_crash_events_video_id_fk", "video_id"),
    )

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
    # DBカラム名は従来の cam_heatmap_path を維持し、コード上では実際の役割である clip_path として扱う。
    clip_path = Column("cam_heatmap_path", String(500), nullable=True)
    # 生成されたCAMクリップが元動画において開始する時刻。以前の分析結果はNULL。
    clip_start_sec = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("AnalysisTask", back_populates="crash_events")
    video = relationship("Video", back_populates="crash_events")
