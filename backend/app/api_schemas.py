"""Pydantic リクエスト/レスポンス スキーマ。"""
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- 認証 ----------
class SignupRequest(BaseModel):
    username: str
    name: str = Field(min_length=1)   # 氏名必須
    email: Optional[str] = None
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    role: str


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class LoginResponse(BaseModel):
    token: str
    user: UserOut


# ---------- イベント (事故区間) ----------
class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    timestamp_sec: float
    frame_number: int
    end_timestamp_sec: Optional[float] = None
    end_frame_number: Optional[int] = None
    crash_prob: Optional[float] = None
    has_clip: bool = False
    has_raw_clip: bool = False   # オーバーレイなしの元事故クリップをダウンロード可能か
    analyzed_at: Optional[datetime] = None   # 分析完了およびクリップ生成日時 (UTC)
    analysis_no: Optional[int] = None        # このクリップを生成した分析の回数番号 (失敗・キャンセル含む、1開始)
    clip_start_sec: Optional[float] = None   # クリップが元動画の何秒から開始するか


class ActiveTaskOut(BaseModel):
    """ログインユーザーの待機・実行中の分析タスク — 画面リロード時も追跡を継続。"""
    task_id: int
    video_id: int
    video_name: str
    recording_date: Optional[date] = None
    camera_location: str = "駐車場"
    status: str
    progress: Optional[float] = None
    eta_sec: Optional[int] = None
    eta_is_min: bool = False
    stage: Optional[str] = None        # 現在のステージ (TaskStatusOut 参照)
    stage_count: Optional[int] = None
    stage_index: Optional[int] = None
    created_at: Optional[datetime] = None


# ---------- 車両自動検出 ----------
class DetectedVehicleBox(BaseModel):
    id: int
    class_name: str
    confidence: float
    bbox: list[int]  # [xmin, ymin, xmax, ymax]


class VehicleDetectionResponse(BaseModel):
    video_id: int
    total_detected: int
    detected_vehicles: list[DetectedVehicleBox] = []


# ---------- 動画 ----------
class LastTaskOut(BaseModel):
    """動画の最新分析タスク — 失敗・中断をUIへ通知し「再試行」を可能にする。"""
    task_id: int
    status: str
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None


class AnalysisHistoryOut(BaseModel):
    """動画の分析履歴 — 再生画面右側の「分析履歴」。時刻はすべてUTC。"""
    analysis_no: int                       # 何回目の分析か (失敗・キャンセル含む、1開始)
    task_id: int
    status: str                            # PENDING/PROCESSING/SUCCESS/FAILURE/CANCELLED
    error_message: Optional[str] = None
    requested_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None  # 完了日時 (UTC)
    event_count: Optional[int] = None       # 検出されたクリップ数


class VideoOut(BaseModel):
    id: int
    video_name: str
    title: Optional[str] = None   # アップロード時のタイトル (未入力時は None)
    recording_date: Optional[date] = None
    camera_location: str = "駐車場"
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    detected_vehicles: Optional[str] = None
    created_at: datetime
    # 保存期間終了日時および残日数 (サーバー時刻基準、切り上げ)。保存期間無効時は None
    expires_at: Optional[datetime] = None
    expires_in_days: Optional[int] = None
    events: list[EventOut] = []
    last_task: Optional[LastTaskOut] = None
    analysis_done: bool = False   # 過去に1回でも分析成功(SUCCESS)しているか
    analysis_history: list[AnalysisHistoryOut] = []   # 動画詳細 (GET /videos/{id}) でのみ設定


# ---------- 分析 ----------
class AnalyzeRequest(BaseModel):
    bbox_xmin: int
    bbox_ymin: int
    bbox_xmax: int
    bbox_ymax: int


class AnalyzeResponse(BaseModel):
    task_id: int
    celery_task_id: Optional[str] = None
    status: str
    # 同一動画で既に待機・実行中の分析が存在する場合は新規作成せず既存タスクを返す
    already_running: bool = False


class TaskStatusOut(BaseModel):
    task_id: int
    status: str
    progress: Optional[float] = None  # 進捗率 % (小数点第1位、PROCESSING時のみ)
    eta_sec: Optional[int] = None    # 現在ステージの推定残り時間 (秒)
    eta_is_min: bool = False         # True の場合は「最低これ以上」
    # 現在のステージ: loading(エンジン初期化) · decode(動画読み込み) · motion(車両動き計算)
    #   · coarse(全体大まかスキャン) · refine(疑い区間詳細スキャン) · scan(全体スキャン)
    #   · events(事故区間整理) · render(事故クリップ生成)
    stage: Optional[str] = None
    stage_count: Optional[int] = None  # coarse: 検出疑い地点数 · refine: 疑い区間数 · render: クリップ数
    stage_index: Optional[int] = None  # refine・render: 現在何件目か (1開始)
    error_message: Optional[str] = None
    events: list[EventOut] = []
