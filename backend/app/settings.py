"""アプリケーション設定 — DB/Redis接続情報、ローカルストレージパス。

サーバー1台にWeb、ワーカー、DB、Redis、ストレージを同一配置する構成を想定。
動画およびクリップはプロジェクトルートの storage/ ディレクトリ (ローカルファイルシステム) に保存する。

本番サーバー環境ではプロジェクトルートの .env で設定をオーバーライド可能 (deploy/env.example 参照)。
"""
import functools
import os
from pathlib import Path
from urllib.parse import quote


def _env_path(name: str, default: Path, base: Path) -> Path:
    """環境変数パス (相対パスの場合はプロジェクトルート基準)。未設定時は default。"""
    value = os.getenv(name, "").strip()
    if not value:
        return default
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


# 使用するバックボーン → モデルコードディレクトリ。ディレクトリごとに config.py / hitandrun_model.py が
# 同名で配置されており、prediction_job が sys.path に追加してインポートする。
# ⚠️ モジュール名が同一のため1つのプロセスには1つのバックボーンのみロード可能。
# 本番モデルは model/ (x3d)。
MODEL_VARIANT_DIRS = {
    "x3d": "model",
    "s3d": "s3d model",
    "slowfast": "slowfast model",
}


class Settings:
    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    # MODEL_VARIANT=s3d|x3d|slowfast (デフォルト x3d)
    MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "x3d").strip().lower()
    if MODEL_VARIANT not in MODEL_VARIANT_DIRS:
        raise ValueError(
            f"MODEL_VARIANT='{MODEL_VARIANT}' はサポートされていません。 "
            f"利用可能な値: {', '.join(MODEL_VARIANT_DIRS)}")
    MODEL_DIR: Path = BASE_DIR / MODEL_VARIANT_DIRS[MODEL_VARIANT]
    # 本番重みファイルのパスは各ディレクトリの config.py 内の SERVICE_WEIGHTS_PATH で管理する。

    # 動画・クリップ・サムネイル保存ディレクトリ。
    STORAGE_DIR: Path = _env_path("STORAGE_DIR", BASE_DIR / "storage", BASE_DIR)
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CLIP_DIR: Path = STORAGE_DIR / "clips"
    THUMBNAIL_DIR: Path = STORAGE_DIR / "thumbnails"

    # DB・Redis 接続先。
    DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip() or (
        "mysql+pymysql://root:"
        + quote(os.getenv("MYSQL_ROOT_PASSWORD", "").strip() or "rootpassword", safe="")
        + "@127.0.0.1:3306/capstone_db")
    REDIS_URL: str = os.getenv("REDIS_URL", "").strip() or (
        f"redis://:{quote(os.getenv('REDIS_PASSWORD', '').strip(), safe='')}@127.0.0.1:6379/0"
        if os.getenv("REDIS_PASSWORD", "").strip() else "redis://127.0.0.1:6379/0")

    # ログイントークン (JWT) 署名キー・有効期間。
    # デフォルト8時間。
    AUTH_TOKEN_TTL_HOURS: float = float(os.getenv("AUTH_TOKEN_TTL_HOURS", "").strip() or "8")

    # 動画保存期間 (日数) — アップロード後この日数を経過すると動画および関連ファイル (事故クリップ・元クリップ・サムネイル・分析履歴) をすべて削除する (app/retention.py)。0以下の場合は削除しない。
    VIDEO_RETENTION_DAYS: int = int(os.getenv("VIDEO_RETENTION_DAYS", "").strip() or "90")

    # 車両検出結果をデバッグ用画像として outputs/ に出力するか (開発確認用)。
    DETECT_DEBUG_IMAGES: bool = os.getenv("DETECT_DEBUG_IMAGES", "").strip().lower() in ("1", "true", "yes")

    # CORS 許可オリジン (デフォルト: Vite 開発サーバー)。
    CORS_ORIGINS = [o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

    def abs_path(self, rel_path: str) -> Path:
        """storage 基準の相対パス → 絶対パス"""
        return self.STORAGE_DIR / rel_path

    def rel_path(self, abs_path) -> str:
        """絶対パス → storage 基準の相対パス (DB保存用)"""
        return str(Path(abs_path).resolve().relative_to(self.STORAGE_DIR))


@functools.lru_cache(maxsize=1)
def auth_secret_key() -> str:
    """JWT 署名キー。トークンを使用するWebサーバーが必要時に読み込みまたは生成。"""
    key = os.getenv("AUTH_SECRET_KEY", "").strip()
    if key:
        return key
    path = Path(__file__).resolve().parents[1] / ".auth_secret"
    if path.exists():
        saved_key = path.read_text(encoding="utf-8").strip()
        if saved_key:
            return saved_key
    import secrets
    key = secrets.token_hex(32)
    path.write_text(key, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


settings = Settings()

# ストレージディレクトリの作成を保証
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CLIP_DIR.mkdir(parents=True, exist_ok=True)
settings.THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
