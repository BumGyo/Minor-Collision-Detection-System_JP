"""アプリケーション設定 — DB/Redis接続情報、ローカルストレージパス。

GPU・大容量ストレージサーバーが存在しない環境を想定。
動画およびクリップはプロジェクトルートの storage/ ディレクトリ (ローカルファイルシステム) に保存する。
"""
import functools
import os
import shlex
from pathlib import Path
from urllib.parse import quote, urlencode

from dotenv import load_dotenv


# backend/.env keeps precedence for local development credentials. The root
# .env is also loaded as a fallback so dev.sh uses the same provider settings
# as Docker Compose and the systemd deployment.
_BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(_BASE_DIR / "backend" / ".env")
load_dotenv(_BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    """Treat unset and blank dotenv values alike; deployment templates contain blanks."""
    return os.getenv(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return default


def _database_url() -> str:
    configured = _env_first("DATABASE_URL", "AZURE_POSTGRESQL_CONNECTIONSTRING")
    if configured:
        if "://" not in configured:
            if ";" in configured:
                parts = (part.strip() for part in configured.split(";"))
            else:
                parts = shlex.split(configured)
            options = {}
            for part in parts:
                key, separator, value = part.partition("=")
                if separator:
                    options[key.strip().lower().replace(" ", "_")] = value.strip()
            host = options.get("host") or options.get("server")
            user = options.get("user") or options.get("username")
            password = options.get("password")
            database = options.get("dbname") or options.get("database")
            if not all((host, user, password, database)):
                raise ValueError(
                    "Azure PostgreSQL connection string must include host, user, password, and database"
                )
            query = {}
            sslmode = options.get("sslmode") or options.get("ssl_mode")
            if sslmode:
                query["sslmode"] = sslmode.lower()
            query_string = f"?{urlencode(query)}" if query else ""
            port = options.get("port", "5432")
            return (
                f"postgresql+psycopg://{quote(user, safe='')}:{quote(password, safe='')}"
                f"@{host}:{port}/{quote(database, safe='')}{query_string}"
            )
        return configured
    user = quote(_env("MYSQL_USER", "root"), safe="")
    password = quote(_env("MYSQL_ROOT_PASSWORD", "rootpassword"), safe="")
    host = _env("MYSQL_HOST", "127.0.0.1")
    port = _env("MYSQL_PORT", "3307")
    database = _env("MYSQL_DATABASE", "capstone_db")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"


def _redis_url() -> str:
    configured = _env("REDIS_URL")
    if configured:
        return configured
    azure_connection = _env("AZURE_REDIS_CONNECTIONSTRING")
    azure_options: dict[str, str] = {}
    azure_host = _env("AZURE_REDIS_HOST")
    azure_port = _env("AZURE_REDIS_PORT")
    if azure_connection and "://" not in azure_connection:
        server, *options = azure_connection.split(",")
        host, separator, server_port = server.strip().rpartition(":")
        if separator:
            azure_host = azure_host or host
            azure_port = azure_port or server_port
        for option in options:
            key, separator, value = option.partition("=")
            if separator:
                azure_options[key.strip().lower()] = value.strip()
    elif azure_connection:
        return azure_connection

    username = _env_first("REDIS_USERNAME", "AZURE_REDIS_USERNAME")
    password = _env_first(
        "REDIS_PASSWORD", "AZURE_REDIS_ACCESS_KEY",
        default=azure_options.get("password", azure_options.get("accesskey", "")),
    )
    auth = ""
    if username:
        auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    elif password:
        auth = f":{quote(password, safe='')}@"
    use_tls = _env_first(
        "REDIS_SSL", "AZURE_REDIS_SSL", default=azure_options.get("ssl", "")
    ).lower() in {"1", "true", "yes"}
    scheme = "rediss" if use_tls else "redis"
    default_port = "6380" if use_tls else "6379"
    host = _env_first("REDIS_HOST", default=azure_host or "127.0.0.1")
    port = _env_first("REDIS_PORT", default=azure_port or default_port)
    database = _env("REDIS_DB", "0")
    return f"{scheme}://{auth}{host}:{port}/{database}"


def _storage_dir() -> Path:
    configured = _env("STORAGE_DIR")
    if not configured:
        return _BASE_DIR / "storage"
    path = Path(configured).expanduser()
    return (path if path.is_absolute() else _BASE_DIR / path).resolve()


def _cors_origins() -> list[str]:
    configured = [
        part.strip() for part in _env("CORS_ORIGINS").split(",") if part.strip()
    ]
    return configured or ["http://localhost:5173", "http://127.0.0.1:5173"]


# 使用するバックボーン → モデルコードフォルダ。フォルダごとに config.py / hitandrun_model.py が
# 同名で配置されており、prediction_job が sys.path に追加してインポートする。
# ⚠️ モジュール名(config, hitandrun_model)がフォルダごとに同一であるため1プロセスには
#    1つのバックボーンのみロード可能。別のバックボーンを使用する場合はワーカーを個別に起動する。
# 本番モデルは model/(x3d)。"s3d model/" はgit除外のローカル比較用であるため
# そのフォルダが存在するPCでのみ MODEL_VARIANT=s3d として利用可能。
MODEL_VARIANT_DIRS = {
    "x3d": "model",
    "s3d": "s3d model",
    "slowfast": "slowfast model",
}


class Settings:
    # capstone-26/backend/app/settings.py → parents[2] == capstone-26
    BASE_DIR: Path = _BASE_DIR

    # MODEL_VARIANT=s3d|x3d|slowfast (デフォルト x3d)
    MODEL_VARIANT: str = _env("MODEL_VARIANT", "x3d").lower()
    if MODEL_VARIANT not in MODEL_VARIANT_DIRS:
        raise ValueError(
            f"MODEL_VARIANT='{MODEL_VARIANT}' はサポートされていません。 "
            f"利用可能な値: {', '.join(MODEL_VARIANT_DIRS)}")
    MODEL_DIR: Path = BASE_DIR / MODEL_VARIANT_DIRS[MODEL_VARIANT]
    # 本番重みファイルのパスは各ディレクトリの config.py 内の SERVICE_WEIGHTS_PATH で管理する。

    STORAGE_DIR: Path = _storage_dir()
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CLIP_DIR: Path = STORAGE_DIR / "clips"
    THUMBNAIL_DIR: Path = STORAGE_DIR / "thumbnails"

    DATABASE_URL: str = _database_url()
    REDIS_URL: str = _redis_url()

    # Existing DB rows store object keys (for example uploads/<uuid>.mp4).
    # Keep the former R2/S3 configuration so those videos remain playable.
    STORAGE_BACKEND: str = _env("STORAGE_BACKEND", "local").lower()
    S3_ENDPOINT_URL: str | None = _env_first(
        "S3_ENDPOINT_URL", "R2_ENDPOINT_URL",
        default=(f"https://{_env('CLOUDFLARE_R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"
                 if _env("CLOUDFLARE_R2_ACCOUNT_ID") else ""),
    ) or None
    S3_BUCKET: str | None = _env_first("S3_BUCKET", "R2_BUCKET", "CLOUDFLARE_R2_BUCKET") or None
    S3_ACCESS_KEY_ID: str | None = _env_first(
        "S3_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID", "CLOUDFLARE_R2_ACCESS_KEY_ID") or None
    S3_SECRET_ACCESS_KEY: str | None = _env_first(
        "S3_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY", "CLOUDFLARE_R2_SECRET_ACCESS_KEY") or None
    S3_REGION: str = _env("S3_REGION", "auto")
    S3_PRESIGNED_URL_EXPIRES: int = _env_int("S3_PRESIGNED_URL_EXPIRES", 3600)

    # ログインCookie(JWT)署名鍵 · 有効時間。
    # 鍵は環境変数 AUTH_SECRET_KEY を優先使用し、未設定時は backend/.auth_secret に一度生成して
    # 永続利用する(再起動後もログインを維持するため)。本ファイルはgitから除外。
    # デフォルト8時間(1日の勤務時間) — CCTV動画(個人情報)を取り扱うためセッション維持時間を短めに設定。
    # 開発時に頻繁な再ログインを避ける場合: AUTH_TOKEN_TTL_HOURS=72 ./dev.sh
    AUTH_TOKEN_TTL_HOURS: float = _env_float("AUTH_TOKEN_TTL_HOURS", 8)

    # アップロードからこの期間が経過した動画・クリップ・分析履歴をクリーンアップ。0以下の場合は無効。
    VIDEO_RETENTION_DAYS: int = _env_int("VIDEO_RETENTION_DAYS", 90)
    # 検知デバッグ画像は元動画の保持期間とは別に残る可能性があるためデフォルト無効。
    DETECT_DEBUG_IMAGES: bool = _env("DETECT_DEBUG_IMAGES").lower() in ("1", "true", "yes")

    # CORS許可オリジン (Vite開発サーバー)
    CORS_ORIGINS = _cors_origins()

    def abs_path(self, rel_path: str) -> Path:
        """storage 基準の相対パス → 絶対パス"""
        return self.STORAGE_DIR / rel_path

    def rel_path(self, abs_path) -> str:
        """絶対パス → storage 基準の相対パス (DB保存用)"""
        return str(Path(abs_path).resolve().relative_to(self.STORAGE_DIR))


@functools.lru_cache(maxsize=1)
def auth_secret_key() -> str:
    """JWT署名鍵。トークンを扱うWebサーバーが初回必要時のみ読み取りまたは生成する
    (ワーカーがimportのみで鍵ファイルを生成しないようにする — 同時起動時の鍵の不一致を防止)。"""
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
