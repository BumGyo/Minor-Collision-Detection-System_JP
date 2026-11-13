"""認証 — パスワードハッシュ化 + ログイントークン(JWT)発行/検証の依存関係。

トークンはサーバー秘密鍵で署名されたJWT(HS256)のため改ざんできず、AUTH_TOKEN_TTL_HOURS が
経過すると失効する。以前の "tok_{user_id}" 形式はユーザーIDを変更するだけで他者に
なりすますことが可能であった。

動画・サムネイル・クリップは <video>・<img>・ダウンロードリンク経由でアクセスされるため Authorization ヘッダーを付与できない。
そのため、ログイン時に「メディア専用トークン」を HttpOnly クッキーとして発行する (get_media_user)。
このトークンはメディアURLでのみ有効であり、他のAPIリクエストでは拒否される。
"""
import functools
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from fastapi import Depends, Header, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models
from app.settings import settings, auth_secret_key


# ── パスワード ───────────────────────────────────────────────────────────
# PBKDF2-SHA256, ユーザーごとのソルト(16バイト), 60万回反復(OWASP 2023 推奨)。
# 保存形式: pbkdf2_sha256$反復回数$ソルトhex$ハッシュhex
# 旧形式(ソルトなしSHA-256 hex)も検証は行い、ログイン成功時に新形式へ自動移行する。
_PBKDF2_PREFIX = "pbkdf2_sha256"
_PBKDF2_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"{_PBKDF2_PREFIX}${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith(_PBKDF2_PREFIX + "$"):
        try:
            _, iterations, salt_hex, digest_hex = password_hash.split("$")
            digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                         bytes.fromhex(salt_hex), int(iterations))
        except ValueError:
            return False
        return hmac.compare_digest(digest.hex(), digest_hex)
    legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()   # 旧フォーマット
    return hmac.compare_digest(legacy, password_hash)


def needs_rehash(password_hash: str) -> bool:
    """旧フォーマットまたは反復回数が現在の基準と異なる場合 True。"""
    return not password_hash.startswith(f"{_PBKDF2_PREFIX}${_PBKDF2_ITERATIONS}$")


@functools.lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return hash_password(secrets.token_hex(8))


def burn_password_check(password: str) -> None:
    """存在しないIDでのログイン時もパスワード検証と同等の時間を消費 — 応答時間からIDの
    存在有無を推測されないようにする(タイミング攻撃防止)。"""
    verify_password(password, _dummy_hash())


# ── ログイントークン ─────────────────────────────────────────────────────────
_JWT_ALG = "HS256"
_MEDIA_SCOPE = "media"
MEDIA_COOKIE = "siot_media"


def create_token(user: "db_models.User", scope: str | None = None,
                 expires_at: datetime | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "iat": now,
        "exp": expires_at or now + timedelta(hours=settings.AUTH_TOKEN_TTL_HOURS),
    }
    if scope:
        payload["scope"] = scope
    return jwt.encode(payload, auth_secret_key(), algorithm=_JWT_ALG)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _decode(token: str) -> dict:
    try:
        # アルゴリズムを固定して 'alg' 改変攻撃を防ぎ、有効期限・対象フィールドを必須とする
        payload = jwt.decode(token, auth_secret_key(), algorithms=[_JWT_ALG],
                             options={"require": ["exp", "sub"]})
        int(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise _unauthorized("ログインの有効期限が切れました。再度ログインしてください。")
    except (jwt.InvalidTokenError, ValueError):
        raise _unauthorized("無効なトークンです。")
    return payload


def bearer_payload(authorization: str | None) -> dict:
    """Authorization: Bearer <token> ヘッダーの通常ログイントークン内容。"""
    if not authorization:
        raise _unauthorized("認証が必要です。")
    payload = _decode(authorization.removeprefix("Bearer ").strip())
    if payload.get("scope"):   # メディア専用トークンでは通常APIを利用不可
        raise _unauthorized("無効なトークンです。")
    return payload


def user_from_payload(payload: dict, db: Session) -> "db_models.User":
    user = db.get(db_models.User, int(payload["sub"]))
    if user is None:
        raise _unauthorized("ユーザーが見つかりません。")
    return user


def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db),
) -> "db_models.User":
    """Authorization: Bearer <token> ヘッダーから現在のユーザーを解決する。"""
    return user_from_payload(bearer_payload(authorization), db)


def get_media_user(
    request: Request,
    authorization: str = Header(None),
    db: Session = Depends(get_db),
) -> "db_models.User":
    """動画・サムネイル・クリップ配信用 — ヘッダーがあればヘッダーで、なければログイン時に発行されたメディア用Cookieで検証。"""
    if authorization:
        return get_current_user(authorization, db)
    token = request.cookies.get(MEDIA_COOKIE)
    if not token:
        raise _unauthorized("認証が必要です。")
    payload = _decode(token)
    if payload.get("scope") != _MEDIA_SCOPE:
        raise _unauthorized("無効なトークンです。")
    return user_from_payload(payload, db)


def _cookie_options(request: Request) -> dict:
    # 動画パス(/api)にのみ送信され、JavaScriptからの読み取り不可、クロスサイトリクエストには付加されない。
    # HTTPSアクセス時(nginx経由時は X-Forwarded-Proto で判定)は HTTPS のみで送信。
    return dict(path="/api", httponly=True, samesite="strict",
                secure=request.url.scheme == "https")


def set_media_cookie(response: Response, request: Request,
                      user: "db_models.User", expires_at: datetime) -> None:
    """通常ログイントークンと同一有効期限のメディア専用トークンをCookieとして設定する。"""
    max_age = max(0, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(MEDIA_COOKIE, create_token(user, _MEDIA_SCOPE, expires_at),
                        max_age=max_age, **_cookie_options(request))


def clear_media_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(MEDIA_COOKIE, **_cookie_options(request))
