"""会員登録 / ログインルーター。"""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.db_connection import get_db
from app import db_models, api_schemas
from app.auth_guard import (
    bearer_payload,
    burn_password_check,
    clear_media_cookie,
    create_token,
    get_current_user,
    hash_password,
    needs_rehash,
    set_media_cookie,
    user_from_payload,
    verify_password,
)
from app.settings import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=api_schemas.UserOut)
def signup(req: api_schemas.SignupRequest, db: Session = Depends(get_db)):
    if db.query(db_models.User).filter(
            db_models.User.username == req.username).first():
        raise HTTPException(status_code=400, detail="既に使用されているユーザーIDです。")
    if req.email and db.query(db_models.User).filter(
            db_models.User.email == req.email).first():
        raise HTTPException(status_code=400, detail="既に使用されているメールアドレスです。")

    user = db_models.User(
        username=req.username,
        password_hash=hash_password(req.password),
        name=req.name,
        email=req.email,
        role="USER",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=api_schemas.LoginResponse)
def login(req: api_schemas.LoginRequest, request: Request, response: Response,
          db: Session = Depends(get_db)):
    user = db.query(db_models.User).filter(
        db_models.User.username == req.username).first()
    if user is None:
        burn_password_check(req.password)
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ユーザーIDまたはパスワードが正しくありません。",
        )
    if needs_rehash(user.password_hash):   # 旧方式の場合は新方式に更新
        user.password_hash = hash_password(req.password)
        db.commit()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.AUTH_TOKEN_TTL_HOURS)
    set_media_cookie(response, request, user, expires_at)   # メディア配信用クッキー設定
    return api_schemas.LoginResponse(token=create_token(user, expires_at=expires_at), user=user)


@router.post("/session", response_model=api_schemas.UserOut)
def start_session(request: Request, response: Response,
                  authorization: str = Header(None), db: Session = Depends(get_db)):
    """画面アクセス時 — 最新のユーザー情報を返し、メディアクッキーを再付与する。"""
    payload = bearer_payload(authorization)
    user = user_from_payload(payload, db)
    set_media_cookie(response, request, user, datetime.fromtimestamp(payload["exp"], timezone.utc))
    return user


@router.post("/logout")
def logout(request: Request, response: Response):
    """メディア用クッキーを削除する (ログイントークンはクライアント側で破棄)。"""
    clear_media_cookie(response, request)
    return {"ok": True}


@router.get("/me", response_model=api_schemas.UserOut)
def get_me(current_user: db_models.User = Depends(get_current_user)):
    """現在ログイン中のユーザー情報を返します。"""
    return current_user


@router.put("/me", response_model=api_schemas.UserOut)
def update_me(
    req: api_schemas.UpdateUserRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """現在ログイン中のユーザープロフィール (氏名、メール、連絡先) を更新します。"""
    if req.email and req.email != current_user.email:
        existing = db.query(db_models.User).filter(
            db_models.User.email == req.email,
            db_models.User.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="既に使用されているメールアドレスです。")
        current_user.email = req.email

    if req.name is not None:
        current_user.name = req.name.strip()
    if req.phone is not None:
        current_user.phone = req.phone.strip()

    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/change-password")
def change_password(
    req: api_schemas.ChangePasswordRequest,
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """現在ログイン中のユーザーのパスワードを変更します。"""
    if not verify_password(req.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=400,
            detail="現在のパスワードが一致しません。",
        )
    if len(req.new_password) < 4:
        raise HTTPException(
            status_code=400,
            detail="新しいパスワードは4文字以上で入力してください。",
        )
    current_user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"message": "パスワードが正常に変更されました。"}
