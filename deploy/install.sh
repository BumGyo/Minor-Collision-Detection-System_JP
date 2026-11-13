#!/usr/bin/env bash
# SIOT サーバーインストール・更新スクリプト — Ubuntu サーバー用。
#
#   sudo ./deploy/install.sh                インストール・更新: UIビルド → DB・Redis → バックエンド・ワーカー → nginx
#   ./deploy/install.sh --dry-run <ディレクトリ> 設定ファイル生成確認用
#
# 前提条件 (deploy/DEPLOY.md 1〜4段階参照): venv311, 重みファイル, プロジェクトルート .env, nginx, docker
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY="${2:?--dry-run の出力先ディレクトリを指定してください}"
fi

log()  { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

env_get() {
  [[ -f "$REPO/.env" ]] || return 0
  sed -n "s/^$1=//p" "$REPO/.env" | tail -n 1 | tr -d '\r'
}

DOMAIN="$(env_get DOMAIN)"
SERVER_NAME="${DOMAIN:-_}"
HTTPS=""
if [[ -n "$DOMAIN" && -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]]; then
  HTTPS=1
fi
NGINX_TEMPLATE="$REPO/deploy/nginx/siot-http.conf"
[[ -n "$HTTPS" ]] && NGINX_TEMPLATE="$REPO/deploy/nginx/siot-https.conf"

render() {
  sed -e "s|__REPO__|$REPO|g" -e "s|__USER__|$RUN_USER|g" -e "s|__DOMAIN__|$SERVER_NAME|g" "$1"
}

# ── Dry-run: 設定ファイルの生成確認 ──────────────────────────────────────
if [[ -n "$DRY" ]]; then
  mkdir -p "$DRY"
  render "$REPO/deploy/systemd/siot-backend.service" > "$DRY/siot-backend.service"
  render "$REPO/deploy/systemd/siot-worker.service" > "$DRY/siot-worker.service"
  render "$NGINX_TEMPLATE" > "$DRY/siot.conf"
  cp "$REPO/deploy/nginx/siot-common.conf" "$DRY/siot-common.conf"
  log "生成された設定ファイルを $DRY に出力しました (実行ユーザー: $RUN_USER, ドメイン: $SERVER_NAME, HTTPS: ${HTTPS:-なし})"
  exit 0
fi

# ── 前提条件確認 ───────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || die "sudo で実行してください: sudo ./deploy/install.sh"
[[ "$RUN_USER" != "root" ]] || die "root ではなく通常ユーザーから sudo で実行してください"
[[ -x "$REPO/venv311/bin/python" ]] || die "venv311 が存在しません — deploy/DEPLOY.md 手順2を参照"
[[ -f "$REPO/.env" ]] || die ".env が存在しません — cp deploy/env.example .env を実行して設定してください"
[[ -n "$(env_get MYSQL_ROOT_PASSWORD)" ]] || die ".env の MYSQL_ROOT_PASSWORD を設定してください"
[[ -n "$(env_get REDIS_PASSWORD)" ]] || die ".env の REDIS_PASSWORD を設定してください"
[[ -n "$(env_get AUTH_SECRET_KEY)" ]] || warn "AUTH_SECRET_KEY が未設定のため backend/.auth_secret を自動生成して使用します"
command -v nginx >/dev/null || die "nginx がインストールされていません — deploy/DEPLOY.md 手順1を参照"
command -v docker >/dev/null || die "docker がインストールされていません — deploy/DEPLOY.md 手順1を参照"

WEIGHTS="$(sudo -u "$RUN_USER" env SERVICE_WEIGHTS="$(env_get SERVICE_WEIGHTS)" \
  "$REPO/venv311/bin/python" -c "import sys; sys.path.insert(0, '$REPO/model'); import config; print(config.SERVICE_WEIGHTS_PATH)")"
[[ -f "$WEIGHTS" ]] || die "サービス重みファイルが存在しません: $WEIGHTS — deploy/DEPLOY.md 手順3を参照"
[[ -f "$REPO/yolo11x.pt" ]] || warn "yolo11x.pt が存在しないため初回車両検出時に自動ダウンロードされます (約110MB)"

chown "$RUN_USER" "$REPO/.env"
chmod 600 "$REPO/.env"

# ── 1) フロントエンドビルド → /var/www/siot ─────────────────────────────
NODE_MAJOR="$(sudo -u "$RUN_USER" bash -lc 'node -v 2>/dev/null' | sed -n 's/^v\([0-9]*\).*/\1/p')"
if [[ -n "$NODE_MAJOR" && "$NODE_MAJOR" -ge 20 ]]; then
  log "フロントエンドビルド (node v$NODE_MAJOR)"
  sudo -u "$RUN_USER" bash -lc "
    set -e; cd '$REPO/ui'
    if [[ ! -f node_modules/.package-lock.json || package-lock.json -nt node_modules/.package-lock.json ]]; then
      npm ci --no-audit --no-fund
    fi
    npm run build"
elif [[ -f "$REPO/ui/dist/index.html" ]]; then
  warn "node 20 以上が見つからないため既存の ui/dist をそのまま配置します"
else
  die "ui/dist が存在せず、node 20 以上もインストールされていません"
fi
install -d -m 755 /var/www/siot
if command -v rsync >/dev/null; then
  rsync -a --delete "$REPO/ui/dist/" /var/www/siot/
else
  find /var/www/siot -mindepth 1 -delete
  cp -r "$REPO/ui/dist/." /var/www/siot/
fi
chmod -R a+rX /var/www/siot

# ── 2) 動画ストレージディレクトリ ────────────────────────────────────────
STORAGE="$(env_get STORAGE_DIR)"
if [[ -n "$STORAGE" ]]; then
  [[ "$STORAGE" = /* ]] || STORAGE="$REPO/$STORAGE"
  install -d -o "$RUN_USER" "$STORAGE"
fi

# ── 3) DB・Redis (docker) ──────────────────────────────────────────────
log "DB・Redis 起動"
docker compose --project-directory "$REPO" -f "$REPO/docker-compose.yml" up -d db redis

# ── 4) バックエンド・ワーカー サービス ──────────────────────────────────
log "バックエンド・ワーカー サービス登録・再起動 (実行ユーザー: $RUN_USER)"
for unit in siot-backend siot-worker; do
  render "$REPO/deploy/systemd/$unit.service" > "/etc/systemd/system/$unit.service"
done
systemctl daemon-reload
systemctl enable siot-backend siot-worker >/dev/null
systemctl restart siot-backend siot-worker

# ── 5) nginx ──────────────────────────────────────────────────────────
log "nginx 設定 (${HTTPS:+HTTPS, }ドメイン: $SERVER_NAME)"
install -m 644 "$REPO/deploy/nginx/siot-common.conf" /etc/nginx/snippets/siot-common.conf
render "$NGINX_TEMPLATE" > /etc/nginx/sites-available/siot.conf
ln -sf /etc/nginx/sites-available/siot.conf /etc/nginx/sites-enabled/siot.conf
if [[ -L /etc/nginx/sites-enabled/default ]]; then
  rm /etc/nginx/sites-enabled/default
  log "nginx デフォルト設定を無効化しました"
fi
nginx -t
systemctl enable nginx >/dev/null
systemctl reload nginx 2>/dev/null || systemctl restart nginx

# ── 6) 起動確認 ────────────────────────────────────────────────────────
log "バックエンド起動待機中 (最大90秒)"
for _ in $(seq 1 45); do
  if curl -fsS http://127.0.0.1:8000/ >/dev/null 2>&1; then
    if [[ -n "$HTTPS" ]]; then URL="https://$DOMAIN"; else URL="http://${DOMAIN:-<サーバーIP>}"; fi
    log "完了 — アクセスURL: $URL"
    [[ -n "$DOMAIN" && -z "$HTTPS" ]] && log "HTTPS は certbot 完了後に本スクリプトを再実行することで有効化されます"
    exit 0
  fi
  sleep 2
done
die "バックエンドが応答しません — journalctl -u siot-backend -n 50 でログを確認してください"
