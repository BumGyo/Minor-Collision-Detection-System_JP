#!/usr/bin/env bash
# SIOT サーバーインストール・更新 — Ubuntu サーバーでコード取得(または git pull)後に実行。複数回実行可能。
#
#   sudo ./deploy/install.sh                インストール・更新: UI ビルド → Redis → バックエンド・ワーカー → nginx
#   ./deploy/install.sh --dry-run <フォルダ>     インストールせず生成される設定ファイルのみ <フォルダ> に出力(確認用)
#
# 前提条件(deploy/DEPLOY.md 1〜4段階): venv311, 重みファイル, プロジェクトルート .env, nginx, docker
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY="${2:?--dry-run の後に結果を出力するフォルダを指定してください}"
fi

log()  { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

# .env から設定値を1つ読み取る (KEY=VALUE 行の末尾の値)
env_get() {
  [[ -f "$REPO/.env" ]] || return 0
  sed -n "s/^$1=//p" "$REPO/.env" | tail -n 1 | tr -d '\r'
}

env_first() {
  local name value
  for name in "$@"; do
    value="$(env_get "$name")"
    if [[ -n "$value" ]]; then
      printf '%s' "$value"
      return 0
    fi
  done
  return 0
}

DOMAIN="$(env_get DOMAIN)"
SERVER_NAME="${DOMAIN:-_}"   # ドメイン未設定時は IP へのアクセスをすべて受信
HTTPS=""
if [[ -n "$DOMAIN" && -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]]; then
  HTTPS=1
fi
NGINX_TEMPLATE="$REPO/deploy/nginx/siot-http.conf"
[[ -n "$HTTPS" ]] && NGINX_TEMPLATE="$REPO/deploy/nginx/siot-https.conf"

render() {
  sed -e "s|__REPO__|$REPO|g" -e "s|__USER__|$RUN_USER|g" -e "s|__DOMAIN__|$SERVER_NAME|g" "$1"
}

# ── 確認用: ファイル生成のみ実行 ──────────────────────────────────────
if [[ -n "$DRY" ]]; then
  mkdir -p "$DRY"
  render "$REPO/deploy/systemd/siot-backend.service" > "$DRY/siot-backend.service"
  render "$REPO/deploy/systemd/siot-worker.service" > "$DRY/siot-worker.service"
  render "$NGINX_TEMPLATE" > "$DRY/siot.conf"
  cp "$REPO/deploy/nginx/siot-common.conf" "$DRY/siot-common.conf"
  log "生成ファイルを $DRY に出力しました (実行アカウント $RUN_USER, ドメイン $SERVER_NAME, HTTPS ${HTTPS:-なし})"
  exit 0
fi

# ── 前提条件の確認 ───────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || die "sudo で実行してください: sudo ./deploy/install.sh"
[[ "$RUN_USER" != "root" ]] || die "root ではなく通常アカウントから sudo で実行してください(サービスがそのアカウントで稼働します)"
[[ -x "$REPO/venv311/bin/python" ]] || die "venv311 が存在しません — deploy/DEPLOY.md ステップ2"
[[ -f "$REPO/.env" ]] || die ".env が存在しません — cp deploy/env.example .env の後に値を設定してください"
DATABASE_URL="$(env_first DATABASE_URL AZURE_POSTGRESQL_CONNECTIONSTRING)"
LOCAL_DATABASE="0"
if [[ -z "$DATABASE_URL" ]]; then
  LOCAL_DATABASE="1"
  warn "DATABASE_URL が空のため loopback 専用ローカル MySQL を使用します"
fi
REDIS_URL="$(env_get REDIS_URL)"
LOCAL_REDIS="1"
if [[ -n "$REDIS_URL" ]]; then
  case "$REDIS_URL" in
    redis://*localhost:*|redis://*127.0.0.1:*|rediss://*localhost:*|rediss://*127.0.0.1:*) ;;
    *) LOCAL_REDIS="0" ;;
  esac
else
  REDIS_HOST="$(env_first REDIS_HOST AZURE_REDIS_HOST)"
  case "$REDIS_HOST" in
    ""|localhost|127.0.0.1|::1) ;;
    *) LOCAL_REDIS="0" ;;
  esac
fi
[[ -n "$(env_get AUTH_SECRET_KEY)" ]] || warn "AUTH_SECRET_KEY が空のため backend/.auth_secret に鍵を自動生成して使用します"
STORAGE_BACKEND="$(env_get STORAGE_BACKEND)"
STORAGE_BACKEND="${STORAGE_BACKEND:-local}"
if [[ "$STORAGE_BACKEND" == "s3" ]]; then
  [[ -n "$(env_first S3_ENDPOINT_URL R2_ENDPOINT_URL CLOUDFLARE_R2_ACCOUNT_ID)" ]] || die ".env に R2 endpoint または CLOUDFLARE_R2_ACCOUNT_ID を設定してください"
  [[ -n "$(env_first S3_BUCKET R2_BUCKET CLOUDFLARE_R2_BUCKET)" ]] || die ".env に S3_BUCKET または CLOUDFLARE_R2_BUCKET を設定してください"
  [[ -n "$(env_first S3_ACCESS_KEY_ID R2_ACCESS_KEY_ID CLOUDFLARE_R2_ACCESS_KEY_ID)" ]] || die ".env に R2 access key を設定してください"
  [[ -n "$(env_first S3_SECRET_ACCESS_KEY R2_SECRET_ACCESS_KEY CLOUDFLARE_R2_SECRET_ACCESS_KEY)" ]] || die ".env に R2 secret key を設定してください"
elif [[ "$STORAGE_BACKEND" != "local" ]]; then
  die ".env の STORAGE_BACKEND は local または s3 である必要があります"
fi
command -v nginx >/dev/null || die "nginx が存在しません — deploy/DEPLOY.md ステップ1"
if [[ "$LOCAL_DATABASE" == "1" || "$LOCAL_REDIS" == "1" ]]; then
  command -v docker >/dev/null || die "ローカル DB/Redis の使用には docker が必要です — deploy/DEPLOY.md ステップ1"
fi

WEIGHTS="$(sudo -u "$RUN_USER" env SERVICE_WEIGHTS="$(env_get SERVICE_WEIGHTS)" \
  "$REPO/venv311/bin/python" -c "import sys; sys.path.insert(0, '$REPO/model'); import config; print(config.SERVICE_WEIGHTS_PATH)")"
[[ -f "$WEIGHTS" ]] || die "本番重みファイルが存在しません: $WEIGHTS — ローカルからコピーしてください(deploy/DEPLOY.md ステップ3)"
[[ -f "$REPO/yolo11x.pt" ]] || warn "yolo11x.pt が存在しないため初回の車両検知時に自動ダウンロードされます(約110MB)"

chown "$RUN_USER" "$REPO/.env"
chmod 600 "$REPO/.env"   # パスワードが含まれるため権限を制限

# ── 1) UI ビルド → /var/www/siot ───────────────────────────────────────
NODE_MAJOR="$(sudo -u "$RUN_USER" bash -lc 'node -v 2>/dev/null' | sed -n 's/^v\([0-9]*\).*/\1/p')"
if [[ -n "$NODE_MAJOR" && "$NODE_MAJOR" -ge 20 ]]; then
  log "UI ビルド (node v$NODE_MAJOR)"
  sudo -u "$RUN_USER" bash -lc "
    set -e; cd '$REPO/ui'
    # package-lock.json 変更時または初回はパッケージを新規インストール
    if [[ ! -f node_modules/.package-lock.json || package-lock.json -nt node_modules/.package-lock.json ]]; then
      npm ci --no-audit --no-fund
    fi
    npm run build"
elif [[ -f "$REPO/ui/dist/index.html" ]]; then
  warn "node 20 以上が存在しないため既存の ui/dist をそのまま使用します(ローカルでビルドしてコピー済みか確認してください)"
else
  die "UI ビルド成果物(ui/dist)がなく node 20 以上も存在しません — deploy/DEPLOY.md ステップ1(Node)またはローカルビルドをコピー"
fi
install -d -m 755 /var/www/siot
if command -v rsync >/dev/null; then
  rsync -a --delete "$REPO/ui/dist/" /var/www/siot/
else
  find /var/www/siot -mindepth 1 -delete
  cp -r "$REPO/ui/dist/." /var/www/siot/
fi
chmod -R a+rX /var/www/siot

# ── 2) 動画保存フォルダ (別ディスクへ移動時は権限を適合) ─────────────
STORAGE="$(env_get STORAGE_DIR)"
if [[ -n "$STORAGE" ]]; then
  [[ "$STORAGE" = /* ]] || STORAGE="$REPO/$STORAGE"
  install -d -o "$RUN_USER" "$STORAGE"
fi

# ── 3) 選択されたローカル DB・Redis (docker) ──────────────────────────────────
LOCAL_SERVICES=()
if [[ "$LOCAL_DATABASE" == "1" ]]; then
  LOCAL_SERVICES+=(db)
fi
if [[ "$LOCAL_REDIS" == "1" ]]; then
  LOCAL_SERVICES+=(redis)
fi
if (( ${#LOCAL_SERVICES[@]} > 0 )); then
  log "ローカルサービス起動: ${LOCAL_SERVICES[*]} (loopback 専用)"
  docker compose --project-directory "$REPO" -f "$REPO/docker-compose.yml" up -d "${LOCAL_SERVICES[@]}"
else
  log "外部 PostgreSQL・Redis 設定を使用します"
fi

# ── 4) バックエンド・ワーカーサービス ──────────────────────────────────────────────
log "バックエンド・ワーカーサービス登録・再起動 (実行アカウント $RUN_USER)"
for unit in siot-backend siot-worker; do
  render "$REPO/deploy/systemd/$unit.service" > "/etc/systemd/system/$unit.service"
done
systemctl daemon-reload
systemctl enable siot-backend siot-worker >/dev/null
systemctl restart siot-backend siot-worker

# ── 5) nginx ──────────────────────────────────────────────────────────
log "nginx 設定 (${HTTPS:+HTTPS, }ドメイン $SERVER_NAME)"
install -m 644 "$REPO/deploy/nginx/siot-common.conf" /etc/nginx/snippets/siot-common.conf
render "$NGINX_TEMPLATE" > /etc/nginx/sites-available/siot.conf
ln -sf /etc/nginx/sites-available/siot.conf /etc/nginx/sites-enabled/siot.conf
if [[ -L /etc/nginx/sites-enabled/default ]]; then
  rm /etc/nginx/sites-enabled/default   # デフォルトウェルカムページの無効化(原本は sites-available/default に保持)
  log "nginx デフォルトウェルカムページを無効化しました"
fi
nginx -t
systemctl enable nginx >/dev/null
systemctl reload nginx 2>/dev/null || systemctl restart nginx

# ── 6) 動作確認 ───────────────────────────────────────────────────────────
log "バックエンドの起動を待機中(最大90秒)"
for _ in $(seq 1 45); do
  if curl -fsS http://127.0.0.1:8000/ >/dev/null 2>&1; then
    if [[ -n "$HTTPS" ]]; then URL="https://$DOMAIN"; else URL="http://${DOMAIN:-<サーバーパブリック IP>}"; fi
    log "完了 — アクセス URL: $URL"
    [[ -n "$DOMAIN" && -z "$HTTPS" ]] && log "HTTPS は deploy/DEPLOY.md ステップ6(certbot) の後に本スクリプトを再実行すると有効化されます"
    exit 0
  fi
  sleep 2
done
die "バックエンドが応答しません — journalctl -u siot-backend -n 50 でエラーログを確認してください"
