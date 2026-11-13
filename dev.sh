#!/usr/bin/env bash
# 開発用サービス一括起動/停止スクリプト — tmuxセッション内にバックエンド・ワーカー・UIを起動。
#
#   ./dev.sh            サービス起動 (起動中の場合はアタッチ)
#   ./dev.sh stop       サービス全停止
#   ./dev.sh restart    再起動
#   ./dev.sh status     ステータス確認
#   ./dev.sh attach     起動中セッションへのアタッチ
#
# ホットリロード対応:
#   · バックエンド: backend/app の .py 保存時 → uvicorn --reload が自動再起動
#   · ワーカー    : backend/app · model の .py 保存時 → watchfiles が自動再起動
#   · UI         : Vite による即時ホットリロード
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="capstone"
PY="$ROOT/venv311/bin"
BACKEND_PORT=8000
UI_PORT=5173

log() { printf '\033[1;36m[dev]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[dev]\033[0m %s\n' "$*" >&2; }

port_in_use() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }

run_worker() {
  cd "$ROOT/backend"
  local pid=""
  local pattern="$PY/celery -A app.worker worker"
  trap 'if [ -n "$pid" ]; then kill -QUIT "$pid" 2>/dev/null || true; ( sleep 5; pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null ) & else exit 0; fi' INT TERM
  if pgrep -f "$pattern" >/dev/null; then
    echo "[dev] 残存ワーカーを終了します: $(pgrep -f "$pattern" | tr '\n' ' ')"
    pkill -QUIT -f "$pattern" || true
    sleep 2
    pkill -KILL -f "$pattern" || true
  fi
  "$PY/celery" -A app.worker worker --concurrency=1 --loglevel=info &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    wait "$pid" || true
  done
}

start() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "既に実行中です — 画面にアタッチします (終了: Ctrl+b d)"
    exec tmux attach -t "$SESSION"
  fi

  [[ -x "$PY/python" ]] || { err "venv311 が存在しません。事前に仮想環境を作成してください。"; exit 1; }
  for p in "$BACKEND_PORT" "$UI_PORT"; do
    if port_in_use "$p"; then
      err "ポート $p は既に使用されています。該当プロセスを停止後、再実行してください。"
      exit 1
    fi
  done
  if [[ ! -d "$ROOT/ui/node_modules" ]]; then
    log "ui/node_modules がないため npm install を実行します"
    (cd "$ROOT/ui" && npm install)
  fi

  log "DB・Redis 確認"
  if ! (cd "$ROOT" && docker compose up -d db redis); then
    err "docker compose 実行失敗 — Docker デーモンが起動しているか確認してください。"
    exit 1
  fi

  log "tmux セッション '$SESSION' にバックエンド・ワーカー・UI を起動します"
  tmux new-session -d -s "$SESSION" -n services -c "$ROOT/backend" \
    "$PY/python -m uvicorn app.main:app --port $BACKEND_PORT --reload --reload-dir app"
  tmux set-option -t "$SESSION" remain-on-exit on
  tmux set-option -t "$SESSION" pane-border-status top
  tmux set-option -t "$SESSION" mouse on
  tmux select-pane -t "$SESSION:services.0" -T "バックエンド :$BACKEND_PORT (自動再起動)"

  tmux split-window -v -t "$SESSION:services" -c "$ROOT/backend" \
    "$PY/watchfiles --filter python --sigint-timeout 15 '$ROOT/dev.sh _worker' app ../model"
  tmux select-pane -t "$SESSION:services.1" -T "Celery ワーカー (自動再起動)"

  tmux split-window -v -t "$SESSION:services" -c "$ROOT/ui" "npm run dev -- --port $UI_PORT --strictPort"
  tmux select-pane -t "$SESSION:services.2" -T "UI :$UI_PORT"
  tmux select-layout -t "$SESSION:services" even-vertical

  log "起動完了 — UI: http://localhost:$UI_PORT · バックエンド: http://localhost:$BACKEND_PORT/docs"
  log "停止: ./dev.sh stop · アタッチ解除: Ctrl+b d"
  if [[ -t 1 ]]; then
    exec tmux attach -t "$SESSION"
  fi
}

stop() {
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    log "実行中のサービスはありません"
    return 0
  fi
  log "サービス停止中…"
  for pane in $(tmux list-panes -t "$SESSION" -F '#{pane_id}'); do
    tmux send-keys -t "$pane" C-c 2>/dev/null || true
  done
  for _ in $(seq 1 20); do
    alive=$(tmux list-panes -t "$SESSION" -F '#{pane_dead}' 2>/dev/null | grep -c 0 || true)
    [[ "$alive" == "0" ]] && break
    sleep 0.5
  done
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  log "停止しました (DB・Redis は継続稼働 — 停止するには: docker compose stop)"
}

status() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "tmux セッション '$SESSION' 実行中"
    tmux list-panes -t "$SESSION" -F '  · #{pane_title} — #{?pane_dead,停止,動作中}'
  else
    log "tmux セッションなし (サービス停止中)"
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$BACKEND_PORT/docs" || true)
  log "バックエンド応答: ${code:-なし}"
  (cd "$ROOT" && docker compose ps --format '  · {{.Service}}: {{.State}}' db redis 2>/dev/null) || true
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  attach)  exec tmux attach -t "$SESSION" ;;
  _worker) run_worker ;;
  *) echo "使用法: ./dev.sh [start|stop|restart|status|attach]"; exit 1 ;;
esac
