#!/usr/bin/env bash
# 開発用サービス一括起動/停止 — tmux セッション1つにバックエンド・ワーカー・UIを3ペインで起動。
#
# データベースは backend/.env の Supabase 設定を使用。本スクリプトはローカル SQL の
# 起動・管理は行わず、Celery ブローカー/結果ストレージである Redis のみ確認・起動する。
#
#   ./dev.sh            サービス起動(起動済みの場合はその画面にアタッチ)
#   ./dev.sh stop       サービス全停止
#   ./dev.sh restart    停止後に再起動
#   ./dev.sh status     ステータス確認
#   ./dev.sh attach     起動中の画面にアタッチ
#
# tmux 終了方法:
#   ./dev.sh stop               本スクリプトが作成したサービスセッションを正常終了
#   tmux kill-session -t capstone  capstone セッションを直接強制終了
#   tmux kill-server            実行中の全 tmux セッションを終了
#
# コード保存時に自動反映(再起動不要):
#   · バックエンド: backend/app の .py 保存 → uvicorn --reload が再起動
#   · ワーカー    : backend/app · model の .py 保存 → watchfiles が再起動
#              ⚠️ 分析実行中の場合はその分析は中断(「サーバー再起動により中断」→ 再実行)
#   · UI         : Vite が保存即座にホットリロード
#
# tmux 操作: Ctrl+b の後に方向キー = ペイン移動 · Ctrl+b d = デタッチ(サービスは継続稼働)
#            Ctrl+b [ = ログを上にスクロール(q で終了)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="capstone"
PY="$ROOT/venv311/bin"
BACKEND_PORT=8000
UI_PORT=5173

log() { printf '\033[1;36m[dev]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[dev]\033[0m %s\n' "$*" >&2; }

port_in_use() {
  # macOS には ss が標準インストールされていないため lsof を先に使用。
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN
  else
    return 1
  fi
}

ensure_redis() {
  if port_in_use 6379; then
    log "Redis 確認 — localhost:6379 稼働中"
    return 0
  fi

  command -v docker >/dev/null 2>&1 || {
    err "Redis が停止しており Docker が見つかりません。Redis を 6379 ポートで先に起動してください。"
    return 1
  }

  log "Redis 起動"
  if ! (cd "$ROOT" && docker compose up -d redis); then
    err "Redis 起動失敗 — Docker Desktop が起動しているか確認してください。"
    return 1
  fi

  for _ in $(seq 1 20); do
    port_in_use 6379 && return 0
    sleep 0.5
  done
  err "Redis が localhost:6379 で準備完了しませんでした。"
  return 1
}

# ワーカーラッパー: watchfiles が再起動用に送信する SIGINT を Celery の即時終了(SIGQUIT)に
# 変換して伝達。SIGINT をそのまま受信すると Celery は進行中分析の完了を待機する
# 「正常終了」を行い、コード修正の反映が遅延し強制終了時に子プロセスが残存する可能性がある。
run_worker() {
  cd "$ROOT/backend"
  # macOS では PyTorch/ONNX/OpenCV ランタイムを Celery prefork 子プロセスで
  # 初期化する際に SIGSEGV が発生する可能性がある。分析は単一ワーカーで順次処理するため
  # フォークを行わない solo プールを使用してモデルランタイムを1プロセス内で維持。
  "$PY/celery" -A app.worker worker --pool=solo --concurrency=1 --loglevel=info &
  local pid=$!
  trap 'kill -QUIT "$pid" 2>/dev/null || true' INT TERM
  while kill -0 "$pid" 2>/dev/null; do
    wait "$pid" || true
  done
}

start() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "既に実行中です — 画面にアタッチします (離脱: Ctrl+b d)"
    exec tmux attach -t "$SESSION"
  fi

  # 前提条件の確認
  [[ -x "$PY/python" ]] || { err "venv311 が存在しません。先に作成してください。"; exit 1; }
  for p in "$BACKEND_PORT" "$UI_PORT"; do
    if port_in_use "$p"; then
      err "ポート $p は既に他のプログラムが使用しています。停止後に再実行してください。"
      exit 1
    fi
  done
  if [[ ! -d "$ROOT/ui/node_modules" ]]; then
    log "ui/node_modules が存在しないため npm install を先に実行します"
    (cd "$ROOT/ui" && npm install)
  fi

  ensure_redis

  log "tmux セッション '$SESSION' にバックエンド・ワーカー・UI を起動します"
  # 先に空のセッションを作成。バックエンドが即時終了してもセッションを維持してエラーログを確認できるよう
  # 実行コマンドより先に remain-on-exit を設定。
  tmux new-session -d -s "$SESSION" -n services -c "$ROOT/backend"
  # プログラムが異常終了してもペインを閉じずエラーログを確認できるように設定
  tmux set-option -t "$SESSION:services" remain-on-exit on
  tmux set-option -t "$SESSION:services" pane-border-status top
  tmux set-option -t "$SESSION" mouse on
  tmux respawn-pane -k -t "$SESSION:services.0" -c "$ROOT/backend" \
    "exec \"$PY/python\" -m uvicorn app.main:app --port $BACKEND_PORT --reload --reload-dir app"
  tmux select-pane -t "$SESSION:services.0" -T "バックエンド :$BACKEND_PORT (自動再起動)"

  # バックエンドの準備完了前に UI が /api/auth/me を呼び出すと Vite プロキシが
  # ECONNREFUSED を記録する。ワーカーと UI はバックエンドのヘルスチェック成功後に
  # 起動し初期起動順序を保証する。
  log "バックエンドの準備待機中…"
  backend_ready=0
  for _ in $(seq 1 60); do
    if curl -fsS --max-time 1 "http://127.0.0.1:$BACKEND_PORT/" >/dev/null 2>&1; then
      backend_ready=1
      break
    fi
    sleep 0.5
  done
  if [[ "$backend_ready" != "1" ]]; then
    err "バックエンドが30秒以内に準備完了しませんでした。バックエンド画面のログを確認してください。"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    exit 1
  fi
  log "バックエンド準備完了"

  tmux split-window -v -t "$SESSION:services" -c "$ROOT/backend" \
    "exec \"$PY/watchfiles\" --filter python --sigint-timeout 15 '\"$ROOT/dev.sh\" _worker' app ../model"
  tmux select-pane -t "$SESSION:services.1" -T "Celery ワーカー (自動再起動)"

  tmux split-window -v -t "$SESSION:services" -c "$ROOT/ui" "npm run dev -- --port $UI_PORT --strictPort"
  tmux select-pane -t "$SESSION:services.2" -T "UI :$UI_PORT"
  tmux select-layout -t "$SESSION:services" even-vertical

  log "起動完了 — UI: http://localhost:$UI_PORT · バックエンド: http://localhost:$BACKEND_PORT/docs"
  log "停止: ./dev.sh stop · デタッチ: Ctrl+b d"
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
  # 各ペインに Ctrl+C を送信して正常終了を試み、終了を待機した後にセッションを閉じる
  for pane in $(tmux list-panes -t "$SESSION" -F '#{pane_id}'); do
    tmux send-keys -t "$pane" C-c 2>/dev/null || true
  done
  for _ in $(seq 1 20); do
    alive=$(tmux list-panes -t "$SESSION" -F '#{pane_dead}' 2>/dev/null | grep -c 0 || true)
    [[ "$alive" == "0" ]] && break
    sleep 0.5
  done
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  log "停止しました (Supabase/R2 と Redis は継続利用可能です)"
}

status() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "tmux セッション '$SESSION' 実行中"
    tmux list-panes -t "$SESSION" -F '  · #{pane_title} — #{?pane_dead,停止,動作中}'
  else
    log "tmux セッションなし(サービス停止中)"
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$BACKEND_PORT/docs" || true)
  log "バックエンド応答: ${code:-なし}"
  if port_in_use 6379; then
    log "Redis 応答: localhost:6379 受信中"
  else
    log "Redis 応答: 受信なし (Celery タスクは実行不可)"
  fi
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  attach)  exec tmux attach -t "$SESSION" ;;
  _worker) run_worker ;;   # 内部用: watchfiles がワーカーを起動する際に使用
  *) echo "使用法: ./dev.sh [start|stop|restart|status|attach]"; exit 1 ;;
esac
