# SIOT サーバーデプロイ手順 (デモ環境用)

UI・バックエンド・分析ワーカーを起動し、Supabase/Azure PostgreSQL と Redis URL、Cloudflare R2(S3互換)またはローカルストレージを環境変数で選択する。

```
ブラウザ ──HTTPS──▶ nginx (80・443)
                     ├─ UI 静的ファイル (/var/www/siot ← ui/dist)
                     └─ /api ──▶ バックエンド uvicorn (127.0.0.1:8000) ──┐
                                                                         ├─▶ PostgreSQL (Supabase/Azure または MySQL)
                                 分析ワーカー Celery ────────────────────┘
                                 (Redis URL, 動画・クリップ: R2 または storage/)
```

- 対象環境: **Ubuntu 22.04 または 24.04** — Oracle Cloud(ARM, CPU 推論) / AWS 等(x86, NVIDIA GPU 推論) 共通手順
- サーバー配置ファイル: 本ディレクトリ(`deploy/`)の nginx・systemd 設定を `install.sh` が自動配置
- 開発用 `dev.sh` はそのまま使用可能(サーバー設定は `.env` が存在する場合のみ適用)
- セキュリティ: 全 API で認証を検証(動画・サムネイル・クリップはログイン時に発行される HttpOnly Cookie により自身のリソースのみアクセス可能)。
  パスワードは PBKDF2(ソルト+60万回)でハッシュ化し、nginx が同一 IP からのログイン・登録を毎分10回に制限
- 動画保存期間: アップロードから90日(`VIDEO_RETENTION_DAYS`)経過した動画および事故クリップ・原本クリップ・サムネイル・分析履歴を
  自動削除(バックエンド起動時に1回、以降1時間ごと)。リストカード上に「○日後に失効」と表示

---

## 0. インフラ構築時の準備

- **ディスク容量**: 動画が蓄積されるため 100GB 以上を推奨
- **ポート開放**: 22(SSH), 80, 443 を開放
  - Oracle: VCN セキュリティリストに 80・443 のインバウンドルールを追加 **+ サーバー内部ファイアウォールも** 開放必須(Oracle Ubuntu イメージは SSH 以外初期拒否設定)
    ```bash
    sudo iptables -I INPUT -p tcp -m multiport --dports 80,443 -m state --state NEW -j ACCEPT
    sudo netfilter-persistent save
    ```
  - AWS: セキュリティグループのインバウンドに 22(自身のIP), 80, 443 を設定
- **ドメイン**(HTTPS 用): DuckDNS 等の無料 DNS でサーバーのパブリック IP を紐付ける。未設定時は `http://<IP>` でのアクセス(分析完了のブラウザ通知は HTTPS 環境のみ動作)

## 1. 依存パッケージのインストール (sudo)

```bash
sudo apt update
sudo apt install -y git curl rsync nginx certbot python3-certbot-nginx ffmpeg libgl1 libglib2.0-0
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER          # 再ログイン後に sudo なしで docker 実行可能

# UI ビルド用 Node 22 — サーバー上でビルドする場合のみ。(ローカルで npm run build した ui/dist をコピーしても可)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

GPU サーバーのみ: NVIDIA ドライバの導入が必須 — `nvidia-smi` で GPU が正常認識されることを確認
(AWS Deep Learning AMI は導入済み。未導入時は `sudo ubuntu-drivers install` 後に再起動)。

## 2. ソースコードの取得と Python 環境構築

```bash
git clone <リポジトリURL> ~/capstone-26 && cd ~/capstone-26
git checkout <デプロイ対象ブランチ>

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv --python 3.11 venv311

# torch を先行インストール — サーバー環境に適合する1行のみ実行
uv pip install --python venv311 --index-url https://download.pytorch.org/whl/cpu   torch==2.12.0 torchvision==0.27.0   # CPU サーバー(Oracle ARM 等)
uv pip install --python venv311 --index-url https://download.pytorch.org/whl/cu126 torch==2.12.0 torchvision==0.27.0   # NVIDIA GPU (nvidia-smi の CUDA Version 12.6以上。13.0以上なら cu130 も対応)

uv pip install --python venv311 -r requirements.txt
```

torch を先行インストールすることで `requirements.txt` 実行時に別バージョンの torch(数GB)の重複ダウンロードを防止。
2026-10-18 確認済み: Ubuntu 22.04 以上であれば ARM・x86 双方とも必要なパッケージが提供されています(torch 2.12.0 に cu128 版は未提供)。

## 3. 重みファイルの配置 (ローカル PC → サーバー)

git 管理外のファイルであるため手動で転送します。

```bash
# ローカル PC の capstone-26 ディレクトリで実行
scp weights/hitandrun_x3d_260922_33ep_earlyY_ptY_0.1999.pth <アカウント>@<サーバー>:~/capstone-26/weights/
scp yolo11x.pt <アカウント>@<サーバー>:~/capstone-26/        # 未配置時は初回の車両検知時に自動ダウンロード
```

別の重みファイルを使用する場合は転送後に `.env` の `SERVICE_WEIGHTS` にパスを指定。

## 4. 設定ファイル `.env` の構成

```bash
cp deploy/env.example .env
nano .env
```

| 項目 | 設定値 |
|---|---|
| `DOMAIN` | ドメイン名(未設定時は空) |
| `DATABASE_URL` | Supabase/Azure PostgreSQL URL。空の場合は loopback 専用ローカル MySQL を使用 |
| `REDIS_URL` | ローカル Redis または Azure Cache URL。Azure は `rediss://` と TLS ポートを使用 |
| `STORAGE_BACKEND` | `s3`(Cloudflare R2/S3) または `local`。R2 は endpoint・bucket・各アクセスキーを全設定 |
| `AUTH_SECRET_KEY` | `openssl rand -hex 32` の生成結果 |

URL内にパスワードを含める際、`@`, `#`, `/`, `:` 等の記号は percent-encode が必要です。ローカル MySQL・Redis コンテナは loopback アドレスにのみポートを開放します。

## 5. インストール — 初回およびコード更新時共通の単一コマンド

```bash
sudo ./deploy/install.sh
```

処理内容: UI ビルド → `/var/www/siot` コピー → Redis(docker) 起動 → バックエンド・ワーカーを systemd 登録・再起動 →
nginx 設定・再起動 → バックエンドヘルスチェック。完了時にアクセス URL を表示。
インストール前に生成される設定ファイルのみを確認する場合: `./deploy/install.sh --dry-run /tmp/siot-check`。

推論ランナーの事前ベンチマーク(未実行でも初回分析時に自動実行 — 事前実行で初回分析を高速化):

```bash
set -a; . ./.env; set +a
venv311/bin/python -m model.main --mode check-backend
```

## 6. HTTPS 化 (ドメイン運用時)

```bash
sudo certbot certonly --nginx -d <ドメイン> --deploy-hook "systemctl reload nginx"
sudo ./deploy/install.sh        # 証明書を検出し HTTPS 設定へ自動切り替え(HTTP は HTTPS へリダイレクト)
```

証明書の自動更新は certbot により自動実行されます。

## 7. 動作確認チェックリスト

- サーバー外部(携帯電話の4G/5G回線等)からのアクセス → ユーザー登録 → アップロード → 車両指定 → 分析 → クリップ再生
- Windows(Chrome・Edge), Mac(Safari) での動画再生
- `sudo reboot` 後の自動復旧確認

```bash
systemctl status siot-backend siot-worker nginx
journalctl -u siot-backend -f        # バックエンドログ確認
journalctl -u siot-worker -f         # 分析ワーカーログ確認
sudo tail -f /var/log/nginx/error.log
```

## 8. コード更新手順

```bash
cd ~/capstone-26 && git pull
uv pip install --python venv311 -r requirements.txt     # requirements.txt 変更時のみ
sudo ./deploy/install.sh
```

実行中だった分析はサービス再起動時に中断され「失敗」に遷移します — UI から再実行してください。

### 既存動画のサムネイルキャッシュ事前生成

既存の大サイズ JPEG を最大 640×360 プレビューにリサイズしサーバーキャッシュに事前投入する場合に実行。

```bash
cd ~/capstone-26/backend
../venv311/bin/python -B -m app.backfill_thumbnails
```

プレビューは R2 の `thumbnails/preview-v1/` およびサーバーの
`STORAGE_DIR/thumbnails/preview-v1/` に保存されます。新規アップロード時は自動生成され、
サーバーキャッシュが空の場合は初回リクエスト時に復元されます。動画削除・保存期間満了クリーンアップ時に同期削除。
サムネイル API はリクエストごとに所有権を検証し、ブラウザの ETag が一致する場合は 304 で再利用。
R2 接続はアクティブタスクが存在せず2分以上アイドル状態の場合、次回タスク実行前に再接続します。

## トラブルシューティング

| 現象 | 確認項目 |
|---|---|
| 接続できない | ステップ0のポート開放(Oracle はセキュリティリスト + OSファイアウォール双方が必要), `systemctl status nginx` |
| 502 Bad Gateway | バックエンド停止 → `journalctl -u siot-backend -n 50` |
| PostgreSQL 接続失敗 | DATABASE_URL のパスワードエンコード・SSL 設定、Supabase/Azure ファイアウォール許可 IP を確認 |
| R2 アップロード・再生失敗 | STORAGE_BACKEND=s3, endpoint・bucket・Access Key・Secret Key を確認 |
| 分析が「待機」から遷移しない | `journalctl -u siot-worker -n 50` |
| アップロード 413 | nginx 設定が未反映 → `sudo ./deploy/install.sh` 再実行。Cloudflare 経由時は Free プラン 100MB 制限に注意 |
| `libGL.so.1` エラー | ステップ1の `libgl1 libglib2.0-0` をインストール |
| 分析が遅い | CPU サーバーは推論負荷が高くなります。`check-backend` の選定ランナーおよび `journalctl -u siot-worker` の処理速度を確認 |
| ログイン時に「リクエストが多すぎます」 | 同一 IP からの短時間での連続試行 — 1分程度待機すると解除されます(siot-http/https.conf の rate 設定) |
| サムネイル・動画が表示されない(401) | 動画 Cookie の欠落 — 再ログイン。HTTPS で発行された Cookie は HTTP 接続では送信されません |

## デモ後の追加推奨タスク

- DB バックアップ(`docker exec ... mysqldump`)と `storage/` バックアップの自動化
- アップロード容量制限の実装(認証済みユーザーによるディスク圧迫防止) · ユーザー登録の招待制化
