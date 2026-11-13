# SIOT サーバーデプロイガイド (デモ運用用)

サーバー1台にフロントエンド・バックエンド・分析ワーカー・DB・Redis を集約し、ユーザーはブラウザからアクセスします。

```
ブラウザ ──HTTPS──▶ nginx (80・443)
                      ├─ フロントエンド静的ファイル (/var/www/siot ← ui/dist)
                      └─ /api ──▶ バックエンド uvicorn (127.0.0.1:8000) ──┐
                                                                          ├─▶ MySQL・Redis (docker, ローカル内)
                                  分析ワーカー Celery ────────────────────┘
                                  (動画・クリップ: storage/)
```

- 対象OS: **Ubuntu 22.04 または 24.04** — Oracle Cloud (ARM, CPU推論) / AWS 等 (x86, NVIDIA GPU推論) 共通手順
- デプロイ設定: `deploy/` ディレクトリ内の nginx・systemd 設定を `install.sh` が自動適用
- セキュリティ: 全APIエンドポイントでJWT認証を実施 (メディアファイルは HttpOnly Cookie による保護)。パスワードは PBKDF2 (ソルト + 60万回反復) で暗号化保存
- 動画保持期間: アップロード後90日 (`VIDEO_RETENTION_DAYS`) 経過した動画およびクリップは自動削除 (1時間周期のバッチ処理)

---

## 0. インフラ準備

- **ディスク容量**: 動画蓄積のため 100GB 以上推奨
- **ポート開放**: 22 (SSH), 80 (HTTP), 443 (HTTPS)
  - Oracle Cloud: VCN セキュリティリスト + サーバー内 iptables の双方を開放
    ```bash
    sudo iptables -I INPUT -p tcp -m multiport --dports 80,443 -m state --state NEW -j ACCEPT
    sudo netfilter-persistent save
    ```
  - AWS: セキュリティグループのインバウンド設定
- **ドメイン** (HTTPS用): ドメインの A レコードをサーバーのパブリック IP に設定

## 1. パッケージインストール (sudo)

```bash
sudo apt update
sudo apt install -y git curl rsync nginx certbot python3-certbot-nginx ffmpeg libgl1 libglib2.0-0
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# フロントエンドビルド用 Node.js 22
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

GPU サーバーの場合: NVIDIA ドライバーのインストールが必要です (`nvidia-smi` で確認可能)。

## 2. リポジトリ取得と Python 環境構築

```bash
git clone <リポジトリURL> ~/capstone-26 && cd ~/capstone-26
git checkout <デプロイブランチ>

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv --python 3.11 venv311

# PyTorch インストール (環境に応じたコマンドを選択)
uv pip install --python venv311 --index-url https://download.pytorch.org/whl/cpu   torch==2.12.0 torchvision==0.27.0   # CPU サーバー
uv pip install --python venv311 --index-url https://download.pytorch.org/whl/cu126 torch==2.12.0 torchvision==0.27.0   # NVIDIA GPU サーバー

uv pip install --python venv311 -r requirements.txt
```

## 3. モデル重みファイルの配置

```bash
# ローカルPCからサーバーへ転送
scp weights/hitandrun_x3d_260922_33ep_earlyY_ptY_0.1999.pth <ユーザー>@<サーバー>:~/capstone-26/weights/
scp yolo11x.pt <ユーザー>@<サーバー>:~/capstone-26/
```

## 4. 環境変数ファイル `.env` の設定

```bash
cp deploy/env.example .env
nano .env
```

| 設定キー | 内容 |
|---|---|
| `DOMAIN` | ドメイン名 (未所持の場合は空欄) |
| `MYSQL_ROOT_PASSWORD`, `REDIS_PASSWORD` | パスワード文字列 |
| `AUTH_SECRET_KEY` | `openssl rand -hex 32` の生成値 |

## 5. インストール・デプロイ実行

```bash
sudo ./deploy/install.sh
```

実行内容: フロントエンドビルド → `/var/www/siot` へ配置 → Docker (DB・Redis) 起動 → バックエンド・ワーカーサービス登録・起動 → nginx 設定適用。

## 6. HTTPS 化 (ドメイン設定時)

```bash
sudo certbot certonly --nginx -d <ドメイン> --deploy-hook "systemctl reload nginx"
sudo ./deploy/install.sh
```

## 7. 稼働確認

```bash
systemctl status siot-backend siot-worker nginx
journalctl -u siot-backend -f        # バックエンドログ
journalctl -u siot-worker -f         # 分析ワーカーログ
```

## 8. コード更新手順

```bash
cd ~/capstone-26 && git pull
uv pip install --python venv311 -r requirements.txt
sudo ./deploy/install.sh
```

## トラブルシューティング

| 現象 | 確認事項 |
|---|---|
| 接続できない | ポート開放状況 (80, 443), `systemctl status nginx` |
| 502 Bad Gateway | バックエンドプロセスの状態 → `journalctl -u siot-backend -n 50` |
| DB Access denied | `.env` の MySQL パスワードの不一致 |
| 分析が PENDING のまま進まない | `journalctl -u siot-worker -n 50` |
| アップロード時 413 エラー | nginx の `client_max_body_size` 設定確認 |
| `libGL.so.1` エラー | `sudo apt install -y libgl1 libglib2.0-0` |
| サムネイル・動画が表示されない (401) | クッキー有効期限切れ。再度ログインを実行 |
