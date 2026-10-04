# SIOT - Minor Collision Detection System

駐車場 CCTV 映像から、物損事故・当て逃げの疑いがある区間を AI で検知し、管理者が確認すべき映像を絞り込むシステムです。

## 構成

- `ui/`: React + Vite の日本語対応 Web UI
- `backend/`: FastAPI API、車両検知、分析ジョブ API
- `model/`: 事故分析モデルと推論コード
- `weights/`: 学習済み 3D Inception モデル重み
- `docker-compose.yml`: MySQL 8 と Redis

## 必要環境

- macOS / Linux / WSL2
- Python 3.11
- Node.js と npm
- Docker Desktop

## 起動方法

```bash
docker compose up -d db redis
python3.11 -m venv venv311
./venv311/bin/python -m pip install -r requirements.txt
./venv311/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

cd backend
../venv311/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

別のターミナルでフロントエンドを起動します。

```bash
cd ui
npm install
npm run dev -- --host 0.0.0.0
```

ブラウザ: `http://localhost:5173/login`

## モデルについて

- 事故分析用の 3D Inception 重みは `weights/` に含まれています。
- 車両検知用の `yolo11x.pt` は GitHub のファイルサイズ制限を避けるためリポジトリには含めていません。`ultralytics` が初回の車両検知時に自動ダウンロードします。
- `storage/`, `outputs/`, `venv311/`, `ui/node_modules/` は実行時に生成されるため Git 管理対象外です。

## データベース

MySQL と Redis は `docker-compose.yml` で起動します。アプリケーションの接続設定は `backend/app/settings.py` を参照してください。
