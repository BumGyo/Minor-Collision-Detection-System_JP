# SIOT - Minor Collision Detection System

ñÒó³íŞ CCTV ç±ßÀª«ªé¡¢ÚªáßŞÀÍº??ªÆÔ±ª²ªÎë÷ª¤ª¬ª¢ªë?Êàªò AI ªÇ?ò±ª·¡¢Î·×âíºª¬ü¬ìãª¹ªÙª­ç±ßÀªòÎíªê?ªà«·«¹«Æ«àªÇª¹¡£

## Ï°à÷

- `ui/`: React + Vite ªÎìíÜâåŞ?? Web UI
- `backend/`: FastAPI API¡¢ó³??ò±¡¢İÂà°«¸«ç«Ö API
- `model/`: ŞÀÍºİÂà°«â«Ç«ëªÈõÏÖå«³?«É
- `weights/`: ?ã§?ªß X3D «â«Ç«ëñìªß
- `docker-compose.yml`: MySQL 8 ªÈ Redis

## ù±é©ü»ÌÑ

- macOS / Linux / WSL2
- Python 3.11
- Node.js ªÈ npm
- Docker Desktop

## ÑÃÔÑÛ°Ûö

```bash
docker compose up -d db redis
python3.11 -m venv venv311
./venv311/bin/python -m pip install -r requirements.txt
./venv311/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

cd backend
../venv311/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Ü¬ªÎ«¿?«ß«Ê«ëªÇ«Õ«í«ó«È«¨«ó«ÉªòÑÃÔÑª·ªŞª¹¡£

```bash
cd ui
npm install
npm run dev -- --host 0.0.0.0
```

«Ö«é«¦«¶: `http://localhost:5173/login`

## «â«Ç«ëªËªÄª¤ªÆ

- ŞÀÍºİÂà°éÄªÎ X3D ñìªßªÏ `weights/` ªËùßªŞªìªÆª¤ªŞª¹¡£
- ó³??ò±éÄªÎ `yolo11x.pt` ªÏ GitHub ªÎ«Õ«¡«¤«ë«µ«¤«ºğ¤ùÚªòù­ª±ªëª¿ªá«ê«İ«¸«È«êªËªÏùßªáªÆª¤ªŞª»ªó¡£`ultralytics` ª¬ôøüŞªÎó³??ò±ãÁªËí»ÔÑ«À«¦«ó«í?«Éª·ªŞª¹¡£
- `storage/`, `outputs/`, `venv311/`, `ui/node_modules/` ªÏ?ú¼ãÁªËßæà÷ªµªìªëª¿ªá Git Î·×â?ßÚèâªÇª¹¡£

## «Ç?«¿«Ù?«¹

MySQL ªÈ Redis ªÏ `docker-compose.yml` ªÇÑÃÔÑª·ªŞª¹¡£«¢«×«ê«±?«·«ç«óªÎïÈ?àâïÒªÏ `backend/app/settings.py` ªò?ğÎª·ªÆª¯ªÀªµª¤¡£
