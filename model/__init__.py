import sys
import os

# model/ ディレクトリを sys.path に追加
# FastAPI バックエンド等から 'from model.hit_and_run import HitAndRun3DCNN' の形式でインポートする際、
# model/ 内部の絶対インポート (import config 等) を正常に動作させる
sys.path.insert(0, os.path.dirname(__file__))
