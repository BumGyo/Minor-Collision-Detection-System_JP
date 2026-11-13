import sys
import os

# model/ フォルダを sys.path に追加
# FastAPIバックエンド等の外部から 'from model.hit_and_run import HitAndRun3DCNN' 形式でインポートする際、
# model/ 内部の絶対インポート(import config 等)が正常に動作するようにする
sys.path.insert(0, os.path.dirname(__file__))
