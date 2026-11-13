import argparse
from pathlib import Path

import torch

import config
from device_utils import get_device, is_channels_last_3d_supported
from hitandrun_model import HitAndRun3DCNN, check_weights

'''
学習開始
python3 -m model.main --mode train                 # A クリップ方式 pre (デフォルト)
python3 -m model.main --mode train --a-mode span   # A クリップ方式 span (config の注記参照)
python3 -m model.main --mode train --scratch       # 事前学習なしのランダム初期化 (scratch)
python3 -m model.main --mode train --scratch --lr 1e-3   # 学習率の直接指定
python3 -m model.main --mode train --balance outputs/s_motion/<日時>/clips.csv   # 空シーン正常の間引き + バッチ毎の衝突均等配分 (名前 _bal)
python3 -m model.main --mode train --balance outputs/s_motion/<日時>/clips.csv --dry-run   # バッチ構成のみ出力 (学習は非実行)

評価開始 (--weights 未指定時は config.EVAL_WEIGHTS_PATH)
python3 -m model.main --mode eval --weights weights/hitandrun_x3d_..._ptY_span_0.1234.pth
python3 -m model.main --mode eval --weights weights/....pth --r 1.5   # クロップ r のみを変更して評価 (今回のみ、結果名 _r1.5)

評価推奨エポックの提示 — 学習実行回フォルダの曲線 CSV から (学習中・中断後も利用可能)。エポック別重みはそのフォルダの epochs/
python3 -m model.main --mode suggest --run outputs/trainlogs/<実行回フォルダ>

推論実行バックエンドの自己診断 — サーバー設置直後に1回実行 (--weights 未指定時はサービス用重み)
python3 -m model.main --mode check-backend

単一動画の分析 — 本番サービスと同様の処理 (事故の疑いがある区間検出 + クリップ抽出)。結果は outputs/detect/<日時>_<動画名>/
python3 -m model.main --mode detect --video data/eval/260828_TA_0001.mp4          # ボックスは同名 .txt の0番車両
python3 -m model.main --mode detect --video video.mp4 --bbox 101,45,198,153         # ボックス直接指定 (元解像度 x1,y1,x2,y2)

予測開始 (開発用: config の動画全体に CAM 合成)
python3 -m model.main --mode predict
'''

def _load_model(weights_path):
    # _load_model は predict/eval (推論) でのみ使用 → INFER デバイスを使用
    check_weights(weights_path)              # 学習時の入力サイズ (_s256)・ヘッドプーリング (_top3) が現在の設定と一致している必要がある
    device = get_device(config.INFER_DEVICE_TYPE)
    model = HitAndRun3DCNN(num_classes=config.MODEL_NUM_CLASSES).to(device)
    if is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last_3d)
    try:
        state_dict = torch.load(
            weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        print(f"重みの読み込み完了: {weights_path}")
    except FileNotFoundError:
        print(f"エラー: 重みファイルが見つかりません -> {weights_path}")
        raise
    return model, device


def run_train(lr=None, dry_run=False):
    from train import train_model
    if not config.DATA_DIR.exists():
        print(f"エラー: データフォルダが見つかりません -> {config.DATA_DIR}")
        return
    print(f"データディレクトリ確認完了: {config.DATA_DIR}. "
          f"{'バッチ構成のみ確認します (--dry-run)' if dry_run else '学習を開始します!'}")
    train_model(learning_rate=lr, save_path=config.TRAIN_BEST_MODEL_SAVE_PATH, dry_run=dry_run)


def run_predict():
    from inference import predict_full_video
    model, _ = _load_model(config.PREDICT_WEIGHTS_PATH)
    out_path, events = predict_full_video(model)
    if out_path is None:
        print("推論失敗: 結果動画が生成されませんでした。")
        return
    if events:
        print(f"\n[イベント要約] 衝突 {len(events)}件検出")
        for i, ev in enumerate(events, 1):
            print(f"  イベント {i}: start_frame={ev['start_frame']}, "
                  f"end_frame={ev['end_frame']}")


def run_detect(video, bbox_arg=None, r=None):
    """単一動画を本番サービスと同様に分析する — 事故の疑いがある区間を検出し、その区間のクリップを生成する。

    r を指定するとクロップ倍率のみを変更して分析 (未指定時は config.R_VALUE)。
    """
    from datetime import datetime
    from inference import predict_events_and_clips
    video = Path(video)
    if not video.exists():
        print(f"エラー: 動画が見つかりません -> {video}")
        return
    if bbox_arg:
        bbox = [int(float(v)) for v in bbox_arg.split(',')]
        if len(bbox) != 4:
            print("エラー: --bbox は x1,y1,x2,y2 の4つの数値である必要があります (例: --bbox 101,45,198,153)")
            return
    else:
        txt = video.with_suffix('.txt')
        boxes = []
        if txt.exists():
            for line in txt.read_text(encoding='utf-8', errors='ignore').splitlines():
                p = [t.strip() for t in line.split(',')]
                if len(p) >= 6 and p[0] == 'car' and p[1] == '0':
                    boxes.append([int(float(v)) for v in p[2:6]])
        if not boxes:
            print(f"エラー: 事故車両ボックスがありません。{txt.name} に 'car,0,x1,y1,x2,y2' 行を記載するか "
                  "--bbox x1,y1,x2,y2 で指定してください")
            return
        bbox = boxes[0]
    model, _ = _load_model(config.DETECT_WEIGHTS_PATH)
    r_value = config.R_VALUE if r is None else r
    r_tag = '' if abs(r_value - 1.0) < 1e-9 else f'_r{r_value:g}'
    out_dir = (Path(config._ROOT) / 'outputs' / 'detect'
               / f"{datetime.now():%y%m%d_%H%M%S}_{video.stem}{r_tag}")
    print(f"分析: {video} | 事故車両ボックス {bbox} | クロップ r {r_value:g} | 出力フォルダ {out_dir}")
    events = predict_events_and_clips(model, video_path=str(video), bbox=tuple(bbox),
                                      output_dir=out_dir, r_value=r_value)
    print(f"\n[結果] 事故の疑いがある区間 {len(events)}件")
    for i, e in enumerate(events, 1):
        print(f"  {i}. {e['start_sec']:.2f}秒 ~ {e['end_sec']:.2f}秒 (フレーム {e['start_frame']}~{e['end_frame']}), "
              f"衝突確率 {e['crash_prob']:.3f}\n     クリップ: {e.get('clip_path')}")


def run_check_backend():
    """本番サービスの推論実行バックエンドを現在のマシンで再診断し、結果を保存する。

    サービスワーカーは初回の分析時に同一の診断を実行するため、未実行でも動作自体は可能。事前に実施しておくと
    初回分析が診断時間分高速化される。重み・パッケージ・ハードウェア変更時にも再実行可能。
    """
    from infer_runtime import get_runner
    model, device = _load_model(config.CHECK_WEIGHTS_PATH)
    runner = get_runner(model, force_check=True)
    print(f"\n本番サービス推論バックエンド: {runner.name} (デバイス {device}) — キャッシュ保存先 {config.INFER_CACHE_DIR}")


def run_eval(r=None):
    """r を指定するとクロップ倍率のみを変更して評価 (従来・サービス判定の双方、今回のみ適用)。"""
    from evaluate import evaluate_folder_accuracy
    model, _ = _load_model(config.EVAL_WEIGHTS_PATH)
    evaluate_folder_accuracy(model, **({} if r is None else {'r_value': r}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hit-and-Run 3D-CNN 実行スクリプト")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "predict", "eval", "check-backend", "detect", "suggest"],
        help="実行モード: train(学習) / predict(単一動画CAM予測) / eval(精度評価) / check-backend(バックエンド診断) / detect(動画分析) / suggest(エポック推奨)",
    )
    parser.add_argument(
        "--a-mode",
        choices=["pre", "span"],
        help="学習専用。衝突(A)クリップ方式 — pre(デフォルト): 衝突動画あたり1ウィンドウ / "
             "span: 接近〜衝突終了を複数ウィンドウで網羅的に学習。未指定時は config.TRAIN_A_MODE",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="学習専用。Kinetics-400 事前学習なしでランダム初期化から学習 (重み名 ptN)。"
             "学習率のデフォルト値は config.TRAIN_SCRATCH_LEARNING_RATE",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="学習専用。学習率 (ヘッド基準)。未指定時は事前学習 config.TRAIN_LEARNING_RATE、"
             "scratch config.TRAIN_SCRATCH_LEARNING_RATE",
    )
    parser.add_argument(
        "--balance",
        metavar="CLIPS_CSV",
        help="学習専用。model/s_motion_check の clips.csv で空シーンの正常を間引き (エポックごとに1/3を巡回利用) "
             "衝突をバッチごとに均等配分。名前に _bal (config.TRAIN_S_MOTION_CSV・TRAIN_SPREAD_A)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="学習専用。データ分割・バッチ構成のみ出力し学習は行わない",
    )
    parser.add_argument(
        "--r",
        type=float,
        help="eval・detect 専用。クロップ倍率 r (正方形クロップの1辺 = 車両ボックスの長辺 × r)。"
             "未指定時は config.R_VALUE。今回の実行にのみ適用され、結果名に _r<値> が付加される (1 以外の時)",
    )
    parser.add_argument("--video", help="detect 専用。分析対象の動画ファイル")
    parser.add_argument("--bbox", help="detect 専用。事故車両ボックス x1,y1,x2,y2 (未指定時は同名 .txt の0番車両)")
    parser.add_argument(
        "--weights",
        help="eval・predict 専用。使用する重みファイル (未指定時は config の EVAL/PREDICT_WEIGHTS_PATH)",
    )
    parser.add_argument(
        "--run",
        help="suggest 専用。学習実行回フォルダ (outputs/trainlogs/<実行回>) — 評価推奨エポックを提示",
    )
    args = parser.parse_args()
    if args.mode == "suggest" and not args.run:
        parser.error("--mode suggest には --run 学習実行回フォルダが必要です。")
    if args.run and args.mode != "suggest":
        parser.error("--run は --mode suggest にのみ使用します。")
    if args.a_mode and args.mode != "train":
        # 推論 (評価・本番サービス) は A 方式と無関係 — span で学習した重みを --weights で指定する
        parser.error("--a-mode は学習(--mode train)にのみ使用します。評価対象の重みは --weights で指定してください。")
    if args.weights and args.mode == "train":
        parser.error("--weights は eval・predict・check-backend・detect にのみ使用します。")
    if args.mode == "detect" and not args.video:
        parser.error("--mode detect には --video 動画パス が必要です。")
    if (args.video or args.bbox) and args.mode != "detect":
        parser.error("--video・--bbox は --mode detect にのみ使用します。")
    if (args.scratch or args.lr is not None) and args.mode != "train":
        parser.error("--scratch・--lr は学習(--mode train)にのみ使用します。")
    if args.r is not None and args.mode not in ("eval", "detect"):
        parser.error("--r は評価(--mode eval)・動画分析(--mode detect)にのみ使用します。学習の r は config で設定します。")
    if args.r is not None and args.r <= 0:
        parser.error("--r は 0 より大きい値である必要があります (通常 1〜3)。")
    if (args.balance or args.dry_run) and args.mode != "train":
        parser.error("--balance・--dry-run は学習(--mode train)にのみ使用します。")
    if args.balance:
        config.TRAIN_S_MOTION_CSV = args.balance
        config.TRAIN_SPREAD_A = True
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr は 0 より大きい値である必要があります。")
    if args.scratch:
        # 識別タグ (ptN) も同時に更新 — config 読み込み時に既に確定している値であるため
        config.PRETRAINED = False
        config.PRETRAIN_TAG = "ptN"
    if args.a_mode:
        config.TRAIN_A_MODE = args.a_mode
    config.CHECK_WEIGHTS_PATH = config.DETECT_WEIGHTS_PATH = config.SERVICE_WEIGHTS_PATH
    if args.weights:
        config.DETECT_WEIGHTS_PATH = Path(args.weights)
        config.EVAL_WEIGHTS_PATH = config.PREDICT_WEIGHTS_PATH = Path(args.weights)
        config.CHECK_WEIGHTS_PATH = Path(args.weights)

    if args.mode == "train":
        run_train(args.lr, dry_run=args.dry_run)
    elif args.mode == "predict":
        run_predict()
    elif args.mode == "eval":
        run_eval(args.r)
    elif args.mode == "check-backend":
        run_check_backend()
    elif args.mode == "detect":
        run_detect(args.video, args.bbox, args.r)
    elif args.mode == "suggest":
        from epoch_suggest import suggest
        for line in suggest(args.run)[1]:
            print(line)
