"""GPU FP16 推論確認 (学習サーバー用) — FP32 と比較して速度と結果の差異を測定する。

    CUDA_VISIBLE_DEVICES=0 python -u -m model.fp16_check               # 速度 + 実際のウィンドウ確率の比較
    CUDA_VISIBLE_DEVICES=0 python -u -m model.fp16_check --full-eval   # + 評価セットを FP32・FP16 でそれぞれ評価

結果は画面および outputs/fp16_check/<日時>/report.txt に出力される。--full-eval の評価ログは
通常通り outputs/evallogs に2つの実行回として記録される。

FP16 を有効化できる基準 (3項目すべて):
  ① 実際のウィンドウで判定閾値 (A_THRESHOLD)・粗探索閾値 (PREDICT_COARSE_PROB) の判定が反転したウィンドウがないこと
  ② --full-eval で従来の判定・サービス判定の指標が FP32 と一致すること
  ③ 速度向上が確認できること
config.INFER_FP16 はデフォルト有効 (2026-10-28 RTX 3080 で検証済み)。他の GPU で結果が異なる場合は本番サービス環境で INFER_FP16=0 を指定。
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from device_utils import is_channels_last_3d_supported
from evaluate import _Tee, evaluate_folder_accuracy
from main import _load_model


def _windows_of(video, txt, n, clip_length, resize):
    """1つの動画から均等に n 個のウィンドウ (正規化前 uint8) — 0番車両ボックス r=1 正方形クロップ (サービスと同様)。"""
    box = None
    for line in open(txt, encoding='utf-8', errors='ignore'):
        p = [t.strip() for t in line.split(',')]
        if len(p) >= 6 and p[0] == 'car' and p[1] == '0':
            box = list(map(int, map(float, p[2:6])))
    if box is None:
        return None, None
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
    cap, frames = cv2.VideoCapture(video), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w, _ = f.shape
        a1, b1, a2, b2 = cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2
        c = f[max(0, b1):min(h, b2), max(0, a1):min(w, a2)]
        c = np.pad(c, ((max(0, -b1), max(0, b2 - h)), (max(0, -a1), max(0, a2 - w)), (0, 0)))
        frames.append(cv2.resize(c, resize))
    cap.release()
    if len(frames) < clip_length:
        return None, None
    frames = np.stack(frames)
    starts = np.unique(np.linspace(0, len(frames) - clip_length, n).round().astype(int))
    return frames, starts


def main():
    ap = argparse.ArgumentParser(description='GPU FP16 推論確認')
    ap.add_argument('--weights', default=str(config.EVAL_WEIGHTS_PATH), help='確認対象の重み (デフォルト: 評価用重み)')
    ap.add_argument('--per-video', type=int, default=24, help='動画ごとに比較するウィンドウ数')
    ap.add_argument('--full-eval', action='store_true', help='評価セットを FP32・FP16 でそれぞれ評価して指標を比較')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('GPU(CUDA)が利用できないため FP16 を確認できません。GPU サーバーで実行してください。')
        sys.exit(1)

    out_dir = Path(config._ROOT) / 'outputs' / 'fp16_check' / datetime.now().strftime('%y%m%d_%H%M')
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / 'report.txt')
    sys.stdout = tee
    try:
        _run(args)
    finally:
        sys.stdout = tee.stdout
        tee.close()
    print(f'\n[ログ] {out_dir / "report.txt"}')


def _run(args):
    config.INFER_DEVICE_TYPE = 'cuda'
    config.EVAL_WEIGHTS_PATH = Path(args.weights)   # --full-eval の評価ログにこの重み名が残るように設定
    model, device = _load_model(args.weights)
    model.eval()
    cap = torch.cuda.get_device_capability(device)
    print(f'GPU: {torch.cuda.get_device_name(device)} (演算能力 {cap[0]}.{cap[1]}) | torch {torch.__version__} | '
          f'CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}')
    print(f'重み: {args.weights}')
    if cap < (7, 0):
        print('⚠️ 演算能力 7.0 未満 (Tensor Core なし) — FP16 による高速化が見込めないため、サービスはこの GPU では FP16 を有効化しません。')
    cl = is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST
    # 本番サービス・評価と同様に cuDNN の最速アルゴリズムをベンチマーク探索させる。
    # 無効のまま速度測定を先行させると、その時点で決定された低速な方式が記憶され、以降の --full-eval に影響する。
    torch.backends.cudnn.benchmark = True
    print(f'cuDNN benchmark 有効 ・ channels_last {"有効" if cl else "無効"}')

    def run(x, fp16, tf32=True):
        old = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
                return model(x).float()
        finally:
            torch.backends.cudnn.allow_tf32 = old

    shape = (3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0])

    # ── ① 速度 ──
    print('\n[速度] 秒あたり処理ウィンドウ数 (サービスバッチ = '
          f'{config.PREDICT_INFER_BATCH_SIZE_GPU}, 評価バッチ = {config.EVAL_INFER_BATCH_SIZE})')
    print(f"{'バッチ':>6} {'FP32':>9} {'FP16':>9} {'FP16/FP32':>10}")
    for bs in (1, 2, 4, 8, 16):
        x = torch.randn((bs, *shape), device=device)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        res = {}
        for fp16 in (False, True):
            for _ in range(3):
                run(x, fp16)
            torch.cuda.synchronize()
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 3.0 or n < 5:
                run(x, fp16)
                n += 1
            torch.cuda.synchronize()
            res[fp16] = n * bs / (time.perf_counter() - t0)
        print(f'{bs:>6} {res[False]:>9.1f} {res[True]:>9.1f} {res[True] / res[False]:>9.2f}x')

    # ── ② 実際のウィンドウ結果比較 ──
    mean = torch.tensor(config.NORM_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD, device=device).view(1, 3, 1, 1, 1)
    folder = Path(config.EVAL_FOLDER_PATH)
    vids = sorted(p for p in folder.glob('*.mp4') if p.with_suffix('.txt').exists()
                  and p.stem not in (getattr(config, 'EXCLUDE_VIDEOS', ()) or ()))   # 10fps 未満の動画は除外
    print(f'\n[結果比較] {folder} 動画 {len(vids)}本 × 最大 {args.per_video}ウィンドウ, FP32 基準')
    all32, all16, allstrict, flips, per_video = [], [], [], [], []
    for v in vids:
        frames, starts = _windows_of(str(v), str(v.with_suffix('.txt')), args.per_video,
                                     config.CLIP_LENGTH, tuple(config.RESIZE))
        if frames is None:
            continue
        p32, p16, pst = [], [], []
        for i in range(0, len(starts), 8):
            s = starts[i:i + 8]
            u8 = torch.from_numpy(np.stack([frames[k:k + config.CLIP_LENGTH] for k in s])).to(device)
            x = (u8.permute(0, 4, 1, 2, 3).float().div(255) - mean) / std
            if cl:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            p32.append(F.softmax(run(x, False), 1)[:, 1].cpu())
            p16.append(F.softmax(run(x, True), 1)[:, 1].cpu())
            pst.append(F.softmax(run(x, False, tf32=False), 1)[:, 1].cpu())
        p32, p16, pst = torch.cat(p32), torch.cat(p16), torch.cat(pst)
        all32.append(p32); all16.append(p16); allstrict.append(pst)
        d = (p16 - p32).abs()
        per_video.append((float(d.max()), v.name))
        for k, s in enumerate(starts):
            for th in (config.A_THRESHOLD, config.PREDICT_COARSE_PROB):
                if (p32[k] > th) != (p16[k] > th):
                    flips.append((v.name, int(s), th, float(p32[k]), float(p16[k])))
    p32, p16, pst = torch.cat(all32), torch.cat(all16), torch.cat(allstrict)
    d16, dtf = (p16 - p32).abs(), (pst - p32).abs()
    print(f'  ウィンドウ数: {len(p32)}個')
    print(f'  FP16 vs FP32       : P(A) 差分 最大 {d16.max():.2e} / 平均 {d16.mean():.2e}')
    print(f'  (参考) TF32無効 FP32 vs 現行 FP32 : 最大 {dtf.max():.2e} / 平均 {dtf.mean():.2e}  '
          f'← GPU FP32 もデフォルトで TF32 のため既に同程度の誤差が存在する')
    print('  差分が大きい動画: ' + ', '.join(f'{n}({m:.1e})' for m, n in sorted(per_video, reverse=True)[:5]))
    f05 = [f for f in flips if f[2] == config.A_THRESHOLD]
    f03 = [f for f in flips if f[2] == config.PREDICT_COARSE_PROB]
    print(f'  判定が反転したウィンドウ: 判定閾値 {config.A_THRESHOLD:g} → {len(f05)}個 / '
          f'粗探索閾値 {config.PREDICT_COARSE_PROB:g} → {len(f03)}個')
    for name, s, th, a, b in flips[:20]:
        print(f'    {name} ウィンドウ開始 {s}f: 基準 {th}  FP32 {a:.4f} → FP16 {b:.4f}')

    # ── ③ 評価セット全体 (任意) ──
    same = None
    if args.full_eval:
        keys = [('base_recall', '① 従来判定 再現率'), ('base_false_alarm', '① 従来判定 誤検知率'),
                ('base_f1', '① 従来判定 F1'), ('svc_video_recall', '② サービス判定 再現率'),
                ('svc_video_false_alarm', '② サービス判定 誤検知率'), ('svc_f1', '② サービス判定 F1'),
                ('svc_hit_recall', '② 位置基準 再現率'), ('svc_false_events', '② 誤検知イベント数'),
                ('svc_err_median_f', '② 開始誤差 中央値(f)'), ('speed_proc_fps', '処理速度 (fps)')]
        got = {}
        for fp16 in (False, True):
            config.INFER_FP16 = fp16
            print(f"\n{'=' * 60}\n[評価セット] {'FP16' if fp16 else 'FP32'}\n{'=' * 60}")
            got[fp16] = evaluate_folder_accuracy(model, folder_path=config.EVAL_FOLDER_PATH) or {}
        config.INFER_FP16 = False
        print(f"\n[評価指標の比較]\n{'指標':<24}{'FP32':>10}{'FP16':>10}")
        for k, label in keys:
            a, b = got[False].get(k), got[True].get(k)
            mark = '' if a == b or k == 'speed_proc_fps' else '  ← 相違'
            print(f'{label:<24}{str(a):>10}{str(b):>10}{mark}')
        same = all(got[False].get(k) == got[True].get(k) for k, _ in keys if k != 'speed_proc_fps')

    # ── 判定 ──
    print('\n[判定結果]')
    ok1 = not flips
    print(f"  ① 判定反転ウィンドウなし: {'合格' if ok1 else f'不合格 ({len(flips)}件)'}")
    if same is not None:
        print(f"  ② 評価指標の一致: {'合格' if same else '不合格'}")
    else:
        print('  ② 評価指標: --full-eval で確認してください')
    print('  ③ 速度向上: 上記の [速度] 表のサービスバッチ行を参照してください')
    if ok1 and same:
        print('  → FP16 を有効化可能です (速度向上がある場合)。デフォルト有効のためそのままで問題ありません (INFER_FP16=0 で無効化可能)')
    elif not ok1 or same is False:
        print('  → FP16 の適用を保留してください (結果に差異が生じるケースがあります)。')


if __name__ == '__main__':
    main()
