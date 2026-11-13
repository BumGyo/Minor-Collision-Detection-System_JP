"""変形耐性テスト — 学習済み拡張と未学習の変形（低解像度・圧縮）に対してモデルの判定がどの程度揺らぐかを測定する。

    python -u -m model.robust_check                      # 評価フォルダ全体、評価重み（GPU推奨、約5分）
    python -u -m model.robust_check --weights <重み.pth>
    python -u -m model.robust_check --videos 4           # 事故・正常を半分ずつ一部のみ（コード点検用）

評価動画(config.EVAL_FOLDER_PATH)ごとにウィンドウを選定する — 事故動画は衝突開始の前後（開始25・20・15・10・5・0フレーム前開始）、
正常動画は動画全体に均等に12個。ウィンドウごとに30フレームを学習・本番運用と同様にクロップ・縮小した後（0番車両ボックス、r=1.0）
変形ごとにP(A)を測定する。変形はクリップ全体で同一値（ノイズは固定グレイン、シード固定）。
  ・学習済み拡張(dataset.py と同等の演算): 左右反転・明度 1.4・暗転 0.6・コントラスト 0.7・彩度 0.7・色相 +0.1・
    グレースケール・ブラー(3×3)・ノイズ σ8
  ・未学習の変形: 低解像度(車両クロップを64・40pxに縮小後再拡大 = 画面内の遠方・小型車)・圧縮(JPEG品質 25)・
    低解像度 64px + 圧縮(小型車を高圧縮CCTVで捉えた状態)
学習済み拡張に対する判定の揺らぎが少なく、未学習の変形に大きく揺らぐ場合、拡張がモデルに正しく反映されている。低解像度で
正常ウィンドウのP(A)が上昇する場合、「ぼやけていれば事故」というショートカット学習を疑う(学習データの低画質動画が事故側に偏在)。
判定はウィンドウ単位 P(A)>0.5 および、動画の選定ウィンドウのいずれか1つでも0.5を超えれば「事故」とみなす簡易判定(本番サービスの判定とは
異なる — 変形間比較用)。結果は画面および outputs/robust_check/<タイムスタンプ>/report.txt ・ windows.csv に出力される。
"""
import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import config
from dataset import HitAndRunDataset
from device_utils import is_channels_last_3d_supported, is_cuda_like
from evaluate import _read_label
from main import _load_model

A_OFFSETS = (25, 20, 15, 10, 5, 0)            # 事故ウィンドウ開始 = 衝突開始 - この値(フレーム)
NOISE_SIGMA = 8.0


class _Tee:
    """画面とファイルの両方に出力する。"""
    def __init__(self, path):
        self.stdout, self.f = sys.stdout, open(path, 'w', encoding='utf-8')

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        self.f.close()


# ───────────────────────── 変形 ─────────────────────────
def _pil(fn):
    """PIL変換をクリップの全フレームに同一値で適用 — dataset._apply_augmentation と同等の演算。"""
    return lambda frames, seed: [np.array(fn(Image.fromarray(f))) for f in frames]


def _noise(frames, seed):
    # 以前の学習ノイズ — フレームごとに新規抽出(ウィンドウごとにシード固定 → 再現性あり)。11-05以降は学習から除外(参考用)
    rng = np.random.default_rng(seed)
    return [np.clip(f.astype(np.float32) + rng.normal(0.0, NOISE_SIGMA, f.shape), 0, 255).astype(np.uint8)
            for f in frames]


def _grain(frames, seed):
    # 現在の学習ノイズ(dataset._apply_augmentation) — クリップ全体で同一グレイン(ウィンドウごとにシード固定)
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, NOISE_SIGMA, frames[0].shape).astype(np.float32)
    return [np.clip(f.astype(np.float32) + g, 0, 255).astype(np.uint8) for f in frames]


def _low(size, jpeg=None):
    """224クロップをsizeに縮小後(カメラ遠方の小型車)再拡大 — 本番サービスが小クロップを拡大するのと同様の補間。"""
    def fn(frames, seed):
        out = []
        for f in frames:
            s = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
            if jpeg:
                s = _jpeg_one(s, jpeg)
            out.append(cv2.resize(s, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR))
        return out
    return fn


def _jpeg_one(f, q):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


TRAINED, UNTRAINED, KEYFRAME = '学習済み拡張', '未学習の変形', 'H.264キーフレーム(画質変動)'
REF = '参考: 以前の学習ノイズ(毎フレームちらつき)'
LIGHT = '照明(夜間)'


def _light(**kw):
    """夜間のように暗転(×0.4)させた後、動的な照明を加算 — dataset.add_light(照明拡張と同等の関数)、値は固定。"""
    def fn(frames, seed):
        from dataset import add_light
        return add_light(list(frames), **kw)
    return fn

H264_CRF = 35


def _h264(crf, gop):
    """クロップ30フレームをH.264で圧縮・展開 — キーフレームはgopフレームごと(ウィンドウ長以上なら先頭のみ = 変動なし)。"""
    def fn(frames, seed):
        from dataset import h264_roundtrip
        return h264_roundtrip(list(frames), crf, gop)
    return fn
CONDITIONS = [
    ('orig', 'オリジナル', None, lambda frames, seed: frames),
    ('flip', '左右反転', TRAINED, _pil(TF.hflip)),
    ('bright', '明度 ×1.4', TRAINED, _pil(lambda im: TF.adjust_brightness(im, 1.4))),
    ('dark', '暗転 ×0.6', TRAINED, _pil(lambda im: TF.adjust_brightness(im, 0.6))),
    ('contrast', 'コントラスト ×0.7', TRAINED, _pil(lambda im: TF.adjust_contrast(im, 0.7))),
    ('sat', '彩度 ×0.7', TRAINED, _pil(lambda im: TF.adjust_saturation(im, 0.7))),
    ('hue', '色相 +0.1', TRAINED, _pil(lambda im: TF.adjust_hue(im, 0.1))),
    ('gray', 'グレースケール', TRAINED, _pil(lambda im: TF.rgb_to_grayscale(im, num_output_channels=3))),
    ('blur', 'ブラー 3×3', TRAINED, _pil(lambda im: TF.gaussian_blur(im, kernel_size=3))),
    ('grain', f'グレインノイズ σ{NOISE_SIGMA:g}', TRAINED, _grain),
    ('low64', '低解像度 64px', UNTRAINED, _low(64)),
    ('low40', '低解像度 40px', UNTRAINED, _low(40)),
    ('jpeg25', 'JPEG品質 25', UNTRAINED, lambda frames, seed: [_jpeg_one(f, 25) for f in frames]),
    ('low64j', '低解像度 64px+圧縮', UNTRAINED, _low(64, jpeg=25)),
    ('noise', f'ちらつきノイズ σ{NOISE_SIGMA:g}', REF, _noise),
]
try:
    import av  # noqa: F401 — H.264変形に必要(dataset.h264_roundtrip)
    CONDITIONS += [
        ('h264g30', f'H.264 圧縮のみ(CRF {H264_CRF})', KEYFRAME, _h264(H264_CRF, 300)),
        ('h264g15', '+ 15フレームごとキーフレーム', KEYFRAME, _h264(H264_CRF, 15)),
        ('h264g10', '+ 10フレームごとキーフレーム', KEYFRAME, _h264(H264_CRF, 10)),
        ('h264g15q', '+ 15フレームごと・CRF 42', KEYFRAME, _h264(42, 15)),
    ]
    KEYFRAME_NOTE = ''
except ImportError:
    KEYFRAME_NOTE = 'PyAV(av) がないため H.264 キーフレーム変形はスキップ — pip install av'


# 照明(2026-11-05): 同一「夜間(×0.4)」に照明のみ加算した場合と比較し、照明によるP(A)の上昇(正常)・低下(事故)を検証
CONDITIONS += [
    ('night', '夜間暗転 ×0.4', LIGHT, _light(kind='none', dark=0.4)),
    ('lsweep', '+ 白色光スイープ', LIGHT, _light(kind='sweep', dark=0.4, peak=160, radius=0.35, t0=8, dur=14,
                                             direction=1, color='white')),
    ('lramp', '+ 温色光照射', LIGHT, _light(kind='ramp', dark=0.4, peak=120, t0=10, dur=8, direction=-1,
                                              color='warm')),
    ('lred', '+ 赤色テールランプスイープ', LIGHT, _light(kind='sweep', dark=0.4, peak=140, radius=0.2, t0=8, dur=14,
                                            direction=-1, color='red')),
]


# ───────────────────────── ウィンドウ選定・読み込み ─────────────────────────
def _windows(total, gt_sf, n_s):
    L = config.CLIP_LENGTH
    last = max(0, total - L)
    if gt_sf is not None:
        return sorted({min(max(0, gt_sf - k), last) for k in A_OFFSETS})
    return sorted({int(round(v)) for v in np.linspace(0, last, n_s)})


def _read_clips(path, bbox, starts):
    """ウィンドウ開始位置リスト → [(T,H,W,3) uint8] — 先頭から順次読み込み(本番同様)、必要フレームのみクロップ・縮小。"""
    L = config.CLIP_LENGTH
    need = {s + i for s in starts for i in range(L)}
    crop_self = SimpleNamespace(resize=tuple(config.RESIZE))
    cap = cv2.VideoCapture(str(path))
    crops, idx = {}, 0
    while idx <= max(need):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in need:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crops[idx] = HitAndRunDataset._crop_and_pad(crop_self, rgb, bbox, config.R_VALUE)
        idx += 1
    cap.release()
    clips = []
    for s in starts:
        fr = [crops[s + i] for i in range(L) if s + i in crops]
        if not fr:
            continue
        while len(fr) < L:                       # 動画が短い場合は最終フレームでパディング(dataset.py と同等)
            fr.append(fr[-1])
        clips.append((s, np.stack(fr)))
    return clips


# ───────────────────────── 推論 ─────────────────────────
def _infer(model, device, clips, batch, cl):
    mean = torch.tensor(config.NORM_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(clips), batch):
            x = torch.from_numpy(np.stack(clips[i:i + batch])).to(device)        # (B,T,H,W,3) uint8
            x = (x.permute(0, 4, 1, 2, 3).float() / 255.0 - mean) / std           # dataset._frames_to_tensor と同等
            if cl:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            out += torch.softmax(model(x).float(), 1)[:, 1].cpu().tolist()
    return out


def _auc(pos, neg):
    """事故ウィンドウが正常ウィンドウよりP(A)が高い確率(同点は0.5) — 判定閾値に依存しない識別能。"""
    if not pos or not neg:
        return float('nan')
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


# ───────────────────────── 実行 ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description='変形耐性テスト')
    ap.add_argument('--weights', default=str(config.EVAL_WEIGHTS_PATH), help='検証対象の重みパス(デフォルト: 評価用重み)')
    ap.add_argument('--folder', default=str(config.EVAL_FOLDER_PATH), help='評価動画フォルダ(mp4 + txt)')
    ap.add_argument('--videos', type=int, default=0, help='事故・正常を各この件数のみ評価(0 = 全件)')
    ap.add_argument('--s-windows', type=int, default=12, help='正常動画ごとに選定するウィンドウ数')
    ap.add_argument('--batch', type=int, default=0, help='推論バッチサイズ(0 = GPU 16 · CPU 4)')
    args = ap.parse_args()

    out_dir = Path(config._ROOT) / 'outputs' / 'robust_check' / datetime.now().strftime('%y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / 'report.txt')
    sys.stdout = tee
    try:
        return _run(args, out_dir)
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[保存] {out_dir / "report.txt"}')


def _run(args, out_dir):
    model, device = _load_model(args.weights)
    model.eval()
    cl = is_channels_last_3d_supported(device) and bool(config.USE_CHANNELS_LAST)
    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True
    batch = args.batch or (16 if is_cuda_like(device) else 4)
    print(f'[変形耐性テスト] {datetime.now():%Y-%m-%d %H:%M} · デバイス {device} · FP32 · バッチ {batch}')
    print(f'重み: {args.weights}')

    folder = Path(args.folder)
    vids = []
    for mp4 in sorted(folder.glob('*.mp4')):
        txt = mp4.with_suffix('.txt')
        if not txt.exists() or mp4.stem in (getattr(config, 'EXCLUDE_VIDEOS', ()) or ()):
            continue   # 10fps未満の動画(config.LOW_FPS_VIDEOS)は除外
        bboxes, gt_sf, _ = _read_label(str(txt))
        if bboxes:
            vids.append((mp4, bboxes.get(config.TARGET_ID, next(iter(bboxes.values()))), gt_sf))
    if args.videos:
        a = [v for v in vids if v[2] is not None][:math.ceil(args.videos / 2)]
        s = [v for v in vids if v[2] is None][:args.videos // 2]
        vids = a + s
    n_a = sum(v[2] is not None for v in vids)
    print(f'動画: {folder} — 事故 {n_a}件 · 正常 {len(vids) - n_a}件 · 変形 {len(CONDITIONS) - 1}種 + オリジナル')
    if KEYFRAME_NOTE:
        print(f'  ⚠️ {KEYFRAME_NOTE}')

    rows = []                                    # (動画, gt, ウィンドウ開始, 変形, P(A))
    t0 = time.perf_counter()
    for vi, (mp4, bbox, gt_sf) in enumerate(vids):
        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        clips = _read_clips(mp4, bbox, _windows(total, gt_sf, args.s_windows))
        gt = 'A' if gt_sf is not None else 'S'
        for key, _, _, fn in CONDITIONS:
            xs = [np.stack(fn(list(c), seed=vi * 1000 + s)) for s, c in clips]
            for (s, _), p in zip(clips, _infer(model, device, xs, batch, cl)):
                rows.append((mp4.name, gt, s, key, p))
        print(f'  {vi + 1}/{len(vids)} {mp4.name} ({gt}) ウィンドウ {len(clips)}件 · {time.perf_counter() - t0:.0f}秒', flush=True)

    with open(out_dir / 'windows.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video', 'gt', 'window_start', 'condition', 'p_a'])
        w.writerows([r[:4] + (f'{r[4]:.6f}',) for r in rows])
    _report(rows)
    return 0


def _report(rows):
    base = {(v, s): p for v, gt, s, k, p in rows if k == 'orig'}
    gt_of = {v: gt for v, gt, *_ in rows}
    n_av = sum(g == 'A' for g in gt_of.values())
    n_sv = len(gt_of) - n_av
    stats = {}
    for key, name, group, _ in CONDITIONS:
        sub = [(v, gt, s, p) for v, gt, s, k, p in rows if k == key]
        pa = [p for v, gt, s, p in sub if gt == 'A']
        ps = [p for v, gt, s, p in sub if gt == 'S']
        d = [abs(p - base[(v, s)]) for v, gt, s, p in sub]
        miss = sum(gt == 'A' and base[(v, s)] > 0.5 >= p for v, gt, s, p in sub)     # 事故ウィンドウ: 正解を検出漏れ
        fa = sum(gt == 'S' and base[(v, s)] <= 0.5 < p for v, gt, s, p in sub)       # 正常ウィンドウ: 新規誤検知
        vmax = {}
        for v, gt, s, p in sub:
            vmax[v] = max(vmax.get(v, 0.0), p)
        stats[key] = dict(name=name, group=group, pa=np.mean(pa) if pa else float('nan'),
                          ps=np.mean(ps) if ps else float('nan'), d=float(np.mean(d)), miss=miss, fa=fa,
                          auc=_auc(pa, ps), hit=sum(gt_of[v] == 'A' and m > 0.5 for v, m in vmax.items()),
                          vfa=sum(gt_of[v] == 'S' and m > 0.5 for v, m in vmax.items()),
                          n_a=len(pa), n_s=len(ps))
    o = stats['orig']
    print(f'\n[結果] ウィンドウ: 事故 {o["n_a"]}件 · 正常 {o["n_s"]}件 (事故動画 {n_av}件 · 正常動画 {n_sv}件)')
    print(f'  {"変形":<18}{"事故窓 P(A)":>11}{"正常窓 P(A)":>11}{"平均変化":>9}{"検出漏れ+":>6}{"誤検知+":>6}'
          f'{"識別能":>8}{"事故動画":>9}{"正常誤検知":>9}')
    group = None
    for key, name, g, _ in CONDITIONS:
        if g != group:
            group = g
            if g:
                print(f'  ── {g}')
        s = stats[key]
        print(f'  {name:<18}{s["pa"]:>11.3f}{s["ps"]:>11.3f}{s["d"]:>9.3f}{s["miss"]:>6}{s["fa"]:>6}'
              f'{s["auc"]:>8.3f}{s["hit"]:>6}/{n_av:<2}{s["vfa"]:>6}/{n_sv:<2}')
    print('  ・平均変化 = オリジナル対比 |P(A) 差分| 平均 · 検出漏れ+ = オリジナルで 0.5 を超過していた事故窓が下回った数 · '
          '誤検知+ = オリジナルで 0.5 以下だった正常窓が超過した数')
    print('  ・識別能 = 事故窓が正常窓よりP(A)が高い確率(1が完全、0.5はランダム) · '
          '事故動画/正常誤検知 = 動画の選定窓のいずれか1つでも 0.5 超過(簡易判定)')

    tr = [stats[k] for k, _, g, _ in CONDITIONS if g == TRAINED]
    un = [stats[k] for k, _, g, _ in CONDITIONS if g == UNTRAINED]
    print('\n[判定]')
    d_tr, d_un = np.mean([s['d'] for s in tr]), np.mean([s['d'] for s in un])
    a_tr, a_un = np.mean([s['auc'] for s in tr]), np.mean([s['auc'] for s in un])
    print(f'  1) 平均変化: 学習済み拡張 {d_tr:.3f} · 未学習の変形 {d_un:.3f} / '
          f'識別能: オリジナル {o["auc"]:.3f} · 学習済み拡張平均 {a_tr:.3f} · 未学習の変形平均 {a_un:.3f}')
    worst_tr = max(tr, key=lambda s: s['d'])
    if d_un > d_tr * 1.5:
        print('     → 学習済み拡張に対する揺らぎが少なく、未学習の変形でより揺らぎます — 拡張がモデルに正しく反映されていると考えられます。')
    else:
        print('     → 学習済み拡張と未学習の変形の揺らぎが同等です — 拡張の効果が顕著ではありません。')
    print(f'     学習済み拡張の中で最も揺らぎが大きい項目: {worst_tr["name"]}(平均変化 {worst_tr["d"]:.3f}, '
          f'検出漏れ+ {worst_tr["miss"]} · 誤検知+ {worst_tr["fa"]})')
    low = [stats[k] for k in ('low64', 'low40', 'jpeg25', 'low64j')]
    rise_s = max(s['ps'] - o['ps'] for s in low)
    rise_a = max(s['pa'] - o['pa'] for s in low)
    fa_low = max(s['fa'] for s in low)
    print('  2) 画質を低下させた場合(低解像度・圧縮)の P(A):')
    print(f'     正常窓 オリジナル {o["ps"]:.3f} → ' + ' / '.join(f'{s["name"]} {s["ps"]:.3f}' for s in low)
          + f' · 新規誤検知窓 最大 {fa_low}件')
    print(f'     事故窓 オリジナル {o["pa"]:.3f} → ' + ' / '.join(f'{s["name"]} {s["pa"]:.3f}' for s in low))
    if rise_s > 0.05 or fa_low >= max(2, 0.05 * o['n_s']):
        print('     → 画質が低下すると正常窓も事故側に傾斜します — 「ぼやけていれば事故」というショートカットが疑われます'
              '(低画質拡張・データ拡充による補強を推奨)。')
    elif rise_a > 0.1:
        print('     → 正常窓はほぼ不変ですが、事故窓のP(A)が画質低下に伴い上昇します — 同一方向の傾向があり、'
              '正常窓がわずかでも曖昧な場合に誤検知へ繋がる可能性があります(低画質拡張での補強を検討)。')
    elif max(s['pa'] for s in low) < o['pa'] - 0.1:
        print('     → 画質が低下すると事故窓を見落とす傾向にあります(ショートカットというより低解像度での情報不足)。')
    else:
        print('     → 画質が低下しても判定に大きな偏りは見られません — ショートカットの兆候は軽微です。')
    if 'h264g30' in stats:
        _report_keyframe(rows, stats, th)
    if 'night' in stats:
        _report_light(rows, stats, th)


def _report_light(rows, stats, th):
    """4) 照明: 同一「夜間」(night)に照明のみを加算した際、正常窓のP(A)が上昇するか(誤検知傾向)・事故窓が低下するか(遮光による見落とし)。"""
    ref = {(v, s): p for v, gt, s, k, p in rows if k == 'night'}
    c = stats['night']
    print(f'  4) 照明(同一夜間 ×0.4 に照明のみ加算) — 夜間: 正常窓 {c["ps"]:.3f} · 事故窓 {c["pa"]:.3f}')
    worst_s, worst_a = None, None
    for key in ('lsweep', 'lramp', 'lred'):
        sub = [(v, gt, s, p) for v, gt, s, k, p in rows if k == key]
        ds = [p - ref[(v, s)] for v, gt, s, p in sub if gt == 'S']
        da = [p - ref[(v, s)] for v, gt, s, p in sub if gt == 'A']
        new_fa = sum(gt == 'S' and ref[(v, s)] <= th < p for v, gt, s, p in sub)
        new_miss = sum(gt == 'A' and ref[(v, s)] > th >= p for v, gt, s, p in sub)
        print(f'     {stats[key]["name"]:<16} 正常窓 P(A) 変化平均 {np.mean(ds):+.3f}(最大 {max(ds):+.3f}) · 新規誤検知窓 {new_fa}件 · '
              f'事故窓 {np.mean(da):+.3f} · 新規見落とし事故窓 {new_miss}件')
        if worst_s is None or np.mean(ds) > worst_s[1]:
            worst_s = (stats[key]['name'], float(np.mean(ds)), new_fa)
        if worst_a is None or np.mean(da) < worst_a[1]:
            worst_a = (stats[key]['name'], float(np.mean(da)), new_miss)
    n_s = stats['orig']['n_s']
    if worst_s[1] > 0.03 or worst_s[2] >= max(2, 0.05 * n_s):
        print(f'     → 照明のみで正常窓が事故側に傾斜します({worst_s[0]}) — 夜間照明の誤検知傾向。照明拡張・夜間正常動画で補強')
    else:
        print('     → 照明による正常窓判定の揺らぎはほぼありません。')
    if worst_a[1] < -0.05:
        print(f'     → 照明が存在すると事故窓 P(A) が低下します({worst_a[0]}) — ヘッドライト点灯車の夜間衝突を見落とすリスクあり')


def _report_keyframe(rows, stats, th):
    """3) キーフレーム画質変動: 「圧縮のみ」(h264g30)に対しキーフレームをウィンドウ内に配置した変形でのP(A)上昇を検証。"""
    ref = {(v, s): p for v, gt, s, k, p in rows if k == 'h264g30'}
    c = stats['h264g30']
    print(f'  3) キーフレーム画質変動(同一H.264圧縮にキーフレームを追加配置) — 圧縮のみ: 正常窓 {c["ps"]:.3f} · 事故窓 {c["pa"]:.3f}')
    worst = None
    for key in ('h264g15', 'h264g10', 'h264g15q'):
        sub = [(v, gt, s, p) for v, gt, s, k, p in rows if k == key]
        ds = [p - ref[(v, s)] for v, gt, s, p in sub if gt == 'S']
        da = [p - ref[(v, s)] for v, gt, s, p in sub if gt == 'A']
        new_fa = sum(gt == 'S' and ref[(v, s)] <= th < p for v, gt, s, p in sub)
        print(f'     {stats[key]["name"]:<18} 正常窓 P(A) 変化平均 {np.mean(ds):+.3f}(最大 {max(ds):+.3f}) · '
              f'事故窓 {np.mean(da):+.3f} · 圧縮のみでは対象外だった正常誤検知窓 {new_fa}件')
        if worst is None or np.mean(ds) > worst[1]:
            worst = (stats[key]['name'], float(np.mean(ds)), new_fa)
    n_s = stats['orig']['n_s']
    if worst[1] > 0.03 or worst[2] >= max(2, 0.05 * n_s):
        print(f'     → キーフレーム画質変動のみで正常窓が事故側に傾斜します({worst[0]}) — 静止シーンの圧縮アーティファクトを揺れとして'
              '誤認する傾向(ReS_0018等の誤検知)。ヘッドプーリングを変更した場合は対照群と比較してください。')
    else:
        print('     → キーフレーム画質変動に対する判定の揺らぎはほぼありません。')


if __name__ == '__main__':
    sys.exit(main())
