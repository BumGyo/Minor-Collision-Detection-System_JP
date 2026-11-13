"""非衝突 (S) 学習クリップの動き量を計測し、静止正常クリップを間引いた際の A:S 比率を分析するスクリプト。

    python -u -m model.s_motion_check
    python -u -m model.s_motion_check --workers 8 --montage
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import config
from dataset import HitAndRunDataset

PIX_T = 15                                  # 動体画素判定の輝度差閾値 (0〜255)
THRESHOLDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)   # 静止判定候補閾値 (動体画素 %)
KEEP_FRACS = (0.0, 0.2, 1 / 3, 0.5)         # 静止 S クリップの残存比率
BATCH = 15                                  # バッチサイズ


class _Crop:
    def __init__(self, resize):
        self.resize = resize

    def __call__(self, frame, bbox):
        return HitAndRunDataset._crop_and_pad(self, frame, list(bbox), config.R_VALUE)


def _gray(crop):
    return cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def _video_task(args):
    mp4, items, clip_len, resize = args
    crop = _Crop(resize)
    boxes = sorted({tuple(b) for _, _, b in items})
    need = max(s for _, s, _ in items) + clip_len
    frac = {b: [] for b in boxes}
    dmean = {b: [] for b in boxes}
    prev = dict.fromkeys(boxes)
    cap = cv2.VideoCapture(mp4)
    t = 0
    while t < need:
        ok, frame = cap.read()
        if not ok:
            break
        for b in boxes:
            g = _gray(crop(frame, b))
            if prev[b] is not None:
                d = cv2.absdiff(g, prev[b])
                frac[b].append(float((d > PIX_T).mean()) * 100.0)
                dmean[b].append(float(d.mean()))
            prev[b] = g
        t += 1
    cap.release()
    out = []
    for idx, s, b in items:
        f = np.asarray(frac[tuple(b)][s:s + clip_len - 1])
        m = np.asarray(dmean[tuple(b)][s:s + clip_len - 1])
        if f.size == 0:
            out.append((idx, float('nan'), float('nan'), float('nan'), 0))
            continue
        third = float(np.sort(f)[-3]) if f.size >= 3 else float(f.max())
        out.append((idx, third, float(f.max()), float(m.mean()), int(f.size)))
    return mp4, t, out


def _scenario(file_name, domain):
    seg = file_name.split('_')
    code = seg[1][-1] if len(seg) > 1 and seg[1] else '?'
    return code if domain == 'rc' else ('A' if code == 'A' else 'S')


def _tile(mp4, s, bbox, clip_len, resize, text):
    crop = _Crop(resize)
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    frames = []
    for _ in range(clip_len):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(crop(f, bbox))
    cap.release()
    h, w = resize[1], resize[0]
    if len(frames) < 2:
        tile = np.zeros((h, w * 2, 3), np.uint8)
    else:
        gs = [_gray(c) for c in frames]
        dmax = np.max([cv2.absdiff(a, b) for a, b in zip(gs[1:], gs[:-1])], axis=0)
        heat = cv2.applyColorMap(np.clip(dmax.astype(np.int32) * 4, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        mid = frames[len(frames) // 2]
        tile = np.hstack([mid, cv2.addWeighted(mid, 0.45, heat, 0.55, 0)])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(tile, (0, 0), (tw + 8, th + 10), (0, 0, 0), -1)
    cv2.putText(tile, text, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _montage(rows, path, clip_len, resize, cols=3):
    tiles = [_tile(r['mp4'], r['start_f'], r['bbox'], clip_len, resize,
                   f"{r['video'][:22]} f{r['start_f']} c{r['car_id']} {r['move']:.2f}%")
             for r in rows]
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser(description='非衝突学習クリップの動き量測定')
    ap.add_argument('--data-dir', default=str(config.DATA_DIR), help='学習データディレクトリ')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 2),
                    help='並行読み込み動画数')
    ap.add_argument('--limit', type=int, default=0, help='指定動画数のみ測定 (テスト用)')
    ap.add_argument('--montage', action='store_true', help='モンタージュ画像保存')
    args = ap.parse_args()

    config.TRAIN_A_MODE = 'pre'
    ds = HitAndRunDataset(data_dir=args.data_dir, augment=False)
    clip_len, resize = ds.clip_length, ds.resize
    rows = []
    for i, s in enumerate(ds.samples):
        dom = 'real' if 'realdata' in os.path.normpath(s['mp4_path']).split(os.sep) else 'rc'
        rows.append({'idx': i, 'mp4': s['mp4_path'], 'video': s['file_name'], 'domain': dom,
                     'scenario': _scenario(s['file_name'], dom), 'label': s['label'],
                     'car_id': s['car_id'], 'start_f': s['start_f'], 'fps': s['fps'],
                     'bbox': tuple(s['bbox'])})
    by_video = {}
    for r in rows:
        by_video.setdefault(r['mp4'], []).append((r['idx'], r['start_f'], r['bbox']))
    videos = sorted(by_video)
    if args.limit and args.limit < len(videos):
        pick = np.linspace(0, len(videos) - 1, args.limit).round().astype(int)
        videos = [videos[i] for i in sorted(set(pick))]
        keep = set(videos)
        rows = [r for r in rows if r['mp4'] in keep]
    print(f"[対象] 動画 {len(videos)}件 · クリップ {len(rows)}件 (A {sum(r['label'] for r in rows)}) "
          f"· 並行数 {args.workers} · データパス {args.data_dir}", flush=True)

    out_dir = Path(config._ROOT) / 'outputs' / 's_motion' / datetime.now().strftime('%y%m%d_%H%M')
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, done, frames = time.time(), 0, 0
    tasks = [(v, by_video[v], clip_len, resize) for v in videos]
    res = {}
    with Pool(args.workers) as pool:
        for mp4, nf, out in pool.imap_unordered(_video_task, tasks):
            done += 1
            frames += nf
            for idx, move, move_max, dm, n in out:
                res[idx] = (move, move_max, dm, n)
            if done % 20 == 0 or done == len(tasks):
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f"  {done}/{len(tasks)} 動画 · {frames / max(el, 1e-6):.0f} フレーム/秒 · "
                      f"経過 {el / 60:.1f}分 · 残り約 {eta / 60:.1f}分", flush=True)
    for r in rows:
        r['move'], r['move_max'], r['diff_mean'], r['n_diffs'] = res.get(r['idx'], (np.nan,) * 3 + (0,))

    with open(out_dir / 'clips.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video', 'domain', 'scenario', 'label', 'car_id', 'start_f', 'fps',
                    'move_pct', 'move_max_pct', 'diff_mean', 'n_diffs'])
        for r in rows:
            w.writerow([r['video'], r['domain'], r['scenario'], r['label'], r['car_id'], r['start_f'],
                        f"{r['fps']:.2f}", f"{r['move']:.4f}", f"{r['move_max']:.4f}",
                        f"{r['diff_mean']:.4f}", r['n_diffs']])

    lines = []

    def say(s=''):
        print(s)
        lines.append(s)

    ok_rows = [r for r in rows if r['n_diffs'] > 0]
    bad = len(rows) - len(ok_rows)
    A = [r for r in ok_rows if r['label'] == 1]
    S = [r for r in ok_rows if r['label'] == 0]
    nA, nS = len(A), len(S)

    def no_a(na, ns):
        return (ns / (na + ns)) ** BATCH if na + ns else float('nan')

    say(f"測定日時 {datetime.now():%Y-%m-%d %H:%M} · 動画 {len(videos)}件 · 処理時間 {time.time() - t0:.0f}秒 · "
        f"動体画素輝度差閾値 > {PIX_T} · 動き量 = クリップ内上位3番目のフレーム値")
    if bad:
        say(f"⚠️ 読み込み失敗クリップ {bad}件 (スキップ)")
    say(f"\n[現状] A {nA} : S {nS} = 1:{nS / max(nA, 1):.1f} · バッチ {BATCH}個中 衝突なしバッチ約 {no_a(nA, nS) * 100:.0f}%")
    for dom in ('rc', 'real'):
        a = sum(r['domain'] == dom for r in A)
        s = sum(r['domain'] == dom for r in S)
        if a or s:
            say(f"  {dom:<4}: A {a} : S {s} = 1:{s / max(a, 1):.1f}")

    say("\n[動き量分布] 動体画素 % (p = パーセンタイル)")
    say(f"  {'グループ':<12}{'クリップ':>7}{'p5':>8}{'p10':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p90':>8}")
    groups = [('A-rc', [r for r in A if r['domain'] == 'rc']), ('A-real', [r for r in A if r['domain'] == 'real'])]
    for sc, name in (('V', '駐車'), ('S', '直進'), ('W', '徘徊')):
        groups.append((f'S-rc-{sc}({name})', [r for r in S if r['domain'] == 'rc' and r['scenario'] == sc]))
    groups.append(('S-real', [r for r in S if r['domain'] == 'real']))
    for name, g in groups:
        if not g:
            continue
        v = np.array([r['move'] for r in g])
        q = np.percentile(v, [5, 10, 25, 50, 75, 90])
        say(f"  {name:<12}{len(g):>7}" + "".join(f"{x:>8.2f}" for x in q))

    say("\n[閾値別] 動き量が閾値未満の S = 「静止」。静止 S を間引いた場合の A:S 比率")
    head = "".join(f"{'残存 ' + (f'{k:.0%}' if k else '0'):>12}" for k in KEEP_FRACS)
    say(f"  {'閾値%':>6}{'静止 S':>14}{'静止 S(rc/real)':>16}{head}{'A中閾値未満':>16}")
    for th in THRESHOLDS:
        empty = [r for r in S if r['move'] < th]
        e_rc = sum(r['domain'] == 'rc' for r in empty)
        e_real = len(empty) - e_rc
        cells = ""
        for k in KEEP_FRACS:
            ns = nS - len(empty) + round(len(empty) * k)
            cells += f"{f'1:{ns / max(nA, 1):.1f} ({no_a(nA, ns) * 100:.0f}%)':>12}"
        a_low = sum(r['move'] < th for r in A)
        say(f"  {th:>6.2f}{f'{len(empty)} ({len(empty) / max(nS, 1):.0%})':>14}{f'{e_rc}/{e_real}':>16}"
            f"{cells}{f'{a_low} ({a_low / max(nA, 1):.0%})':>16}")
    (out_dir / 'report.txt').write_text("\n".join(lines) + "\n", encoding='utf-8')

    if args.montage:
        s_sorted = sorted(S, key=lambda r: r['move'])
        qs = np.linspace(0, len(s_sorted) - 1, 24).round().astype(int) if s_sorted else []
        _montage([s_sorted[i] for i in qs], out_dir / 'montage_S_by_motion.jpg', clip_len, resize)
        _montage(sorted(A, key=lambda r: r['move'])[:12], out_dir / 'montage_A_lowest.jpg', clip_len, resize)
        print("[画像保存] S クリップ動き量昇順 24件 · 動き量が最小の A クリップ 12件")
    print(f"\n[出力ディレクトリ] {out_dir}")


if __name__ == '__main__':
    sys.exit(main())
