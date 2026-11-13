"""判定閾値・後処理・低fps時間軸の検証 — ウィンドウごとに P(A) を一度測定・保存し、サービス判定を複数の設定で再計算・比較する。

    python -u -m model.postproc_check --weights <重み1.pth> [<重み2.pth> ...]   # GPU 推奨 (重み1個あたり約10分)
    python -u -m model.postproc_check --weights ... --no-val            # 評価動画のみ
    python -u -m model.postproc_check --again outputs/postproc_check/<日時>   # 保存済み P(A) で集計のみ再実行 (モデル不使用)

検証する設定項目 (全組み合わせを summary.csv に出力):
  ・判定閾値 A_THRESHOLD 0.2〜0.7。粗探索閾値は config デフォルト式 min(0.3, 0.6×閾値)
  ・後処理単位 — 'frame' 固定フレーム数 ・ 'fps' 粗探索間隔 (10) と最小イベント長の連続 A ウィンドウ数 (7) を 30fps 基準として
    動画の実 fps に応じてスケーリング (inference.fps_scaled_event_params — サービスと同一関数, config.PREDICT_FPS_SCALED_POSTPROC)。
    10fps の場合は探索間隔 3・連続 2窓。マージ間隔は既に秒単位で定義
  ・最小イベント長 35・32・30・29 = 連続 A ウィンドウ 7・4・2・1(=フィルタ無効) (イベント長 = 28 + 連続ウィンドウ数、fps スケーリング時は 30fps 基準)
  ・粗探索の有効/無効 (無効時は全ウィンドウを評価 — 粗探索による取りこぼしを検証)
  ・低fps (20未満) 動画の時間軸処理 — none そのまま ・ repeat フレーム複製 ・ blend 隣接フレーム補間。round(30/fps)倍に拡張し、
    1ウィンドウ (30フレーム) が学習時と同様に約1秒間となるよう調整
対象動画: 評価動画 (config.EVAL_FOLDER_PATH) + 検証実写動画 (重みの実行回 split.json の検証データのうち realdata — 学習未使用動画)。
設定パラメータは検証動画で選定し、評価動画で汎化性能を確認する (評価動画で選定するとスコアが過大評価される)。事故は区間の重複判定でカウント。
サービス判定 (粗探索 → 候補窓周辺の詳細探索 → 状態遷移 → マージ・最小長フィルタ) は inference.predict_events_and_clips の
イベント決定ロジックを抽出し、ウィンドウ別 P(A) から再計算。現行設定と完全一致するケースについては実際のサービス関数と
結果を照合し再現性を確認する (サービス再現確認)。
結果: outputs/postproc_check/<日時>/ — report.txt (標準出力と同一) ・ summary.csv (設定組み合わせ別集計) ・ videos.csv (動画別 P(A)
要約・イベント一覧) ・ probs/<重み>/<eval|val>_<none|repeat|blend>/<動画>.npz (ウィンドウ別 P(A) — 再集計用キャッシュ)
"""
import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from device_utils import is_channels_last_3d_supported, is_cuda_like, predict_batch_size
from evaluate import _Tee, _read_label
from inference import fps_scaled_event_params

L = config.CLIP_LENGTH
LOW_FPS = 20.0
RESAMPLES = ('none', 'repeat', 'blend')
THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
MIN_SPANS = (35, 32, 30, 29)
SETS = (('eval', '評価'), ('val', '検証'))


def _now_post():
    """現在のサービスの後処理単位。"""
    return 'fps' if config.PREDICT_FPS_SCALED_POSTPROC else 'frame'


# ── 動画一覧 ──────────────────────────────────────────────────────────────

def _label(mp4):
    bboxes, sf, ef = _read_label(str(mp4.with_suffix('.txt')))
    if not bboxes:
        return None
    return bboxes.get(config.TARGET_ID, next(iter(bboxes.values()))), sf, ef


def _excluded(name):
    """10fps 未満の動画 (config.EXCLUDE_VIDEOS) であるか判定 — フォルダ内に存在しても除外。"""
    return name in (getattr(config, 'EXCLUDE_VIDEOS', ()) or ())


def _folder_videos(folder):
    out = []
    for mp4 in sorted(Path(folder).glob('*.mp4')):
        if _excluded(mp4.stem):
            continue
        lab = _label(mp4) if mp4.with_suffix('.txt').exists() else None
        if lab:
            out.append((mp4, *lab))
    return out


def _split_path(weights, arg):
    """検証動画リスト — 指定の split.json、重み実行回フォルダ内のファイル、または最新学習のもの。"""
    if arg:
        return Path(arg)
    p = Path(weights).resolve().parent.parent / 'split.json'
    if p.exists():
        return p
    cands = sorted((Path(config._ROOT) / 'outputs' / 'trainlogs').glob('*/split.json'))
    return cands[-1] if cands else None


def _val_videos(split):
    with open(split, encoding='utf-8') as f:
        rels = json.load(f).get('val', [])
    out = []
    for rel in rels:
        rel = rel.replace('\\', '/')
        if not rel.startswith('realdata/'):
            continue
        roots = [Path(r) for r in (getattr(config, 'VAL_DATA_DIR', None), config.DATA_DIR) if r]
        mp4 = next((r / rel for r in roots if (r / rel).exists()), Path(config.DATA_DIR) / rel)
        if _excluded(mp4.stem):
            continue
        lab = _label(mp4) if mp4.exists() and mp4.with_suffix('.txt').exists() else None
        if lab:
            out.append((mp4, *lab))
    return out


# ── ウィンドウ別 P(A) ─────────────────────────────────────────────────────

def _read_crops(mp4, bbox):
    """サービスと同一のクロップ処理 (0番車両ボックス, r=config.R_VALUE, 224) — uint8 RGB リスト。"""
    from inference import _crop_square_and_pad
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(_crop_square_and_pad(rgb, bbox, config.R_VALUE, config.RESIZE)[0])
    cap.release()
    return frames, fps, size


def _resample(frames, factor, mode):
    """時間軸を factor 倍にリサンプリング — repeat: フレーム複製, blend: 隣接フレーム補間。"""
    if factor <= 1 or mode == 'none':
        return frames
    out = []
    for i, f in enumerate(frames):
        if mode == 'repeat':
            out += [f] * factor
            continue
        cur = f.astype(np.float32)
        nxt = frames[min(i + 1, len(frames) - 1)].astype(np.float32)
        for k in range(factor):
            a = k / factor
            out.append(np.clip((1 - a) * cur + a * nxt + 0.5, 0, 255).astype(np.uint8))
    return out


def _window_probs(runner, device, frames, batch):
    """ストライド 1 の全ウィンドウの P(A)。サービスの _clip と同一の正規化。"""
    frames = list(frames)
    while len(frames) < L:                       # 短い動画は末尾フレームでパディング
        frames.append(frames[-1])
    u8 = torch.from_numpy(np.stack(frames)).to(device)          # T,H,W,3
    mean = torch.tensor(config.NORM_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    cl = is_channels_last_3d_supported(device) and bool(config.USE_CHANNELS_LAST)
    nw = len(frames) - L + 1
    pa = np.zeros(nw, dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, nw, batch):
            e = min(nw, s + batch)
            x = torch.stack([u8[w:w + L] for w in range(s, e)]).permute(0, 4, 1, 2, 3).float() / 255.0
            x = ((x - mean) / std).contiguous()
            if cl:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            pa[s:e] = F.softmax(runner(x).float(), dim=1)[:, 1].cpu().numpy()
    del u8
    return pa


def _measure(weights, sets, out_dir, batch_arg):
    """単一の重みで動画セットごとに none・repeat・blend の P(A) を測定し npz で保存。戻り値: (モデル, ランナー名)。"""
    from infer_runtime import get_runner
    from main import _load_model
    model, device = _load_model(weights)
    model.eval()
    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True
    runner = get_runner(model)
    batch = batch_arg or predict_batch_size(device)
    wtag = _wtag(weights)
    print(f'\n[P(A) 測定] {wtag} ・ デバイス {device} ・ ランナー {runner.name} ・ バッチ {batch}', flush=True)
    t0 = time.perf_counter()
    for set_key, vids in sets.items():
        for i, (mp4, bbox, sf, ef) in enumerate(vids):
            crops, fps, size = _read_crops(mp4, bbox)
            if not crops:
                print(f'  [スキップ] {mp4.name} — フレーム読み込み不可')
                continue
            factor = int(round(30.0 / fps)) if fps < LOW_FPS else 1
            msg = []
            for mode in RESAMPLES:
                if mode != 'none' and factor <= 1:
                    continue
                d = out_dir / 'probs' / wtag / f'{set_key}_{mode}'
                d.mkdir(parents=True, exist_ok=True)
                pa = _window_probs(runner, device, _resample(crops, factor if mode != 'none' else 1, mode), batch)
                meta = dict(file=mp4.name, set=set_key, resample=mode, fps=fps,
                            factor=factor if mode != 'none' else 1, n_orig=len(crops),
                            width=size[0], height=size[1], bbox=[int(v) for v in bbox],
                            gt_sf=sf, gt_ef=ef, weights=str(weights))
                np.savez(d / f'{mp4.stem}.npz', pa=pa, meta=json.dumps(meta))
                msg.append(f'{mode} 最大 {pa.max():.2f}')
            print(f'  {set_key} {i + 1}/{len(vids)} {mp4.name} {fps:.1f}fps ・ ' + ' ・ '.join(msg)
                  + f' ・ {time.perf_counter() - t0:.0f}秒', flush=True)
    return model, runner.name


def _wtag(weights):
    return Path(weights).stem.replace('hitandrun_x3d_', '').replace('hitandrun_', '')


def _load_probs(wdir, set_key, mode):
    """{動画名: (pa, meta)}。repeat・blend フォルダには低fps動画のみ格納されるため他は none で補完。"""
    out = {}
    for m in ([mode, 'none'] if mode != 'none' else ['none']):
        for p in sorted((wdir / f'{set_key}_{m}').glob('*.npz')):
            if p.stem not in out and not _excluded(p.stem):
                z = np.load(p)
                out[p.stem] = (z['pa'], json.loads(str(z['meta'])))
    return dict(sorted(out.items()))


# ── サービス判定の再計算 ────────────────────────────────────────────────────

def _events(pa, meta, t, *, coarse_prob=None, fps_scale=None, min_span=None, use_coarse=None):
    """inference.predict_events_and_clips のイベント判定ロジック (ウィンドウ間隔 1)。戻り値: [(開始, 終了)] 元フレーム, 評価ウィンドウ数。"""
    coarse_pad = config.PREDICT_COARSE_PAD
    min_span = config.PREDICT_MIN_EVENT_SPAN_FRAMES if min_span is None else min_span
    use_coarse = config.PREDICT_USE_COARSE_SCAN if use_coarse is None else use_coarse
    if coarse_prob is None:
        coarse_prob = min(0.3, 0.6 * t)
    f = meta['factor']
    fps = meta['fps'] * f
    coarse_step, min_span = fps_scaled_event_params(fps, config.PREDICT_COARSE_STEP, min_span, L, enabled=fps_scale)
    nw = len(pa)
    last = nw - 1
    n_real = meta['n_orig'] * f
    grid = max(1, int(round(coarse_step))) if use_coarse else 1
    if grid > 1:
        ev = np.zeros(nw, dtype=bool)
        ev[::grid] = True
        ev[last] = True                          # 最終ウィンドウを粗探索グリッドに追加
        hot = np.nonzero(ev & (pa > coarse_prob))[0]
        if len(hot):                             # 候補ウィンドウ周辺 ±pad を詳細探索
            cover = np.zeros(nw + 1, dtype=np.int32)
            np.add.at(cover, np.maximum(0, hot - coarse_pad), 1)
            np.add.at(cover, np.minimum(nw, hot + coarse_pad + 1), -1)
            ev |= np.cumsum(cover[:nw]) > 0
    else:
        ev = np.ones(nw, dtype=bool)
    pred = ev & (pa > t)                         # 未評価ウィンドウは非事故扱い
    d = np.diff(np.concatenate(([0], pred.astype(np.int8), [0])))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    events = [[int(s), int(e) + L - 2 if e < nw else n_real - 1] for s, e in zip(starts, ends)]
    gap = int(round(fps * config.PREDICT_EVENT_MERGE_GAP_SEC))
    if events:
        merged = [events[0]]
        for e in events[1:]:
            if e[0] - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], e[1])
            else:
                merged.append(e)
        events = [e for e in merged if e[1] - e[0] >= min_span]
    return [(s // f, e // f) for s, e in events], int(ev.sum())


def _score(data, t, **kw):
    """事故は区間重複で検知判定、誤検知イベント = 正常動画のイベント + 事故動画で衝突と重複しないイベント。"""
    rows = []
    for name, (pa, m) in data.items():
        evs, _ = _events(pa, m, t, **kw)
        if m['gt_sf'] is not None:
            ov = [e for e in evs if e[0] <= m['gt_ef'] and e[1] >= m['gt_sf']]
            caught, false_ev = bool(ov), len(evs) - len(ov)
        else:
            caught, false_ev = None, len(evs)
        rows.append(dict(name=name, gt='A' if m['gt_sf'] is not None else 'S', events=evs,
                         caught=caught, false_ev=false_ev, base_a=bool((pa > t).any())))
    A = [r for r in rows if r['gt'] == 'A']
    S = [r for r in rows if r['gt'] == 'S']
    miss = [r['name'] for r in A if not r['caught']]
    fa = [r['name'] for r in S if r['events']]
    return dict(n=len(rows), n_A=len(A), n_S=len(S), miss=len(miss), fa=len(fa),
                false_ev=sum(r['false_ev'] for r in rows),
                acc=(len(A) - len(miss) + len(S) - len(fa)) / max(1, len(rows)),
                base_miss=sum(not r['base_a'] for r in A), base_fa=sum(r['base_a'] for r in S),
                miss_list=miss, fa_list=fa), rows


def _short(name):
    """260828_TA_0001 → TA_0001。"""
    return name.split('_', 1)[1] if '_' in name else name


# ── サービス再現確認 ────────────────────────────────────────────────────────

def _service_check(model, wtag, sets, probs_root):
    """現行設定通りの再計算が実際のサービス関数 (predict_events_and_clips) と完全に一致するか動画ごとに検証。"""
    from inference import predict_events_and_clips
    tmp = Path(tempfile.mkdtemp(prefix='postproc_check_'))
    print(f'\n[サービス再現確認] {wtag} — 現行設定 (閾値 {config.A_THRESHOLD:g}, 粗探索 {config.PREDICT_COARSE_PROB:g}, '
          f'最小長 {config.PREDICT_MIN_EVENT_SPAN_FRAMES}, 後処理 {_now_post()}) で実際のサービス関数と比較')
    bad = total = 0
    for set_key, vids in sets.items():
        data = _load_probs(probs_root / wtag, set_key, 'none')
        for mp4, bbox, _, _ in vids:
            if mp4.stem not in data:
                continue
            pa, meta = data[mp4.stem]
            mine, _ = _events(pa, meta, config.A_THRESHOLD, coarse_prob=config.PREDICT_COARSE_PROB)
            with contextlib.redirect_stdout(io.StringIO()):
                evs = predict_events_and_clips(model, video_path=str(mp4), bbox=tuple(bbox), output_dir=tmp,
                                                render_clips=False)
            svc = [(e['event_start_frame'], e['event_end_frame']) for e in evs]
            total += 1
            if svc != mine:
                bad += 1
                near = float(np.abs(pa - config.A_THRESHOLD).min())
                print(f'  不一致 {set_key} {mp4.name}: サービス {svc} ・ 再計算 {mine} '
                      f'(閾値に最も近いウィンドウの P(A) 差分 {near:.4f})')
    print(f'  → {total}本中 {total - bad}本 一致' + ('' if not bad else
          ' — 不一致の動画は閾値近傍のウィンドウが FP16 バッチ境界により分岐した可能性を確認してください'))
    return bad, total


# ── 集計・レポート ─────────────────────────────────────────────────────────

def _grid():
    for mode in RESAMPLES:
        for post in ('frame', 'fps'):
            for ms in MIN_SPANS:
                for coarse in (True, False):
                    for t in THRESHOLDS:
                        yield mode, post, ms, coarse, t


def _cfg_name(mode, post, ms, coarse, t):
    """現行設定と異なる項目のみを名前に付与。"""
    parts = [f'閾値 {t:g}']
    if post != _now_post():
        parts.append('fps適合' if post == 'fps' else 'フレーム固定')
    if ms != config.PREDICT_MIN_EVENT_SPAN_FRAMES:
        parts.append(f'最小 {ms}(連続 {ms - (L - 2)}窓)' if ms > L - 1 else f'最小 {ms}(無効)')
    if not coarse:
        parts.append('粗探索無効')
    if mode != 'none':
        parts.append(f'時間軸 {mode}')
    return ' + '.join(parts)


def _fmt(tot):
    return (f"見逃し {tot['miss']:>2}/{tot['n_A']} ・ 誤検知動画 {tot['fa']:>2}/{tot['n_S']} ・ 誤検知イベント {tot['false_ev']:>2} ・ "
            f"正確度 {tot['acc'] * 100:5.1f}% (①見逃し {tot['base_miss']}・誤検知 {tot['base_fa']})")


def _lists(tot):
    return f"見逃し {' '.join(_short(x) for x in tot['miss_list']) or '-'} | 誤検知 {' '.join(_short(x) for x in tot['fa_list']) or '-'}"


def _aggregate(out_dir, probs_root, suffix=''):
    wdirs = sorted(p for p in probs_root.iterdir() if p.is_dir())
    summary, results, caches = [], {}, {}
    for wdir in wdirs:
        cache = {}
        for set_key, _ in SETS:
            for mode in RESAMPLES:
                cache[(set_key, mode)] = _load_probs(wdir, set_key, mode)
        for mode, post, ms, coarse, t in _grid():
            for set_key, _ in SETS:
                data = cache[(set_key, mode)]
                if not data:
                    continue
                if mode != 'none' and not any(m['factor'] > 1 for _, m in data.values()):
                    continue
                tot, _ = _score(data, t, fps_scale=post == 'fps', min_span=ms, use_coarse=coarse)
                key = (wdir.name, mode, post, ms, coarse, t)
                results.setdefault(key, {})[set_key] = tot
                summary.append(dict(weights=wdir.name, set=set_key, resample=mode, postproc=post, min_span=ms,
                                    coarse=int(coarse), threshold=t, n=tot['n'], n_A=tot['n_A'], n_S=tot['n_S'],
                                    miss=tot['miss'], fa_videos=tot['fa'], false_events=tot['false_ev'],
                                    accuracy=round(tot['acc'], 4), base_miss=tot['base_miss'], base_fa=tot['base_fa'],
                                    miss_list=' '.join(_short(x) for x in tot['miss_list']),
                                    fa_list=' '.join(_short(x) for x in tot['fa_list'])))
        _videos_csv(out_dir / f'videos{suffix}.csv', wdir, cache, write_header=wdir == wdirs[0])
        caches[wdir.name] = cache
    if summary:
        with open(out_dir / f'summary{suffix}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    _report(results, [w.name for w in wdirs])
    for wtag, cache in caches.items():
        _wrong_now(wtag, cache)


def _videos_csv(path, wdir, cache, write_header):
    """動画別詳細 — P(A) 最大値・位置、衝突区間内外の最大値、現行設定でのイベント出力等。"""
    cols = ['weights', 'set', 'resample', 'file', 'gt', 'fps', 'factor', 'width', 'height', 'box_w', 'box_h',
            'gt_start', 'gt_end', 'n_windows', 'p_max', 'p_max_frame', 'p_max_gt', 'p_max_outside',
            'events_frame', 'caught_frame', 'events_fps', 'caught_fps']
    with open(path, 'w' if write_header else 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(cols)
        for (set_key, mode), data in cache.items():
            for name, (pa, m) in data.items():
                if mode != 'none' and m['resample'] == 'none':
                    continue
                f_ = m['factor']
                ws = np.arange(len(pa))
                lo, hi = ws // f_, (ws + L - 1) // f_
                if m['gt_sf'] is not None:
                    in_gt = (lo <= m['gt_ef']) & (hi >= m['gt_sf'])
                    p_gt = f'{pa[in_gt].max():.4f}' if in_gt.any() else ''
                    p_out = f'{pa[~in_gt].max():.4f}' if (~in_gt).any() else ''
                else:
                    p_gt, p_out = '', f'{pa.max():.4f}'
                row = [wdir.name, set_key, mode, m['file'], 'A' if m['gt_sf'] is not None else 'S',
                       round(m['fps'], 2), f_, m['width'], m['height'], m['bbox'][2] - m['bbox'][0],
                       m['bbox'][3] - m['bbox'][1], m['gt_sf'], m['gt_ef'], len(pa), f'{pa.max():.4f}',
                       int(pa.argmax()) // f_, p_gt, p_out]
                for fps_scale in (False, True):
                    evs, _ = _events(pa, m, config.A_THRESHOLD, coarse_prob=config.PREDICT_COARSE_PROB,
                                     fps_scale=fps_scale)
                    caught = ('' if m['gt_sf'] is None else
                              int(any(e[0] <= m['gt_ef'] and e[1] >= m['gt_sf'] for e in evs)))
                    row += [' '.join(f'{s}-{e}' for s, e in evs), caught]
                w.writerow(row)


def _wrong_now(wtag, cache):
    """現行設定で誤判定となった動画 — モデルが未検知 (P(A) 低下) か後処理で除外されたか特定するため最大 P(A) を併記。"""
    print(f'\n[{wtag}] 現行設定 (閾値 {config.A_THRESHOLD:g}, 後処理 {_now_post()}) で誤判定となった動画 — fps ・ ボックス ・ '
          f'P(A) 最大値 (事故は衝突区間内窓 / 区間外窓)')
    for set_key, set_name in SETS:
        data = cache.get((set_key, 'none')) or {}
        if not data:
            continue
        _, rows = _score(data, config.A_THRESHOLD, coarse_prob=config.PREDICT_COARSE_PROB)
        wrong = [r for r in rows if (r['gt'] == 'A' and not r['caught']) or (r['gt'] == 'S' and r['events'])]
        night = getattr(config, 'NIGHT_VIDEOS', ()) or ()
        nr = [r for r in rows if r['name'] in night]
        if nr:     # 夜間動画のみ別途集計 (照明誤検知の分析用)
            na = [r for r in nr if r['gt'] == 'A']
            ns = [r for r in nr if r['gt'] == 'S']
            print(f'  {set_name} 夜間 {len(nr)}本: 事故 {len(na)}本中 検知 {sum(1 for r in na if r["caught"])}本 ・ '
                  f'正常 {len(ns)}本中 誤検知 {sum(1 for r in ns if r["events"])}本')
        for r in wrong:
            pa, m = data[r['name']]
            box = f"{m['bbox'][2] - m['bbox'][0]}x{m['bbox'][3] - m['bbox'][1]}"
            if r['gt'] == 'A':
                ws = np.arange(len(pa))
                in_gt = (ws <= m['gt_ef']) & (ws + L - 1 >= m['gt_sf'])
                p_gt = pa[in_gt].max() if in_gt.any() else float('nan')
                p_out = pa[~in_gt].max() if (~in_gt).any() else float('nan')
                why = 'モデル未検知' if p_gt <= config.A_THRESHOLD else 'モデルは反応 — 後処理・粗探索で除外'
                print(f'  {set_name} 見逃し {_short(r["name"]):<9}{"[夜間]" if r["name"] in night else ""} {m["fps"]:5.1f}fps ボックス {box:<8} 衝突区間最大 {p_gt:.2f} '
                      f'・ 区間外 {p_out:.2f} ・ イベント {r["events"] or "-"} — {why}')
            else:
                print(f'  {set_name} 誤検知 {_short(r["name"]):<9}{"[夜間]" if r["name"] in night else ""} {m["fps"]:5.1f}fps ボックス {box:<8} 最大 {pa.max():.2f}'
                      f'(窓 {int(pa.argmax())}) ・ イベント {r["events"]}')


def _report(results, wtags):
    post, ms_now = _now_post(), config.PREDICT_MIN_EVENT_SPAN_FRAMES
    other = 'frame' if post == 'fps' else 'fps'
    for wtag in wtags:
        print(f'\n{"=" * 100}\n[{wtag}]\n{"=" * 100}')
        for set_key, set_name in SETS:
            rows = [(k, v[set_key]) for k, v in results.items() if k[0] == wtag and set_key in v]
            if not rows:
                continue
            n = rows[0][1]
            print(f'\n── {set_name} 動画 {n["n"]}本 (事故 {n["n_A"]} ・ 正常 {n["n_S"]}) ──')
            show = []
            for t in (0.3, 0.4, 0.5, 0.6, 0.7):
                show.append(('none', post, ms_now, True, t))
            for t in (0.4, 0.5, 0.6):
                show.append(('none', other, ms_now, True, t))
            for ms in MIN_SPANS:
                if ms != ms_now:
                    show.append(('none', post, ms, True, 0.5))
            show.append(('none', post, ms_now, False, 0.5))
            for mode in ('repeat', 'blend'):
                for t in (0.4, 0.5):
                    show.append((mode, post, ms_now, True, t))
            for cfg in show:
                tot = results.get((wtag, *cfg), {}).get(set_key)
                if tot:
                    print(f'  {_cfg_name(*cfg):<38} {_fmt(tot)}  {_lists(tot)}')
        # 検証動画で選定した設定 → 評価動画で検証
        both = [(k, v) for k, v in results.items() if k[0] == wtag and 'val' in v and 'eval' in v]
        if not both:
            continue
        cur = results.get((wtag, 'none', post, ms_now, True, config.A_THRESHOLD), {})
        print(f'\n── 検証動画による選定設定 上位8件 (見逃し → 誤検知動画 → 誤検知イベント順) → 評価動画の結果 ──')
        if cur:
            print(f'  {"現行設定":<38} 検証 {_fmt(cur["val"])}\n  {"":<38} 評価 {_fmt(cur["eval"])}')

        def _rank(kv):
            k, v = kv
            t = v['val']
            change = (k[1] != 'none') + (k[2] != post) + (k[3] != ms_now) + (not k[4]) + abs(k[5] - config.A_THRESHOLD)
            return (t['miss'], t['fa'], t['false_ev'], change)
        for k, v in sorted(both, key=_rank)[:8]:
            print(f'  {_cfg_name(*k[1:]):<38} 検証 {_fmt(v["val"])}\n  {"":<38} 評価 {_fmt(v["eval"])}  {_lists(v["eval"])}')
    print('\n・ 正確度 = (区間重複した事故動画 + イベントなしの正常動画) / 全体。①は1ウィンドウでも閾値を超えた場合に事故とみなす判定。')
    print('・ 全組み合わせは summary.csv、動画別 P(A)・イベントは videos.csv、ウィンドウ別 P(A) は probs/ に保存されます。')


# ── 実行 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='判定閾値・後処理・低fps時間軸の検証')
    ap.add_argument('--weights', nargs='+', help='検証対象の重み (複数指定可能)')
    ap.add_argument('--folder', default=str(config.EVAL_FOLDER_PATH), help='評価動画フォルダ (mp4 + txt)')
    ap.add_argument('--split', help='検証動画リスト split.json (デフォルト: 初回重みの実行回フォルダのもの)')
    ap.add_argument('--no-val', action='store_true', help='検証実写動画を除外して評価動画のみ')
    ap.add_argument('--no-service-check', action='store_true', help='実際のサービス関数との再現確認をスキップ')
    ap.add_argument('--batch', type=int, default=0, help='推論バッチサイズ (0 = サービス標準 GPU 8 ・ CPU 1)')
    ap.add_argument('--again', help='過去の結果フォルダ — 保存済み P(A) で集計のみ再実行 (モデル不使用)')
    ap.add_argument('--videos', type=int, default=0, help='事故・正常を均等に指定件数のみ評価 (0 = 全件, デバッグ用)')
    args = ap.parse_args()
    if not args.weights and not args.again:
        ap.error('--weights または --again が必要です')
    if int(config.PREDICT_WINDOW_STRIDE) != 1:
        print(f'⚠️ PREDICT_WINDOW_STRIDE={config.PREDICT_WINDOW_STRIDE} — 本検証はストライド 1 基準で再計算します')

    if args.again:
        out_dir = Path(args.again)
        suffix = '_again_' + datetime.now().strftime('%y%m%d_%H%M%S')
    else:
        out_dir = Path(config._ROOT) / 'outputs' / 'postproc_check' / datetime.now().strftime('%y%m%d_%H%M%S')
        suffix = ''
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / f'report{suffix}.txt')
    sys.stdout = tee
    try:
        return _run(args, out_dir, suffix)
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[保存] {out_dir}')


def _run(args, out_dir, suffix):
    print(f'[判定閾値・後処理検証] {datetime.now():%Y-%m-%d %H:%M}')
    print(f'現行設定: 判定閾値 {config.A_THRESHOLD:g} ・ 粗探索間隔 {config.PREDICT_COARSE_STEP}・閾値 '
          f'{config.PREDICT_COARSE_PROB:g}・周辺 ±{config.PREDICT_COARSE_PAD} ・ マージ {config.PREDICT_EVENT_MERGE_GAP_SEC:g}秒 ・ '
          f'最小長 {config.PREDICT_MIN_EVENT_SPAN_FRAMES}フレーム ・ 後処理 fps 適合 '
          f'{"有効" if config.PREDICT_FPS_SCALED_POSTPROC else "無効"} ・ r {config.R_VALUE:g}')
    probs_root = out_dir / 'probs'
    if not args.again:
        sets = {'eval': _folder_videos(args.folder)}
        print(f'評価動画: {args.folder} — {len(sets["eval"])}本')
        if not args.no_val:
            splits = ([Path(args.split)] if args.split else
                      list(dict.fromkeys(p for p in (_split_path(w, None) for w in args.weights) if p)))
            if not splits:
                print('検証動画: split.json が見つからないためスキップ (--split で指定可能)')
            else:
                per = [_val_videos(p) for p in splits]
                common = set.intersection(*({str(v[0]) for v in vs} for vs in per))
                sets['val'] = [v for v in per[0] if str(v[0]) in common]
                dropped = sorted({v[0].stem for vs in per for v in vs} - {Path(c).stem for c in common})
                print(f'検証動画: {" ・ ".join(str(p) for p in splits)} の検証実写動画 {len(sets["val"])}本 '
                      f'(学習未使用動画' + (f' — 一部の回で学習に使用された {", ".join(dropped)} は除外)' if dropped else ')'))
        if args.videos:
            for k, vids in sets.items():
                sets[k] = ([v for v in vids if v[2] is not None][:(args.videos + 1) // 2]
                           + [v for v in vids if v[2] is None][:args.videos // 2])
            print(f'デバッグ実行: セットごとに {args.videos}本のみ評価')
        print('重み: ' + ' ・ '.join(str(w) for w in args.weights))
        t0 = time.perf_counter()
        for weights in args.weights:
            model, _ = _measure(weights, sets, out_dir, args.batch)
            if not args.no_service_check:
                _service_check(model, _wtag(weights), sets, probs_root)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f'\nP(A) 測定完了 — {time.perf_counter() - t0:.0f}秒')
    _aggregate(out_dir, probs_root, suffix)
    return 0


if __name__ == '__main__':
    sys.exit(main())
