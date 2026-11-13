"""過去の学習のエポック別重みから検証指標を再計算 — 2026-10-29 追加指標を過去セッションにも補完。

学習中の検証（train.validate）と同一関数・同一検証動画（セッションフォルダ split.json）で測定。
  ・既存指標（バランス損失・再現率）も再算出して学習CSVと整合性を検証（検算 — 一致すれば同一検証の再現が確認可能）
  ・初窓再現率 : 衝突開始 TRAIN_A_PRE_FRAMES 前から1ウィンドウ（'pre' 学習の val 再現率と同一定義）
  ・区間再現率 : 接近〜衝突終了ウィンドウ（最大 TRAIN_SPAN_MAX_WINDOWS）のいずれかで A 判定 — 本番判定に近い
  ・識別能     : 衝突サンプル（区間ウィンドウ中の最高スコア）が非衝突クリップより P(A) が高い確率 — 判定閾値に非依存
学習方式（pre/span）によらず定義が同一のため、pre 学習と span 学習をこの指標で相互比較可能。
--threshold で判定閾値を変更して再集計可能（検証データで閾値を策定する場合 — 評価動画では策定しない）。

    python -m model.val_recheck --run outputs/trainlogs/<セッションフォルダ>                  # 全エポック(epochs/)
    python -m model.val_recheck --run outputs/trainlogs/<セッションフォルダ> --epochs 35,41,43 --domain real

成果物: セッションフォルダの val_recheck_<日時>[_th<閾値>].csv · .txt。エポック別重みがない学習（10-22以前）は
work_best・work_cand_ep* のみを対象とする。
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Subset

import config
from dataset import HitAndRunDataset, span_collate
from device_utils import get_device, is_channels_last_3d_supported, is_cuda_like
from evaluate import _Tee
from hitandrun_model import HitAndRun3DCNN, check_weights
from train import _make_loader, validate

# 検証クリップ構成に影響する設定 — 学習当時（config_snapshot.py）と異なる場合検算が一致しない可能性がある
_SAMPLE_KEYS = ('CLIP_LENGTH', 'RESIZE', 'R_VALUE', 'NORM_MEAN', 'NORM_STD', 'TRAIN_A_PRE_FRAMES',
                'TRAIN_A_SLICE_ENABLED', 'TRAIN_A_SLICE_STRIDE', 'TRAIN_S_SLICE_ENABLED', 'TRAIN_S_SLICE_STRIDE',
                'TRAIN_S_MAX_CLIPS_PER_VIDEO', 'TRAIN_S_REAL_USE_CAR1', 'TRAIN_SPAN_WINDOW_STRIDE',
                'TRAIN_SPAN_MAX_WINDOWS', 'TRAIN_SPAN_FIRST_WEIGHT', 'TRAIN_CLASS_WEIGHTS')
_METRICS = ('bal', 'rec', 'prec', 'preRec', 'prePrec', 'spanRec', 'spanPrec', 'auc')


def _snapshot_diff(run_dir):
    """学習当時の設定と現在の設定が異なるキー [(キー, 当時, 現在)]。スナップショットがない場合は None。"""
    snap = run_dir / 'config_snapshot.py'
    if not snap.exists():
        return None
    spec = importlib.util.spec_from_file_location('_run_config_snapshot', snap)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — 旧スナップショットが読めなくとも再計算は実行可能
        print(f'⚠️ 設定スナップショットを読み込めないため比較をスキップします: {exc}')
        return None
    return [(k, getattr(mod, k, None), getattr(config, k, None)) for k in _SAMPLE_KEYS
            if getattr(mod, k, None) != getattr(config, k, None)]


def _weights(run_dir, epochs):
    """[(エポック番号または名称, 重みパス)]。epochs/ がない場合は work_best・work_cand_ep*。"""
    found = []
    for p in sorted((run_dir / 'epochs').glob('*_ep[0-9][0-9][0-9].pth')):
        found.append((int(re.search(r'_ep(\d+)\.pth$', p.name).group(1)), p))
    if not found:
        for p in sorted(run_dir.glob('work_cand_ep*.pth')):
            found.append((int(re.search(r'_ep(\d+)\.pth$', p.name).group(1)), p))
        if (run_dir / 'work_best.pth').exists():
            found.append(('best', run_dir / 'work_best.pth'))
    if epochs:
        found = [(e, p) for e, p in found if e in epochs]
    return found


def _val_indices(dataset, split, data_dir):
    """split.json の検証動画 → (rc サンプル番号, real サンプル番号, 未検出の検証動画件数)。

    split.json は train.py がデータフォルダ基準の相対パス（os.path.relpath）で記録するため同様に照合 —
    学習時のPCとデータフォルダ配置が異なっていても動作可能。
    """
    def _key(p):
        return str(p).replace('\\', '/')     # 異種OS(Windows)で学習したセッションも読込可能

    # 10fps未満の動画(config.EXCLUDE_VIDEOS)は除外 — 2026-11-04以前のセッション検証(ReA_0005含む)も
    # 現在の検証(model/split_lock.json)と同一動画で測定。学習CSVとの検算はその分差分が生じる可能性がある。
    excl = set(getattr(config, 'EXCLUDE_VIDEOS', ()) or ())
    want = {_key(v) for v in split['val']
            if os.path.splitext(os.path.basename(_key(v)))[0] not in excl}
    rc, real, seen = [], [], set()
    roots = [Path(r).resolve() for r in (data_dir if isinstance(data_dir, (list, tuple)) else [data_dir])]
    for i, s in enumerate(dataset.samples):
        vk = Path(s.get('video_key', s['mp4_path'])).resolve()
        root = next((r for r in roots if r in vk.parents), roots[-1])
        key = _key(os.path.relpath(vk, root))
        if key in want:
            seen.add(key)
            (real if 'realdata' in Path(s['mp4_path']).parts else rc).append(i)
    return rc, real, len(want - seen)


def _f(v):
    return '-' if v is None else f'{v:.3f}'


def main():
    ap = argparse.ArgumentParser(description='過去学習の検証指標再計算')
    ap.add_argument('--run', required=True, help='学習セッションフォルダ(outputs/trainlogs/<セッション>)')
    ap.add_argument('--epochs', default='', help='対象エポック(カンマ区切り、例: 35,41,43)。空の場合は全件')
    ap.add_argument('--domain', choices=('all', 'real', 'rc'), default='all',
                    help='検証動画 — real のみ指定時は数倍高速化')
    ap.add_argument('--threshold', type=float, default=None,
                    help=f'判定閾値 P(A)(デフォルト: config.A_THRESHOLD = {config.A_THRESHOLD:g})')
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    if not (run_dir / 'split.json').exists():
        print(f'エラー: セッションフォルダに split.json が存在しません -> {run_dir}')
        return 1
    th = config.A_THRESHOLD if args.threshold is None else args.threshold
    stamp = datetime.now().strftime('%y%m%d_%H%M%S')
    th_tag = '' if abs(th - 0.5) < 1e-9 else f'_th{th:g}'
    out_base = run_dir / f'val_recheck_{stamp}{th_tag}'
    tee = _Tee(out_base.with_suffix('.txt'))
    sys.stdout = tee
    try:
        return _run(args, run_dir, th, out_base)
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[保存] {out_base}.csv · .txt')


def _run(args, run_dir, th, out_base):
    span = '_span' in run_dir.name           # 学習方式 — 既存指標(損失・再現率)を学習時と同様に算出するため
    epochs = {int(e) for e in args.epochs.split(',') if e.strip()} if args.epochs else None
    weights = _weights(run_dir, epochs)
    if not weights:
        print(f'エラー: 測定対象の重みが見つかりません(epochs/ · work_best.pth) -> {run_dir}')
        return 1
    split = json.loads((run_dir / 'split.json').read_text(encoding='utf-8'))
    data_dir = Path(split.get('data_dir') or config.DATA_DIR)
    if not data_dir.exists():
        data_dir = Path(config.DATA_DIR)

    print(f'[検証再計算] {datetime.now():%Y-%m-%d %H:%M} · セッション {run_dir.name} · '
          f'A 方式 {"span" if span else "pre"} · 判定閾値 P(A) {th:g} · 重み {len(weights)}件')
    diff = _snapshot_diff(run_dir)
    if diff:
        print('⚠️ 検証クリップに影響する設定が学習当時と異なります — 検算が一致しない可能性があります:')
        for k, old, new in diff:
            print(f'    {k}: 当時 {old!r} → 現在 {new!r}')

    device = get_device(config.TRAIN_DEVICE_TYPE)
    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True
    channels_last = bool(config.USE_CHANNELS_LAST) and is_channels_last_3d_supported(device)
    # 検証動画の探索優先順位: 検証フォルダ(config.VAL_DATA_DIR, 11-04以降) → セッションの学習フォルダ。
    # 検証動画がすべて検証フォルダに存在する場合は学習フォルダを読み込まない(高速化)。
    roots = []
    vdir = Path(split.get('val_dir') or getattr(config, 'VAL_DATA_DIR', '') or '')
    if str(vdir) not in ('', '.') and not vdir.exists():
        vdir = Path(getattr(config, 'VAL_DATA_DIR', '') or '')
    if str(vdir) not in ('', '.') and vdir.exists():
        roots.append(vdir)
    if not roots or any(not (roots[0] / v.replace('\\', '/')).exists() for v in split['val']):
        roots.append(data_dir)
    dataset = HitAndRunDataset(
        data_dir=roots, augment=False, span_windows=True,
        cache_dir=config.TRAIN_VAL_CACHE_DIR if getattr(config, 'TRAIN_VAL_CACHE', False) else None)
    rc_idx, real_idx, missing = _val_indices(dataset, split, roots)
    print(f'[検証動画] {" + ".join(str(r) for r in roots)} — セッションの検証動画 {len(split["val"])}件中 未検出 {missing}件 · '
          f'クリップ rc {len(rc_idx)} / real {len(real_idx)}')
    if missing:
        print('⚠️ 不足している検証動画があるため学習時と検証セットが異なります(検算に差分が生じる可能性あり)')
    n_ex = sum(os.path.splitext(os.path.basename(v.replace('\\', '/')))[0]
               in (getattr(config, 'EXCLUDE_VIDEOS', ()) or ()) for v in split['val'])
    if n_ex:
        print(f'⚠️ セッション検証中 10fps未満(config.LOW_FPS_VIDEOS) {n_ex}件を除外して測定 — 学習CSV検算に差分が'
              f'生じる場合があります(当時通り測定する場合は EXCLUDE_LOW_FPS=0 を設定し動画を配置)')
    doms = [d for d in ('rc', 'real') if args.domain in ('all', d)]
    idx = {'rc': rc_idx, 'real': real_idx}
    loaders = {d: _make_loader(Subset(dataset, idx[d]), batch_size=config.TRAIN_BATCH_SIZE, shuffle=False,
                               device=device, collate_fn=span_collate) if idx[d] else None for d in doms}

    rows_csv = {}
    log = next(iter(sorted(run_dir.glob('trainlog_*.csv'))), None)
    if log is not None:
        with open(log, encoding='utf-8') as fh:
            rows_csv = {r['epoch']: r for r in csv.DictReader(fh) if r.get('epoch')}

    model = HitAndRun3DCNN(num_classes=config.MODEL_NUM_CLASSES).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    out_rows, n_check, n_ok = [], 0, 0
    t0 = time.perf_counter()
    for k, (ep, path) in enumerate(weights):
        check_weights(path)                  # 学習時の入力サイズ(_s256)・ヘッドプーリング(_top3)が現在設定と一致する必要あり
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        row = {'epoch': ep, 'weights': path.name}
        old = rows_csv.get(str(ep), {})
        notes = []
        for d in doms:
            m = validate(model, loaders[d], device, span=span, criterion=torch.nn.CrossEntropyLoss(),
                         batch_size=config.TRAIN_BATCH_SIZE, first_w=float(config.TRAIN_SPAN_FIRST_WEIGHT),
                         channels_last=channels_last, threshold=th) if loaders[d] else None
            for key in _METRICS:
                row[f'{d}_{key}'] = None if m is None else m[key]
            # 検算: 学習CSVのバランス損失・再現率と一致するか(閾値が0.5以外の場合は再現率を除外)
            if m is not None and old.get(f'val_{d}_bal'):
                ok = abs(float(old[f'val_{d}_bal']) - m['bal']) < 2e-3
                if abs(th - 0.5) < 1e-9 and old.get(f'val_{d}_recallA') and m['rec'] is not None:
                    ok = ok and abs(float(old[f'val_{d}_recallA']) - m['rec']) < 1e-6
                row[f'{d}_check'] = 'ok' if ok else 'diff'
                n_check += 1
                n_ok += ok
                if not ok:
                    notes.append(f"{d} バランス 記録 {float(old[f'val_{d}_bal']):.4f}/再計算 {m['bal']:.4f}")
        out_rows.append(row)
        el = time.perf_counter() - t0
        print(f'  [{k + 1}/{len(weights)}] ep {ep}: '
              + ' | '.join(f"{d} 初窓 {_f(row[f'{d}_preRec'])} 区間 {_f(row[f'{d}_spanRec'])} "
                            f"区間適合 {_f(row[f'{d}_spanPrec'])} 識別能 {_f(row[f'{d}_auc'])}" for d in doms)
              + (f'  ⚠️ {"; ".join(notes)}' if notes else '')
              + f'  · {el / (k + 1):.0f}秒/件, 残り約 {el / (k + 1) * (len(weights) - k - 1) / 60:.1f}分')

    cols = ['epoch', 'weights'] + [f'{d}_{c}' for d in doms for c in (*_METRICS, 'check')]
    with open(f'{out_base}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in out_rows:
            w.writerow(['' if r.get(c) is None else (f'{r[c]:.6f}' if isinstance(r[c], float) else r[c])
                        for c in cols])
    print(f'\n[検算] 学習 CSV と一致 {n_ok}/{n_check}'
          + ('' if n_check else ' (比較対象レコードなし)')
          + ('' if n_ok == n_check else ' — 不一致項目は検証動画・設定・デバイス(CPU/GPU)の差異を確認'))
    dom = 'real' if 'real' in doms and real_idx else doms[0]
    top = sorted((r for r in out_rows if r.get(f'{dom}_spanRec') is not None),
                 key=lambda r: (-r[f'{dom}_spanRec'], -(r[f'{dom}_auc'] or 0)))[:5]
    if top:
        print(f'[{dom} 区間再現率上位] ' + ', '.join(
            f"ep {r['epoch']}({_f(r[f'{dom}_spanRec'])}·識別能 {_f(r[f'{dom}_auc'])})" for r in top)
            + ' — 候補選定であり最終決定は評価動画で実施')
    return 0


if __name__ == '__main__':
    sys.exit(main())

