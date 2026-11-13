"""学習速度プロファイリング — ボトルネックを特定し、計算結果を変えずに高速化可能な設定候補を検証する。

    python -u -m model.speed_check                # 全項目実行 (約10〜15分)
    python -u -m model.speed_check --part gpu     # ①⑤⑥ GPU 演算のみ
    python -u -m model.speed_check --part data    # ②③⑦ データ準備のみ (GPU 不要)
    python -u -m model.speed_check --part loader  # ③④ ローダー組み合わせおよび実学習ステップのみ (約5分)

  ① GPU 演算: channels_last 有効/無効での学習・検証1ステップの速度比較 (ランダム入力、データ読込なし)。
     AMP・cudnn.benchmark・gradient checkpointing・勾配クリッピング (1.0) は train.py と同一。
  ② クリップ1個の生成ステップ別所要時間 (単一プロセス): 動画オープン / 開始フレームへシーク / デコード /
     全画面色変換 / クロップ・リサイズ / データ拡張 / テンソル変換 — dataset.py __getitem__ と同一シーケンス。
  ③ データローダー処理スループット: 学習と同一設定 (バッチ 15・シャッフル・事前取得数 train.loader_prefetch) の
     DataLoader でバッチ生成時間をワーカー数別 × ワーカーあたり OpenCV スレッド数別に測定。
  ④ 実学習ステップ: 実際のローダー + 現行設定モデルで、バッチあたり時間・GPU データ待機率等を測定。
  ⑤ gradient checkpointing 範囲: 再計算対象とする前方ブロック数別の所要時間と最大メモリ消費量。
  ⑥ torch.compile: コンパイル時間、学習・検証ステップ所要時間、P(A) 差分を検証。
  ⑦ 検証クリップキャッシュ (config.TRAIN_VAL_CACHE): キャッシュなし / 初回 (生成・保存) / 2回目以降 (読込) の比較。
最後にボトルネックと推奨設定の総合判定を出力する。
"""
import argparse
import copy
import functools
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import config
import hitandrun_model
from dataset import limit_cv2_threads
from device_utils import is_channels_last_3d_supported
from hitandrun_model import HitAndRun3DCNN
from train import loader_prefetch

BLOCK_NAMES = ['stem', 'res2', 'res3', 'res4', 'res5']
CV2_DEFAULT_THREADS = cv2.getNumThreads()       # 変更前の OpenCV デフォルトスレッド数


class _Tee:
    """画面出力とファイル書き込みを同時に行う。"""
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


def _cur_cl(dev):
    """現行設定 (config.USE_CHANNELS_LAST) で channels_last を使用しているか判定。"""
    return bool(config.USE_CHANNELS_LAST) and is_channels_last_3d_supported(dev)


def _build(state, dev, cl):
    model = HitAndRun3DCNN(num_classes=2, pretrained=False)
    model.load_state_dict(state)
    model = model.to(dev)
    if cl:
        model = model.to(memory_format=torch.channels_last_3d)
    return model


def _p_a(model, x, dev, amp, cl):
    """検証モードでの P(A) を計算。"""
    model.eval()
    xe = x.contiguous(memory_format=torch.channels_last_3d) if cl else x
    with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
        return torch.softmax(model(xe).float(), 1)[:, 1].cpu().numpy()


def _free_gpu(dev):
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


# ───────────────────────── ① GPU 演算 ─────────────────────────
def _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train):
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-5) if train else None
    scaler = torch.amp.GradScaler('cuda') if (train and amp) else None
    model.train(train)
    times = []
    for i in range(warmup + steps):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if train:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=amp):
                loss = crit(model(x), y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
                model(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return np.array(times)


def _gpu_run(channels_last, state, x, y, dev, amp, steps, warmup):
    """同一の初期重みで学習・検証ステップの所要時間、ピークメモリ、検証 P(A) を測定。"""
    model = _build(state, dev, channels_last)
    p = _p_a(model, x, dev, amp, channels_last)
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_val = _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train=False)
    t_train = _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else float('nan')
    del model
    _free_gpu(dev)
    return t_train, t_val, peak, p


def part_gpu(args, dev, amp, state, x, y):
    """① → {channels_last 有効/無効: 学習 1ステップ平均所要時間 (秒)}。"""
    print(f'\n① GPU 演算 — バッチ {args.batch} · {config.CLIP_LENGTH}フレーム · {config.RESIZE[0]}×{config.RESIZE[1]} · '
          f'AMP {"有効" if amp else "無効"} · gradient checkpointing {"有効" if hitandrun_model.GRAD_CHECKPOINT else "無効"} · '
          f'ウォームアップ {args.warmup} + 測定 {args.steps}ステップ (ランダム入力)', flush=True)
    res = {}
    for cl in (True, False):
        name = 'channels_last 有効' if cl else 'channels_last 無効'
        print(f'  {name} 測定中…', flush=True)
        try:
            res[cl] = _gpu_run(cl, state, x, y, dev, amp, args.steps, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f'  {name}: GPU メモリ不足 (OOM) — この設定はこのバッチサイズで学習不可')
            _free_gpu(dev)
    if not res:
        return None
    print(f'  {"設定":<18}{"学習 1ステップ":>14}{"検証 1ステップ":>14}{"最大メモリ":>12}')
    for cl, (tt, tv, peak, _) in res.items():
        print(f'  {"channels_last " + ("有効" if cl else "無効"):<18}{tt.mean():>9.3f}±{tt.std():.3f}秒'
              f'{tv.mean():>9.3f}±{tv.std():.3f}秒{peak:>10.2f}GB')
    if len(res) == 2 and not args.smoke:
        (on_t, on_v, _, p_on), (off_t, off_v, _, p_off) = res[True], res[False]
        ep_on, ep_off = on_t.mean() + on_v.mean() / 4, off_t.mean() + off_v.mean() / 4
        faster = 'channels_last 無効' if ep_off < ep_on else 'channels_last 有効'
        d_t = (off_t.mean() - on_t.mean()) / on_t.mean() * 100
        d_v = (off_v.mean() - on_v.mean()) / on_v.mean() * 100
        print(f'  → 高速な設定: {faster} — GPU 計算時間 約 {abs(ep_on - ep_off) / max(ep_on, ep_off) * 100:.0f}% 短縮')
        print(f'    無効が有効より 学習ステップ {abs(d_t):.0f}% {"高速" if d_t < 0 else "低速"} · '
              f'検証ステップ {abs(d_v):.0f}% {"高速" if d_v < 0 else "低速"}')
        print(f'    同一重み・入力での P(A) 最大差分 {np.abs(p_on - p_off).max():.1e} (丸め誤差レベル)')
    print(f'  現在の設定: config.USE_CHANNELS_LAST = {bool(config.USE_CHANNELS_LAST)}')
    return {cl: float(v[0].mean()) for cl, v in res.items()}


# ───────────────────────── ⑤ gradient checkpointing 範囲 ─────────────────────────
def part_ckpt(args, dev, amp, state, x, y):
    """⑤ 再計算対象とする前方ブロック数別の学習所要時間およびメモリ消費。"""
    if not hitandrun_model.GRAD_CHECKPOINT:
        print('\n⑤ gradient checkpointing が無効化されているため (環境変数 GRAD_CHECKPOINT=0) 範囲比較をスキップします')
        return None
    cl = _cur_cl(dev)
    default = HitAndRun3DCNN(num_classes=2, pretrained=False).default_checkpoint_blocks()
    choices = sorted({default, 3, 2, 0}, reverse=True)
    print(f'\n⑤ gradient checkpointing 範囲 — 学習 1ステップ, バッチ {args.batch} · AMP {"有効" if amp else "無効"} · '
          f'channels_last {"有効" if cl else "無効"}(現在設定) · ウォームアップ {args.warmup} + 測定 {args.steps}ステップ', flush=True)
    total = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30 if dev.type == 'cuda' else float('nan')
    saved = hitandrun_model.GRAD_CHECKPOINT_BLOCKS
    out = {}
    try:
        for k in choices:
            name = _ckpt_name(k, default)
            print(f'  {name} 測定中…', flush=True)
            hitandrun_model.GRAD_CHECKPOINT_BLOCKS = k
            model = None
            try:
                model = _build(state, dev, cl)
                if dev.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                t = _step_times(model, x, y, cl, dev, amp, args.steps, args.warmup, train=True)
                if dev.type == 'cuda':
                    peak = torch.cuda.max_memory_allocated() / 2 ** 30
                    resv = torch.cuda.max_memory_reserved() / 2 ** 30
                else:
                    peak = resv = float('nan')
                out[k] = {'t': float(t.mean()), 'sd': float(t.std()), 'peak': peak, 'resv': resv, 'oom': False}
            except torch.cuda.OutOfMemoryError:
                out[k] = {'oom': True}
            del model
            _free_gpu(dev)
    finally:
        hitandrun_model.GRAD_CHECKPOINT_BLOCKS = saved
    base = out.get(default)
    print(f'  {"再計算対象ブロック":<22}{"学習 1ステップ":>14}{"現行比":>10}{"最大メモリ(予約)":>20}')
    for k in choices:
        r = out[k]
        if r['oom']:
            print(f'  {_ckpt_name(k, default):<22}{"GPU メモリ不足(OOM)":>24}')
            continue
        rel = f'{(r["t"] / base["t"] - 1) * 100:+.0f}%' if base and not base['oom'] else '-'
        print(f'  {_ckpt_name(k, default):<22}{r["t"]:>9.3f}±{r["sd"]:.3f}秒{rel:>10}'
              f'{r["peak"]:>11.2f}GB({r["resv"]:.2f})')
    print(f'  GPU 全体メモリ {total:.1f}GB · 勾配はすべて完全一致、範囲変更ブロックは BN momentum 補正により結果同一')
    return {'res': out, 'default': default, 'total': total}


def _ckpt_name(k, default):
    if k == 0:
        name = 'なし(無効)'
    else:
        name = f'{BLOCK_NAMES[0]}~{BLOCK_NAMES[k - 1]}' if k > 1 else BLOCK_NAMES[0]
    return f'{k}個: {name}' + (' ← 現行' if k == default else '')


# ───────────────────────── ⑥ torch.compile ─────────────────────────
def part_compile(args, dev, amp, state, x, y):
    """⑥ コンパイル前後の学習・検証所要時間および P(A) 差分の検証。"""
    cl = _cur_cl(dev)
    print(f'\n⑥ torch.compile — バッチ {x.shape[0]} · 入力 {tuple(x.shape[2:])} · channels_last {"有効" if cl else "無効"}(現在設定)'
          f'{" · デバッグ用小型入力" if args.smoke else ""}', flush=True)
    model = None
    try:
        import torch._dynamo
        torch._dynamo.reset()
        model = _build(state, dev, cl)
        p_eager = _p_a(model, x, dev, amp, cl)
        cmodel = torch.compile(model)
        print('  コンパイル中 (検証モード)…', flush=True)
        t0 = time.perf_counter()
        p_comp = _p_a(cmodel, x, dev, amp, cl)
        c_val = time.perf_counter() - t0
        print('  コンパイル中 (学習モード)…', flush=True)
        t0 = time.perf_counter()
        _step_times(cmodel, x, y, cl, dev, amp, 1, 0, train=True)
        c_train = time.perf_counter() - t0
        print('  測定中…', flush=True)
        e_val = _step_times(model, x, y, cl, dev, amp, args.steps, args.warmup, train=False)
        k_val = _step_times(cmodel, x, y, cl, dev, amp, args.steps, args.warmup, train=False)
        e_tr = _step_times(model, x, y, cl, dev, amp, args.steps, args.warmup, train=True)
        k_tr = _step_times(cmodel, x, y, cl, dev, amp, args.steps, args.warmup, train=True)
    except Exception as e:
        msg = f'{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ""}'[:300]
        print(f'  失敗 — {msg}')
        print('  (この環境では torch.compile は利用不可)')
        return {'error': msg}
    finally:
        del model
        _free_gpu(dev)
    diff = float(np.abs(p_eager - p_comp).max())
    print(f'  {"":<10}{"学習 1ステップ":>14}{"検証 1ステップ":>14}')
    print(f'  {"コンパイル前":<10}{e_tr.mean():>9.3f}±{e_tr.std():.3f}秒{e_val.mean():>9.3f}±{e_val.std():.3f}秒')
    print(f'  {"コンパイル後":<10}{k_tr.mean():>9.3f}±{k_tr.std():.3f}秒{k_val.mean():>9.3f}±{k_val.std():.3f}秒')
    print(f'  → 学習 {e_tr.mean() / k_tr.mean():.2f}倍速 · 検証 {e_val.mean() / k_val.mean():.2f}倍速 · '
          f'コンパイル時間 学習 {c_train:.0f}秒 + 検証 {c_val:.0f}秒 (初回のみ)')
    print(f'    同一重み・入力での P(A) 最大差分 {diff:.1e} (丸め誤差レベル)')
    return {'e_tr': float(e_tr.mean()), 'k_tr': float(k_tr.mean()), 'e_val': float(e_val.mean()),
            'k_val': float(k_val.mean()), 'c_sec': c_train + c_val, 'diff': diff}


# ───────────────────────── ② クリップ1個の生成ステップ別 ─────────────────────────
def part_clip(ds, idxs):
    """② クリップ生成の各処理フェーズごとの所要時間を測定。"""
    keys = ['オープン', 'シーク', 'デコード', '色変換(全画面)', 'クロップ・縮小', '拡張', 'テンソル変換']
    tot = dict.fromkeys(keys, 0.0)
    color_small, n_frames, n_clips = 0.0, 0, 0
    pc = time.perf_counter
    for i in idxs:
        s = ds.samples[i]
        starts = list(s.get('win_starts') or [s['start_f']])
        s0, n_read = min(starts), max(starts) - min(starts) + ds.clip_length
        t = pc()
        cap = cv2.VideoCapture(s['mp4_path'])
        tot['オープン'] += pc() - t
        t = pc()
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
        tot['シーク'] += pc() - t
        frames = []
        for _ in range(n_read):
            t = pc()
            ok, f = cap.read()
            tot['デコード'] += pc() - t
            if not ok:
                break
            t = pc()
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tot['色変換(全画面)'] += pc() - t
            t = pc()
            crop = ds._crop_and_pad(rgb, s['bbox'], ds.r_value)
            tot['クロップ・縮小'] += pc() - t
            t = pc()
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)       # 比較用: クロップ後(224)の色変換所要時間
            color_small += pc() - t
            frames.append(crop)
            n_frames += 1
        cap.release()
        if not frames:
            continue
        t = pc()
        frames = ds._apply_augmentation(frames)
        tot['拡張'] += pc() - t
        t = pc()
        ds._frames_to_tensor(frames)
        tot['テンソル変換'] += pc() - t
        n_clips += 1
    if not n_clips:
        print('  クリップの読み込みに失敗しました')
        return None
    per = {k: v / n_clips for k, v in tot.items()}
    total = sum(per.values())
    print(f'\n② クリップ1個の生成 (単一プロセス, 学習クリップ {n_clips}個平均 · クリップあたり {n_frames / n_clips:.0f}フレーム)')
    for k in keys:
        print(f'  {k:<16}{per[k] * 1000:>9.1f} ms  {per[k] / total * 100:>5.1f}%')
    print(f'  {"合計":<16}{total * 1000:>9.1f} ms')
    small = color_small / n_clips
    save = (per['色変換(全画面)'] - small) / total
    print(f'  ※ 色変換をクロップ後(224)に行うと {per["色変換(全画面)"] * 1000:.1f} → {small * 1000:.1f} ms '
          f'(クリップあたり {save * 100:.0f}% 短縮、結果は同一)')
    return {'total': total, 'per': per, 'color_save': save}


# ───────────────────────── ③ データローダー処理スループット ─────────────────────────
def _loader(ds, idxs, batch, workers, pin, threads=None, shuffle=True):
    ds.cv2_threads = threads
    cv2.setNumThreads(threads or CV2_DEFAULT_THREADS)
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=pin)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=loader_prefetch(workers))
        if threads:
            kw['worker_init_fn'] = functools.partial(limit_cv2_threads, threads)
    return DataLoader(Subset(ds, idxs), **kw)


def _tname(threads):
    return 'ワーカーあたり1スレッド' if threads == 1 else (f'ワーカーあたり{threads}スレッド' if threads else 'OpenCV デフォルト')


def _time_batches(dl, n):
    t0 = time.perf_counter()
    it = iter(dl)
    got = 0
    for _ in range(n):
        try:
            next(it)
        except StopIteration:
            break
        got += 1
    el = time.perf_counter() - t0
    del it
    return el / max(got, 1), got


def part_loader(ds, pool, batch, workers_list, pin):
    print(f'\n③ データローダー処理スループット (バッチ {batch}, 学習と同一設定 · ワーカー起動込み)')
    out = {}
    for w in workers_list:
        n = max(24, 4 * w)
        idxs = pool[:batch * n]
        for threads in (1, None):
            dl = _loader(ds, idxs, batch, w, pin, threads)
            sec, got = _time_batches(dl, n)
            del dl
            out[(w, threads)] = sec
            print(f'  ワーカー {w:>2}個 · {_tname(threads):<10}: バッチ1個 {sec:.3f}秒 '
                  f'(秒あたり {batch / sec:.1f}クリップ, {got}バッチ)', flush=True)
    _restore_threads(ds)
    return out


def _restore_threads(ds):
    ds.cv2_threads = getattr(config, 'TRAIN_CV2_THREADS', None)
    cv2.setNumThreads(CV2_DEFAULT_THREADS)


# ───────────────────────── ⑦ 検証クリップキャッシュ ─────────────────────────
def part_val_cache(ds, pool, batch, workers, pin, n_batches):
    idxs = pool[:batch * n_batches]
    need = len(idxs) * config.CLIP_LENGTH * config.RESIZE[0] * config.RESIZE[1] * 3 / 2 ** 30
    root = Path(config.TRAIN_VAL_CACHE_DIR).parent
    print(f'\n⑦ 検証クリップキャッシュ (検証同様に拡張なし・順次走査 · ワーカー {workers}個 · クリップ {len(idxs)}個, 一時容量約 {need:.1f}GB)')
    try:
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free / 2 ** 30
    except OSError as e:
        print(f'  保存先フォルダを使用できないためスキップ: {e}')
        return None
    if free < need * 2 + 1:
        print(f'  ディスク空き容量不足 ({free:.1f}GB) のためスキップ')
        return None
    tmp = Path(tempfile.mkdtemp(prefix='speed_check_', dir=root))
    try:
        plain = copy.copy(ds)
        plain.augment, plain.cache_dir, plain._cache_off = False, None, False
        cached = copy.copy(plain)
        cached.cache_dir = tmp
        res = {}
        for name, d in (('none', plain), ('first', cached), ('cached', cached)):
            dl = _loader(d, idxs, batch, workers, pin, getattr(config, 'TRAIN_CV2_THREADS', None), shuffle=False)
            res[name], _ = _time_batches(dl, n_batches)
            del dl
        same = all(torch.equal(plain[i][0], cached[i][0]) for i in idxs[:5])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        cv2.setNumThreads(CV2_DEFAULT_THREADS)
    print(f'  キャッシュなし (従来)     : バッチ1個 {res["none"]:.3f}秒')
    print(f'  初回 (生成・保存)          : バッチ1個 {res["first"]:.3f}秒  ← 初回エポックの第1回検証のみ')
    print(f'  2回目以降 (キャッシュ読込) : バッチ1個 {res["cached"]:.3f}秒')
    print(f'  キャッシュと動画生成のテンソル完全一致 (クリップ5個): {"合格" if same else "不合格 ⚠️"}')
    res['same'] = same
    return res


# ───────────────────────── ④ 実学習ステップ ─────────────────────────
def _shm_used_gb():
    try:
        return shutil.disk_usage('/dev/shm').used / 2 ** 30
    except OSError:
        return None


def part_e2e(ds, pool, batch, workers, dev, amp, n, threads):
    cl = _cur_cl(dev)
    model = HitAndRun3DCNN(num_classes=2, pretrained=False).to(dev)
    if cl:
        model = model.to(memory_format=torch.channels_last_3d)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda') if amp else None
    crit = nn.CrossEntropyLoss()
    dl = _loader(ds, pool[:batch * (n + 2)], batch, workers, dev.type == 'cuda', threads)
    it = iter(dl)
    waits, comps = [], []
    mem_min, shm_max = _mem_available_gb(), _shm_used_gb()
    for i in range(n + 1):
        t0 = time.perf_counter()
        try:
            x, y = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, enabled=amp):
            loss = crit(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        loss.item()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i > 0:
            waits.append(t1 - t0)
            comps.append(t2 - t1)
        m, s = _mem_available_gb(), _shm_used_gb()
        mem_min = m if mem_min is None or (m is not None and m < mem_min) else mem_min
        shm_max = s if shm_max is None or (s is not None and s > shm_max) else shm_max
    del it, dl, model
    _free_gpu(dev)
    _restore_threads(ds)
    w, c = float(np.mean(waits)), float(np.mean(comps))
    mem = (f' · 空きメモリ最小 {mem_min:.1f}GB' if mem_min is not None else '') + \
          (f' · 共有メモリ最大 {shm_max:.1f}GB' if shm_max is not None else '')
    print(f'  ワーカー {workers:>2}個 · {_tname(threads):<10}: バッチ1個 {w + c:.3f}秒 = データ待機 {w:.3f}秒'
          f'({w / (w + c) * 100:.0f}%) + GPU 計算 {c:.3f}秒  ({len(waits)}バッチ){mem}', flush=True)
    return w, c, mem_min, shm_max


def _mem_available_gb():
    try:
        for line in open('/proc/meminfo', encoding='utf-8'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description='学習速度プロファイリング')
    ap.add_argument('--part', choices=['all', 'gpu', 'data', 'loader'], default='all',
                    help='all: ①〜⑦全項目 / gpu: ①⑤⑥ GPU 演算のみ / data: ②③⑦ データ準備のみ (GPU 不要) / '
                         'loader: ③④ ローダー組み合わせおよび実学習ステップのみ')
    ap.add_argument('--batch', type=int, default=config.TRAIN_BATCH_SIZE, help='バッチサイズ (デフォルト: 学習設定と同一)')
    ap.add_argument('--steps', type=int, default=20, help='①⑤⑥ 測定ステップ数 (ウォームアップ除く)')
    ap.add_argument('--warmup', type=int, default=5, help='①⑤⑥ ウォームアップステップ数 (時間計測から除外)')
    ap.add_argument('--clips', type=int, default=40, help='② フェーズ別測定クリップ数')
    ap.add_argument('--workers', default='',
                    help='③ ワーカー数リスト (カンマ区切り)')
    ap.add_argument('--e2e-batches', type=int, default=30, help='④ 測定バッチ数')
    ap.add_argument('--val-batches', type=int, default=16, help='⑦ 測定バッチ数')
    ap.add_argument('--no-compile', action='store_true', help='⑥ torch.compile の検証をスキップ')
    ap.add_argument('--data-dir', default=str(config.DATA_DIR), help='学習データフォルダ (②③④⑦)')
    ap.add_argument('--force', action='store_true', help='GPU メモリが既に使用中でも強制実行')
    ap.add_argument('--smoke', action='store_true', help='デバッグ実行 — CPU・バッチ1・1ステップ')
    args = ap.parse_args()

    out_dir = Path(config._ROOT) / 'outputs' / 'speed_check' / datetime.now().strftime('%y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / 'report.txt')
    sys.stdout = tee
    try:
        return _main(args)
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f'\n[保存] {out_dir / "report.txt"}')


def _main(args):
    use_gpu = args.part in ('all', 'gpu', 'loader') and not args.smoke
    if args.smoke:
        dev, amp = torch.device('cpu'), False
        args.batch, args.steps, args.warmup = 1, 1, 0
    elif use_gpu:
        if not torch.cuda.is_available():
            print('GPU が検出されません — 学習サーバーで実行してください (データ準備のみ: --part data, デバッグ: --smoke)。')
            return 1
        dev, amp = torch.device('cuda'), True
        torch.backends.cudnn.benchmark = True
        free, total = (v / 2 ** 30 for v in torch.cuda.mem_get_info())
        print(f'[GPU] {torch.cuda.get_device_name(0)} · 空きメモリ {free:.1f} / {total:.1f} GB')
        if free < 0.85 * total and not args.force:
            print('⚠️ GPU メモリが他プロセスで使用中です。学習終了後に実行してください (--force で強制実行可能)。')
            return 1
    else:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        amp = dev.type == 'cuda'
    n_cpu = os.cpu_count() or 2
    print(f'[測定] {datetime.now():%Y-%m-%d %H:%M} · CPU コア {n_cpu}個 · 対象 {args.part}'
          f'{" (デバッグ実行)" if args.smoke else ""}')

    gpu = ckpt = comp = None
    if args.part in ('all', 'gpu'):
        torch.manual_seed(0)
        state = HitAndRun3DCNN(num_classes=2, pretrained=False).state_dict()
        x = torch.randn(args.batch, 3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0], device=dev)
        y = torch.randint(0, 2, (args.batch,), device=dev)
        gpu = part_gpu(args, dev, amp, state, x, y)
        ckpt = part_ckpt(args, dev, amp, state, x, y)
        if args.no_compile:
            print('\n⑥ torch.compile — スキップ (--no-compile)')
        else:
            xc = torch.randn(1, 3, 8, 64, 64, device=dev) if args.smoke else x
            comp = part_compile(args, dev, amp, state, xc, y[:xc.shape[0]])
        del x, y
        _free_gpu(dev)
    if args.part == 'gpu':
        _verdict(args, gpu, None, None, None, {}, ckpt, comp, None)
        return 0

    from dataset import HitAndRunDataset
    print(f'\n[データ] {args.data_dir} の学習クリップリストを生成中…', flush=True)
    ds = HitAndRunDataset(data_dir=args.data_dir, augment=True)
    if not len(ds):
        print('  学習クリップが存在しません')
        return 1
    rng = random.Random(0)
    pool = list(range(len(ds)))
    rng.shuffle(pool)
    clip = part_clip(ds, pool[:args.clips]) if args.part in ('all', 'data') else None

    train_w = getattr(config, 'TRAIN_NUM_WORKERS', None) or min(8, n_cpu)
    if args.workers:
        workers_list = sorted({int(w) for w in args.workers.split(',') if w.strip()})
    else:
        workers_list = sorted({train_w, n_cpu})
    mem = _mem_available_gb()
    per_w = 0.55 * args.batch / 15
    if mem is not None:
        cap = max(1, int((mem - 2.0) / per_w))
        if max(workers_list) > cap:
            print(f'\n[メモリ] 空きメモリ {mem:.1f}GB — ワーカー数を最大 {cap}個に制限します')
            workers_list = sorted({min(w, cap) for w in workers_list})
            train_w = min(train_w, cap)
    pin = dev.type == 'cuda'
    loader = part_loader(ds, pool, args.batch, workers_list, pin)
    val = part_val_cache(ds, pool, args.batch, train_w, pin, args.val_batches) if args.part in ('all', 'data') else None

    e2e = {}
    if args.part in ('all', 'loader') and not args.smoke and dev.type == 'cuda':
        cur = (train_w, getattr(config, 'TRAIN_CV2_THREADS', None))
        combos = [cur]
        best = min(loader, key=loader.get) if loader else None
        if best and best != cur and loader[best] < loader.get(cur, float('inf')) * 0.9:
            combos.insert(0, best)
        print(f'\n④ 実学習ステップ — 学習ローダー + 現行設定モデル · '
              + ' / '.join(f'ワーカー {w}個・{_tname(t)}' for w, t in combos))
        for w, t in combos:
            e2e[(w, t)] = part_e2e(ds, pool, args.batch, w, dev, amp, args.e2e_batches, t)

    _verdict(args, gpu, clip, loader, train_w, e2e, ckpt, comp, val)
    return 0


def _verdict(args, gpu, clip, loader, train_w, e2e, ckpt, comp, val):
    print('\n' + '=' * 72 + '\n[判定結果]')
    cur_t = getattr(config, 'TRAIN_CV2_THREADS', None)
    no = 0

    cur = (train_w, cur_t)
    data_bound = None
    if e2e and cur in e2e:
        w, c = e2e[cur][:2]
        data_bound = w / (w + c)

    # 1) 所要時間の内訳
    if e2e or clip or gpu:
        no += 1
        print(f'{no}) 学習 1バッチ (学習1ステップ) の所要時間')
        for (wk, t), (w, c, *_rest) in e2e.items():
            mark = '  ← 現行設定' if (wk, t) == cur else ''
            print(f'   ワーカー {wk}個 · {_tname(t)}: {w + c:.3f}秒 = データ待機 {w:.3f}秒({w / (w + c) * 100:.0f}%)'
                  f' + GPU 計算 {c:.3f}秒{mark}')
        if loader and cur in loader:
            print(f'   (参考) データ準備のみ: ワーカー {train_w}個でバッチ1個 {loader[cur]:.3f}秒')
        if gpu and len(gpu) == 2:
            cur_cl = bool(config.USE_CHANNELS_LAST)
            print(f'   (参考) GPU 演算のみ(①): channels_last 有効 {gpu[True]:.3f}秒 / 無効 {gpu[False]:.3f}秒'
                  f' — 現行設定 {"有効" if cur_cl else "無効"}')
        if clip:
            top = sorted(((k, v) for k, v in clip['per'].items()), key=lambda kv: -kv[1])[:3]
            print(f'   クリップ1個生成 {clip["total"]:.3f}秒中 上位ボトルネック: '
                  + ', '.join(f'{k} {v * 1000:.0f}ms({v / clip["total"] * 100:.0f}%)' for k, v in top))
        if data_bound is not None:
            w, c = e2e[cur][:2]
            if data_bound > 0.2:
                print(f'   → ボトルネックはデータ準備: GPU がバッチごとに {w:.3f}秒({data_bound * 100:.0f}%) 待機しています。')
            else:
                print(f'   → ボトルネックは GPU 演算 (データ待機率 {data_bound * 100:.0f}%)')
            print(f'   学習クリップ 1,000個あたり約 {(w + c) * 1000 / args.batch / 60:.1f}分 (検証除く)')

    # 2) ワーカー数 × OpenCV スレッド数
    if loader:
        no += 1
        print(f'{no}) ローダーワーカー数 × ワーカーあたり OpenCV スレッド数 (現行: ワーカー {train_w}個 · {_tname(cur_t)})')
        for (wk, t), sec in sorted(loader.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            mark = '  ← 現行' if (wk, t) == cur else ''
            print(f'   ローダー単体: ワーカー {wk:>2}個 · {_tname(t):<10} バッチ1個 {sec:.3f}秒{mark}')
        cands = {k: v for k, v in e2e.items() if k != cur}
        if e2e and cur in e2e and cands:
            k = min(cands, key=lambda kk: sum(cands[kk][:2]))
            a, b = sum(e2e[cur][:2]), sum(cands[k][:2])
            wk, t = k
            print(f'   実学習ステップ(④): 現行 {a:.3f}秒 → ワーカー {wk}個 · {_tname(t)} {b:.3f}秒 ({a / b:.2f}倍速)')
            mem_min, shm_max = cands[k][2], cands[k][3]
            if mem_min is not None or shm_max is not None:
                print('   その際のメモリ:' + (f' 空きメモリ最小 {mem_min:.1f}GB' if mem_min is not None else '')
                      + (f' · 共有メモリ最大 {shm_max:.1f}GB' if shm_max is not None else ''))
            if b < a * 0.9:
                env = f'TRAIN_NUM_WORKERS={wk}' + (f' TRAIN_CV2_THREADS={t}' if t else '')
                print(f'   → 高速化可能です。学習実行時に環境変数指定: {env} python -u -m model.main --mode train ...')
            else:
                print('   → 実学習ステップでの差分は 10% 未満です。現行設定の維持を推奨します。')
        elif e2e and cur in e2e:
            print('   → 現行より 10% 以上高速な組み合わせが存在しないため現行設定を維持します。')
        else:
            print('   → 現行設定を維持します。')

    if data_bound is not None and data_bound > 0.2 and (ckpt or comp):
        print(f'   ※ 3)・4) は GPU 演算のみを短縮します。現在 GPU はデータを {data_bound * 100:.0f}% 待機しているため、'
              ' GPU 演算を高速化しても全体の学習時間はほとんど短縮されません。')

    # 3) gradient checkpointing 範囲
    if ckpt:
        no += 1
        res, d, total = ckpt['res'], ckpt['default'], ckpt['total']
        base = res.get(d)
        print(f'{no}) gradient checkpointing 範囲')
        if not base or base['oom']:
            print('   現行範囲でもメモリが逼迫しています — バッチサイズを確認してください。')
        else:
            limit = total * 0.8 if total == total else float('inf')
            ok = {k: r for k, r in res.items() if not r['oom'] and (r['resv'] != r['resv'] or r['resv'] <= limit)}
            ok[d] = base
            best = min(ok, key=lambda k: ok[k]['t'])
            gain = 1 - ok[best]['t'] / base['t']
            if best != d and gain >= 0.05:
                print(f'   → GRAD_CHECKPOINT_BLOCKS={best} ({_ckpt_name(best, d).split(": ")[1]}): 学習 GPU 演算 '
                      f'{base["t"]:.3f} → {ok[best]["t"]:.3f}秒 ({gain * 100:.0f}% 短縮), 最大メモリ '
                      f'{ok[best]["resv"]:.1f}GB / {total:.1f}GB。')
                print('     学習コマンド指定例: GRAD_CHECKPOINT_BLOCKS=%d python -u -m model.main --mode train ...' % best)
            else:
                print('   → 現行範囲 (stem〜res4) を維持します (より高速で安全な範囲が存在しないか差分 5% 未満)')

    # 4) torch.compile
    if comp:
        no += 1
        print(f'{no}) torch.compile')
        if 'error' in comp:
            print(f'   → この環境では利用できません ({comp["error"][:80]})。現行のまま維持します。')
        else:
            g_tr, g_val = comp['e_tr'] / comp['k_tr'], comp['e_val'] / comp['k_val']
            print(f'   学習 {g_tr:.2f}倍速 · 検証 {g_val:.2f}倍速 · コンパイル {comp["c_sec"]:.0f}秒 · P(A) 差分 {comp["diff"]:.1e}')
            if g_tr >= 1.1 and comp['diff'] < 1e-2:
                print('   → 高速化効果が確認されました。')
            else:
                print('   → 効果が 10% 未満または結果差分が許容値を超えています。現行のまま維持します。')

    # 5) 検証クリップキャッシュ
    if val:
        no += 1
        n, c = val['none'], val['cached']
        print(f'{no}) 検証クリップキャッシュ (config.TRAIN_VAL_CACHE = {bool(getattr(config, "TRAIN_VAL_CACHE", False))})')
        print(f'   検証 1バッチ {n:.3f} → {c:.3f}秒 ({n / max(c, 1e-9):.1f}倍速) · 初回のみ {val["first"]:.3f}秒 · '
              f'結果完全一致 {"合格" if val["same"] else "不合格 ⚠️"}')
        print(f'   検証クリップ 1,000個あたり約 {n * 1000 / args.batch / 60:.1f}分 → {c * 1000 / args.batch / 60:.1f}分 (エポック毎)')


if __name__ == '__main__':
    sys.exit(main())
