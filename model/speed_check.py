"""学習速度の診断・ベンチマーク — ボトルネックを特定し、結果を変えずに高速化可能な設定候補を測定する。

    python -u -m model.speed_check                # 全体(約10〜15分)
    python -u -m model.speed_check --part gpu     # ①⑤⑥ GPU計算のみ
    python -u -m model.speed_check --part data    # ②③⑦ データ準備のみ(GPU不要)
    python -u -m model.speed_check --part loader  # ③④ ローダー組み合わせと実際の学習ステップのみ(約5分)

  ① GPU計算: 学習・検証の1ステップを channels_last 有効/無効で測定(ランダム入力、データ読み込みなし)。
     AMP・cudnn.benchmark・gradient checkpointing・勾配クリッピング(1.0)は train.py と同一。
  ② クリップ1個作成の段階別時間(単一プロセス): 動画オープン / 開始フレームへシーク / フレームデコード /
     全画面色変換 / クロップ・縮小 / データ増強 / テンソル変換 — dataset.py の __getitem__ と同一順序。
     「クロップ後の色変換」でどの程度短縮されるかも測定(結果は同一)。
  ③ データローダー処理能力: 学習と同一設定(バッチ15・シャッフル・プリフェッチバッチ数 train.loader_prefetch)のDataLoaderで
     バッチ1個準備時間をワーカー数別(デフォルト: 学習ローダーワーカー数 · CPUコア数) × ワーカーあたりOpenCVスレッド
     (デフォルト / 1個 — config.TRAIN_CV2_THREADS)で測定。
  ④ 実際の学習ステップ: 学習と同一ローダー + 現在の設定のモデルで、現在の設定および③で最も高速だった組み合わせ(10%以上
     高速な場合)それぞれについてバッチあたり時間・GPUデータ待機率・最小空きメモリ・最大共有メモリを測定。
  ⑤ gradient checkpointing 範囲: backward時に再計算する前方ブロック数(現在 4 = stem~res4 / 3 / 2 / 0 = 無効)別
     学習1ステップ時間と最大メモリ(hitandrun_model.GRAD_CHECKPOINT_BLOCKS)。勾配は同一でありBN移動平均も揃える。
  ⑥ torch.compile: コンパイル時間、学習・検証1ステップ時間、同一重み・入力でのP(A)差分。
  ⑦ 検証クリップキャッシュ(config.TRAIN_VAL_CACHE): キャッシュなし / 初回(作成しながら保存) / 2回目以降(読み込み)のバッチ1個
     所要時間と結果の一致確認。一時フォルダに作成し完了後に削除。
最後に入念にボトルネックと推奨設定を判定する。

⚠️ 学習を停止した状態(GPU・CPUがアイドル時)で実行してください — 学習中は両方を共有するため計測時間が不正確になります。
結果は画面および outputs/speed_check/<タイムスタンプ>/report.txt に出力されます。
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
CV2_DEFAULT_THREADS = cv2.getNumThreads()       # 変更前のOpenCVデフォルトスレッド数(③④⑦の後に復元)


class _Tee:
    """画面とファイルに同時に出力する。"""
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
    """現在の設定(config.USE_CHANNELS_LAST)で channels_last を使用しているか。"""
    return bool(config.USE_CHANNELS_LAST) and is_channels_last_3d_supported(dev)


def _build(state, dev, cl):
    model = HitAndRun3DCNN(num_classes=2, pretrained=False)
    model.load_state_dict(state)
    model = model.to(dev)
    if cl:
        model = model.to(memory_format=torch.channels_last_3d)
    return model


def _p_a(model, x, dev, amp, cl):
    """検証モード P(A)。"""
    model.eval()
    xe = x.contiguous(memory_format=torch.channels_last_3d) if cl else x
    with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
        return torch.softmax(model(xe).float(), 1)[:, 1].cpu().numpy()


def _free_gpu(dev):
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


# ───────────────────────── ① GPU計算 ─────────────────────────
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
    """同一の初期重みで学習ステップ・検証ステップの所要時間、最大メモリ、検証P(A)を計測する。"""
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
    """① → {channels_last 有効/無効: 学習1ステップ平均時間(秒)}。"""
    print(f'\n① GPU計算 — バッチ {args.batch} · {config.CLIP_LENGTH}フレーム · {config.RESIZE[0]}×{config.RESIZE[1]} · '
          f'AMP {"有効" if amp else "無効"} · gradient checkpointing {"有効" if hitandrun_model.GRAD_CHECKPOINT else "無効"} · '
          f'ウォームアップ {args.warmup} + 測定 {args.steps}ステップ (ランダム入力)', flush=True)
    res = {}
    for cl in (True, False):
        name = 'channels_last 有効' if cl else 'channels_last 無効'
        print(f'  {name} 測定中…', flush=True)
        try:
            res[cl] = _gpu_run(cl, state, x, y, dev, amp, args.steps, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f'  {name}: GPUメモリ不足(OOM) — この設定はこのバッチサイズで学習できません')
            _free_gpu(dev)
    if not res:
        return None
    print(f'  {"設定":<18}{"学習1ステップ":>14}{"検証1ステップ":>14}{"最大メモリ":>12}')
    for cl, (tt, tv, peak, _) in res.items():
        print(f'  {"channels_last " + ("有効" if cl else "無効"):<18}{tt.mean():>9.3f}±{tt.std():.3f}秒'
              f'{tv.mean():>9.3f}±{tv.std():.3f}秒{peak:>10.2f}GB')
    if len(res) == 2 and not args.smoke:
        (on_t, on_v, _, p_on), (off_t, off_v, _, p_off) = res[True], res[False]
        ep_on, ep_off = on_t.mean() + on_v.mean() / 4, off_t.mean() + off_v.mean() / 4
        faster = 'channels_last 無効' if ep_off < ep_on else 'channels_last 有効'
        d_t = (off_t.mean() - on_t.mean()) / on_t.mean() * 100
        d_v = (off_v.mean() - on_v.mean()) / on_v.mean() * 100
        print(f'  → 高速な方: {faster} — GPU計算時間 約 {abs(ep_on - ep_off) / max(ep_on, ep_off) * 100:.0f}% 短縮')
        print(f'    無効が有効より学習ステップ {abs(d_t):.0f}% {"高速" if d_t < 0 else "低速"} · '
              f'検証ステップ {abs(d_v):.0f}% {"高速" if d_v < 0 else "低速"}')
        print(f'    同一の重み・入力におけるP(A)の最大差 {np.abs(p_on - p_off).max():.1e} (四捨五入レベルであれば同一結果)')
    print(f'  現在の設定: config.USE_CHANNELS_LAST = {bool(config.USE_CHANNELS_LAST)}')
    return {cl: float(v[0].mean()) for cl, v in res.items()}


# ───────────────────────── ⑤ gradient checkpointing 範囲 ─────────────────────────
def part_ckpt(args, dev, amp, state, x, y):
    """⑤ 再計算する前方ブロック数ごとの学習1ステップ時間・最大メモリ → {ブロック数: 結果}, デフォルトブロック数。"""
    if not hitandrun_model.GRAD_CHECKPOINT:
        print('\n⑤ gradient checkpointing が無効化されているため(環境変数 GRAD_CHECKPOINT=0) 範囲比較をスキップします')
        return None
    cl = _cur_cl(dev)
    default = HitAndRun3DCNN(num_classes=2, pretrained=False).default_checkpoint_blocks()
    choices = sorted({default, 3, 2, 0}, reverse=True)
    print(f'\n⑤ gradient checkpointing 範囲 — 学習1ステップ、バッチ {args.batch} · AMP {"有効" if amp else "無効"} · '
          f'channels_last {"有効" if cl else "無効"}(現在の設定) · ウォームアップ {args.warmup} + 測定 {args.steps}ステップ', flush=True)
    total = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30 if dev.type == 'cuda' else float('nan')
    saved = hitandrun_model.GRAD_CHECKPOINT_BLOCKS
    out = {}
    try:
        for k in choices:
            name = _ckpt_name(k, default)
            print(f'  {name} 測定中…', flush=True)
            hitandrun_model.GRAD_CHECKPOINT_BLOCKS = k    # モデル生成時にBN momentumも調整
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
    print(f'  {"再計算対象ブロック":<22}{"学習1ステップ":>14}{"現在比":>10}{"最大メモリ(予約)":>20}')
    for k in choices:
        r = out[k]
        if r['oom']:
            print(f'  {_ckpt_name(k, default):<22}{"GPUメモリ不足(OOM)":>24}')
            continue
        rel = f'{(r["t"] / base["t"] - 1) * 100:+.0f}%' if base and not base['oom'] else '-'
        print(f'  {_ckpt_name(k, default):<22}{r["t"]:>9.3f}±{r["sd"]:.3f}秒{rel:>10}'
              f'{r["peak"]:>11.2f}GB({r["resv"]:.2f})')
    print(f'  GPU総メモリ {total:.1f}GB · 勾配はすべて同一であり、範囲が変わるブロックはBN momentumを調整して結果が同一になるように維持')
    return {'res': out, 'default': default, 'total': total}


def _ckpt_name(k, default):
    if k == 0:
        name = 'なし(無効)'
    else:
        name = f'{BLOCK_NAMES[0]}~{BLOCK_NAMES[k - 1]}' if k > 1 else BLOCK_NAMES[0]
    return f'{k}個: {name}' + (' ← 現在' if k == default else '')


# ───────────────────────── ⑥ torch.compile ─────────────────────────
def part_compile(args, dev, amp, state, x, y):
    """⑥ コンパイル前後の学習・検証1ステップ時間とP(A)の差 → 結果dict、失敗時は {'error': ...}。"""
    cl = _cur_cl(dev)
    print(f'\n⑥ torch.compile — バッチ {x.shape[0]} · 入力 {tuple(x.shape[2:])} · channels_last {"有効" if cl else "無効"}(現在の設定)'
          f'{" · スモークテスト用縮小入力" if args.smoke else ""}', flush=True)
    model = None
    try:
        import torch._dynamo
        torch._dynamo.reset()
        model = _build(state, dev, cl)
        p_eager = _p_a(model, x, dev, amp, cl)
        cmodel = torch.compile(model)
        print('  コンパイル中(検証モード)…', flush=True)
        t0 = time.perf_counter()
        p_comp = _p_a(cmodel, x, dev, amp, cl)            # 初回呼び出し = コンパイル実行
        c_val = time.perf_counter() - t0
        print('  コンパイル中(学習モード)…', flush=True)
        t0 = time.perf_counter()
        _step_times(cmodel, x, y, cl, dev, amp, 1, 0, train=True)
        c_train = time.perf_counter() - t0
        print('  測定中…', flush=True)
        # 同一モデルのコンパイル前(model)・後(cmodel)を交互に測定 — 学習ステップによる重み更新は時間計測に無関係
        e_val = _step_times(model, x, y, cl, dev, amp, args.steps, args.warmup, train=False)
        k_val = _step_times(cmodel, x, y, cl, dev, amp, args.steps, args.warmup, train=False)
        e_tr = _step_times(model, x, y, cl, dev, amp, args.steps, args.warmup, train=True)
        k_tr = _step_times(cmodel, x, y, cl, dev, amp, args.steps, args.warmup, train=True)
    except Exception as e:                                  # コンパイラ未搭載・未対応オペレーション等
        msg = f'{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ""}'[:300]
        print(f'  失敗 — {msg}')
        print('  (この環境では torch.compile を使用できません — 学習は現在の設定のまま実行)')
        return {'error': msg}
    finally:
        del model
        _free_gpu(dev)
    diff = float(np.abs(p_eager - p_comp).max())
    print(f'  {"":<10}{"学習1ステップ":>14}{"検証1ステップ":>14}')
    print(f'  {"コンパイル前":<10}{e_tr.mean():>9.3f}±{e_tr.std():.3f}秒{e_val.mean():>9.3f}±{e_val.std():.3f}秒')
    print(f'  {"コンパイル後":<10}{k_tr.mean():>9.3f}±{k_tr.std():.3f}秒{k_val.mean():>9.3f}±{k_val.std():.3f}秒')
    print(f'  → 学習 {e_tr.mean() / k_tr.mean():.2f}倍 · 検証 {e_val.mean() / k_val.mean():.2f}倍 · '
          f'コンパイル時間 学習 {c_train:.0f}秒 + 検証 {c_val:.0f}秒(学習開始時に1回)')
    print(f'    同一の重み・入力におけるP(A)の最大差 {diff:.1e} (四捨五入レベルであれば同一結果)')
    return {'e_tr': float(e_tr.mean()), 'k_tr': float(k_tr.mean()), 'e_val': float(e_val.mean()),
            'k_val': float(k_val.mean()), 'c_sec': c_train + c_val, 'diff': diff}


# ───────────────────────── ② クリップ1個作成の段階別計測 ─────────────────────────
def part_clip(ds, idxs):
    """② 学習データセットと同一順序でクリップを作成しながら段階別の所要時間を測定 → クリップ1個平均時間(秒)。"""
    keys = ['オープン', 'シーク', 'デコード', '色変換(全画面)', 'クロップ・縮小', 'データ増強', 'テンソル変換']
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
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)       # 比較用: クロップ後(224)の色変換にかかる時間
            color_small += pc() - t
            frames.append(crop)
            n_frames += 1
        cap.release()
        if not frames:
            continue
        t = pc()
        frames = ds._apply_augmentation(frames)
        tot['データ増強'] += pc() - t
        t = pc()
        ds._frames_to_tensor(frames)
        tot['テンソル変換'] += pc() - t
        n_clips += 1
    if not n_clips:
        print('  クリップを読み込めませんでした')
        return None
    per = {k: v / n_clips for k, v in tot.items()}
    total = sum(per.values())
    print(f'\n② クリップ1個作成 (単一プロセス、学習クリップ {n_clips}個平均 · クリップあたりフレーム数 {n_frames / n_clips:.0f}枚)')
    for k in keys:
        print(f'  {k:<16}{per[k] * 1000:>9.1f} ms  {per[k] / total * 100:>5.1f}%')
    print(f'  {"合計":<16}{total * 1000:>9.1f} ms')
    small = color_small / n_clips
    save = (per['色変換(全画面)'] - small) / total
    print(f'  ※ 色変換をクロップ後(224)に行うと {per["色変換(全画面)"] * 1000:.1f} → {small * 1000:.1f} ms '
          f'(クリップあたり {save * 100:.0f}% 短縮、結果は同一)')
    return {'total': total, 'per': per, 'color_save': save}


# ───────────────────────── ③ データローダー処理能力 ─────────────────────────
def _loader(ds, idxs, batch, workers, pin, threads=None, shuffle=True):
    """学習と同一設定のローダー。threads = ワーカーごとのOpenCVスレッド数(None = デフォルト、train.py の _make_loader と同一方式)。"""
    ds.cv2_threads = threads                     # 動画デコードスレッド(ワーカーにデータセットがコピーされる際に渡される)
    # OpenCV計算スレッドは親プロセスで先に決定してワーカーに継承させる(ワーカー内で初回変更時にハングするため —
    # dataset.limit_cv2_threads のコメント参照)。ワーカーはこのローダーを初回実行する際に生成される。
    cv2.setNumThreads(threads or CV2_DEFAULT_THREADS)
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=pin)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=loader_prefetch(workers))   # train.py と同一
        if threads:
            kw['worker_init_fn'] = functools.partial(limit_cv2_threads, threads)
    return DataLoader(Subset(ds, idxs), **kw)


def _tname(threads):
    return 'ワーカーあたり1スレッド' if threads == 1 else (f'ワーカーあたり{threads}スレッド' if threads else 'OpenCVデフォルト')


def _time_batches(dl, n):
    """ローダー作成時からn番目のバッチまで(ワーカー起動時間を含む) → (バッチあたり秒数, 取得バッチ数)。"""
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
    """③ (ワーカー数, OpenCVスレッド)ごとのバッチ1個準備時間(秒)。"""
    print(f'\n③ データローダー処理能力 (バッチ {batch}、学習と同一設定 · ワーカー起動時から計測)')
    out = {}
    for w in workers_list:
        n = max(24, 4 * w)                 # ワーカー数の4倍バッチ — 初期バッチ群が同時に生成される効果を平滑化
        idxs = pool[:batch * n]
        # 1スレッドを先に測定 — 同一動画を2回目に読み込む際のディスクキャッシュの恩恵を現在の設定(デフォルト)側に割り当てるため(保守的評価)
        for threads in (1, None):
            dl = _loader(ds, idxs, batch, w, pin, threads)
            sec, got = _time_batches(dl, n)
            del dl
            out[(w, threads)] = sec
            print(f'  ワーカー {w:>2}個 · {_tname(threads):<10}: バッチ1個 {sec:.3f}秒 '
                  f'(秒あたりクリップ {batch / sec:.1f}個, {got}バッチ)', flush=True)
    _restore_threads(ds)
    return out


def _restore_threads(ds):
    ds.cv2_threads = getattr(config, 'TRAIN_CV2_THREADS', None)
    cv2.setNumThreads(CV2_DEFAULT_THREADS)


# ───────────────────────── ⑦ 検証クリップキャッシュ ─────────────────────────
def part_val_cache(ds, pool, batch, workers, pin, n_batches):
    """⑦ 検証と同様に(増強なし・順次)バッチ1個の所要時間: キャッシュなし / 初回(作成しながら保存) / 2回目以降(読み込み)。"""
    idxs = pool[:batch * n_batches]
    need = len(idxs) * config.CLIP_LENGTH * config.RESIZE[0] * config.RESIZE[1] * 3 / 2 ** 30
    root = Path(config.TRAIN_VAL_CACHE_DIR).parent
    print(f'\n⑦ 検証クリップキャッシュ (検証と同様に増強なしで順次処理 · ワーカー {workers}個 · クリップ {len(idxs)}個, 一時容量 約 {need:.1f}GB)')
    try:
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free / 2 ** 30
    except OSError as e:
        print(f'  キャッシュ保存フォルダを使用できないためスキップします: {e}')
        return None
    if free < need * 2 + 1:
        print(f'  空きディスク容量 {free:.1f}GB — 不足しているためスキップします')
        return None
    tmp = Path(tempfile.mkdtemp(prefix='speed_check_', dir=root))
    try:
        # 学習データセット(増強有効)をコピーして検証用に変更 — クリップリストの再構築を回避するため
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
        shutil.rmtree(tmp, ignore_errors=True)    # 本チェックが作成した一時フォルダのみ削除
        cv2.setNumThreads(CV2_DEFAULT_THREADS)
    print(f'  キャッシュなし(従来)   : バッチ1個 {res["none"]:.3f}秒')
    print(f'  初回(作成しながら保存) : バッチ1個 {res["first"]:.3f}秒  ← 初回学習の初回検証時に1回発生')
    print(f'  2回目以降(キャッシュ読み込み) : バッチ1個 {res["cached"]:.3f}秒')
    print(f'  キャッシュから生成した入力が動画から直接生成したものと一致するか(クリップ5個): {"はい" if same else "いいえ ⚠️"}')
    print('  ※ 直前に書き込んだファイルであるためメモリ(ディスクキャッシュ)から読み出された可能性があります — サーバーメモリがキャッシュ(約5GB)より十分にあれば実運用でも同様です')
    res['same'] = same
    return res


# ───────────────────────── ④ 実際の学習ステップ ─────────────────────────
def _shm_used_gb():
    """共有メモリ(/dev/shm)使用量(GB) — ローダーワーカーが生成したバッチがここに蓄積される。Linux専用、読み取れない場合はNone。"""
    try:
        return shutil.disk_usage('/dev/shm').used / 2 ** 30
    except OSError:
        return None


def part_e2e(ds, pool, batch, workers, dev, amp, n, threads):
    """④ 実際のローダー + モデル(現在の設定)でバッチあたり時間とデータ待機時間を測定 → (待機時間, 計算時間, 最小空きメモリ, 最大共有メモリ)。"""
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
        loss.item()                                  # 学習コードと同様に毎ステップ値を取得(同期化)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i > 0:                                    # 初回バッチ(ワーカー初期化・cudnn自動探索)を除外
            waits.append(t1 - t0)
            comps.append(t2 - t1)
        m, s = _mem_available_gb(), _shm_used_gb()  # プリフェッチされたバッチが蓄積された時点のメモリ使用量
        mem_min = m if mem_min is None or (m is not None and m < mem_min) else mem_min
        shm_max = s if shm_max is None or (s is not None and s > shm_max) else shm_max
    del it, dl, model
    _free_gpu(dev)
    _restore_threads(ds)
    w, c = float(np.mean(waits)), float(np.mean(comps))
    mem = (f' · 最小空きメモリ {mem_min:.1f}GB' if mem_min is not None else '') + \
          (f' · 最大共有メモリ {shm_max:.1f}GB' if shm_max is not None else '')
    print(f'  ワーカー {workers:>2}個 · {_tname(threads):<10}: バッチ1個 {w + c:.3f}秒 = データ待機 {w:.3f}秒'
          f'({w / (w + c) * 100:.0f}%) + GPU計算 {c:.3f}秒  ({len(waits)}バッチ){mem}', flush=True)
    return w, c, mem_min, shm_max


def _mem_available_gb():
    """空きメモリ(GB)。Linux(/proc/meminfo)専用、読み取れない場合はNone。"""
    try:
        for line in open('/proc/meminfo', encoding='utf-8'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description='学習速度の診断・ベンチマーク')
    ap.add_argument('--part', choices=['all', 'gpu', 'data', 'loader'], default='all',
                    help='all: ①~⑦ / gpu: ①⑤⑥ GPU計算のみ / data: ②③⑦ データ準備のみ(GPU不要) / '
                         'loader: ③④ ローダー組み合わせと実際の学習ステップのみ')
    ap.add_argument('--batch', type=int, default=config.TRAIN_BATCH_SIZE, help='バッチサイズ(デフォルト学習と同一)')
    ap.add_argument('--steps', type=int, default=20, help='①⑤⑥ 測定ステップ数(ウォームアップを除く)')
    ap.add_argument('--warmup', type=int, default=5, help='①⑤⑥ ウォームアップステップ(cudnn最適アルゴリズム探索) — 計測から除外')
    ap.add_argument('--clips', type=int, default=40, help='② 段階別に測定するクリップ数')
    ap.add_argument('--workers', default='',
                    help='③ ワーカー数リスト(カンマ区切り)。デフォルト: 学習と同一の min(8, CPU) および CPUコア数')
    ap.add_argument('--e2e-batches', type=int, default=30, help='④ 測定するバッチ数')
    ap.add_argument('--val-batches', type=int, default=16, help='⑦ 測定するバッチ数')
    ap.add_argument('--no-compile', action='store_true', help='⑥ torch.compile をスキップ')
    ap.add_argument('--data-dir', default=str(config.DATA_DIR), help='学習データフォルダ(②③④⑦)')
    ap.add_argument('--force', action='store_true', help='GPUメモリが既に使用されていても強制実行')
    ap.add_argument('--smoke', action='store_true', help='コード検査用 — CPU・バッチ1・1ステップ(時間計測の意味なし)')
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
            print('GPUが検出されません — 学習サーバーで実行してください(データ準備のみ: --part data, 動作確認: --smoke)。')
            return 1
        dev, amp = torch.device('cuda'), True
        torch.backends.cudnn.benchmark = True
        free, total = (v / 2 ** 30 for v in torch.cuda.mem_get_info())
        print(f'[GPU] {torch.cuda.get_device_name(0)} · 空きメモリ {free:.1f} / {total:.1f} GB')
        if free < 0.85 * total and not args.force:
            print('⚠️ GPUメモリを他のプロセス(学習等)が使用しています。学習完了後に実行してください(--force で無視可能)。')
            return 1
    else:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        amp = dev.type == 'cuda'
    n_cpu = os.cpu_count() or 2
    print(f'[測定] {datetime.now():%Y-%m-%d %H:%M} · CPUコア {n_cpu}個 · パート {args.part}'
          f'{" (テスト用)" if args.smoke else ""}')

    gpu = ckpt = comp = None
    if args.part in ('all', 'gpu'):
        torch.manual_seed(0)
        state = HitAndRun3DCNN(num_classes=2, pretrained=False).state_dict()
        x = torch.randn(args.batch, 3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0], device=dev)
        y = torch.randint(0, 2, (args.batch,), device=dev)
        gpu = part_gpu(args, dev, amp, state, x, y)
        ckpt = part_ckpt(args, dev, amp, state, x, y)
        if args.no_compile:
            print('\n⑥ torch.compile — スキップ(--no-compile)')
        else:
            # 動作確認用(CPU)は小さな入力でコードパスのみ検証 — 元のサイズはCPUコンパイル・メモリ負荷が大きすぎるため
            xc = torch.randn(1, 3, 8, 64, 64, device=dev) if args.smoke else x
            comp = part_compile(args, dev, amp, state, xc, y[:xc.shape[0]])
        del x, y
        _free_gpu(dev)
    if args.part == 'gpu':
        _verdict(args, gpu, None, None, None, {}, ckpt, comp, None)
        return 0

    # ②③④⑦ — 学習と同一データセット(増強有効)からランダムにサンプリングしたクリップ
    from dataset import HitAndRunDataset
    print(f'\n[データ] {args.data_dir} の学習クリップ一覧を構築中…', flush=True)
    ds = HitAndRunDataset(data_dir=args.data_dir, augment=True)
    if not len(ds):
        print('  学習クリップが存在しません')
        return 1
    rng = random.Random(0)
    pool = list(range(len(ds)))
    rng.shuffle(pool)
    clip = part_clip(ds, pool[:args.clips]) if args.part in ('all', 'data') else None

    # train.py 学習ローダーと同一のワーカー数(config.TRAIN_NUM_WORKERS, 指定なし時は min(8, CPU))
    train_w = getattr(config, 'TRAIN_NUM_WORKERS', None) or min(8, n_cpu)
    if args.workers:
        workers_list = sorted({int(w) for w in args.workers.split(',') if w.strip()})
    else:
        workers_list = sorted({train_w, n_cpu})
    # ワーカー1基がバッチ1個(約0.55GB, バッチ15)を保持するため、空きメモリから上限を算出
    mem = _mem_available_gb()
    per_w = 0.55 * args.batch / 15
    if mem is not None:
        cap = max(1, int((mem - 2.0) / per_w))
        if max(workers_list) > cap:
            print(f'\n[メモリ] 空きメモリ {mem:.1f}GB — ワーカー {cap}個までに制限します(ワーカーあたり約 {per_w:.2f}GB)')
            workers_list = sorted({min(w, cap) for w in workers_list})
            train_w = min(train_w, cap)
    pin = dev.type == 'cuda'
    loader = part_loader(ds, pool, args.batch, workers_list, pin)
    val = part_val_cache(ds, pool, args.batch, train_w, pin, args.val_batches) if args.part in ('all', 'data') else None

    e2e = {}
    if args.part in ('all', 'loader') and not args.smoke and dev.type == 'cuda':
        # 現在の設定 + ③でローダーが最も高速だった組み合わせ(現在より10%以上高速な場合)。候補を先に測定 — 同一動画を
        # 2回目に読み込む際のディスクキャッシュの恩恵を現在の設定側に割り当てるため(保守的評価)
        cur = (train_w, getattr(config, 'TRAIN_CV2_THREADS', None))
        combos = [cur]
        best = min(loader, key=loader.get) if loader else None
        if best and best != cur and loader[best] < loader.get(cur, float('inf')) * 0.9:
            combos.insert(0, best)
        print(f'\n④ 実際の学習ステップ — 学習と同一ローダー + 現在の設定のモデル · '
              + ' / '.join(f'ワーカー {w}個·{_tname(t)}' for w, t in combos))
        for w, t in combos:
            e2e[(w, t)] = part_e2e(ds, pool, args.batch, w, dev, amp, args.e2e_batches, t)

    _verdict(args, gpu, clip, loader, train_w, e2e, ckpt, comp, val)
    return 0


def _verdict(args, gpu, clip, loader, train_w, e2e, ckpt, comp, val):
    """結果を一目で判断できるように: 何に時間がかかっているか / 変更推奨設定は何か。"""
    print('\n' + '=' * 72 + '\n[判定]')
    cur_t = getattr(config, 'TRAIN_CV2_THREADS', None)
    no = 0

    cur = (train_w, cur_t)
    data_bound = None                               # 現在の設定でGPUがデータを待機している割合
    if e2e and cur in e2e:
        w, c = e2e[cur][:2]
        data_bound = w / (w + c)

    # 1) 何に時間がかかっているか
    if e2e or clip or gpu:
        no += 1
        print(f'{no}) 学習バッチ1個(学習ステップ1回)の所要時間')
        for (wk, t), (w, c, *_rest) in e2e.items():
            mark = '  ← 現在の設定' if (wk, t) == cur else ''
            print(f'   ワーカー {wk}個 · {_tname(t)}: {w + c:.3f}秒 = データ待機 {w:.3f}秒({w / (w + c) * 100:.0f}%)'
                  f' + GPU計算 {c:.3f}秒{mark}')
        if loader and cur in loader:
            print(f'   (参考) データ準備のみ: ワーカー {train_w}個でバッチ1個 {loader[cur]:.3f}秒')
        if gpu and len(gpu) == 2:
            cur_cl = bool(config.USE_CHANNELS_LAST)
            print(f'   (参考) GPU計算のみ(①): channels_last 有効 {gpu[True]:.3f}秒 / 無効 {gpu[False]:.3f}秒'
                  f' — 現在の設定 {"有効" if cur_cl else "無効"}'
                  + ('' if args.smoke or gpu[cur_cl] <= min(gpu.values()) * 1.05
                      else ' ⚠️ 他方の設定が高速(config.USE_CHANNELS_LAST)'))
        if clip:
            top = sorted(((k, v) for k, v in clip['per'].items()), key=lambda kv: -kv[1])[:3]
            print(f'   クリップ1個作成 {clip["total"]:.3f}秒 中で所要時間が長い段階: '
                  + ', '.join(f'{k} {v * 1000:.0f}ms({v / clip["total"] * 100:.0f}%)' for k, v in top))
        if data_bound is not None:
            w, c = e2e[cur][:2]
            if data_bound > 0.2:
                print(f'   → ボトルネックはデータ準備: GPUがバッチごとに {w:.3f}秒({data_bound * 100:.0f}%) データを待機しています。')
            else:
                print(f'   → ボトルネックはGPU計算(データ待機 {data_bound * 100:.0f}%) — データ側を最適化しても全体の速度は向上しません。')
            print(f'   学習クリップ1,000個あたり約 {(w + c) * 1000 / args.batch / 60:.1f}分(検証を除く)')

    # 2) ローダーワーカー数 × ワーカーあたりOpenCVスレッド
    if loader:
        no += 1
        print(f'{no}) ローダーワーカー数 × ワーカーあたりOpenCVスレッド (現在: ワーカー {train_w}個 · {_tname(cur_t)})')
        for (wk, t), sec in sorted(loader.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            mark = '  ← 現在' if (wk, t) == cur else ''
            print(f'   ローダー単体: ワーカー {wk:>2}個 · {_tname(t):<10} バッチ1個 {sec:.3f}秒{mark}')
        cands = {k: v for k, v in e2e.items() if k != cur}
        if e2e and cur in e2e and cands:
            k = min(cands, key=lambda kk: sum(cands[kk][:2]))
            a, b = sum(e2e[cur][:2]), sum(cands[k][:2])
            wk, t = k
            print(f'   実際の学習ステップ(④): 現在 {a:.3f}秒 → ワーカー {wk}個 · {_tname(t)} {b:.3f}秒 ({a / b:.2f}倍)')
            mem_min, shm_max = cands[k][2], cands[k][3]
            if mem_min is not None or shm_max is not None:
                print('   その際のメモリ:' + (f' 最小空きメモリ {mem_min:.1f}GB' if mem_min is not None else '')
                      + (f' · 最大共有メモリ {shm_max:.1f}GB' if shm_max is not None else '')
                      + ' (検証ローダーワーカーはこの計測に含まれていません — 実際の学習時は若干増加します)')
            if b < a * 0.9:
                env = f'TRAIN_NUM_WORKERS={wk}' + (f' TRAIN_CV2_THREADS={t}' if t else '')
                print(f'   → 高速化されます。学習コマンドの先頭に環境変数を指定: {env} python -u -m model.main --mode train ...'
                      f'  (または config) — 結果(画素・順序)は同一です。')
                if mem_min is not None and mem_min < 4:
                    print('     ⚠️ 空きメモリが4GB未満に低下しました — 他のタスクと同時実行するとメモリ不足の可能性があります。')
            else:
                print('   → 実際の学習ステップでは10%未満の差異です。現在の設定のままで問題ありません。')
        elif e2e and cur in e2e:
            print('   → ③で現在より10%以上高速なローダー組み合わせが存在しないため、現在の設定のままとします。')
        elif cur not in loader:
            print(f'   → 現在の設定(ワーカー {train_w}個 · {_tname(cur_t)})を測定していないため比較できません(--workers を確認)。')
        else:
            k = min(loader, key=loader.get)
            if k != cur and loader[k] < loader[cur] * 0.9:
                print(f'   → ローダー単体ではワーカー {k[0]}個 · {_tname(k[1])} が {loader[cur]:.3f} → {loader[k]:.3f}秒で高速です。 '
                      '実際の学習ステップはGPU環境で --part loader を指定して確認してください。')
            else:
                print('   → ローダー単体では現在より10%以上高速な組み合わせが存在しません。')

    if data_bound is not None and data_bound > 0.2 and (ckpt or comp):
        print(f'   ※ 3)・4)はGPU計算のみを短縮します。現在はGPUがデータを {data_bound * 100:.0f}% 待機しているため、GPU計算を短縮しても'
              ' 学習ステップはほぼ高速化されません — データ側を先に最適化した後に再計測してください。')

    # 3) gradient checkpointing 範囲
    if ckpt:
        no += 1
        res, d, total = ckpt['res'], ckpt['default'], ckpt['total']
        base = res.get(d)
        print(f'{no}) gradient checkpointing 範囲')
        if not base or base['oom']:
            print('   現在の範囲でもメモリが不足しています — バッチサイズを確認してください。')
        else:
            # 学習中の他のリソース(検証ローダー・メモリ断片化)を考慮し、GPU全体の80%以内を安全マージンとする
            limit = total * 0.8 if total == total else float('inf')
            ok = {k: r for k, r in res.items() if not r['oom'] and (r['resv'] != r['resv'] or r['resv'] <= limit)}
            ok[d] = base                                   # 現在の範囲は既に使用中のため候補に登録
            best = min(ok, key=lambda k: ok[k]['t'])
            gain = 1 - ok[best]['t'] / base['t']
            if best != d and gain >= 0.05:
                print(f'   → GRAD_CHECKPOINT_BLOCKS={best} ({_ckpt_name(best, d).split(": ")[1]}): 学習GPU計算 '
                      f'{base["t"]:.3f} → {ok[best]["t"]:.3f}秒({gain * 100:.0f}% 短縮), 最大メモリ '
                      f'{ok[best]["resv"]:.1f}GB / {total:.1f}GB.')
                print('     学習コマンドの先頭に環境変数を指定: GRAD_CHECKPOINT_BLOCKS=%d python -u -m model.main --mode train ...' % best)
                print('     勾配は同一であり、BN移動平均も揃えるため結果は同一です。⚠️ span学習(--a-mode span)は1ステップ内のウィンドウが')
                print('     最大2倍となるためメモリ消費が増加します — span時は現在の範囲を維持するか個別測定してください。')
            else:
                print('   → 現在の範囲(stem~res4)を維持します(より高速で安全な範囲が存在しない、または5%未満の微差のため)'
                      + (f'。安全基準: 予約メモリ {limit:.1f}GB 以下' if limit != float('inf') else ''))

    # 4) torch.compile
    if comp:
        no += 1
        print(f'{no}) torch.compile')
        if 'error' in comp:
            print(f'   → この環境では使用できません({comp["error"][:80]})。現在の設定のままとします。')
        else:
            g_tr, g_val = comp['e_tr'] / comp['k_tr'], comp['e_val'] / comp['k_val']
            print(f'   学習 {g_tr:.2f}倍 · 検証 {g_val:.2f}倍 · コンパイル {comp["c_sec"]:.0f}秒 · P(A)差 {comp["diff"]:.1e}')
            if g_tr >= 1.1 and comp['diff'] < 1e-2:
                print('   → 有効な効果が確認されました。学習コードに有効化設定を追加する価値があります(現状未実装 — 結果は丸め誤差レベルの差のみ)。')
            else:
                print('   → 改善効果が10%未満であるか結果の乖離が大きいため、現在の設定のままとします。')

    # 5) 検証クリップキャッシュ
    if val:
        no += 1
        n, c = val['none'], val['cached']
        print(f'{no}) 検証クリップキャッシュ(config.TRAIN_VAL_CACHE = {bool(getattr(config, "TRAIN_VAL_CACHE", False))})')
        print(f'   検証バッチ1個 {n:.3f} → {c:.3f}秒 ({n / max(c, 1e-9):.1f}倍) · 初回1回 {val["first"]:.3f}秒 · '
              f'結果一致 {"はい" if val["same"] else "いいえ ⚠️"}')
        print(f'   検証クリップ1,000個あたり約 {n * 1000 / args.batch / 60:.1f}分 → {c * 1000 / args.batch / 60:.1f}分(エポックごと)')


if __name__ == '__main__':
    sys.exit(main())
