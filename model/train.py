import csv
import functools
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import config
from dataset import HitAndRunDataset, limit_cv2_threads, span_collate
from device_utils import get_device, is_cuda_like, is_channels_last_3d_supported
import hitandrun_model
from hitandrun_model import HitAndRun3DCNN
from infer_runtime import is_accident
from runlog import make_run_dir, write_run_info
from train_balance import build as build_balance, resolve_csv


class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        # 最終ファイル名命名規則に使用する値 — best重みが保存されたエポック(1-index)
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'早期終了カウント: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if val_loss < self.val_loss_min:
            print(
                f'評価指標の改善 ({self.val_loss_min:.6f} --> {val_loss:.6f})。モデルを保存中...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
            self.best_epoch = epoch


def _unique_path(path):
    """pathが既に存在する場合は末尾に _2, _3 … を付与したパス（他の学習結果を上書きしないようにする）。"""
    path, n = Path(path), 1
    out = path
    while out.exists():
        n += 1
        out = path.with_name(f'{path.stem}_{n}{path.suffix}')
    return out


def loader_prefetch(workers):
    """ワーカーごとに事前生成しておくバッチ数 — ワーカー数に関わらず全体で約32バッチになるように調整する。

    バッチ1つが共有メモリ約0.27GB（バッチサイズ15）のため、ワーカーを24に増やして4バッチずつ保持すると最大96バッチ（約26GB）まで
    蓄積する可能性がある。ワーカー8なら従来同様4、24なら1。
    """
    return max(1, 32 // max(1, workers))


def span_weights(sizes, first_w, device):
    """span: ウィンドウごとの（重み、所属サンプル番号）。サンプルごとの重みの合計 = 1。

    ウィンドウ1つ（非衝突・短い衝突）なら1、複数の場合は最初のウィンドウ first_w + 残りが 1 − first_w を等分する。
    """
    w, owner = [], []
    for i, k in enumerate(sizes.tolist()):
        w += [1.0] if k == 1 else [first_w] + [(1.0 - first_w) / (k - 1)] * (k - 1)
        owner += [i] * k
    return (torch.tensor(w, device=device),
            torch.tensor(owner, device=device, dtype=torch.long))


def _run_windows(model, x, batch_size, channels_last=False):
    """検証: ウィンドウをバッチサイズ単位に分割して推論する（fp32、'pre' 検証と同条件）。

    spanはウィンドウ数がサンプル数より多いため、一度に入力するとメモリ不足になる可能性がある。→ ロジット (N, 2)
    """
    outs = []
    for i in range(0, x.size(0), batch_size):
        chunk = x[i:i + batch_size]
        if channels_last:
            chunk = chunk.contiguous(memory_format=torch.channels_last_3d)
        outs.append(model(chunk).float())
    return torch.cat(outs)


def _max_window(logits, sizes):
    """サンプルごとに衝突スコア（logit_A − logit_S）が最も高いウィンドウの位置（全ウィンドウ基準）。"""
    score = logits[:, 1] - logits[:, 0]
    picks, off = [], 0
    for k in sizes.tolist():
        picks.append(off + int(score[off:off + k].argmax()))
        off += k
    return picks


def _auc(pos, neg):
    """衝突サンプルのスコアが非衝突サンプルより高くなる確率（同点は0.5） — 判定しきい値に依存しない識別力。"""
    if not pos or not neg:
        return None
    p, n = torch.tensor(pos)[:, None], torch.tensor(neg)[None, :]
    return float((p > n).double().mean() + 0.5 * (p == n).double().mean())


def validate(model, loader, device, *, span, criterion, batch_size, class_w=None,
             first_w=0.5, channels_last=False, threshold=None):
    """1つの検証ローダー → 指標dict（ローダーがなければNone）。学習中の検証とval_recheckが共用。

    span              : 学習がspan方式かどうか — 損失・従来判定を学習方式に沿って算出。
    loss/acc/rec/prec : 従来の指標。判定は P(A) > 基準（config.A_THRESHOLD、デフォルト 0.5 = argmax）。
                        'pre' は先頭ウィンドウ、span は1つでもAならA（スコアが最も高いウィンドウ）。
    lossA / lossS     : 衝突・非衝突クリップそれぞれの平均損失（重みなしCE）。
    bal               : (lossA + lossS) / 2 — クラス均衡損失。
    preRec/prePrec    : 学習方式に関わらず先頭ウィンドウ1つ（衝突開始 TRAIN_A_PRE_FRAMES フレーム前から）で判定。
                        'pre' 学習の rec/prec と同一。
    spanRec/spanPrec  : 学習方式に関わらず接近〜衝突終了ウィンドウ（最大 TRAIN_SPAN_MAX_WINDOWS）のうち
                        1つでもAならA — 実サービス判定に近い（区間再現率）。span 学習の rec/prec と同一。
    auc               : 識別力 — 衝突サンプル（区間ウィンドウ中の最高スコア）が非衝突クリップよりP(A)が高い確率。
                        判定基準に依存しない（1が完璧、0.5はランダム）。
    ローダーがウィンドウまとめ（span_collate）でない場合、サンプルごとにウィンドウ1つのためpre・span指標は同一。

    ⚠️ val_real は51クリップ中40件が非衝突のため、単純な損失の78%が「非衝突を
       どれだけ確信しているか」で決まってしまう。目標は衝突を見逃さないことなので、
       best判定には両クラスを均等に評価する bal を使用する。
       acc も A:S の不均衡によりすべてSと予測しても高くなってしまうため、recall を重視する。
    """
    if loader is None:
        return None
    cuda_like = is_cuda_like(device)
    model.eval()
    vloss, correct = 0.0, 0
    cnt = {k: [0, 0, 0] for k in ('base', 'pre', 'span')}   # 判定方式別 [tp, fp, fn] — A(=1) 基準
    sum_a = sum_s = 0.0
    n_a = n_s = 0
    score_a, score_s = [], []
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                inputs, sizes, labels = batch
            else:
                inputs, labels = batch
                sizes = torch.ones(labels.size(0), dtype=torch.long)
            inputs = inputs.to(device, non_blocking=cuda_like)
            labels = labels.to(device, non_blocking=cuda_like)
            logits = _run_windows(model, inputs, batch_size, channels_last)
            first = (torch.cumsum(sizes, 0) - sizes).tolist()    # 各サンプルの先頭ウィンドウ位置
            best = _max_window(logits, sizes)
            if span:
                # 損失: 学習と同じ重み（先頭ウィンドウ + 残りウィンドウ）でサンプルごとの損失を算出。
                # 判定(acc・recall・prec): サービス同様に1つでもAならA —
                #   衝突スコアが最も高いウィンドウで判定する。
                w, owner = span_weights(sizes, first_w, device)
                ce = torch.zeros(labels.size(0), device=device).index_add_(
                    0, owner, w * F.cross_entropy(logits, labels[owner], reduction='none'))
                outputs = logits[best]
                cls_w = (class_w[labels] if class_w is not None
                         else torch.ones_like(ce))
                vloss += ((cls_w * ce).sum() / cls_w.sum()).item() * labels.size(0)
            else:
                outputs = logits[first]
                vloss += criterion(outputs, labels).item() * labels.size(0)
                ce = F.cross_entropy(outputs.float(), labels, reduction='none')
            is_a = labels == 1
            sum_a += ce[is_a].sum().item()
            sum_s += ce[~is_a].sum().item()
            n_a += int(is_a.sum().item())
            n_s += int((~is_a).sum().item())
            for key, out in (('base', outputs), ('pre', logits[first]), ('span', logits[best])):
                preds = is_accident(out, threshold)
                if key == 'base':
                    correct += torch.sum(preds == is_a).item()
                c = cnt[key]
                c[0] += torch.sum(preds & is_a).item()
                c[1] += torch.sum(preds & ~is_a).item()
                c[2] += torch.sum(~preds & is_a).item()
            score = (logits[best][:, 1] - logits[best][:, 0]).tolist()
            for sc, a in zip(score, is_a.tolist()):
                (score_a if a else score_s).append(sc)
    n = len(loader.dataset)
    loss_a = sum_a / n_a if n_a else None
    loss_s = sum_s / n_s if n_s else None

    def _rp(key):
        tp, fp, fn = cnt[key]
        return (tp / (tp + fn) if (tp + fn) else None,
                tp / (tp + fp) if (tp + fp) else None)

    (rec, prec), (pre_rec, pre_prec), (span_rec, span_prec) = _rp('base'), _rp('pre'), _rp('span')
    return {
        'loss': vloss / n, 'acc': correct / n,
        'rec': rec, 'prec': prec,
        'lossA': loss_a, 'lossS': loss_s,
        # 検証セットに一方のクラスが存在しない場合は均衡損失を計算できない
        'bal': (loss_a + loss_s) / 2 if (n_a and n_s) else None,
        'preRec': pre_rec, 'prePrec': pre_prec, 'spanRec': span_rec, 'spanPrec': span_prec,
        'auc': _auc(score_a, score_s),
    }


def _make_loader(dataset, batch_size, shuffle, device, collate_fn=None, batch_sampler=None, workers=None):
    # CUDA: マルチワーカー + prefetch 有効化（① Ryzen 7 5700X 16スレッド基準）
    # CPU : Windows マルチプロセッシングの競合防止およびデバイスコンテキスト共有不可の問題のため単一プロセスを使用
    # workers: ワーカー数（None = min(8, CPU)）。学習ローダーのみ config.TRAIN_NUM_WORKERS を渡す。
    if is_cuda_like(device):
        num_workers = workers or min(8, os.cpu_count() or 2)
    else:
        num_workers = 0
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': is_cuda_like(device),
    }
    if batch_sampler is not None:
        # バッチ構成（train_balance）がエポックごとにバッチを直接組み立てる — batch_size・shuffle の代わり
        kwargs['batch_sampler'] = batch_sampler
    else:
        kwargs.update({'batch_size': batch_size, 'shuffle': shuffle})
    if collate_fn is not None:
        kwargs['collate_fn'] = collate_fn
    if num_workers > 0:
        # ② prefetch — GPU演算中に次のバッチを先行準備（ワーカー8なら4、全体約32バッチ — loader_prefetch）
        kwargs.update({'persistent_workers': True, 'prefetch_factor': loader_prefetch(num_workers)})
        # ③ ワーカーごとのOpenCVスレッド数（config.TRAIN_CV2_THREADS、デフォルト None = OpenCV既定値）。
        #    親プロセスで事前に決定してワーカーへ継承させる（ワーカー内で初回変更すると停止する — dataset.py コメント）。
        #    動画デコードスレッドは dataset.py が同一値で初期化する。
        threads = getattr(config, 'TRAIN_CV2_THREADS', None)
        if threads:
            limit_cv2_threads(threads)
            kwargs['worker_init_fn'] = functools.partial(limit_cv2_threads, threads)
    return DataLoader(dataset, **kwargs)


def train_model(
    data_dir=config.DATA_DIR,
    num_classes=config.MODEL_NUM_CLASSES,
    batch_size=config.TRAIN_BATCH_SIZE,
    num_epochs=config.TRAIN_NUM_EPOCHS,
    clip_length=config.CLIP_LENGTH,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    save_path=config.TRAIN_BEST_MODEL_SAVE_PATH,
    train_split_ratio=config.TRAIN_SPLIT_RATIO,
    early_stopping_patience=config.TRAIN_EARLY_STOPPING_PATIENCE,
    learning_rate=None,
    use_amp=config.USE_AMP,
    use_channels_last=config.USE_CHANNELS_LAST,
    dry_run=False,
):
    """dry_run=True の場合はデータ分割・バッチ構成のみを出力して学習は行わない（実行回フォルダも作成しない）。"""
    device = get_device(config.TRAIN_DEVICE_TYPE)
    cuda_like = is_cuda_like(device)
    # 学習率: 指定がない場合は事前学習（ファインチューニング）とscratchで既定値が異なる（config コメント参照）。
    # 事前学習時はバックボーンを ×0.1 と保守的に、scratch 時は保護すべき事前学習重みがないため
    # バックボーンもヘッドと同一の学習率で学習する。
    if learning_rate is None:
        learning_rate = (config.TRAIN_LEARNING_RATE if config.PRETRAINED
                         else config.TRAIN_SCRATCH_LEARNING_RATE)
    backbone_lr_mult = 0.1 if config.PRETRAINED else 1.0
    print(f"使用中のデバイス: {device}")

    if cuda_like:
        torch.backends.cudnn.benchmark = True

    # 学習用（増強あり） / 検証用（増強なし）データセットをそれぞれ生成し、同一インデックスで分割。
    # → 検証セットは毎エポック決定論的となり、val loss が安定し best model 選択の信頼性が向上する。
    train_dataset = HitAndRunDataset(
        data_dir=data_dir, clip_length=clip_length, r_value=r_value,
        resize=resize, augment=True,
    )
    # 検証フォルダ（config.VAL_DATA_DIR）に動画が存在する場合、そのフォルダ全体が検証用、学習フォルダ全体が学習用 — 分割計算なし
    # （2026-11-04〜、python -m model.val_move --apply で移動済み）。空の場合は学習フォルダ内で分割（後述）。
    val_dir = getattr(config, 'VAL_DATA_DIR', None)
    val_dir = Path(val_dir) if val_dir else None
    use_val_dir = val_dir is not None and val_dir.exists() and any(val_dir.rglob('*.mp4'))
    # 検証クリップは毎エポック同一のため、初回生成分を保存して再読み込み（config.TRAIN_VAL_CACHE、結果は同一）
    val_dataset = HitAndRunDataset(
        data_dir=val_dir if use_val_dir else data_dir, clip_length=clip_length, r_value=r_value,
        resize=resize, augment=False,
        cache_dir=config.TRAIN_VAL_CACHE_DIR if getattr(config, 'TRAIN_VAL_CACHE', False) else None,
        # 衝突サンプルを接近〜衝突終了ウィンドウのまとめに — 従来指標は先頭ウィンドウ(= 'pre' のウィンドウ)、区間再現率は全ウィンドウ
        span_windows=not getattr(config, 'TRAIN_A_SLICE_ENABLED', False),
    )
    # 層化分割（stratified）: グループを細分化し、各グループを train_split_ratio(8:2) で分割。
    #   rc  : 方向(N/S/L/R) × クラス(A/S) → 最大8グループ（各方向・クラスが train/val に均等に分布）
    #   real: クラス(A/S)のみ → 2グループ（実動画には方向の概念がない）
    #   → train は全グループの学習分（80%）を統合して学習、val は rc用 / real用 を分離してそれぞれ測定。
    def _domain(mp4_path):
        parts = os.path.normpath(mp4_path).split(os.sep)
        return 'real' if 'realdata' in parts else 'rc'

    def _direction(file_name):
        # ファイル名例: 220510_LA_0001 → 2番目のトークンの先頭文字が方向(N/L/R/S)
        seg = file_name.split('_')
        return seg[1][0] if len(seg) > 1 and seg[1] else '?'

    def _scenario(file_name):
        # 2番目のトークンの末尾文字がシナリオ/クラスコード
        #   rc: A(衝突)/V(駐車)/S(直進)/W(徘徊),  real: A(衝突)/S(非衝突)
        seg = file_name.split('_')
        return seg[1][-1] if len(seg) > 1 and seg[1] else '?'

    # 層化グループ: rc = 方向(N/L/R/S) × シナリオ(A/V/S/W) → 最大16グループ、
    #              real = シナリオ(A/S) → 2グループ。各グループを train_split_ratio(8:2) で分割。
    #   ※ 実際の学習ラベルはそのままtxtベースの2値（A vs 非A） — シナリオは「分割の均衡」にのみ使用。
    # ⚠️ 分割は必ず「動画単位」で行う。非衝突動画は1つの動画から複数のクリップが
    #    生成されるが（Sスライシング）、それらのクリップが train と val に分かれると
    #    同じシーンを既に学習した状態で検証することになり、検証スコアが過大評価される（データリーク）。
    groups = {}  # (domain, direction, scenario) -> {video_key: [sample_idx, ...]}
    for i, s in enumerate(train_dataset.samples):
        dom = _domain(s['mp4_path'])
        direction = _direction(s['file_name']) if dom == 'rc' else '-'
        key = (dom, direction, _scenario(s['file_name']))
        vkey = s.get('video_key', s['mp4_path'])
        groups.setdefault(key, {}).setdefault(vkey, []).append(i)

    train_idx, val_rc_idx, val_real_idx = [], [], []
    split_videos = {'train': [], 'val': []}  # 実行フォルダに残す動画リスト（後でリーク確認用）
    n_train_vid = n_val_vid = 0
    # 検証固定ファイル（config.TRAIN_SPLIT_LOCK）が存在する場合、そのリストが検証用、残りがすべて学習用 — フォルダに動画を
    # 追加・削除しても検証セットが再シャッフルされない。存在しない場合は従来通りグループごとに seed 42 で 8:2。
    lock_path = getattr(config, 'TRAIN_SPLIT_LOCK', None)
    lock_path = Path(lock_path) if lock_path and Path(lock_path).exists() else None
    lock_val = set()
    if lock_path is not None:
        with open(lock_path, encoding='utf-8') as _f:
            lock_val = {v.replace('\\', '/') for v in json.load(_f)['val']}
    if use_val_dir:
        # 検証 = 検証フォルダ全体。固定リストの検証動画が学習フォルダに残っている場合（未移動）、学習に混入しないよう除外する。
        left = []
        for (dom, direction, scen), vid_map in sorted(groups.items()):
            for vk in sorted(vid_map):
                if os.path.relpath(vk, data_dir).replace('\\', '/') in lock_val:
                    left.append(vk)
                    continue
                train_idx += vid_map[vk]
                split_videos['train'].append(vk)
                n_train_vid += 1
        val_keys = []
        for i, s in enumerate(val_dataset.samples):
            (val_real_idx if _domain(s['mp4_path']) == 'real' else val_rc_idx).append(i)
            vk = s.get('video_key', s['mp4_path'])
            if not val_keys or val_keys[-1] != vk:
                val_keys.append(vk)
        split_videos['val'] = list(dict.fromkeys(val_keys))
        n_val_vid = len(split_videos['val'])
        val_rel = {os.path.relpath(v, val_dir).replace('\\', '/') for v in split_videos['val']}
        not_moved = sorted(lock_val - val_rel)
        print(f"[分割] 検証フォルダ {val_dir} — 検証動画 {n_val_vid}件（学習フォルダ {data_dir} は全件学習）"
              + (f" · 固定リスト {len(lock_val)}件中、検証フォルダ未存在 {len(not_moved)}件" if lock_path and not_moved else ""))
        if left:
            print(f"[分割] ⚠️ 固定リストの検証動画 {len(left)}件が学習フォルダに残っているため学習から除外しました — "
                  f"python -m model.val_move --apply で移動してください: "
                  + ', '.join(os.path.splitext(os.path.basename(v))[0] for v in left[:10])
                  + (' …' if len(left) > 10 else ''))
    elif lock_path is not None:
        seen = set()
        for (dom, direction, scen), vid_map in sorted(groups.items()):
            for vk in sorted(vid_map):
                rel = os.path.relpath(vk, data_dir).replace('\\', '/')
                if rel in lock_val:
                    seen.add(rel)
                    (val_real_idx if dom == 'real' else val_rc_idx).extend(vid_map[vk])
                    split_videos['val'].append(vk)
                    n_val_vid += 1
                else:
                    train_idx += vid_map[vk]
                    split_videos['train'].append(vk)
                    n_train_vid += 1
        missing = sorted(lock_val - seen)
        miss_real = [os.path.splitext(os.path.basename(m))[0] for m in missing if m.startswith('realdata/')]
        print(f"[分割] 検証固定({lock_path.name}): リスト {len(lock_val)}件中、データフォルダに存在 {len(seen)}件"
              + (f" · ⚠️ 未存在 {len(missing)}件（実動画 {', '.join(miss_real) or '0件'}, "
                 f"rc {len(missing) - len(miss_real)}件） — 検証セットが基準と異なります" if missing else ""))
    else:
        if getattr(config, 'TRAIN_SPLIT_LOCK', None):
            print(f"[分割] ⚠️ 検証固定ファイルが見つからないため従来方式(seed 42)で分割します -> {config.TRAIN_SPLIT_LOCK}")
        gen = torch.Generator().manual_seed(42)  # 再現可能な分割（seed固定）
        for (dom, direction, scen), vid_map in sorted(groups.items()):
            vkeys = sorted(vid_map)                      # 動画リスト（グループ内）
            order = torch.randperm(len(vkeys), generator=gen).tolist()
            shuffled = [vkeys[i] for i in order]
            cut = int(train_split_ratio * len(vkeys))
            for vk in shuffled[:cut]:                    # 動画丸ごと train
                train_idx += vid_map[vk]
                split_videos['train'].append(vk)
            for vk in shuffled[cut:]:                    # 動画丸ごと val
                (val_real_idx if dom == 'real' else val_rc_idx).extend(vid_map[vk])
                split_videos['val'].append(vk)
            n_train_vid += cut
            n_val_vid += len(vkeys) - cut

    # 検証の代わりに学習へ投入する動画（config.TRAIN_FORCE_TRAIN_VIDEOS） — 分割自体は維持し、検証に選ばれたもののみを移動して
    # 他の動画の分割が変化しないようにする
    force = {str(n) for n in (getattr(config, 'TRAIN_FORCE_TRAIN_VIDEOS', ()) or ())}
    forced = []
    if use_val_dir:
        # フォルダ分割後はファイル配置がそのまま分割となる — 検証フォルダに含まれている場合は通知のみ行う
        in_val = sorted(os.path.splitext(os.path.basename(v))[0] for v in split_videos['val']
                        if os.path.splitext(os.path.basename(v))[0] in force)
        if in_val:
            print(f"[分割] ⚠️ TRAIN_FORCE_TRAIN_VIDEOS {', '.join(in_val)} が検証フォルダに存在します — 学習フォルダへ移動してください")
        force = set()
    for vk in list(split_videos['val']):
        if os.path.splitext(os.path.basename(vk))[0] in force:
            idxs = [i for i, s in enumerate(train_dataset.samples) if s.get('video_key', s['mp4_path']) == vk]
            moved = set(idxs)
            val_rc_idx[:] = [i for i in val_rc_idx if i not in moved]
            val_real_idx[:] = [i for i in val_real_idx if i not in moved]
            train_idx += idxs
            split_videos['val'].remove(vk)
            split_videos['train'].append(vk)
            n_train_vid += 1
            n_val_vid -= 1
            forced.append(os.path.splitext(os.path.basename(vk))[0])
    if force:
        missing = sorted(force - set(forced) - {os.path.splitext(os.path.basename(v))[0]
                                                  for v in split_videos['train']})
        print(f"[分割] 学習側固定(config.TRAIN_FORCE_TRAIN_VIDEOS): 検証から移動 {', '.join(forced) or 'なし'}"
              + (f" · ⚠️ データ内に未存在 {', '.join(missing)}" if missing else ''))

    # 10fps未満の動画（config.EXCLUDE_VIDEOS） — 通常は lowfps_move で事前にフォルダ外へ退避済み。
    # まだ残っている環境（移動前のサーバー等）でも同様の学習結果になるよう、分割後に再度除外する（他動画の分割は維持）。
    excl = set(getattr(config, 'EXCLUDE_VIDEOS', ()) or ())
    excluded = []
    if excl:
        def _stem(v):
            return os.path.splitext(os.path.basename(v))[0]
        drop = {i for i, s in enumerate(train_dataset.samples)
                if _stem(s.get('video_key', s['mp4_path'])) in excl}
        # 検証番号は検証フォルダ使用時は val_dataset 基準、それ以外は学習データセットと同一番号
        drop_val = ({i for i, s in enumerate(val_dataset.samples) if _stem(s.get('video_key', s['mp4_path'])) in excl}
                    if use_val_dir else drop)
        if drop or drop_val:
            train_idx[:] = [i for i in train_idx if i not in drop]
            val_rc_idx[:] = [i for i in val_rc_idx if i not in drop_val]
            val_real_idx[:] = [i for i in val_real_idx if i not in drop_val]
            for part in ('train', 'val'):
                gone = [v for v in split_videos[part] if _stem(v) in excl]
                split_videos[part] = [v for v in split_videos[part] if _stem(v) not in excl]
                excluded += [_stem(v) for v in gone]
                if part == 'train':
                    n_train_vid -= len(gone)
                else:
                    n_val_vid -= len(gone)
            print(f"[除外] 10fps未満(config.LOW_FPS_VIDEOS)のうちデータフォルダに残存していた {len(excluded)}件を除外: "
                  f"{', '.join(sorted(excluded))} — python -m model.lowfps_move --apply で移動してください")

    print(f"[分割] 動画 {n_train_vid + n_val_vid}件 → train {n_train_vid} / "
          f"val {n_val_vid} (動画単位、リークなし)")
    print(f"[分割] クリップ train {len(train_idx)} / val_rc {len(val_rc_idx)} / "
          f"val_real {len(val_real_idx)}")
    print("  グループ別件数: " + ", ".join(
        f"{d}{'' if dr == '-' else '-' + dr}-{sc}:{len(v)}"
        for (d, dr, sc), v in sorted(groups.items())))
    if val_dataset.cache_dir is not None:
        have, n_val, gb = val_dataset.cache_status(val_rc_idx + val_real_idx)
        print(f"[検証キャッシュ] {val_dataset.cache_dir} — クリップ {n_val}件中 {have}件存在 · 全体約 {gb:.1f}GB "
              f"({'全件揃っているため初回検証から読み込み' if have == n_val else '未キャッシュのクリップは初回検証時に生成して保存'})")
    else:
        print("[検証キャッシュ] 無効(config.TRAIN_VAL_CACHE) — 毎エポック動画から検証クリップを生成します")
    if getattr(config, 'TRAIN_NUM_WORKERS', None):
        print(f"[ローダー] 学習ワーカー {config.TRAIN_NUM_WORKERS}基(config.TRAIN_NUM_WORKERS) · ワーカーごとに事前準備 "
              f"{loader_prefetch(config.TRAIN_NUM_WORKERS)}バッチ")
    if getattr(config, 'TRAIN_CV2_THREADS', None):
        print(f"[ローダー] ワーカーごとのOpenCVスレッド数 {config.TRAIN_CV2_THREADS}(config.TRAIN_CV2_THREADS)")
    if hitandrun_model.GRAD_CHECKPOINT and hitandrun_model.GRAD_CHECKPOINT_BLOCKS is not None:
        print(f"[チェックポインティング] 先頭ブロック {hitandrun_model.GRAD_CHECKPOINT_BLOCKS}個のみ再計算"
              f"(GRAD_CHECKPOINT_BLOCKS, 既定値 4) — 変更ブロックはBN momentumを整合させ結果同一")

    # ── A クリップ方式 (config.TRAIN_A_MODE) ──
    # 'span' の場合、衝突サンプル1件が「接近〜衝突終了」を包含する複数のウィンドウとなる。ウィンドウ全件をAとして学習するが、
    # 衝突1件あたりの重みの合計は1（先頭ウィンドウ TRAIN_SPAN_FIRST_WEIGHT + 残りが残余の重みを等分）。
    # 実行回フォルダ・重み名に _span を付与して 'pre' の結果と混同しないようにする。
    span = train_dataset.span
    run_tag = f"{config.PRETRAIN_TAG}_span" if span else config.PRETRAIN_TAG
    collate = span_collate if span else None
    first_w = float(getattr(config, 'TRAIN_SPAN_FIRST_WEIGHT', 0.5))
    step_cap = int(getattr(config, 'TRAIN_SPAN_MAX_WINDOWS_PER_STEP', None) or batch_size * 2)
    if span:
        print(f"[A方式] span — 接近〜衝突終了を複数ウィンドウで全件学習（衝突1件の重み合計 1: "
              f"先頭ウィンドウ {first_w:g} + 残りウィンドウが {1 - first_w:g} を等分）、非衝突はウィンドウ1つ")
        for dom in ('rc', 'real'):
            ws = [s['win_starts'] for s in train_dataset.samples
                  if s['label'] == 1 and _domain(s['mp4_path']) == dom and 'win_starts' in s]
            if not ws:
                continue
            ks = [len(w) for w in ws]
            gap = max((w[i + 1] - w[i] for w in ws for i in range(len(w) - 1)), default=0)
            print(f"  {dom}: 衝突 {len(ws)}件 — ウィンドウ数分布 "
                  f"{dict(sorted((k, ks.count(k)) for k in set(ks)))}, 最大ウィンドウ間隔 {gap}フレーム"
                  + (" ⚠️ 30以上の空隙があるため区間にギャップあり" if gap >= clip_length else ""))
    else:
        print("[A方式] pre — 衝突動画あたりウィンドウ1つ（衝突開始 "
              f"{getattr(config, 'TRAIN_A_PRE_FRAMES', 0)}フレーム前から）")
    if val_dataset.span_windows:
        print(f"[検証指標] 従来（学習方式準拠） + 先頭ウィンドウ再現率・区間再現率（接近〜衝突終了ウィンドウ最大 "
              f"{getattr(config, 'TRAIN_SPAN_MAX_WINDOWS', 6)}個のうち1つでもA）·識別力 — "
              f"判定基準 P(A) {config.A_THRESHOLD:g}")
    else:
        print("[検証指標] Aスライシング(TRAIN_A_SLICE_ENABLED)有効のため区間再現率 = 先頭ウィンドウ再現率")
    # 実動画 S の 1号車クリップ (config.TRAIN_S_REAL_USE_CAR1)
    car1 = [s for s in train_dataset.samples if s.get('car_id') == 1 and s['label'] == 0]
    s_car_info = (f"0号車 + 実動画Sは1号車も（1号車クリップ {len(car1)}件、"
                  f"動画 {len({s['video_key'] for s in car1})}件）" if car1 else "0号車のみ")
    print(f"[Sクリップ対象車両] {s_car_info}")

    # ── 学習バッチ構成 (config.TRAIN_S_MOTION_CSV · TRAIN_SPREAD_A, デフォルト無効) ──
    # 学習側のみ変更し、検証ローダーは維持。有効時は名前に _bal を付与して従来の学習と区別。
    motion_csv = getattr(config, 'TRAIN_S_MOTION_CSV', None)
    spread_a = bool(getattr(config, 'TRAIN_SPREAD_A', False))
    balance_on = bool(motion_csv) or spread_a
    batch_sampler, balance_lines = None, []
    if balance_on:
        if motion_csv:
            motion_csv = resolve_csv(motion_csv)
            if not motion_csv.exists():
                raise FileNotFoundError(f"TRAIN_S_MOTION_CSV ファイルが存在しません: {motion_csv}")
        batch_sampler, balance_lines = build_balance(
            train_dataset.samples, train_idx, batch_size, csv_path=motion_csv,
            empty_pct=float(getattr(config, 'TRAIN_S_EMPTY_MOTION_PCT', 0.25)),
            empty_keep=float(getattr(config, 'TRAIN_S_EMPTY_KEEP', 1 / 3)),
            spread_a=spread_a)
        run_tag = f"{run_tag}_bal"
        for line in balance_lines:
            print(f"[バッチ] {line}")
    else:
        n_a = sum(train_dataset.samples[i]['label'] == 1 for i in train_idx)
        n_s = len(train_idx) - n_a
        nb = math.ceil(len(train_idx) / batch_size)
        print(f"[バッチ] 既定(ランダム) — エポックあたり衝突 {n_a} · 正常 {n_s} = 1:{n_s / max(n_a, 1):.1f} · "
              f"バッチ {nb}個 · 衝突ゼロバッチ期待値 {nb * (n_s / len(train_idx)) ** batch_size:.0f}個")

    # ── 低画質（圧縮）増強 (config.TRAIN_LQ_AUG_PROB, デフォルト無効) ── 有効時は名前に _lq を付与して区別
    lq_p = float(getattr(config, 'TRAIN_LQ_AUG_PROB', 0) or 0)
    lq_default = (abs(float(config.TRAIN_LQ_SIZE_PROB) - 0.5) < 1e-9 and tuple(config.TRAIN_LQ_CRF_RANGE) == (28, 42))
    if lq_p > 0:
        # 10-29 の設定通りなら _lq、縮小率・CRF上限を変更した緩和設定なら _lqw（混同防止）
        run_tag = f"{run_tag}_{'lq' if lq_default else 'lqw'}"
        lo, hi = config.TRAIN_LQ_SIZE_RANGE
        c_lo, c_hi = config.TRAIN_LQ_CRF_RANGE
        print(f"[低画質増強] 有効 — 確率 {lq_p:g} · 原本クロップ {config.TRAIN_LQ_MIN_CROP}px 以上のみ · H.264 CRF {c_lo}~{c_hi}"
              f" · {config.TRAIN_LQ_SIZE_PROB:.0%}は {lo}~{hi}px に縮小後圧縮")

    # ── 光増強 (config.TRAIN_LIGHT_AUG_PROB, デフォルト無効) ── 有効時は名前に _light
    light_p = float(getattr(config, 'TRAIN_LIGHT_AUG_PROB', 0) or 0)
    if light_p > 0:
        run_tag = f"{run_tag}_light"
        print(f"[光増強] 有効 — 確率 {light_p:g} · 事故・正常で同一確率 · スキャン 60%/照射 40% · "
              f"{config.TRAIN_LIGHT_DARK_PROB:.0%}は事前に暗転処理(0.25~0.6)")

    # ── 入力サイズ (config.INPUT_SIZE, 既定 224) ── 異なる場合は名前に _s<サイズ> — 重み読み込み時にこの表記でサイズを整合
    if config.RESIZE[0] != 224:
        run_tag = f"{run_tag}_s{config.RESIZE[0]}"
        print(f"[入力サイズ] {config.RESIZE[0]}×{config.RESIZE[1]} (既定 224) — 評価・サービス側も INPUT_SIZE={config.RESIZE[0]} に設定してください")

    # ── ヘッド時間プーリング (config.HEAD_TEMPORAL_POOL, 既定 avg) ── 異なる場合は名前に _top3·_avgtop3 — 重み読み込み時にこの表記で整合
    pool = getattr(config, 'HEAD_TEMPORAL_POOL', 'avg')
    if pool != 'avg':
        run_tag = f"{run_tag}_{pool}"
        print(f"[ヘッド時間プーリング] {pool}(チャンネル・グリッドごとに時間軸上位 {config.HEAD_TEMPORAL_TOPK}個"
              f"{' 平均' if pool == 'top3' else ' 平均と30時点平均の平均'}) — 評価・サービス側も HEAD_TEMPORAL_POOL={pool} に設定してください")

    # ── fps増強 (config.TRAIN_FPS_AUG_PROB, デフォルト無効) ── 有効時は名前に _fps を付与して区別
    fps_p = float(getattr(config, 'TRAIN_FPS_AUG_PROB', 0) or 0)
    fps_ks = tuple(getattr(config, 'TRAIN_FPS_AUG_STRIDES', (2, 3)))
    if fps_p > 0:
        run_tag = f"{run_tag}_fps"
        print(f"[fps増強] 有効 — 確率 {fps_p:g} · {'·'.join(map(str, fps_ks))}フレームごとに1枚（30fps原本なら "
              f"{'·'.join(f'{30 / k:g}' for k in fps_ks)}fps 相当） · 結果 {config.TRAIN_FPS_AUG_MIN_FPS:g}fps 未満・"
              f"短尺動画・ウィンドウまとめサンプルはそのまま維持")

    if dry_run:
        if batch_sampler is not None:
            for ep in range(3):
                bl = batch_sampler.batches(ep)
                n_a_per = [sum(train_dataset.samples[train_idx[p]]['label'] == 1 for p in b) for b in bl]
                print(f"[点検] {ep + 1}エポック: バッチ {len(bl)}個 · クリップ {sum(len(b) for b in bl)}件 · "
                      f"バッチあたり衝突 {dict(sorted((k, n_a_per.count(k)) for k in set(n_a_per)))} · "
                      f"重複 {sum(len(b) for b in bl) - len({p for b in bl for p in b})}件")
        print("[点検] --dry-run のため学習を実行せずに終了します。")
        return None

    train_loader = _make_loader(
        Subset(train_dataset, train_idx), batch_size=batch_size,
        shuffle=True, device=device, collate_fn=collate, batch_sampler=batch_sampler,
        workers=getattr(config, 'TRAIN_NUM_WORKERS', None))
    val_collate = span_collate if val_dataset.span_windows else None
    val_rc_loader = _make_loader(
        Subset(val_dataset, val_rc_idx), batch_size=batch_size,
        shuffle=False, device=device, collate_fn=val_collate) if val_rc_idx else None
    val_real_loader = _make_loader(
        Subset(val_dataset, val_real_idx), batch_size=batch_size,
        shuffle=False, device=device, collate_fn=val_collate) if val_real_idx else None

    # ── 今回の学習実行回フォルダ ──────────────────────────────────────
    # outputs/trainlogs/<YYMMDD_HHMM>_<モデル>_<ptY|ptN>[_span]/ に今回の学習成果物を集約。
    # 重みファイル名にも同一時刻を付与し、どの実行回で生成された重みかを照合可能にする。
    run_started = datetime.now()
    run_dir, run_stamp = make_run_dir(
        config.TRAIN_LOG_DIR, config.MODEL_NAME, run_tag, run_started)
    print(f"[ログ] 今回の学習成果物フォルダ: {run_dir}")
    # 学習中の best・候補重み（作業用）はこの実行フォルダ内に配置する。実行フォルダは学習ごとに新規作成
    # されるため、同一条件（ptY同士など）の学習を並行実行しても互いのファイルを上書き・削除しない。
    # 学習完了時に命名規則ファイル名で weights_dir(= save_path の親フォルダ)へ移動する。
    # 途中で中断された学習の best はこのフォルダの work_best.pth に残る。
    weights_dir = Path(save_path).parent
    work_best = run_dir / 'work_best.pth'
    # エポック別重み(config.TRAIN_SAVE_ALL_EPOCHS) — 実行フォルダ内のため学習間で競合しない
    epochs_dir = run_dir / 'epochs' if getattr(config, 'TRAIN_SAVE_ALL_EPOCHS', False) else None
    if epochs_dir is not None:
        epochs_dir.mkdir(exist_ok=True)
    # エポックファイル名 = hitandrun_<モデル>_<開始時刻>[_n]_<ptY|ptN>[_span][_bal]_ep<番号>.pth
    #   → フォルダ外へ取り出してもどの学習（実行フォルダ <時刻>[_n]_<モデル>_<条件>）の何エポックか判別可能。
    #   同一分に同一条件で開始されフォルダ名に _2 が付与された場合はその番号も反映（名称重複防止）。
    run_dup = run_dir.name[len(f"{run_stamp}_{config.MODEL_NAME}_{run_tag}"):]
    epoch_prefix = f"hitandrun_{config.MODEL_NAME}_{run_stamp}{run_dup}_{run_tag}"
    # 当時の設定をそのまま保持 — 将来 config.py が変更されてもこの実行回の条件を確認可能
    shutil.copy2(config.__file__, run_dir / 'config_snapshot.py')
    # 学習/検証に使用した動画リスト — 評価動画との重複がなかったかを後から検証可能
    with open(run_dir / 'split.json', 'w', encoding='utf-8') as _f:
        root = Path(data_dir)
        json.dump({
            'data_dir': str(root),
            'n_train_videos': len(split_videos['train']),
            'n_val_videos': len(split_videos['val']),
            # 検証フォルダ使用時、val パスはそのフォルダ基準の相対パス（学習フォルダと同様の rcdata/·realdata/ 構造）
            'val_dir': str(val_dir) if use_val_dir else None,
            'split_lock': str(lock_path) if lock_path is not None else None,
            'excluded_in_folder': sorted(excluded),   # フォルダに残存していたためリスト照合で除外した10fps未満動画
            'train': sorted(os.path.relpath(v, root) for v in split_videos['train']),
            'val': sorted(os.path.relpath(v, val_dir if use_val_dir else root) for v in split_videos['val']),
        }, _f, ensure_ascii=False, indent=2)
    # 猶予期間(config.TRAIN_GRACE_EPOCHS — scratch既定 20, 事前学習 0): この間は早期終了・LR減衰・best選定を保留
    grace = getattr(config, 'TRAIN_GRACE_EPOCHS', None)
    if grace is None:
        grace = 0 if config.PRETRAINED else int(getattr(config, 'TRAIN_SCRATCH_GRACE_EPOCHS', 0))
    grace = max(0, min(int(grace), num_epochs - 1))     # 最終まで猶予だとbestが存在せず最終重みを決定できない
    if grace:
        print(f"[猶予期間] 最初の {grace}エポックは早期終了・学習率減衰・best選定を行わずに学習（エポック別重みは保存） — "
              f"以降から早期終了 {early_stopping_patience}・学習率半減を新規カウント")
    run_info = [
        ('種別', '学習'),
        ('モデル', config.MODEL_NAME),
        ('事前学習', f"{config.PRETRAIN_TAG} ("
                     f"{'Kinetics-400 事前学習' if config.PRETRAINED else 'scratch, ランダム初期化'})"),
        ('学習率', f"ヘッド {learning_rate:g} / バックボーン {learning_rate * backbone_lr_mult:g}"),
        ('Sクリップ対象車両', s_car_info),
        ('バッチ構成', ' / '.join(l.strip() for l in balance_lines) if balance_on
                     else '既定(ランダムシャッフル, 正常全件)'),
        ('Aクリップ方式', (f'span (接近〜衝突終了を複数ウィンドウで全件学習, 先頭ウィンドウ重み {first_w:g})'
                       if span else f"pre (衝突開始 {getattr(config, 'TRAIN_A_PRE_FRAMES', 0)}"
                            "フレーム前からウィンドウ1つ)")),
        ('開始', run_started.strftime('%Y-%m-%d %H:%M:%S')),
        ('終了', '(学習中または中断)'),
        ('最終重み', '(学習完了時に記録)'),
        ('作業中best重み', f'{work_best} (学習完了時に最終重みへ移動)'),
        ('エポック別重み', f'{epochs_dir} (毎エポック)' if epochs_dir is not None else '保存なし'),
        # エポック時間は本設定が同一の学習間でのみ比較（無効化時がRTX 3080で約2.3倍高速、結果は同一）
        ('GPUメモリ配置 channels_last',
         '有効' if (use_channels_last and is_channels_last_3d_supported(device)) else '無効'),
        ('データフォルダ', str(data_dir)),
        ('動画数', f"学習 {len(split_videos['train'])} / 検証 {len(split_videos['val'])}"),
        ('判定基準 P(A)', f'{config.A_THRESHOLD:g} (検証再現率・適合率・suggest F1 — 評価指標は基準に非依存)'),
        ('猶予期間', f'{grace}エポック (早期終了・学習率減衰・best選定なし)' if grace else 'なし'),
        ('入力サイズ', f'{config.RESIZE[0]}×{config.RESIZE[1]}'),
        ('ヘッド時間プーリング', pool if pool == 'avg' else f'{pool} (上位 {config.HEAD_TEMPORAL_TOPK}個)'),
        ('検証動画', f'検証フォルダ {val_dir}' if use_val_dir else
                    (f'固定リスト {lock_path.name}' if lock_path is not None else 'seed 42 分割')),
        ('10fps未満除外', f"有効 (リスト {len(getattr(config, 'LOW_FPS_VIDEOS', ()))}件, フォルダ残存除外 {len(excluded)}件)"
                          if getattr(config, 'EXCLUDE_VIDEOS', None) else '無効'),
        ('学習側固定動画', ', '.join(forced) if forced else 'なし'),
        ('クロップ r', f'{r_value:g} (学習中にrを広げていたr増強は2026-11-05削除 — それ以前の学習は40%をr 1.5~3)'),
        ('低画質増強', (f'確率 {lq_p:g} · 縮小率 {config.TRAIN_LQ_SIZE_PROB:g} · CRF {config.TRAIN_LQ_CRF_RANGE[0]}~'
                    f'{config.TRAIN_LQ_CRF_RANGE[1]}') if lq_p > 0 else '無効'),
        ('fps増強', f"確率 {fps_p:g} · {'·'.join(map(str, fps_ks))}フレームに1枚" if fps_p > 0 else '無効'),
        ('光増強', f'確率 {light_p:g} (事故・正常同一確率)' if light_p > 0 else '無効'),
    ]
    write_run_info(run_dir, run_info)

    # ── 学習曲線ログ (CSV) ───────────────────────────────────────
    # ファイル名にもモデル・事前学習有無・時刻を含め、実行フォルダ外でも判別可能。
    log_path = run_dir / (
        f"trainlog_{config.MODEL_NAME}_{run_tag}_{run_stamp}.csv")
    with open(log_path, 'w', newline='', encoding='utf-8') as _f:
        csv.writer(_f).writerow([
            'epoch', 'lr', 'train_loss',
            'val_rc_loss', 'val_rc_acc', 'val_rc_recallA', 'val_rc_precA',
            'val_real_loss', 'val_real_acc', 'val_real_recallA', 'val_real_precA',
            # ↓ クラス別平均損失と均衡損失、best評価指標、本エポック保存フラグ
            'val_rc_lossA', 'val_rc_lossS', 'val_rc_bal',
            'val_real_lossA', 'val_real_lossS', 'val_real_bal',
            'monitor', 'saved_best', 'saved_cand',
            # ↓ 2026-10-29 追加(train.validate): 先頭ウィンドウ1個再現率・適合率、区間(接近〜衝突終了ウィンドウのいずれか)
            #   再現率・適合率、識別力。学習方式(pre/span)に関わらず同一定義のため相互比較可能。
            'val_rc_preRecA', 'val_rc_prePrecA', 'val_rc_spanRecA', 'val_rc_spanPrecA', 'val_rc_auc',
            'val_real_preRecA', 'val_real_prePrecA', 'val_real_spanRecA', 'val_real_spanPrecA', 'val_real_auc',
            'grace',    # 1 = 猶予期間（早期終了・LR減衰・best選定なし — suggestで非推奨）
        ])
    print(f"[ログ] 学習曲線 CSV: {log_path}")

    def _csv(v):
        """None（該当valセットなし / 分母0）は空文字で記録。"""
        return '' if v is None else f'{v:.6f}'

    # 学習時は Kinetics-400 事前学習重みで初期化 (config.PRETRAINED)
    model = HitAndRun3DCNN(
        num_classes=num_classes, pretrained=config.PRETRAINED).to(device)

    # AMP, GradScaler: CUDA/ROCm 共通サポート
    # channels_last_3d: NVIDIA CUDA 専用 (ROCm 未対応)
    amp_enabled = use_amp and cuda_like
    channels_last_enabled = use_channels_last and is_channels_last_3d_supported(
        device)

    if channels_last_enabled:
        model = model.to(memory_format=torch.channels_last_3d)

    # SスライシングによりA:Sの不均衡が拡大するとモデルがSへ偏る可能性があるため、必要に応じて
    # config.TRAIN_CLASS_WEIGHTS で重みを付与（既定 None = 従来と同一）。
    _cw = getattr(config, 'TRAIN_CLASS_WEIGHTS', None)
    if _cw:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(_cw, dtype=torch.float32, device=device))
        print(f"[損失] クラス重み適用: {tuple(_cw)}")
    else:
        criterion = nn.CrossEntropyLoss()
    # ファインチューニングの標準慣行: 事前学習済みバックボーン(features)は低学習率(×0.1)で保守的に、
    # 新規初期化された分類ヘッド(head_conv)は基本学習率で学習（scratch時は両方基本学習率）。
    # AdamW = decoupled weight decay。過学習抑制の正則化はここで設定する。
    # (Adam + weight_decay はL2が適応的LRにより歪められ意図通りに動作しない)
    # パラメータを (事前学習/新規) × (decay/no-decay) の4グループに分類。
    #  · 事前学習部分は LR ×0.1 で保守的に（従来の動作を維持、scratch時は ×1）
    #  · ⚠️ model.named_parameters() を走査する必要がある。features/head_conv のみ
    #    列挙するとその間にあるモジュール（x3d の head_pool 0.97M）が丸ごと漏れて
    #    学習されない。
    #  · BN scale/bias および全 bias は weight decay 除外 — 標準的なプラクティス。
    #    BN の γ が 0 に引っ張られると該当チャンネル出力が消滅する。3D CNN は BN テンソルが
    #    154〜220個あるため影響が大きい。
    _norm_ids = set()
    for _m in model.modules():
        if isinstance(_m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d,
                           nn.GroupNorm, nn.LayerNorm)):
            _norm_ids.update(id(p) for p in _m.parameters(recurse=False))

    _g = {'new_d': [], 'new_n': [], 'pre_d': [], 'pre_n': []}
    for _name, _p in model.named_parameters():
        if not _p.requires_grad:
            continue
        # 新規初期化される分類ヘッドのみ基本LR、残り（事前学習分）は ×0.1
        _kind = 'new' if _name.startswith('head_conv') else 'pre'
        _decay = 'n' if (id(_p) in _norm_ids or _name.endswith('bias')) else 'd'
        _g[f'{_kind}_{_decay}'].append(_p)

    _wd = config.TRAIN_WEIGHT_DECAY
    print(f"[オプティマイザ] バックボーン {sum(p.numel() for p in _g['pre_d'])/1e6:.2f}M(decay) + "
          f"{sum(p.numel() for p in _g['pre_n'])/1e3:.1f}K(no-decay) LR {learning_rate * backbone_lr_mult:g} / "
          f"ヘッド {sum(p.numel() for p in _g['new_d'] + _g['new_n'])/1e3:.1f}K LR {learning_rate:g}, "
          f"weight_decay={_wd} ({'事前学習' if config.PRETRAINED else 'scratch'})")
    optimizer = optim.AdamW([
        {'params': _g['pre_d'], 'lr': learning_rate * backbone_lr_mult, 'weight_decay': _wd},
        {'params': _g['pre_n'], 'lr': learning_rate * backbone_lr_mult, 'weight_decay': 0.0},
        {'params': _g['new_n'], 'lr': learning_rate, 'weight_decay': 0.0},
        {'params': _g['new_d'], 'lr': learning_rate, 'weight_decay': _wd},
    ])

    # ── 安全な学習安定化機能 (モデル構造・損失関数不変、性能劣化なし) ──
    # 1) ReduceLROnPlateau: val loss が停滞した際に LR を半減させて収束を促す。
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    # 2) gradient clipping: 勾配ノルムの上限値（発散/振動の抑制）。必要に応じて調整。
    grad_clip_norm = 1.0

    if amp_enabled:
        scaler = torch.amp.GradScaler("cuda")

    early_stopping = EarlyStopping(
        patience=early_stopping_patience, path=work_best)

    # ── best評価指標 / 候補保存 (config の TRAIN_MONITOR_* コメント参照) ──
    # 評価指標 = val_real 「クラス均衡損失」の直近Nエポック平均。best保存・LR減衰・
    # 早期終了がすべてこの単一値に基づいて連動する。
    smooth_n = max(1, int(getattr(config, 'TRAIN_MONITOR_SMOOTH', 3)))
    topk = max(1, int(getattr(config, 'TRAIN_TOPK_CANDIDATES', 3)))
    min_gap = max(1, int(getattr(config, 'TRAIN_TOPK_MIN_GAP', 3)))
    bal_hist = []
    cands = []   # [{'epoch', 'score', 'path'}] — 互いに min_gap 以上離れた上位 topk
    # 候補も実行フォルダ内に配置（新規フォルダのため過去の実行ファイルと混同しない）

    def _update_candidates(epoch_no, score):
        """epoch_no の重みを候補として保持するか判定。保存した場合は True。

        単純に上位K件を選択すると 33・34・35 のように連続したエポック、すなわち実質的に同一のモデルが
        選ばれてしまう。そのため近接（min_gap未満）の候補間ではより良好な1件のみを残す。
        最良のエポックは決して除外されないため、1位は常に best と一致する。
        """
        near = [c for c in cands if abs(c['epoch'] - epoch_no) < min_gap]
        if near:
            if score >= min(c['score'] for c in near):
                return False
            for c in near:
                c['path'].unlink(missing_ok=True)
                cands.remove(c)
        elif len(cands) >= topk:
            worst = max(cands, key=lambda c: c['score'])
            if score >= worst['score']:
                return False
            worst['path'].unlink(missing_ok=True)
            cands.remove(worst)
        path = run_dir / f'work_cand_ep{epoch_no}.pth'
        torch.save(model.state_dict(), path)
        cands.append({'epoch': epoch_no, 'score': score, 'path': path})
        return True

    plot_warned = False

    _cw_t = torch.tensor(_cw, dtype=torch.float32, device=device) if _cw else None

    def _span_step(inputs, sizes, labels):
        """span学習の1ステップ → バッチ損失（サンプル平均、'pre' の損失と同尺度）。

        損失 = Σ サンプル (クラス重み × Σ ウィンドウ重み × ウィンドウCE) / Σ クラス重み
             — 全サンプルがウィンドウ1つの場合は nn.CrossEntropyLoss と同一。
        ウィンドウ数が step_cap を超える場合はシャッフルして均等に分割し逆伝播させ、勾配を加算して一度に更新。
        （分割しても損失・勾配は同一であり、BN統計のみチャンク単位で計算される）
        """
        w, owner = span_weights(sizes, first_w, device)
        cls_w = (_cw_t[labels] if _cw_t is not None
                 else torch.ones(labels.size(0), device=device))
        coef = cls_w[owner] * w / cls_w.sum()
        n = inputs.size(0)
        if n <= step_cap:
            chunks = [torch.arange(n, device=device)]
        else:
            # 1つのチャンクに同一衝突のウィンドウが集中しないようシャッフル（BN統計の偏り防止）
            chunks = torch.randperm(n, device=device).chunk(math.ceil(n / step_cap))
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for idx in chunks:
            x = inputs[idx]
            if channels_last_enabled:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast(device.type, enabled=amp_enabled):
                out = model(x)
            ce = F.cross_entropy(out.float(), labels[owner[idx]], reduction='none')
            loss = (coef[idx] * ce).sum()
            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total += loss.item()
        if amp_enabled:
            scaler.unscale_(optimizer)  # クリッピング前のスケール解除 (AMP)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        return total

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        n_seen = 0      # 本エポックで実際に使用したクリップ数（バッチ構成が空シーンを間引く場合は全体より少なくなる）
        for batch in train_loader:
            non_blocking = cuda_like
            if span:
                inputs, sizes, labels = batch
                inputs = inputs.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                train_loss += _span_step(inputs, sizes, labels) * labels.size(0)
                n_seen += labels.size(0)
                continue
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if channels_last_enabled:
                inputs = inputs.contiguous(
                    memory_format=torch.channels_last_3d)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # クリッピング前のスケール解除 (AMP)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm)
                optimizer.step()

            train_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        avg_train_loss = train_loss / max(n_seen, 1)

        # rc用/real用のvalをそれぞれ分離して性能測定（validate — 従来指標 + 区間再現率・識別力）
        def _evaluate(loader):
            return validate(model, loader, device, span=span, criterion=criterion,
                            batch_size=batch_size, class_w=_cw_t, first_w=first_w,
                            channels_last=channels_last_enabled)

        rc = _evaluate(val_rc_loader)
        real = _evaluate(val_real_loader)

        def _g(d, k):
            return d[k] if d else None

        # ── 評価指標 ── 実動画(val_real)が目標のため優先。存在しない場合(rcのみ学習)はrc。
        # 1エポックの値は変動するため(収束後標準偏差 0.013)、直近Nエポックの平均値を使用。
        ref = real if real is not None else rc
        bal_hist.append(ref['bal'] if ref['bal'] is not None else ref['loss'])
        window = bal_hist[-smooth_n:]
        monitor = sum(window) / len(window)
        in_grace = epoch + 1 <= grace
        if not in_grace:
            scheduler.step(monitor)
        current_lr = optimizer.param_groups[-1]['lr']  # ヘッドLR表示

        def _fmt(d):
            if d is None:
                return 'N/A'
            def _v(k):
                return f"{d[k]:.3f}" if d.get(k) is not None else '-'
            # 区間 = 接近〜衝突終了ウィンドウのいずれかがAである再現率 · 識別 = 識別力（基準非依存）
            return f"{d['loss']:.4f}/{d['acc']:.4f}/A{_v('rec')} 区間A{_v('spanRec')} 識別{_v('auc')}"
        print(f'Epoch [{epoch+1}/{num_epochs}] Train {avg_train_loss:.4f} | '
              f'val_rc(L/Acc/Arecall) {_fmt(rc)} | '
              f'val_real(L/Acc/Arecall) {_fmt(real)} | '
              f'評価指標 {monitor:.4f} | LR {current_lr:.2e}')

        # best保存・早期終了判定 → 候補保存（猶予期間中は行わない）
        if in_grace:
            saved_best = saved_cand = False
            print(f'猶予期間 {epoch + 1}/{grace} — 早期終了・学習率減衰・best選定なし')
        else:
            prev_best = early_stopping.best_epoch
            early_stopping(monitor, model, epoch + 1)
            saved_best = (early_stopping.best_epoch == epoch + 1
                          and prev_best != epoch + 1)
            saved_cand = _update_candidates(epoch + 1, monitor)
        # エポック別重み — 完了後にログを確認し数点を評価動画で比較して選定
        if epochs_dir is not None:
            torch.save(model.state_dict(), epochs_dir / f"{epoch_prefix}_ep{epoch + 1:03d}.pth")

        # エポック別の曲線をCSVに追記 — 学習が途中で中断しても保持されるよう毎回 append。
        # 過学習判定（train↓ val↑）とハイパーパラメータ比較の根拠となる。
        with open(log_path, 'a', newline='', encoding='utf-8') as _f:
            csv.writer(_f).writerow([
                epoch + 1, f'{current_lr:.3e}', f'{avg_train_loss:.6f}',
                _csv(_g(rc, 'loss')), _csv(_g(rc, 'acc')),
                _csv(_g(rc, 'rec')), _csv(_g(rc, 'prec')),
                _csv(_g(real, 'loss')), _csv(_g(real, 'acc')),
                _csv(_g(real, 'rec')), _csv(_g(real, 'prec')),
                _csv(_g(rc, 'lossA')), _csv(_g(rc, 'lossS')), _csv(_g(rc, 'bal')),
                _csv(_g(real, 'lossA')), _csv(_g(real, 'lossS')), _csv(_g(real, 'bal')),
                f'{monitor:.6f}', int(saved_best), int(saved_cand),
                *[_csv(_g(d, k)) for d in (rc, real)
                  for k in ('preRec', 'prePrec', 'spanRec', 'spanPrec', 'auc')],
                int(in_grace),
            ])

        # 学習曲線グラフの更新 — 失敗しても学習は継続
        try:
            from trainplot import plot_trainlog
            plot_trainlog(log_path)
        except Exception as exc:  # noqa: BLE001
            if not plot_warned:
                print(f'[グラフ] 生成失敗(学習は継続): {exc}')
                plot_warned = True

        if early_stopping.early_stop:
            print("早期終了条件を満たしました。学習を中断します。")
            break

    # ── 最終重みファイル命名規則の適用 ───────────────────────────────
    # hitandrun_[モデル名]_[YYMMDD_HHMM]_[エポック数]ep_[earlyY|earlyN]_[ptY|ptN][_span]_[損失率]
    #   · モデル名 : config.MODEL_NAME (s3d / x3d / slowfast) — フォルダ別判別用
    #   · 時刻   : 学習開始時刻 = 実行フォルダ名の時刻 (例: 2026-07-24 14:05 → 260724_1405)
    #             ⚠️ 2026-10-11 以前のファイルは「学習終了日」(YYMMDD)のみ付与。
    #   · エポック : best重みが保存されたエポック (例: 50 → 50ep)
    #   · early  : 早期終了で停止した場合は earlyY、それ以外は earlyN
    #   · _span  : Aクリップ方式が span の場合のみ付与(--a-mode span)
    #   · 損失率 : 保存当時の「評価指標」（val_real 均衡損失の移動平均）値。
    #             ⚠️ 2026-09-29 以前のファイルは単純な val_real loss であったため相互比較不可。
    date_str = run_stamp
    epoch_str = f'{early_stopping.best_epoch}ep'
    early_str = 'earlyY' if early_stopping.early_stop else 'earlyN'
    loss_str = f'{early_stopping.val_loss_min:.4f}'
    final_name = (f'hitandrun_{config.MODEL_NAME}_{date_str}_'
                  f'{epoch_str}_{early_str}_{run_tag}_{loss_str}.pth')
    # 学習中は実行フォルダの work_best.pth として保存しておき、終了時に規則ファイル名へ移動
    # (実行フォルダと weights が別ディスクでも動作するよう shutil.move。同名が存在する場合は _2 を付与)
    final_path = _unique_path(weights_dir / final_name)
    shutil.move(str(work_best), str(final_path))
    print(f'\n最終重み保存: {final_path}')

    # ── 候補の整理 ──
    # 1位は上記最終ファイルと同一重みのため作業用候補ファイルを削除。残りは
    # 順位・エポック・指標値を付与して保持 → 学習後、学習未使用の動画でサービス
    # パス評価を実行して最終選定を行う（val_real で選別したため val_real で再比較すると
    # その51クリップに偶然適合した方が再度優位になってしまうため）。
    cand_info = []
    for rank, c in enumerate(sorted(cands, key=lambda c: c['score']), 1):
        if c['epoch'] == early_stopping.best_epoch:
            c['path'].unlink(missing_ok=True)
            dest_name = final_path.name
        else:
            dest = _unique_path(weights_dir / (
                f"hitandrun_{config.MODEL_NAME}_{date_str}_{run_tag}_"
                f"cand{rank}_ep{c['epoch']}_{c['score']:.4f}.pth"))
            shutil.move(str(c['path']), str(dest))
            dest_name = dest.name
            print(f'候補 {rank}位保存: {dest}')
        cand_info.append({'rank': rank, 'epoch': c['epoch'],
                          'score': round(c['score'], 6), 'file': dest_name})

    run_finished = datetime.now()
    info = dict(run_info)
    info['終了'] = run_finished.strftime('%Y-%m-%d %H:%M:%S')
    info['最終重み'] = str(final_path)
    info['作業中best重み'] = '(最終重みへ移動完了)'
    info['bestエポック'] = (f"{early_stopping.best_epoch} "
                          f"({'早期終了' if early_stopping.early_stop else '最終エポックまで学習'})")
    info['候補重み'] = ', '.join(c['file'] for c in cand_info if c['rank'] > 1) or 'なし'
    # 評価推奨エポックの提示（評価指標・当該エポック均衡損失・F1上位） — 最終選定は評価動画にて実施
    from epoch_suggest import suggest
    suggest_eps, suggest_lines = suggest(run_dir)
    print()
    for line in suggest_lines:
        print(line)
    info['評価推奨エポック'] = ', '.join(map(str, suggest_eps)) or 'なし'
    write_run_info(run_dir, list(info.items()))

    # 結果サマリー — グラフがbest・候補を表示する際にも参照
    result_path = log_path.with_name(log_path.stem + '.result.json')
    with open(result_path, 'w', encoding='utf-8') as _f:
        json.dump({
            'started': run_started.strftime('%Y-%m-%d %H:%M:%S'),
            'finished': run_finished.strftime('%Y-%m-%d %H:%M:%S'),
            'run_dir': str(run_dir),
            'best_epoch': early_stopping.best_epoch,
            'early_stop': bool(early_stopping.early_stop),
            'final_weights': final_path.name,
            'weights_dir': str(final_path.parent),
            'monitor': f'val_real class-balanced loss, {smooth_n}-epoch avg',
            'a_mode': 'span' if span else 'pre',
            'balance': ({'motion_csv': str(motion_csv) if motion_csv else None,
                          'empty_pct': getattr(config, 'TRAIN_S_EMPTY_MOTION_PCT', None),
                          'empty_keep': getattr(config, 'TRAIN_S_EMPTY_KEEP', None),
                          'spread_a': spread_a} if balance_on else None),
            'candidates': cand_info,
            'epochs_dir': str(epochs_dir) if epochs_dir is not None else None,
            'eval_suggest': suggest_eps,
        }, _f, ensure_ascii=False, indent=2)
    try:
        from trainplot import plot_trainlog
        png = plot_trainlog(log_path)
        print(f'[グラフ] 学習曲線: {png}')
    except Exception as exc:  # noqa: BLE001
        print(f'[グラフ] 生成失敗: {exc}')

    state_dict = torch.load(final_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
