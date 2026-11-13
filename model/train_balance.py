"""学習バッチ構成 — 静止正常クリップの間引き (ローテーション) + バッチ間衝突サンプルの均等配分。

config.TRAIN_S_MOTION_CSV / TRAIN_SPREAD_A 有効時のみ train.py で使用。
学習セット (train_idx) のみに適用し、検証セットには影響を与えない。
"""
import csv
import math
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler

import config


def resolve_csv(path):
    """相対パスはプロジェクトルート基準。"""
    p = Path(path)
    return p if p.is_absolute() else Path(config._ROOT) / p


def load_motion(csv_path):
    """clips.csv → {(動画名, 車両番号, 開始フレーム): 動き量 %} (正常クリップのみ)。"""
    motion = {}
    with open(csv_path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['label'] != '0':
                continue
            try:
                motion[(r['video'], int(r['car_id']), int(r['start_f']))] = float(r['move_pct'])
            except ValueError:
                continue
    return motion


class BalancedBatchSampler(Sampler):
    """学習 Subset のインデックスからエポックごとのバッチリストを生成。"""

    def __init__(self, a_pos, s_move_pos, s_empty_pos, batch_size, empty_keep=1.0,
                 spread_a=False, seed=42):
        self.a = np.asarray(a_pos, dtype=np.int64)
        self.s_move = np.asarray(s_move_pos, dtype=np.int64)
        self.bs = int(batch_size)
        self.spread_a = bool(spread_a)
        self.seed = int(seed)
        self.epoch = 0
        empty = np.asarray(s_empty_pos, dtype=np.int64)
        self.empty_order = np.random.RandomState(seed).permutation(empty)
        self.k_empty = (len(empty) if empty_keep >= 1
                        else int(round(len(empty) * max(0.0, float(empty_keep)))))

    def counts(self):
        n_s = len(self.s_move) + self.k_empty
        n = len(self.a) + n_s
        nb = math.ceil(n / self.bs) if n else 0
        if self.spread_a:
            no_a = max(0, nb - len(self.a))
        else:
            no_a = nb * ((n_s / n) ** self.bs) if n else 0
        return {'a': len(self.a), 's_move': len(self.s_move), 's_empty': self.k_empty,
                's_empty_total': len(self.empty_order), 'n': n, 'batches': nb, 'no_a_batches': no_a}

    def _empty_chunk(self, epoch):
        m, k = len(self.empty_order), self.k_empty
        if k >= m:
            return self.empty_order
        if k == 0:
            return self.empty_order[:0]
        start = (epoch * k) % m
        idx = (start + np.arange(k)) % m
        return self.empty_order[idx]

    def batches(self, epoch):
        rng = np.random.RandomState(self.seed + 1000 + epoch)
        a = rng.permutation(self.a)
        s = rng.permutation(np.concatenate([self.s_move, self._empty_chunk(epoch)]))
        n = len(a) + len(s)
        if n == 0:
            return []
        nb = math.ceil(n / self.bs)
        if not self.spread_a:
            allp = rng.permutation(np.concatenate([a, s]))
            return [allp[i * self.bs:(i + 1) * self.bs].tolist() for i in range(nb)]
        sizes = [self.bs] * (nb - 1) + [n - self.bs * (nb - 1)]
        cnt = [0] * nb
        order = rng.permutation(nb)
        k = 0
        for _ in range(len(a)):
            while cnt[order[k % nb]] >= sizes[order[k % nb]]:
                k += 1
            cnt[order[k % nb]] += 1
            k += 1
        out, ia, is_ = [], 0, 0
        for i in range(nb):
            b = a[ia:ia + cnt[i]].tolist() + s[is_:is_ + sizes[i] - cnt[i]].tolist()
            ia += cnt[i]
            is_ += sizes[i] - cnt[i]
            out.append(rng.permutation(b).tolist())
        return out

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        yield from self.batches(epoch)

    def __len__(self):
        return self.counts()['batches']


def build(train_samples, train_idx, batch_size, csv_path=None, empty_pct=0.25,
          empty_keep=1 / 3, spread_a=False, seed=42):
    """train.py 向け: サンプラーとログ要約を生成。"""
    motion = load_motion(csv_path) if csv_path else {}
    a_pos, s_move, s_empty = [], [], []
    matched = 0
    for pos, i in enumerate(train_idx):
        s = train_samples[i]
        if s['label'] == 1:
            a_pos.append(pos)
            continue
        m = motion.get((s['file_name'], int(s.get('car_id', 0)), int(s['start_f'])))
        if m is not None:
            matched += 1
        (s_empty if (m is not None and m < empty_pct) else s_move).append(pos)
    if not csv_path:
        s_move, s_empty = s_move + s_empty, []
    sampler = BalancedBatchSampler(a_pos, s_move, s_empty, batch_size,
                                   empty_keep=empty_keep if csv_path else 1.0,
                                   spread_a=spread_a, seed=seed)
    c = sampler.counts()
    n_s_all = len(s_move) + len(s_empty)
    lines = []
    if csv_path:
        lines.append(f"静止正常クリップ間引き: 動き量 {empty_pct:g}% 未満をエポックごとに {empty_keep:.0%} ずつローテーション — "
                     f"学習正常 {n_s_all}件中 静止 {len(s_empty)}件 ({len(s_empty) / max(n_s_all, 1):.0%}), "
                     f"CSVマッチ {matched}/{n_s_all}")
        if matched < n_s_all:
            lines.append(f"  ⚠️ CSV未記載の正常 {n_s_all - matched}件は「動きあり」として常時使用")
    lines.append(f"バッチ構成: 衝突 {'各バッチ均等配置' if spread_a else 'ランダム配置'} — エポックあたり "
                 f"衝突 {c['a']} · 正常 {c['s_move'] + c['s_empty']} (動きあり {c['s_move']} + 静止 {c['s_empty']}) "
                 f"= 1:{(c['s_move'] + c['s_empty']) / max(c['a'], 1):.1f} · バッチ数 {c['batches']}個 · "
                 f"衝突なしバッチ {c['no_a_batches']:.0f}個 ({c['no_a_batches'] / max(c['batches'], 1):.0%}"
                 f"{'' if spread_a else ', ランダム期待値'})")
    return sampler, lines
