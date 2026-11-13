"""学習中に保存された最良重みを用いて「検証セットのA確率分布」を直接確認するスクリプト。

recall/acc は argmax (閾値0.5) 基準のため、モデルが A 確率を 0.1 → 0.45 へ
向上させている最中も評価値上は 0 のままとなる。「学習進行中」なのか「全く学習できていない」のか
判別できないため、確率値を直接確認する。

  · Aサンプルの平均A確率 vs Sサンプルの平均A確率 → 差が広がっていれば判別能力を獲得中
  · AUC → 閾値に依存しない判別力 (0.5 = ランダム、1.0 = 完全判別)
  · 閾値別 recall/precision → 0.5 以外の閾値での検出感度

使用法:
    cd model && python3 ../check_probs.py ../weights/hitandrun_x3d_260922_33ep_earlyY_ptY_0.1999.pth
"""
from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())          # モデルディレクトリ内での実行を前提

import config                             # noqa: E402
from dataset import HitAndRunDataset      # noqa: E402
from device_utils import get_device       # noqa: E402
from hitandrun_model import HitAndRun3DCNN  # noqa: E402


def _split_val(dataset, ratio=config.TRAIN_SPLIT_RATIO):
    """train.py と同一の階層的動画単位分割 (seed 42) → (val_rc, val_real) インデックス。"""
    groups: dict = {}
    for i, s in enumerate(dataset.samples):
        parts = os.path.normpath(s['mp4_path']).split(os.sep)
        dom = 'real' if 'realdata' in parts else 'rc'
        name = s['file_name']
        seg = name.split('_')
        if dom == 'rc':
            direction = seg[1][0] if len(seg) > 1 and seg[1] else '-'
            scen = seg[1][-1] if len(seg) > 1 and seg[1] else '-'
        else:
            direction = '-'
            scen = seg[1][-1] if len(seg) > 1 and seg[1] else '-'
        key = (dom, direction, scen)
        groups.setdefault(key, {}).setdefault(s.get('video_key', s['mp4_path']), []).append(i)

    gen = torch.Generator().manual_seed(42)
    val_rc, val_real = [], []
    for (dom, _d, _s), vid_map in sorted(groups.items()):
        vkeys = sorted(vid_map)
        order = torch.randperm(len(vkeys), generator=gen).tolist()
        shuffled = [vkeys[i] for i in order]
        cut = int(ratio * len(vkeys))
        for vk in shuffled[cut:]:
            (val_real if dom == 'real' else val_rc).extend(vid_map[vk])
    return val_rc, val_real


def _auc(pos, neg):
    """Mann-Whitney U 検定による AUC 計算。pos/neg = A・SサンプルのA確率リスト。"""
    if not pos or not neg:
        return None
    ranked = sorted([(p, 1) for p in pos] + [(n, 0) for n in neg])
    # 同点は平均順位で処理
    r, i, s_pos = 0.0, 0, 0.0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if ranked[k][1] == 1:
                s_pos += avg_rank
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (s_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    weights = sys.argv[1]
    device = get_device(config.INFER_DEVICE_TYPE)

    ds_eval = HitAndRunDataset(augment=False)      # 検証時はデータ拡張なし
    val_rc, val_real = _split_val(ds_eval)
    print(f"[分割] val_rc {len(val_rc)}クリップ / val_real {len(val_real)}クリップ (seed 42)")

    model = HitAndRun3DCNN(num_classes=config.MODEL_NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.eval()
    print(f"[重み] {weights}\n")

    for name, idxs in [("val_rc", val_rc), ("val_real", val_real)]:
        if not idxs:
            continue
        loader = DataLoader(Subset(ds_eval, idxs),
                            batch_size=config.TRAIN_BATCH_SIZE, shuffle=False)
        a_sample_probs, s_sample_probs = [], []
        with torch.inference_mode():
            for x, y in loader:
                p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu()
                for prob, lab in zip(p.tolist(), y.tolist()):
                    (a_sample_probs if lab == 1 else s_sample_probs).append(prob)

        print(f"══ {name}  (A {len(a_sample_probs)} / S {len(s_sample_probs)})")
        if a_sample_probs:
            print(f"   AサンプルのA確率 : 平均 {sum(a_sample_probs)/len(a_sample_probs):.4f}  "
                  f"最小 {min(a_sample_probs):.4f}  最大 {max(a_sample_probs):.4f}")
        print(f"   SサンプルのA確率 : 平均 {sum(s_sample_probs)/len(s_sample_probs):.4f}  "
              f"最小 {min(s_sample_probs):.4f}  最大 {max(s_sample_probs):.4f}")
        if a_sample_probs:
            gap = sum(a_sample_probs)/len(a_sample_probs) - sum(s_sample_probs)/len(s_sample_probs)
            auc = _auc(a_sample_probs, s_sample_probs)
            print(f"   判別力         : 平均差 {gap:+.4f}   AUC {auc:.4f}"
                  f"   {'← 判別中' if auc and auc > 0.6 else '← ほぼランダム'}")
            print(f"   {'閾値':>8} {'recall':>8} {'precision':>10} {'予測A数':>8}")
            for th in (0.5, 0.4, 0.3, 0.2, 0.1):
                tp = sum(1 for p in a_sample_probs if p >= th)
                fp = sum(1 for p in s_sample_probs if p >= th)
                rec = tp / len(a_sample_probs)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                mark = "  ← 現在の推論基準" if th == 0.5 else ""
                print(f"   {th:>8.1f} {rec:>8.3f} {prec:>10.3f} {tp+fp:>8}{mark}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
