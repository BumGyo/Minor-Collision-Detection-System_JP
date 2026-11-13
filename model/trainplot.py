"""学習曲線 CSV → PNG グラフ生成。

train.py が各エポック終了時に呼び出し `outputs/trainlogs/<セッション>/<ログ名>.png` を更新する。

3分割プロット (x軸 = エポック):
  1) 損失 — train / val_rc / val_real、および最良判定指標 (存在する場合)
  2) 検証衝突(A) recall (実線) · precision (点線) — val_real / val_rc
  3) ヘッド学習率 (対数スケール) — ReduceLROnPlateau の減衰履歴
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

C_REAL = '#2a78d6'    # val_real — 実車対象
C_RC = '#eb6834'      # val_rc
C_TRAIN = '#1baf7a'   # train
INK = '#0b0b0b'       # テキスト・best表示
INK2 = '#52514e'      # 補助テキスト・候補・学習率
GRID = '#e4e3df'
SURFACE = '#fcfcfb'

JA_FONTS = ('Meiryo', 'MS Gothic', 'Yu Gothic', 'Hiragino Sans', 'Noto Sans CJK JP', 'Noto Sans JP', 'Malgun Gothic')
JA_FONT_FILES = ('/mnt/c/Windows/Fonts/meiryo.ttc', '/mnt/c/Windows/Fonts/msgothic.ttc')

LABELS = {
    'ja': {
        'train': '学習 (拡張・Dropout有効)', 'train_end': '学習',
        'val_rc': 'RC検証', 'val_real': '実車検証',
        'monitor': '監視指標: 実車検証バランスド損失移動平均',
        'loss': '損失', 'recall': '{} 再現率', 'precision': '{} 適合率',
        'collision': '衝突(A)判定\n閾値 0.5', 'lr': 'ヘッド学習率', 'epoch': 'エポック',
        'best': 'best {}エポック', 'cand': '候補 {}エポック', 'early_stop': '早期終了',
        'why_metric': '監視指標最小',
        'why_old_real': '旧基準: 実車検証損失最小',
        'why_old_rc': '旧基準: RC検証損失最小',
    },
    'en': {
        'train': 'train (aug + dropout on)', 'train_end': 'train',
        'val_rc': 'val_rc', 'val_real': 'val_real',
        'monitor': 'selection metric: val_real class-balanced loss, moving avg',
        'loss': 'loss', 'recall': '{} recall', 'precision': '{} precision',
        'collision': 'collision (A)\n@ threshold 0.5', 'lr': 'head LR', 'epoch': 'epoch',
        'best': 'best ep{}', 'cand': 'cand ep{}', 'early_stop': 'early stop',
        'why_metric': 'min selection metric',
        'why_old_real': 'old rule: min val_real loss',
        'why_old_rc': 'old rule: min val_rc loss',
    },
}


def _japanese_font():
    """利用可能な日本語フォント名。存在しない場合は None。"""
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for name in JA_FONTS:
        if name in names:
            return name
    for path in JA_FONT_FILES:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            return fm.FontProperties(fname=path).get_name()
    return None


def _load(csv_path: Path):
    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    def col(key):
        out = []
        for r in rows:
            v = r.get(key)
            out.append(float(v) if v not in (None, '') else math.nan)
        return out
    return rows, col


def _pick_best(rows, col, result):
    """best エポック判定: 結果ファイル (.result.json) → 監視指標列 → 旧基準。"""
    if result.get('best_epoch'):
        return int(result['best_epoch']), 'metric'
    ep = [int(r['epoch']) for r in rows]
    for key, why in (('monitor', 'metric'),
                     ('val_real_loss', 'old_real'),
                     ('val_rc_loss', 'old_rc')):
        vals = col(key) if key in rows[0] else []
        pairs = [(v, e) for v, e in zip(vals, ep) if not math.isnan(v)]
        if pairs:
            return min(pairs)[1], why
    return None, ''


def _end_label(ax, xs, ys, text, color):
    pts = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if not pts:
        return
    x, y = pts[-1]
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords='offset points',
                va='center', fontsize=8.5, color=INK2)
    ax.plot([x], [y], 'o', ms=4, color=color, mec=SURFACE, mew=1.5, zorder=4)


def plot_trainlog(csv_path, out_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    rows, col = _load(csv_path)
    if not rows:
        return None
    out_path = Path(out_path) if out_path else csv_path.with_suffix('.png')
    side = csv_path.with_name(csv_path.stem + '.result.json')
    result = json.loads(side.read_text(encoding='utf-8')) if side.exists() else {}

    ep = [int(r['epoch']) for r in rows]
    best, best_why = _pick_best(rows, col, result)
    cands = [int(c['epoch']) for c in result.get('candidates', [])
             if int(c['epoch']) != best]

    font = _japanese_font()
    L = LABELS['ja' if font else 'en']
    rc = {'font.family': [font, 'DejaVu Sans'], 'axes.unicode_minus': False} if font else {}
    with plt.rc_context(rc):
        fig = _draw(plt, L, csv_path, rows, col, ep, best, best_why, cands, result)
        fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return out_path


def _draw(plt, L, csv_path, rows, col, ep, best, best_why, cands, result):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 9.2), sharex=True,
        gridspec_kw={'height_ratios': [3, 2, 1.1], 'hspace': 0.22})
    fig.patch.set_facecolor(SURFACE)

    # ── 1) 損失 ──
    ax1.plot(ep, col('train_loss'), color=C_TRAIN, lw=2, label=L['train'])
    ax1.plot(ep, col('val_rc_loss'), color=C_RC, lw=2, label=L['val_rc'])
    ax1.plot(ep, col('val_real_loss'), color=C_REAL, lw=2, label=L['val_real'])
    _end_label(ax1, ep, col('train_loss'), L['train_end'], C_TRAIN)
    _end_label(ax1, ep, col('val_rc_loss'), L['val_rc'], C_RC)
    _end_label(ax1, ep, col('val_real_loss'), L['val_real'], C_REAL)
    if 'monitor' in rows[0]:
        ax1.plot(ep, col('monitor'), color=C_REAL, lw=2.2, ls=(0, (5, 3)),
                 label=L['monitor'])
    ax1.set_ylabel(L['loss'], color=INK2)
    ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8.5,
               frameon=False, labelcolor=INK2, handlelength=2.6, columnspacing=1.4)

    # ── 2) 検証衝突 recall / precision (0.5 閾値) ──
    for key, color in (('val_real', C_REAL), ('val_rc', C_RC)):
        ax2.plot(ep, col(f'{key}_recallA'), color=color, lw=2,
                 label=L['recall'].format(L[key]))
        ax2.plot(ep, col(f'{key}_precA'), color=color, lw=1.6, ls=(0, (2, 2)),
                 label=L['precision'].format(L[key]))
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_ylabel(L['collision'], color=INK2, linespacing=1.6)
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8.5,
               frameon=False, labelcolor=INK2, handlelength=2.6, columnspacing=1.4)

    # ── 3) 学習率 ──
    ax3.step(ep, col('lr'), where='post', color=INK2, lw=1.8)
    ax3.set_yscale('log')
    from matplotlib.ticker import LogFormatter
    ax3.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    ax3.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False))
    ax3.set_ylabel(L['lr'], color=INK2)
    ax3.set_xlabel(L['epoch'], color=INK2)

    # ── best / 候補表示 ──
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(which='both', colors=INK2, labelsize=8.5)
        if best is not None:
            ax.axvline(best, color=INK, lw=1.2, alpha=0.8, zorder=1)
        for c in cands:
            ax.axvline(c, color=INK2, lw=1.1, ls=(0, (1, 2)), zorder=1)
    bottom = ax1.get_ylim()[0]
    if best is not None:
        ax1.annotate(L['best'].format(best), (best, bottom), xytext=(3, 4),
                     textcoords='offset points', fontsize=8.5, color=INK)
    for c in cands:
        ax1.annotate(L['cand'].format(c), (c, bottom), xytext=(3, 16),
                     textcoords='offset points', fontsize=8, color=INK2)

    title = csv_path.stem
    if best is not None:
        title += f"   —   {L['best'].format(best)} ({L['why_' + best_why]})"
    if result.get('early_stop'):
        title += f"   {L['early_stop']}"
    fig.suptitle(title, x=0.01, y=0.955, ha='left', fontsize=11, color=INK)
    return fig


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for p in sys.argv[1:]:
        out = plot_trainlog(p)
        print(f'{p} → {out}' if out else f'{p}: データ行なし、スキップ')
