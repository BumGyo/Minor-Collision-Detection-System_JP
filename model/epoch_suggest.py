"""学習曲線 CSV から評価比較対象のエポックを選定するスクリプト。

選定基準:
  ① 監視指標 (3エポック平均)
  ② 該当エポックのバランスド損失
  ③ 該当エポックの F1 スコア

使用法:
    python -m model.main --mode suggest --run outputs/trainlogs/<セッションディレクトリ>
"""
import csv
from pathlib import Path

import config


def _f(row, col):
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def _f1(row, dom):
    """該当エポック検証の F1 スコア。"""
    r, p = _f(row, f'val_{dom}_recallA'), _f(row, f'val_{dom}_precA')
    if not r or not p:
        return 0.0
    return 2 * p * r / (p + r)


def _top(rows, key, k, gap):
    """key が最小の順に、既選定エポックと gap 以上離れたものを k 個抽出。"""
    picked = []
    for r in sorted((r for r in rows if key(r) is not None), key=key):
        e = int(r['epoch'])
        if all(abs(e - q) >= gap for q in picked):
            picked.append(e)
        if len(picked) == k:
            break
    return picked


def suggest(run_dir, k=None, gap=None):
    """(推薦エポックリスト, ログ行リスト)。"""
    k = int(k or getattr(config, 'TRAIN_SUGGEST_TOPK', 2))
    gap = int(gap or getattr(config, 'TRAIN_TOPK_MIN_GAP', 3))
    run_dir = Path(run_dir)
    csv_path = run_dir if run_dir.suffix == '.csv' else next(iter(sorted(run_dir.glob('trainlog_*.csv'))), None)
    if csv_path is None or not csv_path.exists():
        return [], [f"学習曲線 CSV が見つかりません: {run_dir}"]
    run_dir = csv_path.parent
    with open(csv_path, encoding='utf-8') as fh:
        rows = [r for r in csv.DictReader(fh) if r.get('epoch') and r.get('grace') != '1']
    if not rows:
        return [], ["学習ログが空です"]
    dom = 'real' if any(_f(r, 'val_real_bal') is not None for r in rows) else 'rc'
    bal = f'val_{dom}_bal'
    picks = {
        '監視指標': _top(rows, lambda r: _f(r, 'monitor'), k, gap),
        'バランス損失': _top(rows, lambda r: _f(r, bal), k, gap),
        'F1': _top(rows, lambda r: (-_f1(r, dom), _f(r, bal) if _f(r, bal) is not None else 9.0), k, gap),
    }
    pick = sorted({e for v in picks.values() for e in v})
    by_ep = {int(r['epoch']): r for r in rows}
    ep_dir = run_dir / 'epochs'

    def _file(e):
        hit = sorted(ep_dir.glob(f'*_ep{e:03d}.pth')) if ep_dir.exists() else []
        return hit[0] if hit else None

    def _n(v):
        return '-' if v is None else f'{v:.2f}'

    lines = [f"[評価推奨] 各基準の上位 {k}件 (間隔 {gap}エポック以上) — ① 監視指標 ② {dom}バランス損失 ③ {dom} F1",
             f"  {'エポック':>6}{'監視指標':>10}{dom + ' バランス':>10}{dom + ' 再現率':>11}{dom + ' 適合率':>11}"
             f"{dom + ' F1':>8}{'区間再現率':>9}{'判別力':>7}{'rc バランス':>9}  {'選定基準':<16}重み"]
    files = []
    for e in pick:
        r = by_ep[e]
        fp = _file(e)
        files.append(fp)
        why = '·'.join(name for name, eps in picks.items() if e in eps)
        lines.append(f"  {e:>6}{_n(_f(r, 'monitor')):>10}{_n(_f(r, bal)):>10}"
                     f"{_n(_f(r, f'val_{dom}_recallA')):>11}{_n(_f(r, f'val_{dom}_precA')):>11}"
                     f"{_n(_f1(r, dom)):>8}{_n(_f(r, f'val_{dom}_spanRecA')):>9}{_n(_f(r, f'val_{dom}_auc')):>7}"
                     f"{_n(_f(r, 'val_rc_bal')):>9}  {why:<16}"
                     f"{fp.name if fp else '(重みファイルなし)'}")
    have = [p for p in files if p]
    if have:
        from hitandrun_model import weights_head_pool, weights_input_size
        size = weights_input_size(have[0].name)
        pool = weights_head_pool(have[0].name)
        env = (f"INPUT_SIZE={size} " if size else "") + (f"HEAD_TEMPORAL_POOL={pool} " if pool != 'avg' else "")
        lines.append("  評価実行コマンド (outputs/console/ にログ出力):")
        lines.append("    mkdir -p outputs/console && for w in \\")
        lines += [f"      {p} \\" for p in have[:-1]] + [f"      {have[-1]}; do"]
        lines.append(f"      {env}python -u -m model.main --mode eval --weights \"$w\"; done 2>&1 "
                     f"| tee outputs/console/eval_{run_dir.name}_$(date +%y%m%d_%H%M).log")
    return pick, lines
