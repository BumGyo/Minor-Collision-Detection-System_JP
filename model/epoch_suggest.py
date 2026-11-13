"""学習曲線CSVから「評価動画で比較検証すべきエポック」を選定する。

評価指標（検証バランス損失の3エポック移動平均）は実動画の検証衝突数が少ない（約12件）ため変動しやすく、移動平均の特性上
保存が約1エポック遅延して追従する。また損失基準は判定閾値（P(A) 0.5）に変化がなくとも確率値の上昇のみで
改善判定となるため、再現率は同一のまま誤検知が増加したエポックが選定される場合がある（2026-10-22学習: 評価指標
1位 44エポック 7/12・誤検知6 vs 38エポック 7/12・誤検知2）。そのため3基準の上位を合算して候補を抽出する。
  ① 評価指標（3エポック移動平均） ② 当該エポックのバランス損失 ③ 当該エポックのF1スコア（本番同様0.5判定の再現率・適合率）
各基準ごとに上位k件（config.TRAIN_SUGGEST_TOPK）を、互いに TRAIN_TOPK_MIN_GAP エポック以上離れたもののみ
選定する（近接エポックは実質的に同一モデル）。検証衝突数が少なくいずれの基準も揺らぐため、これらはあくまで「候補」であり、
最終選定は評価動画（本番判定・正常動画誤検知）によって行う。エポック別重みはセッションフォルダ内の epochs/ に格納される。

    python -m model.main --mode suggest --run outputs/trainlogs/<セッションフォルダ>   # 学習中・中断後も実行可能
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
    """当該エポック検証のF1スコア。衝突と判定されたクリップがない場合（適合率が空値）は0。"""
    r, p = _f(row, f'val_{dom}_recallA'), _f(row, f'val_{dom}_precA')
    if not r or not p:
        return 0.0
    return 2 * p * r / (p + r)


def _top(rows, key, k, gap):
    """keyの昇順で、既に選定済みのエポックとgap以上離れているもののみk件抽出。"""
    picked = []
    for r in sorted((r for r in rows if key(r) is not None), key=key):
        e = int(r['epoch'])
        if all(abs(e - q) >= gap for q in picked):
            picked.append(e)
        if len(picked) == k:
            break
    return picked


def suggest(run_dir, k=None, gap=None):
    """(推奨エポック一覧, 出力行リスト)。run_dir はセッションフォルダ（またはその中の trainlog CSV）。"""
    k = int(k or getattr(config, 'TRAIN_SUGGEST_TOPK', 2))
    gap = int(gap or getattr(config, 'TRAIN_TOPK_MIN_GAP', 3))
    run_dir = Path(run_dir)
    csv_path = run_dir if run_dir.suffix == '.csv' else next(iter(sorted(run_dir.glob('trainlog_*.csv'))), None)
    if csv_path is None or not csv_path.exists():
        return [], [f"学習曲線CSVが存在しません: {run_dir}"]
    run_dir = csv_path.parent
    with open(csv_path, encoding='utf-8') as fh:
        rows = [r for r in csv.DictReader(fh) if r.get('epoch')]
    if not rows:
        return [], ["学習曲線がまだ空です"]
    # 実動画検証が存在する場合はそれを優先、なければRC検証
    dom = 'real' if any(_f(r, 'val_real_bal') is not None for r in rows) else 'rc'
    bal = f'val_{dom}_bal'
    picks = {
        '評価指標': _top(rows, lambda r: _f(r, 'monitor'), k, gap),
        'バランス損失': _top(rows, lambda r: _f(r, bal), k, gap),
        # F1が同値の場合はバランス損失が低い方を優先
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

    lines = [f"[評価推奨] 各基準の上位 {k}件（互いに {gap}エポック以上離隔） — ① 評価指標 ② 当該エポック {dom} "
             f"バランス損失 ③ 当該エポック {dom} F1。候補選定であり最終決定は評価動画で実施",
             f"  {'エポック':>6}{'評価指標':>10}{dom + ' バランス':>10}{dom + ' 再現率':>11}{dom + ' 適合率':>11}"
             f"{dom + ' F1':>8}{'rc バランス':>9}  {'選定基準':<16}重み"]
    files = []
    for e in pick:
        r = by_ep[e]
        fp = _file(e)
        files.append(fp)
        why = '·'.join(name for name, eps in picks.items() if e in eps)
        lines.append(f"  {e:>6}{_n(_f(r, 'monitor')):>10}{_n(_f(r, bal)):>10}"
                     f"{_n(_f(r, f'val_{dom}_recallA')):>11}{_n(_f(r, f'val_{dom}_precA')):>11}"
                     f"{_n(_f1(r, dom)):>8}{_n(_f(r, 'val_rc_bal')):>9}  {why:<16}"
                     f"{fp.name if fp else '(なし — エポック別保存前の学習)'}")
    have = [p for p in files if p]
    if have:
        from hitandrun_model import weights_head_pool, weights_input_size
        lines.append("  評価コマンド（重み学習設定を自動調整します）:")
        for path in have:
            size = weights_input_size(path.name)
            pool = weights_head_pool(path.name)
            env = (f"INPUT_SIZE={size} " if size else "") + (f"HEAD_TEMPORAL_POOL={pool} " if pool != 'avg' else "")
            lines.append(f'    {env}python -u -m model.main --mode eval --weights "{path}"')
    return pick, lines
