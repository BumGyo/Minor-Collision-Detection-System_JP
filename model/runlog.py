"""学習・評価セッション別出力ディレクトリ管理。

学習や評価を実行するたびに `<親ディレクトリ>/<YYMMDD_HHMM>_<モデル>_<ptY|ptN>` ディレクトリを
新規作成し、そのセッションの生成ファイルをすべて格納する。ディレクトリ名から実行日時やモデル構成を
判別可能にし、他のセッション成果物との混同を防ぐ。
"""
from datetime import datetime
from pathlib import Path


def make_run_dir(parent, model_name, tag, when=None):
    """セッションディレクトリを作成し、(パス, 'YYMMDD_HHMM' 文字列) を返す。

    同一分内に複数回実行された場合は `_2`, `_3` などの連番を付与して上書きを回避する。
    """
    when = when or datetime.now()
    stamp = when.strftime('%y%m%d_%H%M')
    base = Path(parent) / f'{stamp}_{model_name}_{tag}'
    base.parent.mkdir(parents=True, exist_ok=True)
    path, n = base, 1
    while True:
        try:
            path.mkdir()
            return path, stamp
        except FileExistsError:
            n += 1
            path = base.with_name(f'{base.name}_{n}')


def write_run_info(run_dir, items):
    """セッションディレクトリに run_info.txt を出力 (既存の場合は上書き)。

    items は (キー, 値) のタプルリスト。
    """
    lines = [f'{k}: {"" if v is None else v}' for k, v in items]
    Path(run_dir, 'run_info.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
