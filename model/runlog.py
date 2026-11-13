"""学習・評価1回分の成果物フォルダ。

学習や評価を1回実行するたびに `<親フォルダ>/<YYMMDD_HHMM>_<モデル名>_<ptY|ptN>` フォルダを
新規作成し、その実行回の成果物をすべて格納する。フォルダ名のみで実行日時・モデル種別を
把握可能とし、他セッションのファイルとの混在を防止する。
"""
from datetime import datetime
from pathlib import Path


def make_run_dir(parent, model_name, tag, when=None):
    """セッションフォルダを作成し、(フォルダパス, 'YYMMDD_HHMM' 文字列) を返却する。

    同一分内に複数回実行した場合は `_2`, `_3` などの連番を付与し、既存セッションの上書きを防止する。
    """
    when = when or datetime.now()
    stamp = when.strftime('%y%m%d_%H%M')
    base = Path(parent) / f'{stamp}_{model_name}_{tag}'
    base.parent.mkdir(parents=True, exist_ok=True)
    path, n = base, 1
    while True:
        # 存在確認後に作成するのではなく、「作成を試行し、既に存在する場合は連番を進める」方式を採用。
        # 2つの学習プロセスが同一時刻に開始されても、同一フォルダを共用する衝突を防止する。
        try:
            path.mkdir()
            return path, stamp
        except FileExistsError:
            n += 1
            path = base.with_name(f'{base.name}_{n}')


def write_run_info(run_dir, items):
    """セッションフォルダに run_info.txt を出力する（既存の場合は最新内容に更新）。

    items は (項目名, 設定値) のタプルリスト。適用モデル、実行日時、出力重みファイル等の
    メタ情報を、フォルダを開くだけで即座に確認可能にする。
    """
    lines = [f'{k}: {"" if v is None else v}' for k, v in items]
    Path(run_dir, 'run_info.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
