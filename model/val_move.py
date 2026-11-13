"""検証動画を学習フォルダから検証フォルダへ移動する (2026-11-04)。

    python -m model.val_move            # 移動対象のプレビュー表示のみ
    python -m model.val_move --apply    # 実際に移動を実行 (mp4 + txt)
    python -m model.val_move --restore --apply   # 学習フォルダへ復元

移動対象リスト = config.TRAIN_SPLIT_LOCK(model/split_lock.json) の検証動画189件 (学習フォルダ基準の相対パス)。
config.DATA_DIR/<相対パス> → config.VAL_DATA_DIR/<同一相対パス> (rcdata/・realdata/ のディレクトリ構造を維持)。
削除は行わず移動のみ実行。移動先に同名ファイルが既に存在する場合はその動画をスキップ。
移動後の学習は検証フォルダ全体を検証セットとして使用 (train.py)。検証クリップキャッシュはパス変更に伴い初回検証時に再作成。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import config


def main():
    ap = argparse.ArgumentParser(description='検証動画を検証フォルダへ移動')
    ap.add_argument('--apply', action='store_true', help='実際に移動を実行(指定しない場合はプレビューのみ)')
    ap.add_argument('--restore', action='store_true', help='学習フォルダへ復元')
    args = ap.parse_args()

    lock = Path(config.TRAIN_SPLIT_LOCK or '')
    if not lock.is_file():
        print(f'エラー: 検証固定ファイルが見つかりません -> {lock}')
        return 1
    rels = [v.replace('\\', '/') for v in json.loads(lock.read_text(encoding='utf-8'))['val']]
    train_root, val_root = Path(config.DATA_DIR), Path(config.VAL_DATA_DIR)
    src_root, dst_root = (val_root, train_root) if args.restore else (train_root, val_root)

    moves, absent = [], []
    for rel in rels:
        mp4 = src_root / rel
        if not mp4.exists():
            absent.append(rel)
            continue
        for src in (mp4, mp4.with_suffix('.txt')):
            if src.exists():
                moves.append((src, dst_root / Path(rel).with_suffix(src.suffix)))
    clash = {d for _, d in moves if d.exists()}
    n_real = sum(1 for s, _ in moves if s.suffix == '.mp4' and 'realdata' in s.parts)
    n_rc = sum(1 for s, _ in moves if s.suffix == '.mp4') - n_real
    print(f"[検証動画 {'復元' if args.restore else '移動'}] {lock.name} リスト {len(rels)}件 · "
          f"{src_root} → {dst_root} · 移動対象動画 {n_real + n_rc}件(実車 {n_real}, rc {n_rc}) · ファイル {len(moves)}件")
    for src, dst in moves:
        if src.suffix == '.mp4' and ('realdata' in src.parts or len(moves) <= 60):
            print(f"  {src.relative_to(src_root)}" + ('  ⚠️ 移動先に既に存在 — スキップ' if dst in clash else ''))
    if absent:
        print(f"  (移動先に既に存在または未検出: {len(absent)}件"
              + (f" — 例: {', '.join(Path(a).stem for a in absent[:5])}" if absent else '') + ")")
    if not args.apply:
        print('プレビュー表示のみ完了しました。実際に移動するには --apply を指定してください')
        return 0
    done = 0
    for src, dst in moves:
        if dst in clash:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done += 1
    print(f'移動完了: ファイル {done}件' + (f' · スキップ {len(clash)}件' if clash else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
