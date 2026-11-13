"""検証動画を学習ディレクトリから検証ディレクトリへ移動 (2026-11-04)。

    python -m model.val_move            # 移動対象の確認 (Dry-run)
    python -m model.val_move --apply    # 実際に移動 (mp4 + txt)
    python -m model.val_move --restore --apply   # 学習ディレクトリへ復帰
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import config


def main():
    ap = argparse.ArgumentParser(description='検証動画を検証ディレクトリへ移動')
    ap.add_argument('--apply', action='store_true', help='実際に移動を実行 (未指定時は確認のみ)')
    ap.add_argument('--restore', action='store_true', help='学習ディレクトリへ復帰')
    args = ap.parse_args()

    lock = Path(config.TRAIN_SPLIT_LOCK or '')
    if not lock.is_file():
        print(f'エラー: 検証分割ロックファイルが見つかりません -> {lock}')
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
    print(f"[検証動画 {'復帰' if args.restore else '移動'}] {lock.name} リスト {len(rels)}件 · "
          f"{src_root} → {dst_root} · 移動対象動画 {n_real + n_rc}件 (実写 {n_real}, RC {n_rc}) · ファイル {len(moves)}件")
    for src, dst in moves:
        if src.suffix == '.mp4' and ('realdata' in src.parts or len(moves) <= 60):
            print(f"  {src.relative_to(src_root)}" + ('  ⚠️ 移動先に既に存在 — スキップ' if dst in clash else ''))
    if absent:
        print(f"  (移動先に既に存在するかローカルにない: {len(absent)}件"
              + (f" — 例: {', '.join(Path(a).stem for a in absent[:5])}" if absent else '') + ')')
    if not args.apply:
        print('確認のみ完了しました。実際に実行するには --apply を指定してください。')
        return 0
    done = 0
    for src, dst in moves:
        if dst in clash:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done += 1
    print(f'完了: ファイル {done}件' + (f' · スキップ {len(clash)}件' if clash else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
