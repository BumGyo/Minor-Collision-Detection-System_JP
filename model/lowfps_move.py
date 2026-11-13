"""実際の画面更新が10fps未満の動画(config.LOW_FPS_VIDEOS)を学習・評価で使用しないフォルダへ移動する (2026-11-04)。

    python -m model.lowfps_move            # 移動対象のプレビュー表示のみ
    python -m model.lowfps_move --apply    # 実際に移動を実行 (mp4 + txt)
    python -m model.lowfps_move --restore --apply   # 元の場所へ復元

学習フォルダ(config.DATA_DIR) → config.LOW_FPS_EXCLUDED_DIR/train/<同一相対パス>,
評価フォルダ(config.EVAL_FOLDER_PATH) → config.LOW_FPS_EXCLUDED_DIR/eval/<同一相対パス>。
削除は行わず移動のみ実行。同名ファイルが移動先に既に存在する場合はその動画をスキップ。
検証セットは config.TRAIN_SPLIT_LOCK で固定されているため、移動しても他動画の分割は変更されない。
"""
import argparse
import shutil
import sys
from pathlib import Path

import config


def _plan(restore):
    names = set(config.LOW_FPS_VIDEOS)
    base = Path(config.LOW_FPS_EXCLUDED_DIR)
    moves = []
    for part, root in (('train', Path(config.DATA_DIR)), ('eval', Path(config.EVAL_FOLDER_PATH))):
        src_root, dst_root = (base / part, root) if restore else (root, base / part)
        if not src_root.exists():
            continue
        for mp4 in sorted(src_root.rglob('*.mp4')):
            if mp4.stem not in names:
                continue
            rel = mp4.relative_to(src_root)
            for src in (mp4, mp4.with_suffix('.txt')):
                if src.exists():
                    moves.append((part, mp4.stem, src, dst_root / rel.with_suffix(src.suffix)))
    return moves


def main():
    ap = argparse.ArgumentParser(description='10fps未満の動画を学習・評価対象外フォルダへ移動')
    ap.add_argument('--apply', action='store_true', help='実際に移動を実行(指定しない場合はプレビューのみ)')
    ap.add_argument('--restore', action='store_true', help='元の場所へ復元')
    args = ap.parse_args()

    moves = _plan(args.restore)
    found = {stem for _, stem, _, _ in moves}
    absent = sorted(set(config.LOW_FPS_VIDEOS) - found)
    print(f"[10fps未満の動画 {'復元' if args.restore else '移動'}] リスト {len(config.LOW_FPS_VIDEOS)}件 · "
          f"移動対象動画 {len(found)}件(ファイル {len(moves)}件) · 移動先 "
          f"{config.LOW_FPS_EXCLUDED_DIR if not args.restore else '元のフォルダ'}")
    clash = [d for _, _, _, d in moves if d.exists()]
    for part, stem, src, dst in moves:
        print(f"  {part:5s} {src} -> {dst}" + ('  ⚠️ 既に存在 — スキップ' if dst in clash else ''))
    if absent:
        print(f"  (既に移動済みまたは未検出: {', '.join(absent)})")
    if not args.apply:
        print('プレビュー表示のみ完了しました。実際に移動するには --apply を指定してください')
        return 0
    done = 0
    for part, stem, src, dst in moves:
        if dst in clash:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done += 1
    print(f'移動完了: ファイル {done}件' + (f' · スキップ {len(clash)}件' if clash else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
