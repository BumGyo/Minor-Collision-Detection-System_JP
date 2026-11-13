"""実効フレームレートが 10fps 未満の動画 (config.LOW_FPS_VIDEOS) を学習・評価対象外ディレクトリへ退避 (2026-11-04)。

    python -m model.lowfps_move            # 移動対象の確認 (Dry-run)
    python -m model.lowfps_move --apply    # 実際に移動 (mp4 + txt)
    python -m model.lowfps_move --restore --apply   # 元の位置へ復帰
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
    ap = argparse.ArgumentParser(description='10fps未満の動画を学習・評価対象外ディレクトリへ移動')
    ap.add_argument('--apply', action='store_true', help='実際に移動を実行 (未指定時は確認のみ)')
    ap.add_argument('--restore', action='store_true', help='元の位置へ復帰')
    args = ap.parse_args()

    moves = _plan(args.restore)
    found = {stem for _, stem, _, _ in moves}
    absent = sorted(set(config.LOW_FPS_VIDEOS) - found)
    print(f"[10fps未満動画 {'復帰' if args.restore else '退避'}] リスト {len(config.LOW_FPS_VIDEOS)}件 · "
          f"移動対象動画 {len(found)}件 (ファイル {len(moves)}件) · 移動先 "
          f"{config.LOW_FPS_EXCLUDED_DIR if not args.restore else '元のディレクトリ'}")
    clash = [d for _, _, _, d in moves if d.exists()]
    for part, stem, src, dst in moves:
        print(f"  {part:5s} {src} -> {dst}" + ('  ⚠️ 既に存在 — スキップ' if dst in clash else ''))
    if absent:
        print(f"  (移動済みまたはローカルに存在しない: {', '.join(absent)})")
    if not args.apply:
        print('確認のみ完了しました。実際に実行するには --apply を指定してください。')
        return 0
    done = 0
    for part, stem, src, dst in moves:
        if dst in clash:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done += 1
    print(f'完了: ファイル {done}件' + (f' · スキップ {len(clash)}件' if clash else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
