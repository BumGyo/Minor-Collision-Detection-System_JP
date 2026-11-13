"""事故クリップが元動画の何秒から開始するか (crash_events.clip_start_sec) が未設定の区間を補完する — 1回実行用のクリーンアップスクリプト。

再生画面はこの値を使用して赤色区間にCAMクリップを元動画の時刻に合わせて同期重畳する。本値の保存を開始した2026-10-27以前の分析は
空となっているが、当時のクリップ前後の余白が時期によって異なり(15・30フレーム)区間時刻から逆算すると最大0.5秒のずれが生じる(18番動画 実測)。
そのためクリップのフレームを元動画のフレームと直接照合する: クリップが区間を含む範囲の開始位置候補ごとに画面差分を測定し
最小の位置を選定する。合成なしの元クリップ(_raw)が存在する場合はそれを用いて全画面を、ない場合は表示クリップからボックス・ヒートマップが
描画されたbbox領域を除外して比較する(クリップ下部のS/Aバーは常に除外)。2位との差が小さい場合は不確実なためスキップする —
動きがbbox内に限定される動画はbboxを除外すると位置を特定できないため。

    cd backend && ../venv311/bin/python -m app.backfill_clip_start            # 検出のみ (変更なし)
    cd backend && ../venv311/bin/python -m app.backfill_clip_start --apply    # 未設定の値のみ補完
"""
import sys

import cv2
import numpy as np

from app import db_models
from app.db_connection import SessionLocal
from app.routers.videos import raw_clip_key
from app.object_storage import get_storage, materialize

SMALL = (160, 90)      # 比較用縮小サイズ
MIN_RATIO = 1.3        # 2位 (3フレーム以上離れた位置) 差分 ÷ 1位 差分 — これ未満の場合は確度不足としてスキップ


def _gray(frame, mask):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.resize(g * mask, SMALL, interpolation=cv2.INTER_AREA)


def find_clip_start(video_path, clip_path, width, height, bbox, first, last, mask_bbox=True):
    """クリップ先頭フレーム = 元動画の何フレーム目か → (フレーム番号, 2位÷1位比率, fps)。見つからない場合は (None, 理由, fps)。
    mask_bbox: bbox 領域を除外して比較 (ボックス・ヒートマップ描画済みクリップ)。元クリップ(_raw)は False。"""
    cap = cv2.VideoCapture(str(clip_path))
    clip = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        clip.append(f[:height])                    # 下部S/Aステータスバー除外
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0        # 分析 (inference.py) と同一 fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = len(clip)
    if n == 0:
        cap.release()
        return None, 'クリップ読み込み失敗', fps
    # クリップは対象区間 (first〜lastフレーム) を含む → 開始候補は last-n+1 〜 first (動画範囲内)
    lo, hi = max(0, last - n + 1), min(first, max(0, total - n))
    if lo > hi:
        lo, hi = max(0, min(lo, hi) - 30), max(lo, hi) + 30
    mask = np.ones((height, width), np.float32)
    if mask_bbox:
        x1, y1, x2, y2 = bbox
        m = max(6, int(0.05 * max(x2 - x1, y2 - y1)))
        mask[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m] = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    orig = []
    for _ in range(hi - lo + n):
        ok, f = cap.read()
        if not ok:
            break
        orig.append(_gray(f, mask))
    cap.release()
    small = [_gray(f, mask) for f in clip]
    ks = range(0, n, 2)
    errs = {}
    for s in range(lo, hi + 1):
        if s - lo + n > len(orig):
            break
        errs[s] = float(np.mean([np.mean((small[k] - orig[s - lo + k]) ** 2) for k in ks]))
    if not errs:
        return None, '元動画からクリップ範囲の読み込み失敗', fps
    best = min(errs, key=errs.get)
    # 2位は3フレーム以上離れた位置から選定 — 直隣のフレーム同士が同点の場合は同一フレームが連続する動画であるため(37番動画:
    # 連続フレームの49%が同一)、どちらであっても画面は同一。遠方まで類似している場合(静止画面)は確実に特定不能
    far = [e for s, e in errs.items() if abs(s - best) >= 3]
    if not far:
        return best, float('inf'), fps
    ratio = min(far) / max(errs[best], 1e-6)
    return (best, ratio, fps) if ratio >= MIN_RATIO else (None, f'確度不足 (2位÷1位: {ratio:.2f})', fps)


def main(apply=False):
    db = SessionLocal()
    storage = get_storage()
    filled = skipped = 0
    try:
        events = (db.query(db_models.CrashEvent)
                  .filter(db_models.CrashEvent.clip_start_sec.is_(None),
                          db_models.CrashEvent.clip_path.isnot(None))
                  .order_by(db_models.CrashEvent.id).all())
        print(f"開始時刻が未設定の事故区間 {len(events)}件 ({'適用' if apply else '検出のみ — 反映するには --apply'})")
        for ev in events:
            video, task = db.get(db_models.Video, ev.video_id), ev.task
            if not storage.exists(ev.clip_path):
                print(f"  区間 {ev.id} (動画 {ev.video_id}): スキップ — クリップファイルなし")
                skipped += 1
                continue
            raw_key = raw_clip_key(ev)
            # 元クリップが存在する場合は画面全体で、存在しない場合は表示クリップからbboxを除外して比較
            attempts = ([(raw_key, False, '元クリップ')] if raw_key and storage.exists(raw_key) else [])
            attempts += [(ev.clip_path, True, '表示クリップ')]
            for key, mask_bbox, label in attempts:
                with materialize(storage, video.video_path, suffix=Path(video.video_path).suffix) as source_path, \
                        materialize(storage, key, suffix=Path(key).suffix) as clip_path:
                    start, ratio, fps = find_clip_start(
                        source_path, clip_path, video.width, video.height,
                        (task.bbox_xmin, task.bbox_ymin, task.bbox_xmax, task.bbox_ymax),
                        ev.frame_number, ev.end_frame_number if ev.end_frame_number is not None else ev.frame_number,
                        mask_bbox=mask_bbox)
                if start is not None:
                    break
            if start is None:
                print(f"  区間 {ev.id} (動画 {ev.video_id}): スキップ — {ratio}")
                skipped += 1
                continue
            print(f"  区間 {ev.id} (動画 {ev.video_id}): クリップ開始 {start}フレーム = {start / fps:.3f}秒 "
                  f"({label}で比較, 2位÷1位比率: {ratio:.1f})")
            if apply:
                ev.clip_start_sec = start / fps
                db.commit()
            filled += 1
    finally:
        db.close()
    print(f"{'適用' if apply else '検出'} {filled}件 · スキップ {skipped}件")


if __name__ == '__main__':
    main(apply='--apply' in sys.argv[1:])
