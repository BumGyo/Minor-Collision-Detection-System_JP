import os
import queue
import shutil
import subprocess
import threading
import time
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

import config
from device_utils import is_cuda_like, is_channels_last_3d_supported, predict_batch_size
from infer_runtime import get_runner

# システムの ffmpeg (H.264/libx264) パス — 存在する場合はクリップを汎用再生可能な mp4 として生成
_FFMPEG = shutil.which("ffmpeg")
# システムにffmpegがない場合(macOS・Windowsの既定状態)、imageio-ffmpeg同梱のffmpegを使用する。
# クリップ変換にffmpegがないと分析全体が失敗するため requirements に追加済み。
# そのバイナリもlibx264を含み同様のH.264変換が可能(Linux 0.6.0 実測)。
if not _FFMPEG:
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

activation = {}

# 進捗度(0.0〜1.0)区間配分 — 段階ごとに「実測」進捗度を報告しUIが不正確な推定値を
# 使用しないようにする(不正確な推定 → 実測値切り替え時のプログレスバージャンプを防止)。
# 進捗率報告: 残り時間(秒)は実際に計測した速度(ウィンドウ1個推論時間、クリップフレーム処理時間)で
# 計算し、進捗率は「経過 ÷ (経過 + 残り時間)」で算出。従来のように段階ごとに固定割合を
# 割り振ると実時間と大幅に乖離した(TA_0002 実測: 動画読み込みは全時間の0.3%なのにバー24%、
# 密な探索は80%なのにバー33% → 残り時間が序盤は短く、終盤は長く表示される)。
# 推論速度の計測前(動画読み込み・オプティカルフロー)は以下の小さな固定値のみを使用。
PROG_DECODE_END = 0.02    # フレームデコード/クロップ
PROG_FLOW_END = 0.03      # オプティカルフロー(衝撃時点判定用)
# 3D-CNN完了時の進捗率「暫定値」。大まかな走査完了で疑惑区間が判明後、残りのバーを「詳細走査」と
# 「事故クリップ生成」の所要時間比率で再分割(クリップ生成が99%付近に集中して見えなくなるのを防止)。
PROG_INFER_END = 0.90
# クリップレンダリング速度計測前の推定値: フレームあたり時間 ≈ デコード・クロップ時間の本倍数
# (原本解像度デコード + ヒートマップ合成 + エンコード + H.264変換; TA_0002 CPU 実測 約8倍)
RENDER_COST_PER_DECODE = 8.0
# 0.95〜1.0 = 事故区間CAMクリップレンダリング


def _open_clip_writer(dest_stem, fps, size):
    """クリップをmp4v一時ファイルにレンダリングするVideoWriterを開く。戻り値: (writer, rendered_path)。

    ブラウザ互換の最終変換(H.264/mp4)は _finalize_clip が担当する。
    """
    rendered = f"{dest_stem}.render.mp4"
    writer = cv2.VideoWriter(rendered, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return writer, rendered


def _finalize_clip(rendered_path, dest_stem):
    """mp4v一時ファイルをH.264/mp4(yuv420p+faststart)に再エンコードし最終パス(str)を返す。

    ffmpeg(libx264)が**必須**です(システム標準優先、なければimageio-ffmpeg)。H.264/mp4に統一することでブラウザ・OS(Chrome・Firefox・
    Safari・iOS)を問わず再生可能となるため。ffmpegがない場合は明確なエラーを送出する。
    """
    if not _FFMPEG:
        try:
            os.remove(rendered_path)
        except OSError:
            pass
        raise RuntimeError(
            "ffmpeg がインストールされていません — ブラウザ互換クリップ (H.264) の生成に必要です。"
            "pip install imageio-ffmpeg でインストールするか、"
            "システムにインストールしてください: sudo apt install -y ffmpeg (または brew install ffmpeg)")
    final = f"{dest_stem}.mp4"
    subprocess.run(
        [_FFMPEG, "-y", "-loglevel", "error", "-i", rendered_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", final],
        check=True)
    os.remove(rendered_path)
    return final


def make_activation_hook(name):
    """forward hook — 該当層の出力 (特徴マップ) を activation[name] に格納する。"""
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def _frames_to_video_tensor(frames):
    # 正規化統計量は学習と同一である必要がある — configで一括管理(Kinetics-400)
    mean = torch.tensor(config.NORM_MEAN,
                        dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD,
                       dtype=torch.float32).view(3, 1, 1, 1)
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
    return (tensor - mean) / std


def _crop_square_and_pad(frame, bbox, r, resize):
    h, w, _ = frame.shape
    x_min, y_min, x_max, y_max = bbox
    vw, vh = (x_max - x_min) * r, (y_max - y_min) * r
    cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2
    side = int(max(vw, vh))
    nx1, ny1 = cx - side // 2, cy - side // 2
    nx2, ny2 = cx + side // 2, cy + side // 2
    v_x1, v_y1 = max(0, nx1), max(0, ny1)
    v_x2, v_y2 = min(w, nx2), min(h, ny2)
    p_l, p_t = max(0, -nx1), max(0, -ny1)
    p_r, p_b = max(0, nx2 - w), max(0, ny2 - h)
    cropped = frame[v_y1:v_y2, v_x1:v_x2]
    if p_l > 0 or p_t > 0 or p_r > 0 or p_b > 0:
        cropped = np.pad(
            cropped, ((p_t, p_b), (p_l, p_r), (0, 0)), mode='constant')
    return cv2.resize(cropped, resize, interpolation=cv2.INTER_LINEAR), (nx1, ny1, nx2, ny2)


def _video_writer_worker(q, out_video):
    """VideoWriter を別スレッドで実行 — 推論とディスク I/O を並列化"""
    while True:
        frame = q.get()
        if frame is None:
            break
        out_video.write(frame)


def _state_bar_height(frame_h):
    """動画下部に付加するステータスバーの高さ — 動画の高さの約6%(最小36px)。
    H.264(yuv420p)変換は縦横が偶数である必要があるため、全体の高さが偶数になるよう調整。"""
    bar_h = max(36, int(round(frame_h * 0.06)))
    if (frame_h + bar_h) % 2:
        bar_h += 1
    return bar_h


def _add_state_bar(frame, state):
    """フレーム「下部」にステータスバーを付加した新しいフレームを返す(S/A表示)。

    従来はS/Aを動画左上に重ねて描画していたため、車両がその位置にあると隠れてしまった。
    バーを動画枠外に配置することで動画内容は1ピクセルも隠蔽されない。
    state=0 → S · NORMAL (緑), state=1 → A · ACCIDENT (赤)
    (OpenCVフォントは日本語を描画できないため英語で表記 — サーバーに日本語フォントがなくても動作可能)
    """
    h, w = frame.shape[:2]
    bar_h = _state_bar_height(h)
    bar = np.full((bar_h, w, 3), (24, 24, 24), dtype=np.uint8)
    color = (40, 40, 220) if state == 1 else (60, 170, 60)       # BGR: 赤 / 緑
    label, word = ("A", "ACCIDENT") if state == 1 else ("S", "NORMAL")
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(4, bar_h // 6)
    thick = max(1, bar_h // 20)
    # 左側: カラーバッジ + 白文字 (A/S)
    side = bar_h - 2 * pad
    x0 = 2 * pad
    cv2.rectangle(bar, (x0, pad), (x0 + side, pad + side), color, -1)
    scale = (side * 0.62) / cv2.getTextSize(label, font, 1.0, thick)[0][1]
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick + 1)
    cv2.putText(bar, label, (x0 + (side - tw) // 2, pad + (side + th) // 2),
                font, scale, (255, 255, 255), thick + 1, cv2.LINE_AA)
    # 右側: ステータステキスト
    wscale = (bar_h * 0.40) / cv2.getTextSize(word, font, 1.0, thick)[0][1]
    (_, wh), _ = cv2.getTextSize(word, font, wscale, thick)
    cv2.putText(bar, word, (x0 + side + 2 * pad, (bar_h + wh) // 2),
                font, wscale, color, thick, cv2.LINE_AA)
    return np.vstack([frame, bar])


def _draw_collision_cam(frame, cam, square, box, min_rel=config.PREDICT_CLIP_CAM_MIN,
                        alpha=config.PREDICT_CLIP_CAM_ALPHA, gamma=config.PREDICT_CLIP_CAM_GAMMA):
    """事故判定の根拠が強い領域(衝突部位)にのみCAMヒートマップを描画(frameを直接更新)。

    cam   : 正方形クロップの7×7グリッドごとの 'A − S' ロジット。グリッド平均が判定ロジット差であり、大きいほど事故根拠が強い。
    square: 正方形クロップの元座標 (x1, y1, x2, y2) — 動画外にはみ出た領域を含む
    box   : 描画対象bboxの元座標 (x1, y1, x2, y2) — 動画内にクリッピング済み
    bbox内で最大値の min_rel 倍以上の領域のみ、強いほど濃く(最大 alpha, 変化度 gamma)描画し、残りは元画像のまま維持。
    配色は外側 水色 → 緑 → 黄 → 中央 赤のリング状 — 参考仕様に準拠(衝突地点に小さな虹色のリング、2026-10-27)。
    JETの極端に青い端部(0〜0.2)は除外 — 境界部が濃紺になると塗布箇所が打撲痕のように見えるため。
    (従来はbbox全体にJETを40%一律オーバーレイし、根拠の弱い領域まで青く染まっていた)
    """
    sx1, sy1, sx2, sy2 = square
    bx1, by1, bx2, by2 = box
    if bx2 <= bx1 or by2 <= by1:
        return
    big = cv2.resize(np.asarray(cam, dtype=np.float32), (sx2 - sx1, sy2 - sy1), interpolation=cv2.INTER_CUBIC)
    # bbox部分のみ — 四捨五入で正方形より1px溢れる可能性があるため切り取り後にbboxサイズに適合
    part = big[max(0, by1 - sy1):max(0, by2 - sy1), max(0, bx1 - sx1):max(0, bx2 - sx1)]
    if part.size == 0:
        return
    part = cv2.resize(part, (bx2 - bx1, by2 - by1))
    peak = float(part.max())
    if peak <= 0:
        return
    t = np.clip((part / peak - min_rel) / (1 - min_rel), 0, 1)
    a = (t ** gamma * alpha)[..., None]
    color = cv2.applyColorMap(((0.2 + 0.8 * t) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    roi = frame[by1:by2, bx1:bx2]
    frame[by1:by2, bx1:bx2] = (roi * (1 - a) + color * a).astype(np.uint8)


def predict_full_video(
    model,
    video_path=config.PREDICT_VIDEO_PATH,
    txt_path=config.PREDICT_TXT_PATH,
    target_id=config.TARGET_ID,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    output_dir=config.PREDICT_OUTPUT_DIR,
    infer_batch_size=None,
    window_stride=config.PREDICT_WINDOW_STRIDE,
    display_offset=config.PREDICT_DISPLAY_OFFSET,
):
    """単一動画に対してスライディングウィンドウ推論および CAM 重畳動画を出力する。

    Returns:
        tuple(Path, list[dict]):
            - 出力動画パス
            - イベントリスト。各項目: {'start_frame': int, 'end_frame': int}
              (1つの動画に複数の当て逃げイベントが存在する場合に全件返却)
    """
    video_path = str(video_path)
    txt_path = str(txt_path)
    output_dir = Path(output_dir)

    print(f"[{os.path.basename(video_path)}] 全画面分析開始...")
    device = next(model.parameters()).device
    # バッチサイズ — 指定なし時はデバイスに応じた値(CPU 1 · GPU 8, config)
    infer_batch_size = infer_batch_size or predict_batch_size(device)
    model.eval()
    window_stride = max(1, int(window_stride))

    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True

    handle = model.cam_layer.register_forward_hook(
        make_activation_hook('cam_layer'))
    out_video = None
    write_queue = None
    writer_thread = None

    # 複数イベント追跡用状態変数
    events = []          # [{'start_frame': int, 'end_frame': int}, ...]
    display_state = 0    # 現在の表示状態: 0=S(正常), 1=A(衝突)
    prev_state = 0       # 直前のウィンドウ予測 (S→A遷移検知用)
    event_start = None   # 進行中イベントの開始フレーム

    try:
        if not os.path.exists(txt_path):
            print(f"テキストファイルが見つかりません: {txt_path}")
            return None, []

        bboxes = {}
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6 and parts[0] == 'car':
                    bboxes[int(parts[1])] = [
                        int(parts[2]), int(parts[3]),
                        int(parts[4]), int(parts[5]),
                    ]

        if target_id not in bboxes:
            print(f"ID {target_id}番 車両の座標がありません。存在する ID: {list(bboxes.keys())}")
            return None, []

        target_bbox = bboxes[target_id]

        cap = cv2.VideoCapture(video_path)
        original_full_frames = []
        processed_frames = []

        ret, first_frame = cap.read()
        if not ret:
            return None, []
        _, (rx1, ry1, rx2, ry2) = _crop_square_and_pad(
            first_frame, target_bbox, r_value, resize)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_full_frames.append(frame_rgb)
            processed, _ = _crop_square_and_pad(
                frame_rgb, target_bbox, r_value, resize)
            processed_frames.append(processed)
        cap.release()

        if not processed_frames:
            return None, []

        while len(processed_frames) < clip_length:
            processed_frames.append(processed_frames[-1])
            original_full_frames.append(original_full_frames[-1])

        full_video_tensor = _frames_to_video_tensor(
            processed_frames).to(device)

        orig_h, orig_w = original_full_frames[0].shape[:2]
        out_path = output_dir / f'final_{os.path.basename(video_path)}'
        os.makedirs(out_path.parent, exist_ok=True)
        out_stem = str(out_path.with_suffix(""))
        out_video, out_rendered = _open_clip_writer(
            out_stem, 30.0, (orig_w, orig_h + _state_bar_height(orig_h)))

        write_queue = queue.Queue(maxsize=64)
        writer_thread = threading.Thread(
            target=_video_writer_worker,
            args=(write_queue, out_video),
            daemon=True,
        )
        writer_thread.start()

        # CAMオーバーレイ・ボックスを正方形クロップではなく「原本bbox」サイズで描画。
        # (変数名 v_n* は維持しつつ座標を原本bbox基準で定義 — 以降の rectangle/
        #  putText/heatmap のスライシングがすべて自動的に原本bbox領域を対象とする)
        v_nx1, v_ny1 = max(0, target_bbox[0]), max(0, target_bbox[1])
        v_nx2, v_ny2 = min(orig_w, target_bbox[2]), min(orig_h, target_bbox[3])

        # ウィンドウ判定は「ウィンドウ開始 + display_offset」フレームに描画(configコメント参照)。
        display_offset = max(0, min(int(display_offset), clip_length - 1))
        # 先頭 display_offset 個のフレーム: まだ判定を描画するウィンドウがない → 状態Sとして出力
        for i in range(min(display_offset, len(original_full_frames))):
            f = cv2.cvtColor(original_full_frames[i], cv2.COLOR_RGB2BGR)
            cv2.rectangle(f, (v_nx1, v_ny1), (v_nx2, v_ny2), (0, 255, 0), 2)
            write_queue.put(_add_state_bar(f, display_state))
        next_frame_to_write = min(display_offset, len(original_full_frames))

        print(f"バッチスライディングウィンドウ推論およびヒートマップ生成中... (stride={window_stride})")
        num_windows = full_video_tensor.size(1) - (clip_length - 1)
        window_starts = list(range(0, num_windows, window_stride))

        with torch.inference_mode():
            for batch_start in range(0, len(window_starts), infer_batch_size):
                batch_window_starts = window_starts[batch_start:batch_start +
                                                    infer_batch_size]

                clips = torch.stack(
                    [full_video_tensor[:, i:i + clip_length, :, :] for i in batch_window_starts])
                if is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST:
                    clips = clips.contiguous(
                        memory_format=torch.channels_last_3d)

                outputs = model(clips)
                probs = F.softmax(outputs, dim=1)
                pred_classes = outputs.argmax(dim=1)
                feat_maps = activation['cam_layer']

                for offset, window_idx in enumerate(batch_window_starts):
                    # イベント境界はウィンドウ終端(frame_idx)基準、判定・CAM表示は disp_idx に描画
                    frame_idx = window_idx + clip_length - 1
                    disp_idx = window_idx + display_offset
                    pred_class = int(pred_classes[offset].item())
                    conf = probs[offset, pred_class].item() * 100

                    # ── イベント状態遷移検知 ──────────────────────────────
                    if prev_state == 0 and pred_class == 1:
                        # S → A: 新規衝突イベント開始
                        event_start = window_idx
                    elif prev_state == 1 and pred_class == 0:
                        # A → S: 衝突イベント終了
                        if event_start is not None:
                            events.append({
                                'start_frame': event_start,
                                'end_frame': frame_idx - 1,
                            })
                            event_start = None
                    prev_state = pred_class
                    display_state = pred_class
                    # ────────────────────────────────────────────────────────

                    # ⑤ CAM演算のGPU維持
                    feat_map = feat_maps[offset]
                    weight = model.head_conv.weight[pred_class]
                    cam = F.relu(torch.sum(weight * feat_map, dim=0))
                    cam_2d = torch.mean(cam, dim=0)
                    cam_min, cam_max = cam_2d.min(), cam_2d.max()
                    cam_2d = (cam_2d - cam_min) / (cam_max - cam_min + 1e-8)
                    cam_np = (cam_2d * 255).byte().cpu().numpy()

                    heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
                    heatmap = cv2.resize(heatmap, (rx2 - rx1, ry2 - ry1))
                    heatmap_valid = heatmap[
                        v_ny1 - ry1:(v_ny1 - ry1) + (v_ny2 - v_ny1),
                        v_nx1 - rx1:(v_nx1 - rx1) + (v_nx2 - v_nx1),
                    ]

                    # スキップされたフレーム: 直前の display_state をそのまま表示
                    for skipped_idx in range(next_frame_to_write, disp_idx):
                        sf = cv2.cvtColor(
                            original_full_frames[skipped_idx], cv2.COLOR_RGB2BGR)
                        cv2.rectangle(sf, (v_nx1, v_ny1),
                                      (v_nx2, v_ny2), (0, 255, 0), 2)
                        write_queue.put(_add_state_bar(sf, display_state))

                    # 現在の予測フレーム
                    final_frame = cv2.cvtColor(
                        original_full_frames[disp_idx], cv2.COLOR_RGB2BGR)
                    roi = final_frame[v_ny1:v_ny2, v_nx1:v_nx2]

                    if pred_class == 1:
                        # 四捨五入の1px差を吸収するためヒートマップをbbox領域サイズに適合
                        hm = cv2.resize(heatmap_valid, (roi.shape[1], roi.shape[0]))
                        final_frame[v_ny1:v_ny2, v_nx1:v_nx2] = cv2.addWeighted(
                            roi, 0.6, hm, 0.4, 0)
                        bbox_color = (0, 0, 255)
                        conf_text = f"Accident ({conf:.1f}%)"
                    else:
                        bbox_color = (0, 255, 0)
                        conf_text = f"Normal ({conf:.1f}%)"

                    cv2.rectangle(final_frame, (v_nx1, v_ny1),
                                  (v_nx2, v_ny2), bbox_color, 3)
                    # 信頼度はbbox上部に小さく表示
                    cv2.putText(final_frame, conf_text,
                                (v_nx1, max(v_ny1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
                    # クラス状態(S / A)は動画下部バーに表示
                    write_queue.put(_add_state_bar(final_frame, pred_class))
                    next_frame_to_write = disp_idx + 1

        # 最終予測以降の残りフレーム: 最終 display_state を維持
        for skipped_idx in range(next_frame_to_write, len(original_full_frames)):
            sf = cv2.cvtColor(
                original_full_frames[skipped_idx], cv2.COLOR_RGB2BGR)
            cv2.rectangle(sf, (v_nx1, v_ny1), (v_nx2, v_ny2), (0, 255, 0), 2)
            write_queue.put(_add_state_bar(sf, display_state))

        # 動画終端までA状態が維持された場合のイベントクローズ
        if prev_state == 1 and event_start is not None:
            events.append({
                'start_frame': event_start,
                'end_frame': len(original_full_frames) - 1,
            })

        write_queue.put(None)
        writer_thread.join()
        out_video.release()
        out_video = None
        # ブラウザ互換最終版(H.264/mp4)へ変換
        out_path = _finalize_clip(out_rendered, out_stem)

        # 結果出力
        print(f"\n分析完了! 出力ファイル: {out_path}")
        if events:
            print(f"検出された衝突イベント: {len(events)}件")
            for i, ev in enumerate(events, 1):
                print(f"  イベント {i}: 開始フレーム {ev['start_frame']}, "
                      f"終了フレーム {ev['end_frame']}")
        else:
            print("検出された衝突イベントなし (Normal)")

        return out_path, events

    finally:
        handle.remove()
        if writer_thread is not None and writer_thread.is_alive():
            try:
                write_queue.put_nowait(None)
            except queue.Full:
                pass
            writer_thread.join(timeout=5)
        if out_video is not None:
            out_video.release()


def fps_scaled_event_params(fps, coarse_step=None, min_span=None, clip_length=None, enabled=None):
    """30fps基準の粗い探索間隔とイベント最小長さを動画fpsに適合させる。"""
    coarse_step = config.PREDICT_COARSE_STEP if coarse_step is None else coarse_step
    min_span = config.PREDICT_MIN_EVENT_SPAN_FRAMES if min_span is None else min_span
    clip_length = config.CLIP_LENGTH if clip_length is None else clip_length
    enabled = config.PREDICT_FPS_SCALED_POSTPROC if enabled is None else enabled
    if not enabled or not fps or fps <= 0:
        return int(coarse_step), int(min_span)

    scale = float(fps) / float(config.PREDICT_POSTPROC_REF_FPS)
    coarse_step = max(1, int(round(coarse_step * scale)))
    base_span = clip_length - 2
    consecutive_windows = min_span - base_span
    if consecutive_windows > 1:
        min_span = base_span + max(1, int(round(consecutive_windows * scale)))
    return coarse_step, int(min_span)


def predict_events_and_clips(
    model,
    video_path,
    bbox,
    output_dir,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    infer_batch_size=None,
    window_stride=config.PREDICT_WINDOW_STRIDE,
    display_offset=config.PREDICT_DISPLAY_OFFSET,
    cam_overlay=config.PREDICT_CLIP_CAM_OVERLAY,
    clip_pad_frames=config.PREDICT_CLIP_PAD_FRAMES,
    progress_callback=None,
    use_coarse_scan=config.PREDICT_USE_COARSE_SCAN,
    coarse_step=config.PREDICT_COARSE_STEP,
    coarse_prob=config.PREDICT_COARSE_PROB,
    coarse_pad=config.PREDICT_COARSE_PAD,
    render_clips=True,
    stats=None,
    clip_name=None,
):
    """Webサービス用: 単一対象車両(bbox)に対して事故疑惑区間のみを検知し、
    該当区間に対してのみ短いCAMオーバーレイクリップを生成する。

    全長の処理済み動画を再生成しないため、長時間の動画に対しても高効率。

    Args:
        model: ロード済みの HitAndRun3DCNN
        video_path (str|Path): 元動画パス
        bbox (tuple|list): 対象車両座標 (xmin, ymin, xmax, ymax) — 元解像度基準
        output_dir (str|Path): クリップ保存フォルダ
        clip_pad_frames (int): 区間前後に付加する余白フレーム数
        use_coarse_scan (bool): Trueなら粗い探索(coarse_step間隔)後に P(A) >
            coarse_prob のウィンドウ周辺 ±coarse_pad のみ密に走査。Falseなら全スキャン。
        clip_name (str|None): クリップファイル名の接頭辞('<clip_name>_event1.mp4')。未指定時は動画ファイル名。
            サービスは分析ごとに個別名を指定 — 同一動画の再分析時にも前回のクリップを上書きしないようにする。
        render_clips (bool): FalseならCAMクリップを生成せずイベントリストのみ返却
            (clip_path=None)。評価において「サービスと同一の推論パス」でイベントを抽出する際に
            使用 — レンダリングは評価に不要であり最も処理負荷の高い段階であるため。
        stats (dict|None): 指定時、段階別処理時間(秒)とウィンドウ数を格納(評価時の速度
            計測用)。キー: frames, decode_sec, flow_sec, infer_sec, render_sec,
            windows_total, windows_eval.

    Returns:
        list[dict]: イベントリスト。各項目:
            {
                'start_frame', 'end_frame',
                'start_sec', 'end_sec',
                'crash_prob',        # 区間代表確率 (0~1)
                'clip_path',         # 生成されたCAMクリップパス (Path)
            }
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    window_stride = max(1, int(window_stride))
    display_offset = max(0, min(int(display_offset), clip_length - 1))

    device = next(model.parameters()).device
    # バッチサイズ — 指定なし時はデバイスに応じた値(CPU 1 · GPU 8, config)
    infer_batch_size = infer_batch_size or predict_batch_size(device)
    model.eval()
    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True

    target_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    # 3D-CNN実行ランナー — サーバー環境に最適な最速ランナー(infer_runtime.py、初回のみ選定)。
    # CAMヒートマップも同一ランナーが判定と同時に出力(runner.with_cam)。クリップ非生成時(評価)は受領しない。
    runner = get_runner(model)
    cam_overlay = cam_overlay and render_clips
    # ── 動画を最初から最後まで1度走査しながら即座に判定(ストリーミング処理) ──────────────
    # 従来は224クロップフレームを動画全体分メモリに保持した後にfloatテンソルに変換して判定していたため
    # フレームあたり約2MB(1時間動画で約200GB)を消費していた。現在は直近数十フレームのみを保持し、
    #  · 粗い探索ウィンドウ(coarse_step間隔)はフレームが揃い次第判定
    #  · 疑惑ウィンドウ周辺(±coarse_pad)のウィンドウはそのフレームがバッファに残っている間に判定。
    # 判定対象ウィンドウの集合と2個ずつバッチ化する順序は従来と同一 → 結果は完全一致。
    #   粗いウィンドウ: 動画順にバッチ化し、末尾ウィンドウがグリッド外なら最後尾に追加(従来と同一)。
    #   密なウィンドウ: 全体をソートした順序でバッチ化。今後出現する疑惑ウィンドウは「未判定の
    #   粗いウィンドウ - coarse_pad」以降にのみ発生するため、それより前のウィンドウは順序が確定。
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    coarse_step, min_span = fps_scaled_event_params(
        fps, coarse_step, config.PREDICT_MIN_EVENT_SPAN_FRAMES, clip_length
    )

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return []
    orig_h, orig_w = first_frame.shape[:2]
    _, (rx1, ry1, rx2, ry2) = _crop_square_and_pad(
        first_frame, target_bbox, r_value, resize)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 原本bbox領域(フレーム境界にクランプ) — CAMオーバーレイ・ボックスをユーザー指定の
    # 実際のbboxサイズで描画するために使用(正方形クロップサイズではない)。
    bx1, by1 = max(0, target_bbox[0]), max(0, target_bbox[1])
    bx2, by2 = min(orig_w, target_bbox[2]), min(orig_h, target_bbox[3])

    # 被害車両bboxを224クロップ座標系に換算 → 衝撃「時点」判定専用ROI。
    # (正方形クロップ余白を通過する加害車両モーションに衝撃時点が引っ張られないようにする)
    _cw = max(1, rx2 - rx1)
    _ch = max(1, ry2 - ry1)
    _sx, _sy = resize[0] / _cw, resize[1] / _ch
    _roi = (
        max(0, min(resize[0] - 1, int((bx1 - rx1) * _sx))),
        max(0, min(resize[1] - 1, int((by1 - ry1) * _sy))),
        max(1, min(resize[0], int((bx2 - rx1) * _sx))),
        max(1, min(resize[1], int((by2 - ry1) * _sy))),
    )

    def _report(frac=None, eta=None, eta_min=False, stage=None, count=None, index=None):
        """進捗率(0〜1)・現在ステージ・(ステージの残り時間)を報告。画面は進捗率とステージ名を表示。

        stage: decode(動画読み込み) · motion(車両の動き計算) · coarse(動画全体の走査、
               count=これまでに検出された疑惑箇所数) · refine(疑惑区間の詳細走査、index/count)
               · scan(粗い探索なしで全走査) · events(事故区間整理) · render(事故クリップ
               生成、index/count)
        進捗率は後戻りしない(下記の _run_windows コメント参照)。100%はタスク完了(SUCCESS)でのみ
        表示されるよう0.99で止める。eta は「現在のステージ」の残り時間(秒、不明時はNone)。"""
        if progress_callback is None:
            return
        if eta is not None:
            eta = max(0.0, float(eta))
        progress_callback(min(0.99, frac or 0.0), eta, eta_min, stage, count, index)

    events = []          # [{'start_frame','end_frame'}, ...]
    # 表示フレーム(ウィンドウ開始 + display_offset) → (グリッド別 'A − S' ロジット 7×7 または None, prob) (事故予測ウィンドウのみ)
    accident_overlays = {}
    # window_idx → pred_class / P(A) (3D-CNNを実際に実行したウィンドウのみ)
    window_pred = {}
    window_prob_a = {}

    _mean = np.asarray(config.NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
    _std = np.asarray(config.NORM_STD, dtype=np.float32).reshape(3, 1, 1)
    # フレーム番号 → 正規化済み224クロップ (3, H, W) float32 テンソル。必要な直近フレームのみ保持。
    # バッファ投入時に1度のみ正規化し、密なウィンドウで同一フレームの重複処理を回避。
    buf = {}
    pool = []

    def _normalize(rgb):
        t = pool.pop() if pool else torch.empty((3,) + rgb.shape[:2], dtype=torch.float32)
        a = t.numpy()
        np.copyto(a, rgb.transpose(2, 0, 1), casting='unsafe')
        np.divide(a, np.float32(255.0), out=a)
        np.subtract(a, _mean, out=a)
        np.divide(a, _std, out=a)
        return t

    def _clip(w):
        """ウィンドウ w の入力テンソル (3, T, H, W) — 従来の「全体テンソルを切り取ったもの」と同一値。"""
        return torch.stack([buf[i] for i in range(w, w + clip_length)], dim=1)

    speed = {"n": 0, "sec": 0.0, "first": True}
    infer_sec = [0.0]
    infer_prog = [PROG_FLOW_END]

    def _run_batch(batch_window_starts, clips_cpu):
        """ウィンドウバッチを3D-CNNに投入して予測・P(A)・CAMを格納(従来の _run_windows の1バッチ分)。"""
        t_batch = time.time()
        with torch.inference_mode():
            clips = torch.stack(clips_cpu).to(device)
            if is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST:
                clips = clips.contiguous(memory_format=torch.channels_last_3d)
            if cam_overlay:
                outputs, cams = runner.with_cam(clips)
            else:
                outputs, cams = runner(clips), None
            probs = F.softmax(outputs, dim=1)
            pred_classes = outputs.argmax(dim=1)
            for offset, window_idx in enumerate(batch_window_starts):
                # 判定・CAMはウィンドウ終端ではなく「ウィンドウ開始 + display_offset」フレームに
                # 描画 — ウィンドウ末尾に付加するとA表示が実際の衝突より遅延するため(configコメント参照)。
                disp_idx = window_idx + display_offset
                pred_class = int(pred_classes[offset].item())
                prob = probs[offset, pred_class].item()
                window_pred[window_idx] = pred_class
                window_prob_a[window_idx] = probs[offset, 1].item()
                if pred_class == 1:
                    # ヒートマップはクリップ生成時に描画(_draw_collision_cam)。ここではグリッド別 'A − S' ロジット
                    # 49個のみ保持し、事故区間が長くてもbboxサイズ画像を蓄積しないようにする
                    cam_as = (None if cams is None
                              else (cams[offset, 1] - cams[offset, 0]).float().cpu().numpy())
                    accident_overlays[disp_idx] = (cam_as, prob)
        dt = time.time() - t_batch
        infer_sec[0] += dt
        if speed["first"]:
            speed["first"] = False
        else:
            speed["n"] += len(batch_window_starts)
            speed["sec"] += dt

    k = max(1, int(round(coarse_step / window_stride))) if use_coarse_scan else 1
    grid = k * window_stride
    refine_on = k > 1
    coarse_q = []
    next_grid = 0
    coarse_list = []
    hot = []
    refine_pending = set()
    refine_held = []
    refine_done = []

    def _hot_regions():
        """疑惑区間数 — 連続した「詳細走査範囲」の塊(画面上の「疑惑地点 N箇所」)。"""
        n, end = 0, None
        for h in hot:
            lo = max(0, h - coarse_pad)
            if end is None or lo > end:
                n += 1
            end = h + coarse_pad
        return n

    def _flush_coarse(final=False):
        while coarse_q and (len(coarse_q) >= infer_batch_size or final):
            ws = coarse_q[:infer_batch_size]
            del coarse_q[:len(ws)]
            _run_batch(ws, [_clip(w) for w in ws])
            coarse_list.extend(ws)
            for w in ws:
                if refine_on and window_prob_a[w] > coarse_prob:
                    hot.append(w)
                    lo = max(0, w - coarse_pad)
                    lo += (-lo) % window_stride
                    for r in range(lo, w + coarse_pad + 1, window_stride):
                        if r % grid != 0 and r not in window_pred:
                            refine_pending.add(r)

    def _future_coarse(cur_frame):
        """今後判定する粗いウィンドウの下限 — キュー先頭、次回グリッドウィンドウ、および未確定の
        「動画末尾ウィンドウ」(現在入力された最終ウィンドウ以降)となり得る最古の位置。"""
        return min(coarse_q[0] if coarse_q else next_grid, next_grid,
                   max(0, cur_frame - clip_length + 1))

    def _flush_refine(cur_frame, final=False, last_start=None):
        """順序が確定しフレームが準備された密なウィンドウをソート順に2個ずつ判定する。"""
        if final:
            ready_limit = last_start
            order_limit = float("inf")
        else:
            ready_limit = cur_frame - clip_length + 1
            order_limit = _future_coarse(cur_frame) - coarse_pad
        for w in sorted(refine_pending):
            if w > ready_limit or w >= order_limit:
                break
            refine_pending.discard(w)
            refine_held.append((w, _clip(w)))
            if len(refine_held) == infer_batch_size:
                ws = [x[0] for x in refine_held]
                _run_batch(ws, [x[1] for x in refine_held])
                refine_done.extend(ws)
                refine_held.clear()
        if final and refine_held:
            ws = [x[0] for x in refine_held]
            _run_batch(ws, [x[1] for x in refine_held])
            refine_done.extend(ws)
            refine_held.clear()

    # 衝撃時点用オプティカルフロー — 読み込み順に同一間隔(flow_step)で計算し数値のみ保持
    flow_step = max(1, int(config.PREDICT_FLOW_SAMPLE_STEP))
    flow_mag_l, flow_roi_l = [], []
    flow_ok = True
    prev_gray = None
    qx1, qy1, qx2, qy2 = _roi

    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    decode_sec = flow_sec = 0.0
    t_stream = time.time()
    n_read = 0
    last_frame = None
    max_buf = [0]
    stage_name = 'coarse' if k > 1 else 'scan'
    _report(0.0, stage='decode')
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed, _ = _crop_square_and_pad(frame_rgb, target_bbox, r_value, resize)
        i = n_read
        buf[i] = _normalize(processed)
        last_frame = buf[i]
        n_read += 1
        decode_sec += time.time() - t0

        # オプティカルフロー(flow_step間隔フレームペア) — イベント内の「衝撃時点」(動きが最大の
        # フレーム)を選定するためにのみ使用(事前選別は微小衝突を取りこぼすためモデル粗探索に変更)。
        # クロップ全体(mag)と被害車両bbox領域(mag_roi)の2系列を生成し時点はbbox側を参照 —
        # 正方形クロップ余白を通過する加害車両モーションが被害車両の微細な揺れを圧倒するため
        # クロップ全体で判定すると「接触」ではなく「加害車両進入」を指していた(実測 -52/+105フレーム)。
        t0 = time.time()
        flow_mag_l.append(0.0)
        flow_roi_l.append(0.0)
        if flow_ok and i % flow_step == 0:
            try:
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    fmag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    m = float(fmag.mean())
                    sub = fmag[qy1:qy2, qx1:qx2]
                    mr = float(sub.mean()) if sub.size else m
                    for j in range(i - flow_step + 1, i + 1):
                        flow_mag_l[j] = m
                        flow_roi_l[j] = mr
                prev_gray = gray
            except Exception:
                flow_ok = False
        flow_sec += time.time() - t0

        # フレームが揃った粗いウィンドウをキューに投入し、バッチが満杯になり次第判定
        while next_grid + clip_length - 1 <= i:
            coarse_q.append(next_grid)
            next_grid += grid
        _flush_coarse()
        if refine_on:
            _flush_refine(i)

        # 不要になった過去フレームを破棄 — 今後判定する粗いウィンドウとその周辺(±pad)の
        # 密なウィンドウ、および未判定の密なウィンドウに必要なフレームのみを保持
        need = _future_coarse(i)
        if refine_on:
            need = max(0, need - coarse_pad)
            if refine_pending:
                need = min(need, min(refine_pending))
        for old in [f for f in buf if f < need]:
            pool.append(buf.pop(old))
        max_buf[0] = max(max_buf[0], len(buf))

        # 進捗率: 動画の読み込み割合(残り時間 = 残りフレーム × これまでのフレームあたり時間)。
        # 10フレームごとに通知 — 長尺動画では1%が数十秒となるため1%刻みではバーが停止して見える
        # (実際の更新頻度は prediction_job が0.4秒間隔に制御)
        if total_frames_hint > 0 and n_read % 10 == 0:
            frac = min(1.0, n_read / total_frames_hint)
            infer_prog[0] = PROG_FLOW_END + (PROG_INFER_END - PROG_FLOW_END) * frac
            per = (time.time() - t_stream) / n_read
            _report(infer_prog[0], eta=(total_frames_hint - n_read) * per,
                    stage=stage_name, count=_hot_regions() if k > 1 else None)
    cap.release()

    if n_read == 0:
        return []
    real_frame_count = n_read
    # ウィンドウ長より短い動画は最終フレームを反復して1ウィンドウ分を充填(従来と同一)
    n_total = n_read
    while n_total < clip_length:
        buf[n_total] = last_frame
        n_total += 1
        while next_grid + clip_length - 1 <= n_total - 1:
            coarse_q.append(next_grid)
            next_grid += grid
    decode_sec_per_frame = decode_sec / max(1, n_read)

    num_windows = n_total - (clip_length - 1)
    window_starts = list(range(0, max(0, num_windows), window_stride))
    last_start = window_starts[-1]
    # 動画末尾ウィンドウが粗いグリッドにない場合、粗いウィンドウとして最後尾に追加(従来と同一)
    if last_start % grid != 0:
        refine_pending.discard(last_start)
        coarse_q.append(last_start)
    _flush_coarse(final=True)
    if refine_on:
        _report(infer_prog[0], stage='refine', count=_hot_regions())
        _flush_refine(None, final=True, last_start=last_start)
    infer_prog[0] = PROG_INFER_END
    buf.clear()

    flow_mag = flow_mag_roi = None
    if flow_ok and n_total >= clip_length + 2:
        flow_mag = np.asarray(flow_mag_l, dtype=np.float32)
        flow_mag_roi = np.asarray(flow_roi_l, dtype=np.float32)

    _report(infer_prog[0], stage='events')
    print(f"[第2フェーズ] 粗探索 {len(coarse_list)}個 (間隔 {grid}) → "
          f"候補 {len(hot)}個 周辺 {len(refine_done)}個 追加 | 全 "
          f"{len(window_starts)} ウィンドウ中 3D-CNN 評価 {len(window_pred)}個 "
          f"({100.0 * len(window_pred) / max(1, len(window_starts)):.0f}%) "
          f"| 動画 {real_frame_count}フレーム, バッファ最大 {max_buf[0]}枚")

    # Phase B: 全ウィンドウの順序に従い状態遷移マシン(S→A→S)でイベント区間を確定。
    # 未評価ウィンドウは非事故(0)とみなす。
    prev_state = 0
    event_start = None
    for window_idx in window_starts:
        frame_idx = window_idx + clip_length - 1
        pred_class = window_pred.get(window_idx, 0)
        if prev_state == 0 and pred_class == 1:
            event_start = window_idx
        elif prev_state == 1 and pred_class == 0:
            if event_start is not None:
                events.append({'start_frame': event_start,
                               'end_frame': frame_idx - 1})
                event_start = None
        prev_state = pred_class

    # 動画末尾までA状態が維持された場合のイベントクローズ
    if prev_state == 1 and event_start is not None:
        events.append({'start_frame': event_start,
                       'end_frame': real_frame_count - 1})

    # ── イベント後処理: 結合 → 長さフィルタ (1つの衝突が複数に分裂するのを防止) ──
    # 順序が重要: 先に結合することで「1つの衝突の断片群」が1つに統合されて長さを
    # 回復し、その後に残った孤立した瞬間的ノイズのみが長さフィルタで除去される。
    _n_raw = len(events)
    # 結合間隔は秒単位の設定を該当動画のfpsに換算(fps 10〜60が混在)
    merge_gap_frames = int(round(fps * config.PREDICT_EVENT_MERGE_GAP_SEC))
    if events:
        events.sort(key=lambda e: e['start_frame'])
        merged = [events[0]]
        for ev in events[1:]:
            gap = ev['start_frame'] - merged[-1]['end_frame']
            if gap <= merge_gap_frames:
                merged[-1]['end_frame'] = max(
                    merged[-1]['end_frame'], ev['end_frame'])
            else:
                merged.append(ev)
        events = [e for e in merged
                  if (e['end_frame'] - e['start_frame'])
                  >= min_span]
    if _n_raw != len(events):
        print(f"[イベント後処理] {_n_raw}件 → マージ・最小長フィルタ後 {len(events)}件")

    # ── 第2パス: イベント区間のフレームのみ動画から再読み込みしてCAMクリップをレンダリング ──────
    # 元フレームをRAMに保持せず、各事故区間のみ cap.set でシークして読み込む。
    results = []
    base_name = clip_name or os.path.splitext(os.path.basename(video_path))[0]
    render_cap = (cv2.VideoCapture(video_path)
                  if (events and render_clips) else None)
    # クリップレンダリング残り時間 = 残りフレーム × フレームあたり時間(計測前はデコード基準推定)
    render_total = sum(
        min(real_frame_count - 1, e['end_frame'] + clip_pad_frames)
        - max(0, e['start_frame'] - clip_pad_frames) + 1
        for e in events) if render_clips else 0
    render_done = [0]
    render_idx = [0]
    render_start = infer_prog[0]
    t_render = time.time()

    def _report_render():
        if render_done[0] >= 15:
            per = (time.time() - t_render) / render_done[0]
        else:
            per = decode_sec_per_frame * RENDER_COST_PER_DECODE
        frac = render_start + (0.99 - render_start) * render_done[0] / max(1, render_total)
        _report(frac, eta=(render_total - render_done[0]) * per, stage='render',
                count=len(events), index=render_idx[0])

    try:
        for ev_idx, ev in enumerate(events, 1):
            render_idx[0] = ev_idx
            _report_render()
            start_f = ev['start_frame']
            end_f = ev['end_frame']
            clip_start = max(0, start_f - clip_pad_frames)
            clip_end = min(real_frame_count - 1, end_f + clip_pad_frames)

            # 区間代表確率 = 区間内事故フレーム確率の最大値
            probs_in_event = [p for f, (_, p) in accident_overlays.items()
                              if start_f <= f <= end_f]
            crash_prob = max(probs_in_event) if probs_in_event else None

            clip_path = None
            raw_clip_path = None
            if render_clips:
                clip_stem = str(output_dir / f'{base_name}_event{ev_idx}')
                writer, rendered_path = _open_clip_writer(
                    clip_stem, fps, (orig_w, orig_h + _state_bar_height(orig_h)))
                # 原本事故クリップ: 同一フレーム区間をオーバーレイなしでそのまま保存
                # (ユーザーがバウンディングボックス・ステータス表示のない原本を別途ダウンロード可能にするため)
                raw_stem = f'{clip_stem}_raw'
                raw_writer, raw_rendered = _open_clip_writer(raw_stem, fps, (orig_w, orig_h))


                # 事故区間の開始フレームへシーク後に順次デコード
                # Aとして描画する区間 = 本イベントで判定が描画された最初のフレーム 〜 最終フレーム。
                # (イベント開始 start_f は先頭Aウィンドウの「開始」であるため、ここからAを描画すると
                #  判定が未確定の前半部がSとして描画されたり早すぎる表示となる)
                a_keys = sorted(k for k in accident_overlays if start_f <= k <= end_f)
                a_first = a_keys[0] if a_keys else None
                a_last = a_keys[-1] + window_stride - 1 if a_keys else None
                render_cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
                last_cam = None
                # ボックス枠線の太さは解像度に応じて調整(1080p以下 2px · 1440p 3px · 4K 4px)。
                # 以前は事故 3px / 正常 2px 固定で、低解像度動画で車両を過度に覆っていた。
                # 1px は H.264(yuv420p、色情報が半分解像度)で圧縮すると赤線が滲むため最小2px。
                box_thick = max(2, int(round(orig_h / 540)))
                for f in range(clip_start, clip_end + 1):
                    ret, frame = render_cap.read()
                    if not ret:
                        break
                    raw_writer.write(frame)
                    if f in accident_overlays and accident_overlays[f][0] is not None:
                        last_cam = accident_overlays[f][0]
                    in_event = a_first is not None and a_first <= f <= a_last
                    if in_event:
                        if last_cam is not None:
                            # 衝突部位にのみヒートマップ — ボックス枠線(事故 赤・正常 緑)はその上に描画
                            _draw_collision_cam(frame, last_cam, (rx1, ry1, rx2, ry2), (bx1, by1, bx2, by2))
                        cv2.rectangle(frame, (bx1, by1),
                                      (bx2, by2), (0, 0, 255), box_thick, cv2.LINE_AA)
                        state = 1
                    else:
                        cv2.rectangle(frame, (bx1, by1),
                                      (bx2, by2), (0, 255, 0), box_thick, cv2.LINE_AA)
                        state = 0
                    # S/A は動画枠外(下部バー)に表示 — 車両を隠さないように配置
                    writer.write(_add_state_bar(frame, state))
                    render_done[0] += 1
                    if render_done[0] % 15 == 0:
                        _report_render()
                writer.release()
                raw_writer.release()
                # ブラウザ互換最終版(H.264/mp4)へ変換
                clip_path = _finalize_clip(rendered_path, clip_stem)
                raw_clip_path = _finalize_clip(raw_rendered, raw_stem)

            # ── 事故「時点」確定: イベント区間内のフロー最大点(=衝撃瞬間) ──
            # モデルは「ウィンドウ最終フレーム=衝突終了」として学習されておりウィンドウ開始
            # (start_f = F-29)は体系的に早すぎる。ラベル検証の結果、フロー
            # ピークが衝突区間に 5/5 的中(ラベル開始 +3〜4フレーム)し最も高精度であった。
            # (単純な立ち上がりエッジは「接近車両」モーションを先に捉えて最大 -56フレーム
            #  乖離したため不採用)
            # 時点判定は「被害車両bbox限定」フローで行う(なければクロップ全体へフォールバック)
            _mag_for_impact = (flow_mag_roi if flow_mag_roi is not None
                               else flow_mag)
            impact_f = start_f
            if _mag_for_impact is not None and end_f >= start_f:
                seg = _mag_for_impact[
                    start_f:min(end_f + 1, len(_mag_for_impact))]
                if seg.size:
                    impact_f = start_f + int(np.argmax(seg))

            results.append({
                # start_* = 事故発生(衝撃)時点 — UIタイムライン/DB記録用
                'start_frame': impact_f,
                'start_sec': impact_f / fps,
                'end_frame': end_f,
                'end_sec': end_f / fps,
                # 参考用: モデルがAと判定した全区間(クリップレンダリング範囲基準)
                'event_start_frame': start_f,
                'event_end_frame': end_f,
                'crash_prob': crash_prob,
                'clip_path': clip_path,
                'raw_clip_path': raw_clip_path,   # 合成なしの原本事故クリップ(ファイル名 = クリップ + '_raw')
                # クリップ先頭フレーム = 原本のこのフレーム — 再生画面が元動画時間に合わせてCAMクリップを重ねて同期再生する際に使用
                'clip_start_frame': clip_start,
                'clip_start_sec': clip_start / fps,
            })
    finally:
        if render_cap is not None:
            render_cap.release()

    _report(0.99, eta=0.0, stage='render' if events and render_clips else 'events',
            count=len(events) if events and render_clips else None,
            index=len(events) if events and render_clips else None)
    if stats is not None:
        stats.update(frames=real_frame_count, decode_sec=decode_sec, flow_sec=flow_sec,
                     infer_sec=infer_sec[0], render_sec=time.time() - t_render,
                     windows_total=len(window_starts), windows_eval=len(window_pred),
                     backend=runner.name)
    print(f"[predict_events_and_clips] {len(results)}件 事故区間クリップ生成完了")
    return results
