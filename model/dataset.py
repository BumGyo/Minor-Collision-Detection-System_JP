import os
import glob
import hashlib
import inspect
import math
from pathlib import Path
import cv2
import numpy as np
import torch
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

import config


def _open_video(path, threads=None):
    """動画の読み込み。threads を指定した場合、デコードスレッド数を制限する (config.TRAIN_CV2_THREADS —
    DataLoader の複数ワーカーが CPU コアを共有する際)。設定できない場合はデフォルト方式で開く。デコード結果の画素は同一。"""
    if threads and hasattr(cv2, 'CAP_PROP_N_THREADS'):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, int(threads)])
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(path)


def h264_roundtrip(frames, crf, gop):
    """RGB フレームリストをメモリ上で H.264 (libx264) 圧縮・展開し、同一フレーム数で返す。

    実際の防犯カメラのようにキーフレーム間は前フレームとの差分のみが記録され、静止部分は維持され動きのある領域でブロックノイズが発生する。
    Bフレームなし (zerolatency)・スレッド数 1 (DataLoader ワーカーがコアを分散利用)。解像度は偶数である必要がある (yuv420p)。
    """
    import io
    import av
    h, w = frames[0].shape[:2]
    buf = io.BytesIO()
    out = av.open(buf, mode='w', format='h264')
    try:
        st = out.add_stream('libx264', rate=30)
        st.width, st.height, st.pix_fmt = w, h, 'yuv420p'
        st.options = {'crf': str(int(crf)), 'preset': 'ultrafast', 'tune': 'zerolatency',
                      'g': str(int(gop)), 'threads': '1'}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format='rgb24')):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    finally:
        out.close()
    buf.seek(0)
    inp = av.open(buf, mode='r', format='h264')
    try:
        dec = [fr.to_ndarray(format='rgb24') for fr in inp.decode(video=0)]
    finally:
        inp.close()
    if len(dec) != len(frames):
        raise ValueError(f'H.264 復元フレーム数が一致しません ({len(dec)} != {len(frames)})')
    return dec


LIGHT_COLORS = {'white': (255, 255, 255), 'warm': (255, 236, 200), 'cool': (215, 232, 255), 'red': (255, 60, 50)}


def add_light(frames, kind, dark=1.0, peak=150.0, radius=0.35, t0=8, dur=14, direction=1, color='white', y=0.55):
    """クロップフレーム (RGB uint8, 同一解像度) に動的な光を重畳する — 照明拡張・robust_check 照明バリエーション共通 (2026-11-05)。

    dark     : 事前に乗算する明度倍率 (夜間シミュレーション、1.0 で等倍)
    kind     : 'sweep' = 楕円形の光の塊が t0 から dur フレームかけて一方の端から反対側へ通過 (すれ違う車のヘッドライトが照射)
               'ramp'  = t0 から dur フレームかけて徐々に明るくなり維持、片側がより明るいグラデーション (接近する車の照射)
    peak     : 加算する明度最大値 (0〜255), radius: 光の横方向半径 (クロップ一辺に対する比率), y: 光の中心高さ (クロップに対する比率)
    direction: 1 = 左→右 (ramp は左側が明るい), -1 = 反対。color: LIGHT_COLORS のキー
    光の形状はフレーム間で滑らかに連続し車体の輪郭は維持される — 明度変化 ≠ 車両の揺れであることを学習させる目的。
    """
    n = len(frames)
    h, w = frames[0].shape[:2]
    col = np.array(LIGHT_COLORS.get(color, LIGHT_COLORS['white']), np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx, sy = max(2.0, radius * w), max(2.0, radius * w * 0.6)
    # クロップが画面外に出て黒埋めされた領域 (全フレームで 0) は背景シーンではないため光を受けない
    keep = (~np.all(np.stack(frames) == 0, axis=(0, 3)))[..., None].astype(np.float32)
    out = []
    for t, f in enumerate(frames):
        img = f.astype(np.float32) * dark
        u = (t - t0) / max(1, dur)
        if kind == 'sweep' and 0.0 <= u <= 1.0:
            cx = (-sx + u * (w + 2 * sx)) if direction > 0 else (w + sx - u * (w + 2 * sx))
            g = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - y * h) / sy) ** 2) / 2.0)
            amp = peak * math.sin(math.pi * u) ** 0.5           # 出現・退出時に滑らかに減衰
            img = img + amp * g[..., None] * col * keep
        elif kind == 'ramp' and u >= 0.0:
            s = min(1.0, u)
            s = s * s * (3 - 2 * s)                               # 滑らかな明度上昇
            grad = (xx / max(1, w - 1)) if direction < 0 else (1.0 - xx / max(1, w - 1))
            img = img + peak * s * (0.35 + 0.65 * grad)[..., None] * col * keep
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def limit_cv2_threads(n, _worker_id=None):
    """OpenCV 演算 (色変換・リサイズ) のスレッド数を n に制限。DataLoader 生成前に親プロセスで一度呼び出し、
    DataLoader の worker_init_fn にも渡す (functools.partial(limit_cv2_threads, n))。

    ⚠️ fork されたワーカー内で初回変更するとデッドロックする可能性がある (親が既に OpenCV スレッドを起動している場合)。
       そのため親プロセスで先行設定してワーカーに継承させ、ワーカー側では値が異なる場合のみ (spawn — Windows/macOS) 更新する。
    """
    if cv2.getNumThreads() != int(n):
        cv2.setNumThreads(int(n))


class HitAndRunDataset(Dataset):
    def __init__(
        self,
        data_dir=config.DATA_DIR,
        clip_length=config.CLIP_LENGTH,
        r_value=config.R_VALUE,
        resize=config.RESIZE,
        augment=True,
        cache_dir=None,
        span_windows=False,
    ):
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.r_value = r_value
        self.resize = resize
        # augment=True → 学習用 (ランダムデータ拡張適用), False → 検証用 (拡張なし、決定的)
        self.augment = augment
        # クリップキャッシュ保存フォルダ (検証専用, config.TRAIN_VAL_CACHE)。拡張時は毎回クリップが変化するため使用不可。
        if augment and cache_dir:
            raise ValueError('クリップキャッシュは拡張なし(検証)データセットでのみ使用可能です')
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_off = False
        self._lq_off = False         # 低画質拡張が失敗した場合 (PyAV 未導入等) は該当ワーカーで無効化して継続
        # 動画デコードスレッド数 (None = OpenCV デフォルト)。ワーカー側の OpenCV 演算スレッド数は train.py が制御。
        self.cv2_threads = getattr(config, 'TRAIN_CV2_THREADS', None)
        # A クリップ方式 (config.TRAIN_A_MODE)。'span' の場合はサンプルごとにウィンドウ群 (K, 3, T, H, W) を
        # 返し (非衝突は K=1)、DataLoader は span_collate でバッチ化する。
        a_mode = getattr(config, 'TRAIN_A_MODE', 'pre')
        if a_mode not in ('pre', 'span'):
            raise ValueError(f"TRAIN_A_MODE は 'pre' または 'span' である必要があります: {a_mode!r}")
        self.span = a_mode == 'span'
        # span_windows=True (検証専用): 'pre' 学習時であっても衝突サンプルを span と同様のウィンドウ群として生成する。
        # 初回ウィンドウが 'pre' のウィンドウと一致するため、従来指標は初窓で・区間再現率は全ウィンドウで測定 (train.validate)。
        # サンプル数・順序は維持される (衝突動画あたり1個)。A スライシング (TRAIN_A_SLICE_ENABLED) はサンプル数が変化するため併用不可。
        if span_windows and not self.span and getattr(config, 'TRAIN_A_SLICE_ENABLED', False):
            raise ValueError('span_windows は TRAIN_A_SLICE_ENABLED と併用できません (学習・検証のサンプル順序が変化するため)')
        self.span_windows = self.span or bool(span_windows)
        # 正規化統計量は事前学習 (Kinetics-400) と同様に config で一括管理
        self.mean = torch.tensor(
            config.NORM_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(
            config.NORM_STD, dtype=torch.float32).view(3, 1, 1, 1)

        # data_dir 配下のサブフォルダ (rcdata/realdata 等) から再帰的に mp4 を収集する。
        # data_dir は複数フォルダ (リスト) でも指定可能 — val_recheck が検証フォルダ + 学習フォルダを同時に読み込む場合等。
        dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
        self.mp4_paths = sorted(
            p for d in dirs for p in glob.glob(os.path.join(str(d), '**', '*.mp4'), recursive=True))
        self.samples = self._build_index()

    def __len__(self):
        return len(self.samples)

    def _build_index(self):
        samples = []
        for mp4_path in self.mp4_paths:
            # txt は同一パスの拡張子を変更したもの (rcdata/realdata 共通ルール)
            txt_path = os.path.splitext(mp4_path)[0] + ".txt"
            file_name = os.path.splitext(os.path.basename(mp4_path))[0]
            bboxes, action = self._parse_annotation(txt_path)

            if action is not None:
                class_str = action[0]
                target_id = int(float(action[1]))
                start_f = int(float(action[2]))
                # 衝突終了フレーム — A スライシングが区間外に出ないよう制御
                end_f = int(float(action[3])) if len(action) > 3 else start_f
            else:
                class_str = 'S'
                target_id = 0
                start_f = 0
                end_f = 0

            if target_id not in bboxes:
                target_id = next(iter(bboxes), 0)
            target_bbox = bboxes.get(
                target_id, [0, 0, self.resize[0], self.resize[1]])
            label = 1 if class_str == 'A' else 0

            # 動画メタデータは1回のみ取得 (S スライシングの総フレーム数 + ジッター換算用 fps)。
            total, fps = self._video_meta(mp4_path)

            def _add(sf, wins=None, car=None):
                car = target_id if car is None else car
                sample = {
                    'file_name': file_name,
                    'video_key': mp4_path,   # 分割を「動画単位」で束ねるためのキー
                    'mp4_path': mp4_path,
                    'label': label,
                    'start_f': sf,
                    'bbox': target_bbox if car == target_id else bboxes[car],
                    'car_id': car,           # クロップ基準車両番号
                    'fps': fps,              # 時間ジッターを「秒」基準で揃えるため
                    'total': total,          # fps 拡張が k倍で読み込み可能か確認用
                }
                if wins is not None:
                    sample['win_starts'] = wins  # span: 区間を網羅する各ウィンドウの開始フレーム
                samples.append(sample)

            # S クリップを抽出する車両: デフォルトは対象車両 (0番) のみ。実写動画 S は1番車両も抽出可能
            # (config.TRAIN_S_REAL_USE_CAR1)。RC S は0番のみ。
            s_cars = [target_id]
            if (label == 0 and getattr(config, 'TRAIN_S_REAL_USE_CAR1', False)
                    and 'realdata' in os.path.normpath(mp4_path).split(os.sep)
                    and 1 in bboxes and target_id != 1):
                s_cars.append(1)

            if label == 1:
                # 衝突 (A): ウィンドウを start_f より TRAIN_A_PRE_FRAMES 前から開始し
                # 「接近 → 衝突 → 揺れ」が1ウィンドウに収まるようにする (config 注記参照)。
                pre = int(getattr(config, 'TRAIN_A_PRE_FRAMES', 0))
                a_start = max(0, start_f - pre)
                a_stride = max(1, int(getattr(config, 'TRAIN_A_SLICE_STRIDE', 15)))
                last = end_f - self.clip_length + 1      # ウィンドウ末尾 == end_f となる開始位置
                if self.span_windows:
                    # span: 接近から衝突終了までをカバーするウィンドウ群を1サンプルとして保持
                    wins = self._span_starts(a_start, last, total)
                    _add(wins[0], wins)
                elif (getattr(config, 'TRAIN_A_SLICE_ENABLED', False)
                        and last > a_start):
                    # (デフォルト off) 区間内でウィンドウをスライドさせ A クリップを増やすパス。
                    # 初窓のみ接近を含み、以降は「衝突中」となる。
                    starts = list(range(a_start, last + 1, a_stride))
                    if starts[-1] != last:
                        starts.append(last)   # 最終ウィンドウは衝突終了位置に整列
                    for sf in starts:
                        _add(sf)
                else:
                    _add(a_start)
            elif not getattr(config, 'TRAIN_S_SLICE_ENABLED', False):
                for car in s_cars:
                    _add(start_f, car=car)
            else:
                # 非衝突 (S) は動画全体をクリップ長単位で分割して学習に投入する。
                # (以前は先頭30フレームのみであり「車が通過するが衝突ではない」ケースを
                #  学習できず誤検知の直接原因となっていた — config 注記参照)
                stride = max(1, int(getattr(config, 'TRAIN_S_SLICE_STRIDE',
                                            self.clip_length)))
                cap_n = int(getattr(config, 'TRAIN_S_MAX_CLIPS_PER_VIDEO', 0))
                starts = list(range(0, max(1, total - self.clip_length + 1), stride))
                if not starts:
                    starts = [0]
                if cap_n > 0:
                    starts = starts[:cap_n]
                for car in s_cars:
                    for sf in starts:
                        _add(sf, car=car)
        return samples

    def _span_starts(self, a_start, last, total):
        """span ウィンドウ開始点群 — 初窓は a_start (接近)、最終窓は末尾が衝突終了位置。

        last : 末尾が衝突終了に合致するウィンドウの開始位置。動画末尾を超えないようクリッピング。
        区間が1ウィンドウに収まる場合 (last <= a_start) は 1ウィンドウ = 'pre' 方式と等価。
        それ以外は TRAIN_SPAN_WINDOW_STRIDE 以下の間隔で区間を均等分割し、
        ウィンドウ数が TRAIN_SPAN_MAX_WINDOWS を超える場合は間隔を広げてその個数でカバーする。
        """
        if total > 0:
            last = min(last, total - self.clip_length)
        if last <= a_start:
            return [a_start]
        stride = max(1, int(getattr(config, 'TRAIN_SPAN_WINDOW_STRIDE', 15)))
        max_n = max(2, int(getattr(config, 'TRAIN_SPAN_MAX_WINDOWS', 6)))
        d = last - a_start
        n = min(max_n, 1 + math.ceil(d / stride))
        return [a_start + round(i * d / (n - 1)) for i in range(n)]

    @staticmethod
    def _video_meta(mp4_path):
        """(総フレーム数, fps)。メタデータのみを読み込むため高速。取得失敗時は fps=30 とみなす。"""
        cap = cv2.VideoCapture(mp4_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return n, (fps if fps > 1.0 else 30.0)

    def _parse_annotation(self, txt_path):
        bboxes = {}
        action = None
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if not parts or len(parts) < 2:
                        continue
                    if parts[0] == 'car' and len(parts) >= 6:
                        bboxes[int(parts[1])] = [int(parts[2]), int(
                            parts[3]), int(parts[4]), int(parts[5])]
                    elif parts[0] in ['A', 'S']:
                        action = parts
        return bboxes, action

    def _crop_and_pad(self, frame, bbox, r):
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = bbox

        veh_w, veh_h = (x_max - x_min) * r, (y_max - y_min) * r
        cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2

        square_size = int(max(veh_w, veh_h))
        new_x_min, new_y_min = cx - square_size // 2, cy - square_size // 2
        new_x_max, new_y_max = cx + square_size // 2, cy + square_size // 2

        pad_left = max(0, -new_x_min)
        pad_top = max(0, -new_y_min)
        pad_right = max(0, new_x_max - w)
        pad_bottom = max(0, new_y_max - h)

        valid_x_min = max(0, new_x_min)
        valid_y_min = max(0, new_y_min)
        valid_x_max = min(w, new_x_max)
        valid_y_max = min(h, new_y_max)
        cropped_frame = frame[valid_y_min:valid_y_max, valid_x_min:valid_x_max]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped_frame = np.pad(
                cropped_frame,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0,
            )

        return cv2.resize(cropped_frame, self.resize, interpolation=cv2.INTER_LINEAR)

    def _apply_augmentation(self, frames):
        # 実際の駐車場防犯カメラ環境に対応したデータ拡張 — 強度・確率の根拠は config の注記参照
        # 変換パラメータはクリップごとに1回サンプリングし全フレームに同一適用 (時間的一貫性を保持)
        # ノイズもクリップ全体で同一パターン (フレームごとに変えるとチラつきが発生し微小な揺れが隠蔽されるため)

        do_hflip = random.random() > 0.5

        # 明度・コントラスト・彩度・色相: 照明変化、天候、カメラ特性の差異をシミュレーション
        do_jitter = random.random() > 0.5
        brightness = random.uniform(*getattr(config, 'TRAIN_AUG_BRIGHTNESS', (0.4, 1.4)))   # 夜間・地下駐車場 〜 明るい屋外
        contrast = random.uniform(0.7, 1.3)   # 低コントラスト (曇天) 〜 高コントラスト (直射日光)
        saturation = random.uniform(0.7, 1.3)   # カメラごとの彩度特性の差
        hue = random.uniform(-0.1, 0.1)  # ホワイトバランス・色温度の差異

        # 夜間赤外線 (IR) カメラ: 実質モノクロ映像 → 色情報なしでも動作判定できるよう学習
        do_gray = random.random() < getattr(config, 'TRAIN_AUG_GRAY_PROB', 0.2)
        # 低画質カメラ: ピンボケ・圧縮劣化 (ブラー)、夜間ゲインによる固定パターンノイズをシミュレーション
        do_blur = random.random() < 0.2
        do_noise = random.random() < getattr(config, 'TRAIN_AUG_NOISE_PROB', 0.2)
        noise_sigma = random.uniform(*getattr(config, 'TRAIN_AUG_NOISE_SIGMA', (2.0, 8.0)))
        grain = None
        if do_noise:
            # クリップ全体で同一のノイズパターン — モノクロ時は明度ノイズのみ
            h, w = frames[0].shape[:2]
            grain = np.random.normal(0.0, noise_sigma, (h, w, 1 if do_gray else 3)).astype(np.float32)

        aug_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            if do_hflip:
                img = TF.hflip(img)
            if do_jitter:
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
                img = TF.adjust_hue(img, hue)
            if do_gray:
                img = TF.rgb_to_grayscale(img, num_output_channels=3)
            if do_blur:
                img = TF.gaussian_blur(img, kernel_size=3)
            arr = np.array(img)
            if grain is not None:
                arr = np.clip(arr.astype(np.float32) + grain, 0, 255).astype(np.uint8)
            aug_frames.append(arr)

        return aug_frames

    def _apply_light(self, frames):
        """照明拡張 (config.TRAIN_LIGHT_AUG_PROB, 学習専用)。事故・正常を問わず同確率で適用し、クリップごとに
        スイープ・照射の種別および強度・サイズ・方向・色・タイミングをサンプリング。確率 0 の場合は乱数を消費しない。"""
        p = float(getattr(config, 'TRAIN_LIGHT_AUG_PROB', 0.0) or 0.0)
        if p <= 0 or random.random() >= p:
            return frames
        n = len(frames)
        dur = random.randint(6, 30)
        dark = random.uniform(0.25, 0.6) if random.random() < getattr(config, 'TRAIN_LIGHT_DARK_PROB', 0.7) else 1.0
        return add_light(frames, 'sweep' if random.random() < 0.6 else 'ramp', dark=dark,
                         peak=random.uniform(60, 200), radius=random.uniform(0.2, 0.6),
                         t0=random.randint(-dur // 2, n - dur // 2), dur=dur, direction=random.choice((1, -1)),
                         color=random.choices(('white', 'warm', 'cool', 'red'), weights=(4, 3, 2, 1))[0],
                         y=random.uniform(0.3, 0.8))

    def _apply_low_quality(self, frames, crop_side):
        """低画質拡張 (config.TRAIN_LQ_AUG_PROB, 学習専用)。高品質クリップのみ H.264 で圧縮・展開し、
        TRAIN_LQ_SIZE_PROB の比率で事前に縮小・再拡大を実施。パラメータはクリップ単位で一定。"""
        p = float(getattr(config, 'TRAIN_LQ_AUG_PROB', 0.0) or 0.0)
        if (p <= 0 or self._lq_off or crop_side < getattr(config, 'TRAIN_LQ_MIN_CROP', 128)
                or random.random() >= p):
            return frames
        h, w = frames[0].shape[:2]
        size = None
        if random.random() < getattr(config, 'TRAIN_LQ_SIZE_PROB', 0.5):
            lo, hi = getattr(config, 'TRAIN_LQ_SIZE_RANGE', (48, 128))
            hi = min(hi, int(crop_side * 0.75))      # 元サイズより確実に縮小
            if hi >= lo:
                size = random.randint(lo // 2, hi // 2) * 2
        crf = random.randint(*getattr(config, 'TRAIN_LQ_CRF_RANGE', (28, 42)))
        gop = random.choice((10, 15, 30, 60))         # 防犯カメラのキーフレーム間隔 (0.3〜2秒)
        src = [cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames] if size else frames
        try:
            out = h264_roundtrip(src, crf, gop)
        except Exception as e:                        # PyAV・libx264 未導入等 — 該当ワーカーは拡張なしで継続
            self._lq_off = True
            print(f'⚠️ 低画質拡張を使用できないため無効化して継続します: {type(e).__name__}: {e}')
            return frames
        if size:
            out = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in out]
        return out

    def _read_frames(self, mp4_path, s0, n_read, bbox, r):
        """s0 から n_read フレームを読み込み、クロップおよびリサイズした uint8 RGB フレームリストを返す。動画が短い場合は最終フレームでパディング。"""
        cap = _open_video(mp4_path, self.cv2_threads)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0)

        for _ in range(n_read):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self._crop_and_pad(frame, bbox, r))
        cap.release()

        while len(frames) < n_read:
            frames.append(
                frames[-1] if frames else np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8))
        return frames

    # ── クリップキャッシュ (検証専用, config.TRAIN_VAL_CACHE) ──
    # 検証は拡張を行わずエポック毎に同一クリップを評価するため、クロップ・リサイズ済みの uint8 配列を保存して再利用する。
    # キー = クリップ生成ロジック (_read_frames・_crop_and_pad)・OpenCV バージョン + 動画ファイル (パス・サイズ・更新日時)
    #      + 開始フレーム・フレーム長・ボックス・r・サイズ — いずれかが変更されると再生成される。
    _cache_code = None

    @classmethod
    def _cache_code_key(cls):
        if cls._cache_code is None:
            src = inspect.getsource(cls._read_frames) + inspect.getsource(cls._crop_and_pad)
            cls._cache_code = hashlib.sha1(f'{src}|{cv2.__version__}'.encode()).hexdigest()
        return cls._cache_code

    def _cache_path(self, mp4_path, s0, n_read, bbox, r):
        if self.cache_dir is None or self._cache_off:
            return None
        try:
            st = os.stat(mp4_path)
        except OSError:
            return None
        key = repr((self._cache_code_key(), os.path.abspath(mp4_path), st.st_size, st.st_mtime_ns,
                    int(s0), int(n_read), tuple(int(v) for v in bbox), float(r), tuple(self.resize)))
        return self.cache_dir / f'{hashlib.sha1(key.encode()).hexdigest()}.npy'

    def _cache_load(self, path, n_read):
        if path is None or not path.exists():
            return None
        try:
            arr = np.load(path)
        except (OSError, ValueError):
            return None                       # 読み込み不能なファイル — 再生成して上書き
        if arr.dtype != np.uint8 or arr.shape != (n_read, self.resize[1], self.resize[0], 3):
            return None
        return arr

    def _cache_save(self, path, frames):
        if path is None:
            return
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, 'wb') as f:
                np.save(f, np.stack(frames, axis=0))
            os.replace(tmp, path)   # 書き込み完了後にアトミック置換し、他ワーカーの不完全読み込みを防止
        except OSError as e:
            # ディスク満杯等 — 該当ワーカーはキャッシュ保存を中断して処理継続 (検証結果自体は同一)
            self._cache_off = True
            print(f'⚠️ 検証クリップキャッシュの保存に失敗したため保存なしで継続します: {e}')
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def cache_status(self, indices):
        """キャッシュ状況 (キャッシュ済みクリップ数, 総クリップ数, 全生成時の総容量 GB) — 学習開始時の表示用。"""
        have, size = 0, 0
        for i in indices:
            s = self.samples[i]
            starts = list(s.get('win_starts') or [s['start_f']])
            s0, n_read = min(starts), max(starts) - min(starts) + self.clip_length
            p = self._cache_path(s['mp4_path'], s0, n_read, s['bbox'], self.r_value)
            have += bool(p is not None and p.exists())
            size += n_read * self.resize[0] * self.resize[1] * 3
        return have, len(indices), size / 2 ** 30

    def _fps_aug_stride(self, sample, starts):
        """fps 拡張 (config.TRAIN_FPS_AUG_PROB, 学習専用) で間引くフレーム間隔。未適用の場合は 1。
        確率 0 の場合は乱数を消費せず従来の動作を保持。"""
        p = float(getattr(config, 'TRAIN_FPS_AUG_PROB', 0.0) or 0.0)
        if p <= 0 or len(starts) != 1 or random.random() >= p:
            return 1
        fps = float(sample.get('fps', 30.0))
        ks = [k for k in getattr(config, 'TRAIN_FPS_AUG_STRIDES', (2, 3))
              if k > 1 and fps / k >= getattr(config, 'TRAIN_FPS_AUG_MIN_FPS', 8.0)]
        if not ks:
            return 1
        k = random.choice(ks)
        total = int(sample.get('total', 0) or 0)
        if total and total < (self.clip_length - 1) * k + 1:
            return 1                           # 動画が短く k倍で読めない場合は通常読み込み
        return k

    def _read_strided(self, sample, start, k, bbox, r):
        """start 周辺から (clip_length − 1)·k + 1 フレームを読み込み、kフレームごとに1枚抽出 → clip_length 枚。
        事故クリップは衝突開始が通常通り TRAIN_A_PRE_FRAMES 番目に位置するよう前方を k倍に拡張。"""
        n = (self.clip_length - 1) * k + 1
        s0 = start
        if sample['label'] == 1:
            s0 = start - int(getattr(config, 'TRAIN_A_PRE_FRAMES', 0)) * (k - 1)
        total = int(sample.get('total', 0) or 0)
        if total:
            s0 = min(s0, total - n)            # 動画末尾を超えないようクリッピング
        s0 = max(0, s0)
        return self._read_frames(sample['mp4_path'], s0, n, bbox, r)[::k]

    def _frames_to_tensor(self, frames):
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (video_tensor - self.mean) / self.std

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # ウィンドウ開始点群。'pre' モードおよび非衝突サンプルは1ウィンドウ。
        starts = list(sample.get('win_starts') or [sample['start_f']])
        bbox = sample['bbox']

        if self.augment:
            # ① 時間オフセットジッター: 学習クリップが常に「衝突開始 = ウィンドウ先頭」であると
            #    推論 (スライディングウィンドウ) 時に衝突がウィンドウ中央に来た際に入力分布が乖離する。
            #    開始位置を前後に微小シフトさせ衝突位置をウィンドウ内で多様化。
            jitter = int(round(sample.get('fps', 30.0)
                               * getattr(config, 'TRAIN_TIME_JITTER_SEC', 0.33)))
            if jitter > 0:
                # span ウィンドウ群は間隔を保ったまま全体をシフト (先頭ウィンドウが 0 未満にならないよう制御)
                d = max(random.randint(-jitter, jitter), -min(starts))
                starts = [s + d for s in starts]
            # ② bbox ジッター: 本番サービスではユーザーがバウンディングボックスを手動指定する。
            #    GT 座標のみで学習すると手動矩形との分布差が生じるため、
            #    中心移動 (±5%)・サイズ倍率 (0.9〜1.15) で不正確なボックスをシミュレーション。
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            cx = (x1 + x2) // 2 + int(bw * random.uniform(-0.05, 0.05))
            cy = (y1 + y2) // 2 + int(bh * random.uniform(-0.05, 0.05))
            s = random.uniform(0.9, 1.15)
            hw, hh = int(bw * s / 2), int(bh * s / 2)
            bbox = [cx - hw, cy - hh, cx + hw, cy + hh]

        # クロップ倍率は学習・検証・サービス共通で self.r_value (=config.R_VALUE, 1.0)
        r = self.r_value

        # 初回ウィンドウ開始から最終ウィンドウ末尾までを一括読み込み (1ウィンドウ時は clip_length フレーム)。
        # 全ウィンドウが同一のボックス・r・拡張パラメータを共有する。
        s0 = min(starts)
        k = self._fps_aug_stride(sample, starts) if self.augment else 1
        if k > 1:
            frames = self._read_strided(sample, s0, k, bbox, r)
        else:
            n_read = max(starts) - s0 + self.clip_length
            cache = self._cache_path(sample['mp4_path'], s0, n_read, bbox, r)
            frames = self._cache_load(cache, n_read)
            if frames is None:
                frames = self._read_frames(sample['mp4_path'], s0, n_read, bbox, r)
                self._cache_save(cache, frames)

        # データ拡張は学習セットにのみ適用 (検証セットは再現性のため拡張を行わない)
        if self.augment:
            frames = self._apply_light(frames)
            frames = self._apply_augmentation(frames)
            frames = self._apply_low_quality(frames, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * r)

        video = self._frames_to_tensor(frames)
        label = torch.tensor(sample['label'], dtype=torch.long)
        if not self.span_windows:
            return video, label
        # (K, 3, T, H, W) — 区間をカバーする各ウィンドウ
        return torch.stack([video[:, s - s0:s - s0 + self.clip_length]
                            for s in starts]), label


def span_collate(batch):
    """span バッチ生成: サンプルごとにウィンドウ数が異なるためウィンドウを連結し、各サンプルのウィンドウ数を付与して返却。

    戻り値: (ウィンドウ群 (ΣK, 3, T, H, W), サンプルごとのウィンドウ数 (B,), ラベル (B,))
    """
    clips = torch.cat([b[0] for b in batch])
    sizes = torch.tensor([b[0].shape[0] for b in batch], dtype=torch.long)
    labels = torch.stack([b[1] for b in batch])
    return clips, sizes, labels
