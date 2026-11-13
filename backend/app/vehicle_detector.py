"""リアルタイム車両検出モジュール (Webサービス /detect-vehicles で使用)。

YOLO (推奨: segmentation モデル) でフレーム内の車両を検出し、segmentation mask の
実際の輪郭から車両エッジにフィットしたバウンディングボックスを計算する。detection モデル使用時は
マスクが存在しないため YOLO detection bbox にフォールバックする。

- エントリーポイント: get_detector(...) で検出器をプロセスごとに1回ロード・再利用。
- 戻り値: VehicleDetection(class_name, confidence, bbox=(x1,y1,x2,y2)) のリスト。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency 案内用
    raise SystemExit(
        "ultralyticsがインストールされていません。 "
        "次のコマンドでインストールしてください: pip install ultralytics opencv-python"
    ) from exc


DEFAULT_VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")


@dataclass
class VehicleDetection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


class VehicleBBoxDetector:
    """YOLO-seg mask 輪郭から車両 bbox をタイトに再計算する。"""

    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        conf: float = 0.35,
        vehicle_classes: Iterable[str] = DEFAULT_VEHICLE_CLASSES,
        mask_threshold: float = 0.6,
        min_area: int = 100,
        nms_iou: float = 0.35,
        contain_threshold: float = 0.85,
        imgsz: int = 640,
        enhance_night: bool = False,
        dynamic_imgsz: bool = False,
        imgsz_min: int = 640,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        # imgsz の意味:
        #   dynamic_imgsz=False → 常にこの値で推論 (固定)。
        #   dynamic_imgsz=True  → フレーム長辺に合わせた動的 imgsz の「上限 (cap)」。
        #     eff = clamp(round32(long_side), imgsz_min, imgsz)
        #     → 高解像度は過度なダウンスケール(情報損失)を防ぎ、低解像度は無駄なアップスケールを防ぐ。
        self.imgsz = imgsz              # 推論入力解像度 (動的モードでは上限)
        self.dynamic_imgsz = dynamic_imgsz
        self.imgsz_min = imgsz_min
        self.enhance_night = enhance_night  # 暗いフレームに対する CLAHE/ガンマ補正前処理の有無
        self.vehicle_classes = set(vehicle_classes)
        self.mask_threshold = mask_threshold  # マスク二値化閾値 (↑ = bboxがよりタイトに)
        self.min_area = min_area
        self.nms_iou = nms_iou
        self.contain_threshold = contain_threshold

    @staticmethod
    def _clip_box(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width - 1))
        y2 = max(0, min(int(round(y2)), height - 1))
        return x1, y1, x2, y2

    @staticmethod
    def _box_area(bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @classmethod
    def _intersection_area(
        cls,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> int:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        return cls._box_area((x1, y1, x2, y2))

    @classmethod
    def _iou(
        cls,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        intersection = cls._intersection_area(first, second)
        union = cls._box_area(first) + cls._box_area(second) - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _contained_ratio(
        cls,
        inner: tuple[int, int, int, int],
        outer: tuple[int, int, int, int],
    ) -> float:
        inner_area = cls._box_area(inner)
        if inner_area <= 0:
            return 0.0
        return cls._intersection_area(inner, outer) / inner_area

    def _deduplicate_detections(
        self,
        detections: list[VehicleDetection],
    ) -> list[VehicleDetection]:
        """同一車両に複数の bbox が生成された場合、信頼度 (confidence) の高い bbox のみを残す。"""
        kept: list[VehicleDetection] = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for kept_det in kept:
                iou = self._iou(det.bbox, kept_det.bbox)
                contain = self._contained_ratio(det.bbox, kept_det.bbox)
                if iou >= self.nms_iou or contain >= self.contain_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return sorted(kept, key=lambda det: (det.bbox[0], det.bbox[1]))

    def _bbox_from_mask(
        self,
        mask: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        """Segmentation mask の実際の正ピクセル外郭から bbox を計算する。"""
        binary = (mask > self.mask_threshold).astype(np.uint8)
        if binary.shape[:2] != (frame_height, frame_width):
            binary = cv2.resize(
                binary,
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )

        # 小さなノイズ穴や途切れをクローズ処理で補正
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        return self._clip_box(x, y, x + w - 1, y + h - 1, frame_width, frame_height)

    @staticmethod
    def _enhance_low_light(frame_bgr: np.ndarray, brightness_thresh: int = 80) -> np.ndarray:
        """暗い (夜間) フレームの場合、CLAHE (コントラスト強調) + ガンマ補正で車両視認性を高める。

        明るい昼間フレームはそのまま返す (過補正防止)。輝度平均が閾値未満の場合のみ適用。
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) >= brightness_thresh:
            return frame_bgr
        # CLAHE は LAB の L (輝度) チャネルのみに適用 → 色の歪みなくコントラストのみ向上
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        # ガンマ補正 (<1) → 暗部を明るく持ち上げる
        gamma = 0.7
        table = np.array([((i / 255.0) ** gamma) * 255
                          for i in range(256)], dtype=np.uint8)
        return cv2.LUT(enhanced, table)

    def _effective_imgsz(self, width: int, height: int) -> int:
        """動的モードの場合、フレーム長辺を32の倍数にスナップし [imgsz_min, imgsz] に収める。"""
        if not self.dynamic_imgsz:
            return self.imgsz
        long_side = max(width, height)
        snapped = int(round(long_side / 32.0)) * 32
        return max(self.imgsz_min, min(snapped, self.imgsz))

    def detect_frame(self, frame_bgr: np.ndarray) -> list[VehicleDetection]:
        if self.enhance_night:
            frame_bgr = self._enhance_low_light(frame_bgr)
        height, width = frame_bgr.shape[:2]
        eff_imgsz = self._effective_imgsz(width, height)
        result = self.model(frame_bgr, conf=self.conf,
                             imgsz=eff_imgsz, verbose=False)[0]

        if result.boxes is None:
            return []

        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        detections: list[VehicleDetection] = []

        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            if class_name not in self.vehicle_classes:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            tight_bbox = None
            if masks is not None and idx < len(masks):
                tight_bbox = self._bbox_from_mask(masks[idx], width, height)

            if tight_bbox is None:
                tight_bbox = self._clip_box(x1, y1, x2, y2, width, height)

            bx1, by1, bx2, by2 = tight_bbox
            if bx2 <= bx1 or by2 <= by1:
                continue

            detections.append(
                VehicleDetection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=tight_bbox,
                )
            )

        detections = self._deduplicate_detections(detections)
        return sorted(detections, key=lambda det: (det.bbox[0], det.bbox[1]))


def read_source_frame(source: Path, frame_index: int = 0) -> np.ndarray:
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(source))
        if frame is None:
            raise FileNotFoundError(f"画像を読み込めません: {source}")
        return frame

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を読み込めません: {source}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        frame_index = max(0, min(frame_index, frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"{frame_index}番目のフレームを読み込めませんでした: {source}")
    return frame


# プロセスごとに検出器 (YOLOモデル) を1回のみロードして再利用 — リクエストごとの再ロードオーバーヘッドを排除。
_detector_cache: dict = {}


def get_detector(
    model_path: str = "yolo11x.pt",
    conf: float = 0.15,
    imgsz: int = 1536,
    enhance_night: bool = True,
    mask_threshold: float = 0.6,
    dynamic_imgsz: bool = True,
    imgsz_min: int = 640,
) -> "VehicleBBoxDetector":
    """設定ごとに VehicleBBoxDetector をキャッシュして返す (同一設定なら再利用)。

    YOLO の重みロード (約数秒) をリクエストごとに繰り返さないようプロセス全体で保持。
    デフォルトは動的 imgsz (上限 1536): フレーム解像度に合わせて imgsz を決定し、
    高解像度の情報損失と低解像度の無駄なアップスケールを同時に削減する。
    """
    key = (model_path, conf, imgsz, enhance_night, mask_threshold,
           dynamic_imgsz, imgsz_min)
    detector = _detector_cache.get(key)
    if detector is None:
        detector = VehicleBBoxDetector(
            model_path=model_path, conf=conf, imgsz=imgsz,
            enhance_night=enhance_night, mask_threshold=mask_threshold,
            dynamic_imgsz=dynamic_imgsz, imgsz_min=imgsz_min)
        _detector_cache[key] = detector
    return detector
