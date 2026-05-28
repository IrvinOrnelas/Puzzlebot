import cv2
import numpy as np

from yolo import get_signs

# IDs from _cls_to_sign_map in yolo.py:
#   1=left, 2=right, 3=forward, 4=stop, 5=yield, 6=roadwork
SIGN_NAMES = {
    1: "Turn Left",
    2: "Turn Right",
    3: "Go Straight",
    4: "Stop",
    6: "Workers",
}
RELEVANT_SIGNS = set(SIGN_NAMES.keys())


class TrafficSignDetection:
    """
    Pipeline with two detection stages:
      1. Image quality check  — Laplacian-variance blur metric.
         Blurry frames are skipped to avoid unreliable detections.
      2. CNN detection        — YOLOv10 (best.pt) identifies traffic signs.

    Usage
    -----
    detector = TrafficSignDetection(blur_threshold=100.0)

    # Process one frame (numpy BGR array):
    annotated, front_sign = detector.process(frame)

    # front_sign is one of: "Stop", "Workers", "Go Straight",
    #                        "Turn Left", "Turn Right", or None.
    """

    def __init__(self, blur_threshold: float = 100.0):
        self.blur_threshold = blur_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, str | None]:
        """Run the full pipeline on a single BGR frame.

        Returns
        -------
        annotated : np.ndarray
            Copy of the frame with bounding boxes and overlay text drawn.
        front_sign : str or None
            Name of the traffic sign most likely in front of the robot,
            or None if the frame is blurry / no relevant sign detected.
        """
        annotated = frame.copy()

        # --- Stage 1: blur metric ---
        blurry, blur_score = self._is_blurry(frame)

        front_sign = None

        if blurry:
            cv2.putText(
                annotated,
                f"BLUR: {blur_score:.1f} (skipping detection)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            # --- Stage 2: YOLO (CNN) ---
            boxes, sign_types, confidences, _ = get_signs(annotated, drawing_frame=annotated)

            cv2.putText(
                annotated,
                f"BLUR: {blur_score:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            front_sign = self._get_front_sign(boxes, sign_types, confidences)

        # --- Overlay: sign in front ---
        self._draw_front_sign(annotated, front_sign)

        return annotated, front_sign

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_blurry(self, frame: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return score < self.blur_threshold, score

    @staticmethod
    def _box_area(box) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _get_front_sign(boxes, sign_types, confidences) -> str | None:
        best_name = None
        best_area = -1
        for box, stype, conf in zip(boxes, sign_types, confidences):
            if int(stype) not in RELEVANT_SIGNS:
                continue
            area = TrafficSignDetection._box_area(box)
            if area > best_area:
                best_area = area
                best_name = SIGN_NAMES[int(stype)]
        return best_name

    @staticmethod
    def _draw_front_sign(frame: np.ndarray, front_sign: str | None) -> None:
        h, w = frame.shape[:2]
        label = f"IN FRONT: {front_sign}" if front_sign else "IN FRONT: None"
        color = (0, 255, 255) if front_sign else (180, 180, 180)
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
