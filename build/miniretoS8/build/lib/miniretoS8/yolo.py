import os
import cv2
import numpy as np
from ultralytics import YOLO

def process_frame(frame, drawing_frame=None):
    """
    Run YOLOv8 inference on a single frame.

    Carga preferentemente el modelo TensorRT (best.engine) para máximo
    rendimiento en Jetson Nano. Si no existe, usa best.pt como fallback.

    Args:
        frame (np.ndarray): The input image (BGR) on which to run inference.
        drawing_frame (np.ndarray or None): If provided, drawing_frame is where
            bounding boxes and labels will be drawn. Typically the same as `frame`.
            If None, no drawing occurs.

    Returns:
        boxes (np.ndarray of shape (N, 4)): Array of [x1, y1, x2, y2] per detection.
        class_ids (np.ndarray of shape (N,)): Raw class ID integers.
        confidences (np.ndarray of shape (N,)): Confidence scores (0.0 – 1.0).
        class_names (list of str or None): List of class name strings, or None
            for any index where no mapping is available.
    """
    # 1. Run inference (verbose=False suppresses Ultralytics console output)
    results = _model(frame, verbose=False)
    r = results[0]

    # 2. If no detections at all, return empty arrays and list
    if not hasattr(r, "boxes") or len(r.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=int),  # boxes
            np.zeros((0,), dtype=int),    # class_ids
            np.zeros((0,), dtype=float),  # confidences
            []                             # class_names
        )

    # 3. Extract boxes, class IDs, confidences
    #    r.boxes.xyxy: Tensor shape (N, 4)
    #    r.boxes.cls:  Tensor shape (N,)
    #    r.boxes.conf: Tensor shape (N,)
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    confidences = r.boxes.conf.cpu().numpy()

    # 4. Build a parallel list of class names (or None)
    class_names = []
    for cid in class_ids:
        if _names is not None:
            # _names might be a dict or a list
            if isinstance(_names, dict):
                name = _names.get(int(cid), None)
            else:
                name = _names[int(cid)] if 0 <= int(cid) < len(_names) else None
        else:
            name = None
        class_names.append(name)

    # 5. Draw boxes & labels on drawing_frame if provided
    if drawing_frame is not None:
        draw_detections(boxes, class_ids, confidences, class_names, drawing_frame)

    return boxes, class_ids, confidences, class_names

def draw_detections(boxes, class_ids, confidences, class_names, drawing_frame):
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cid = class_ids[idx]
        conf = confidences[idx]
        cname = class_names[idx]

        # Fallback to numeric ID if name is unavailable
        label = f"{cname} ({cid})" if cname is not None else f"ID {cid}"
        text = f"{label}: {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(drawing_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        # Put label
        y_text = y1 - 10 if y1 - 10 > 10 else y1 + 15
        cv2.putText(
            drawing_frame,
            text,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

def get_signs(frame, drawing_frame=None):
    # Get the raw detections from the YOLOv10 model
    boxes, class_ids, confidences, class_names = process_frame(frame)

    # Zip
    detections = list(zip(boxes, class_ids, confidences, class_names))

    # Filter out detections that are not traffic signs
    traffic_signs = [d for d in detections if d[1] in _cls_to_sign_map]

    # Map class IDs to traffic sign IDs
    traffic_signs = [(box, _cls_to_sign_map[cid], conf, cname) for box, cid, conf, cname in traffic_signs]

    # Unzip
    boxes, sign_types, confidences, class_names = zip(*traffic_signs) if traffic_signs else ([], [], [], [])
    boxes = np.array(boxes, dtype=int)
    sign_types = np.array(sign_types, dtype=int)
    confidences = np.array(confidences, dtype=float)
    class_names = list(class_names)  # Convert to list for consistency

    # Draw the traffic signs on the drawing_frame if provided
    if drawing_frame is not None:
        draw_detections(boxes, sign_types, confidences, class_names, drawing_frame)

    return boxes, sign_types, confidences, class_names

# ---------------------------------------------------------------------------
# Globals — modelo YOLOv8 con TensorRT (best.engine) o PyTorch (best.pt)
# ---------------------------------------------------------------------------
_S7_DIR      = '/home/puzzlebot/ros2_ws/src/miniretoS7/miniretoS7'
_ENGINE_PATH = os.path.join(_S7_DIR, 'best.engine')
_PT_PATH     = os.path.join(_S7_DIR, 'best.pt')

if os.path.exists(_ENGINE_PATH):
    MODEL_PATH = _ENGINE_PATH
    print(f"[yolo] Cargando modelo TensorRT: {MODEL_PATH}")
else:
    MODEL_PATH = _PT_PATH
    print(f"[yolo] best.engine no encontrado — usando PyTorch: {MODEL_PATH}")

_model = YOLO(MODEL_PATH, task="detect")
_names = getattr(_model, "names", None)
_cls_to_sign_map = {
    7: 1, # left
    2: 2, # right
    0: 3, # forward
    5: 4, # stop
    1: 5, # yield
    6: 6, # roadwork
}
