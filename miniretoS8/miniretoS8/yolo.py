import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# YOLO ultra-far wrapper para miniretoS8
# Mantiene API compatible:
#   configure(...)
#   process_frame(frame, drawing_frame=None)
#   get_signs(frame, drawing_frame=None)
# ---------------------------------------------------------------------------

# IDs que usa el resto del sistema:
#   0=nada, 1=izq, 2=der, 3=adelante, 4=stop,
#   5=yield/ceda, 6=roadwork, 7=roundabout
_CLS_TO_SIGN_MAP: Dict[int, int] = {
    7: 1,  # left
    2: 2,  # right
    0: 3,  # forward
    5: 4,  # stop
    1: 5,  # yield / give-way
    6: 6,  # roadwork
    3: 7,  # roundabout
}

_MODEL: Optional[YOLO] = None
_NAMES = None
_MODEL_PATH: Optional[str] = None

_CONF = 0.05  # Más bajo para detectar señales lejanas
_IMGSZ = 1280  # Imagen más grande
_IOU = 0.45
_MAX_DET = 60
_AUGMENT = False

# Modo lejano: además del ROI completo, corre inferencia en recortes ampliados.
_FAR_MODE = True
_UPSCALE = 3.0  # Ampliación más agresiva
_TILE_MODE = 'aggressive'  # off | light | aggressive - más agresivo
_MIN_PATCH_W = 80
_MIN_PATCH_H = 80


def _candidate_model_paths() -> List[Path]:
    paths: List[Path] = []
    env_path = os.environ.get('YOLO_MODEL_PATH', '').strip()
    if env_path:
        paths.append(Path(env_path).expanduser())

    for base in (
        Path('/home/puzzlebot/ros2_ws/src/miniretoS8/miniretoS8'),
        Path('/home/puzzlebot/ros2_ws/src/miniretoS8'),
        Path('/home/puzzlebot/ros2_ws/src/miniretoS7/miniretoS7'),
        Path('/home/puzzlebot/ros2_ws/src/miniretoS7'),
    ):
        paths.append(base / 'best.engine')
        paths.append(base / 'best.pt')

    unique: List[Path] = []
    seen = set()
    for p in paths:
        ps = str(p)
        if ps not in seen:
            seen.add(ps)
            unique.append(p)
    return unique


def _find_model_path() -> str:
    for p in _candidate_model_paths():
        if p.exists():
            return str(p)
    candidates = '\n  - '.join(str(p) for p in _candidate_model_paths())
    raise FileNotFoundError(
        '[yolo] No encontré best.engine ni best.pt. Probé:\n  - ' + candidates +
        '\nTambién puedes usar: export YOLO_MODEL_PATH=/ruta/a/best.engine'
    )


def configure(
    model_path: Optional[str] = None,
    conf: float = 0.10,
    imgsz: int = 960,
    iou: float = 0.45,
    max_det: int = 60,
    far_mode: bool = True,
    upscale: float = 2.0,
    tile_mode: str = 'light',
    augment: bool = False,
) -> None:
    """Configura inferencia antes del primer frame."""
    global _MODEL, _NAMES, _MODEL_PATH
    global _CONF, _IMGSZ, _IOU, _MAX_DET, _FAR_MODE, _UPSCALE, _TILE_MODE, _AUGMENT

    _CONF = float(conf)
    _IMGSZ = int(imgsz)
    _IOU = float(iou)
    _MAX_DET = int(max_det)
    _FAR_MODE = bool(far_mode)
    _UPSCALE = max(1.0, float(upscale))
    _TILE_MODE = str(tile_mode).strip().lower()
    _AUGMENT = bool(augment)

    selected = str(Path(model_path).expanduser()) if model_path else _find_model_path()
    if _MODEL is not None and _MODEL_PATH == selected:
        return

    print(f'[yolo] Cargando modelo: {selected}')
    _MODEL = YOLO(selected, task='detect')
    _NAMES = getattr(_MODEL, 'names', None)
    _MODEL_PATH = selected
    print(f'[yolo] Clases del modelo: {_NAMES}')
    print(
        f'[yolo] ultra-far conf={_CONF} imgsz={_IMGSZ} iou={_IOU} '
        f'far={_FAR_MODE} upscale={_UPSCALE} tile={_TILE_MODE} augment={_AUGMENT}'
    )


def _ensure_model() -> YOLO:
    if _MODEL is None:
        configure()
    assert _MODEL is not None
    return _MODEL


def set_class_map(new_map: Dict[int, int]) -> None:
    _CLS_TO_SIGN_MAP.clear()
    _CLS_TO_SIGN_MAP.update({int(k): int(v) for k, v in new_map.items()})


def _class_name(cid: int):
    if _NAMES is None:
        return None
    if isinstance(_NAMES, dict):
        return _NAMES.get(int(cid), None)
    if 0 <= int(cid) < len(_NAMES):
        return _NAMES[int(cid)]
    return None


def _predict_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    model = _ensure_model()
    results = model(
        img,
        verbose=False,
        conf=_CONF,
        iou=_IOU,
        imgsz=_IMGSZ,
        max_det=_MAX_DET,
        augment=_AUGMENT,
    )
    r = results[0]
    if not hasattr(r, 'boxes') or len(r.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
            [],
        )

    boxes = r.boxes.xyxy.cpu().numpy().astype(float)
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    confidences = r.boxes.conf.cpu().numpy().astype(float)
    names = [_class_name(int(cid)) for cid in class_ids]
    return boxes, class_ids, confidences, names


def _clip_box(box: Sequence[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-6 else 0.0


def _nms_class_aware(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    names: list,
    iou_thr: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    if len(boxes) == 0:
        return boxes.astype(int), class_ids, confidences, names

    keep: List[int] = []
    for cid in sorted(set(int(c) for c in class_ids)):
        idxs = [i for i, c in enumerate(class_ids) if int(c) == cid]
        idxs.sort(key=lambda i: float(confidences[i]), reverse=True)
        selected: List[int] = []
        while idxs:
            cur = idxs.pop(0)
            selected.append(cur)
            idxs = [i for i in idxs if _iou(boxes[cur], boxes[i]) < iou_thr]
        keep.extend(selected)

    keep.sort(key=lambda i: float(confidences[i]), reverse=True)
    return (
        boxes[keep].astype(int),
        class_ids[keep].astype(int),
        confidences[keep].astype(float),
        [names[i] for i in keep],
    )


def _windows_for_ultra_far(w: int, h: int) -> List[Tuple[int, int, int, int, float, str]]:
    """Devuelve recortes (x0, y0, x1, y1, upscale, etiqueta)."""
    windows: List[Tuple[int, int, int, int, float, str]] = [(0, 0, w, h, 1.0, 'full')]
    if not _FAR_MODE or _TILE_MODE in ('off', 'none', 'false'):
        return windows

    s = _UPSCALE
    # Recortes grandes con overlap. Al agrandar el recorte, una señal lejana ocupa más pixeles relativos.
    windows.extend([
        (0, 0, w, int(0.82 * h), s, 'upper'),
        (0, 0, int(0.62 * w), h, s, 'left'),
        (int(0.38 * w), 0, w, h, s, 'right'),
        (int(0.18 * w), 0, int(0.82 * w), h, s, 'center'),
    ])

    if _TILE_MODE in ('aggressive', 'full', 'max'):
        windows.extend([
            (0, 0, int(0.54 * w), int(0.62 * h), s, 'ul'),
            (int(0.46 * w), 0, w, int(0.62 * h), s, 'ur'),
            (0, int(0.38 * h), int(0.54 * w), h, s, 'll'),
            (int(0.46 * w), int(0.38 * h), w, h, s, 'lr'),
            (int(0.25 * w), 0, int(0.75 * w), int(0.70 * h), max(1.0, s + 0.3), 'center_top'),
        ])
    return windows


def _infer_patch(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(x0 + 1, min(w, int(x1)))
    y1 = max(y0 + 1, min(h, int(y1)))
    patch = frame[y0:y1, x0:x1]
    ph, pw = patch.shape[:2]
    if pw < _MIN_PATCH_W or ph < _MIN_PATCH_H:
        return (
            np.zeros((0, 4), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
            [],
        )

    if scale > 1.01:
        patch_in = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        patch_in = patch

    boxes, class_ids, confidences, names = _predict_image(patch_in)
    if len(boxes) == 0:
        return boxes, class_ids, confidences, names

    boxes = boxes.astype(float)
    boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale + x0
    boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale + y0
    boxes = np.array([_clip_box(b, w, h) for b in boxes], dtype=float)
    return boxes, class_ids, confidences, names


def process_frame(
    frame: np.ndarray,
    drawing_frame: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Corre YOLO con recortes multi-escala y regresa detecciones en coordenadas del frame recibido."""
    h, w = frame.shape[:2]

    all_boxes: List[np.ndarray] = []
    all_cls: List[np.ndarray] = []
    all_conf: List[np.ndarray] = []
    all_names: List = []

    for x0, y0, x1, y1, scale, tag in _windows_for_ultra_far(w, h):
        boxes, class_ids, confidences, names = _infer_patch(frame, x0, y0, x1, y1, scale)
        if len(boxes) == 0:
            continue
        all_boxes.append(boxes)
        all_cls.append(class_ids)
        all_conf.append(confidences)
        all_names.extend(names)

        if drawing_frame is not None and tag != 'full':
            cv2.rectangle(drawing_frame, (x0, y0), (x1, y1), (80, 80, 80), 1)
            cv2.putText(drawing_frame, tag, (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (80, 80, 80), 1, cv2.LINE_AA)

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
            [],
        )

    boxes = np.vstack(all_boxes)
    class_ids = np.concatenate(all_cls)
    confidences = np.concatenate(all_conf)
    boxes, class_ids, confidences, names = _nms_class_aware(
        boxes, class_ids, confidences, all_names, iou_thr=max(0.35, _IOU)
    )

    if drawing_frame is not None:
        draw_detections(boxes, class_ids, confidences, names, drawing_frame, mapped=False)

    return boxes, class_ids, confidences, names


def draw_detections(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    class_names: list,
    drawing_frame: np.ndarray,
    mapped: bool = False,
) -> None:
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cid = int(class_ids[idx])
        conf = float(confidences[idx])
        cname = class_names[idx] if idx < len(class_names) else None
        label = f'{cname} ({cid})' if cname is not None else f'ID {cid}'
        if mapped:
            label = f'sign {cid}'
        text = f'{label}: {conf:.2f}'

        cv2.rectangle(drawing_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        y_text = y1 - 8 if y1 - 8 > 10 else y1 + 16
        cv2.putText(drawing_frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)


def get_signs(
    frame: np.ndarray,
    drawing_frame: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    boxes, class_ids, confidences, class_names = process_frame(frame)

    traffic_signs = []
    for box, cid, conf, cname in zip(boxes, class_ids, confidences, class_names):
        cid_int = int(cid)
        if cid_int in _CLS_TO_SIGN_MAP:
            traffic_signs.append((box, _CLS_TO_SIGN_MAP[cid_int], float(conf), cname))

    if traffic_signs:
        boxes_out, sign_types, conf_out, names_out = zip(*traffic_signs)
        boxes_arr = np.array(boxes_out, dtype=int)
        sign_arr = np.array(sign_types, dtype=int)
        conf_arr = np.array(conf_out, dtype=float)
        names_list = list(names_out)
    else:
        boxes_arr = np.zeros((0, 4), dtype=int)
        sign_arr = np.zeros((0,), dtype=int)
        conf_arr = np.zeros((0,), dtype=float)
        names_list = []

    if drawing_frame is not None and len(sign_arr) > 0:
        draw_detections(boxes_arr, sign_arr, conf_arr, names_list, drawing_frame, mapped=True)

    return boxes_arr, sign_arr, conf_arr, names_list