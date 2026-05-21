from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Resultados de visión
# -----------------------------

@dataclass
class LineResult:
    vis: np.ndarray
    direction: float          # [-1, 1], negativo=izq, positivo=der
    confidence: float         # [0, 1]
    line_found: bool
    zebra_detected: bool
    center_x: int
    debug: Dict[str, float]


@dataclass
class TrafficLightResult:
    color: str                # 'red', 'yellow', 'green', 'none'
    speed_multiplier: float   # red=0.0, yellow=0.5, green/none=1.0
    confidence: float
    area: float
    bbox: Optional[Tuple[int, int, int, int]]


# -----------------------------
# Detección de línea
# -----------------------------

def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _threshold_dark(roi: np.ndarray) -> np.ndarray:
    """Detecta zonas oscuras de manera adaptable para aguantar cambios de luz."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # El blockSize se adapta al tamaño del recorte. No conviene dejar 501 fijo
    # porque con imágenes chicas puede fallar o volverse muy lento.
    block = _odd(min(151, max(41, roi.shape[1] // 5)))
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=8,
    )

    # Refuerzo por valor bajo en HSV: ayuda cuando la línea negra no contrasta
    # igual en toda la pista.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark_by_value = cv2.inRange(v, 0, 95)

    mask = cv2.bitwise_or(adaptive, dark_by_value)

    # Limpieza ligera. Close conecta pedazos de línea; open quita puntitos.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return mask


def _find_runs(binary_1d: np.ndarray):
    """Regresa segmentos [inicio, fin] donde binary_1d es True."""
    runs = []
    in_run = False
    start = 0
    for i, val in enumerate(binary_1d):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(binary_1d) - 1))
    return runs


def _zebra_score(mask: np.ndarray) -> Tuple[bool, float]:
    """
    Detecta patrón tipo paso de cebra.

    No se usa para detener por sí solo. Se usa para bajar velocidad y evitar
    que una franja horizontal gigante se tome como línea principal.
    """
    roi_h, roi_w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_bars = 0
    horizontal_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        wide = w > 0.45 * roi_w
        flat = h < 0.22 * roi_h
        not_too_thin = h > 3
        if wide and flat and not_too_thin:
            horizontal_bars += 1
            horizontal_area += area

    # También revisamos bandas horizontales con mucha cobertura oscura.
    band_hits = 0
    n_bands = 8
    for i in range(n_bands):
        y0 = int(i * roi_h / n_bands)
        y1 = int((i + 1) * roi_h / n_bands)
        band = mask[y0:y1, :]
        coverage = cv2.countNonZero(band) / float(band.size)
        if coverage > 0.34:
            band_hits += 1

    score = min(1.0, (horizontal_bars / 4.0) * 0.6 + (band_hits / 5.0) * 0.4)
    return score > 0.45, score


def _scan_band_centers(mask: np.ndarray, previous_cx_roi: Optional[float] = None):
    """
    Busca centros de línea por bandas horizontales.

    Esto es más robusto que tomar el contorno más grande porque los pasos de
    cebra producen contornos horizontales muy grandes. Aquí se ignoran bandas
    con demasiada cobertura y segmentos exageradamente anchos.
    """
    roi_h, roi_w = mask.shape[:2]
    centers = []

    # Bandas de abajo hacia arriba. Lo cercano al robot pesa más.
    band_ranges = [
        (0.82, 1.00, 1.00),
        (0.68, 0.84, 0.85),
        (0.54, 0.70, 0.65),
        (0.40, 0.56, 0.45),
        (0.26, 0.42, 0.30),
    ]

    expected = previous_cx_roi if previous_cx_roi is not None else roi_w / 2.0

    for r0, r1, bottom_weight in band_ranges:
        y0 = int(r0 * roi_h)
        y1 = int(r1 * roi_h)
        band = mask[y0:y1, :]
        if band.size == 0:
            continue

        coverage = cv2.countNonZero(band) / float(band.size)
        if coverage > 0.42:
            # Muy probablemente zebra, sombra grande o ruido de piso.
            continue

        col = np.sum(band > 0, axis=0).astype(np.float32)
        if float(np.max(col)) < 3.0:
            continue

        col = cv2.GaussianBlur(col.reshape(1, -1), (1, 31), 0).flatten()
        threshold = max(3.0, 0.32 * float(np.max(col)))
        runs = _find_runs(col > threshold)

        best = None
        best_score = -1.0
        for x0, x1 in runs:
            width = x1 - x0 + 1
            if width < 5:
                continue
            if width > 0.58 * roi_w:
                # Segmentos demasiado anchos suelen ser zebra/piso oscuro.
                continue

            cx = 0.5 * (x0 + x1)
            peak = float(np.max(col[x0:x1 + 1]))
            proximity = 1.0 - min(1.0, abs(cx - expected) / (0.5 * roi_w))
            width_quality = 1.0 - min(1.0, width / (0.58 * roi_w))
            score = bottom_weight * (0.55 * peak + 0.35 * proximity * peak + 0.10 * width_quality * peak)

            if score > best_score:
                best_score = score
                best = (cx, score, y0, y1, width)

        if best is not None:
            centers.append(best)
            # La línea debe ser continua; el siguiente segmento esperado queda
            # cerca del centro encontrado en esta banda.
            expected = best[0]

    return centers


def _fallback_contour_center(mask: np.ndarray, previous_cx_roi: Optional[float] = None):
    roi_h, roi_w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    expected = previous_cx_roi if previous_cx_roi is not None else roi_w / 2.0
    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0.65 * roi_w and h < 0.30 * roi_h:
            continue
        if h < 6:
            continue

        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = float(M['m10'] / M['m00'])
        bottom_bonus = (y + h) / roi_h
        proximity = 1.0 - min(1.0, abs(cx - expected) / (0.5 * roi_w))
        score = area * (0.55 * bottom_bonus + 0.45 * proximity)
        if score > best_score:
            best_score = score
            best = (cx, min(1.0, area / (0.08 * roi_w * roi_h)))

    return best


def find_line(
    img: np.ndarray,
    previous_direction: Optional[float] = None,
    roi_y_start: float = 0.58,
    roi_x_margin: float = 0.08,
) -> LineResult:
    """
    Detecta línea oscura de forma robusta.

    Returns:
        LineResult con direction en [-1, +1].
        Negativo = línea a la izquierda, positivo = línea a la derecha.
    """
    h, w = img.shape[:2]
    crop_y0 = int(h * roi_y_start)
    crop_x0 = int(w * roi_x_margin)
    crop_x1 = int(w * (1.0 - roi_x_margin))

    roi = img[crop_y0:h, crop_x0:crop_x1].copy()
    mask = _threshold_dark(roi)
    roi_h, roi_w = mask.shape[:2]

    previous_cx_roi = None
    if previous_direction is not None:
        prev_cx_full = (previous_direction + 1.0) * (w / 2.0)
        previous_cx_roi = float(prev_cx_full - crop_x0)
        previous_cx_roi = max(0.0, min(float(roi_w - 1), previous_cx_roi))

    zebra_detected, zebra = _zebra_score(mask)
    centers = _scan_band_centers(mask, previous_cx_roi)

    line_found = False
    confidence = 0.0
    cx_roi = roi_w / 2.0

    if centers:
        # Promedio pesado: las bandas inferiores pesan más.
        weighted_sum = sum(c * s for c, s, *_ in centers)
        weight_total = sum(s for _, s, *_ in centers)
        cx_roi = weighted_sum / max(1e-6, weight_total)
        confidence = min(1.0, len(centers) / 4.0 + min(0.35, weight_total / (roi_h * 40.0)))
        line_found = confidence > 0.15
    else:
        fallback = _fallback_contour_center(mask, previous_cx_roi)
        if fallback is not None:
            cx_roi, confidence = fallback
            line_found = confidence > 0.12

    cx_full = int(round(cx_roi + crop_x0))
    cx_full = max(0, min(w - 1, cx_full))
    direction = (cx_full / (w / 2.0)) - 1.0
    direction = max(-1.0, min(1.0, float(direction)))

    vis = img.copy()

    # ROI de búsqueda
    cv2.rectangle(vis, (crop_x0, crop_y0), (crop_x1, h - 1), (0, 255, 0), 1)

    # Centros detectados por banda
    for c, score, y0, y1, width in centers:
        x = int(c + crop_x0)
        yy0 = int(y0 + crop_y0)
        yy1 = int(y1 + crop_y0)
        cv2.circle(vis, (x, (yy0 + yy1) // 2), 4, (255, 0, 255), -1)
        cv2.line(vis, (x, yy0), (x, yy1), (255, 0, 255), 1)

    if line_found:
        cv2.line(vis, (cx_full, crop_y0), (cx_full, h - 1), (0, 0, 255), 2)
    else:
        cv2.line(vis, (w // 2, crop_y0), (w // 2, h - 1), (128, 128, 128), 1)

    if zebra_detected:
        cv2.putText(
            vis,
            f'ZEBRA score={zebra:.2f}',
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2,
        )

    return LineResult(
        vis=vis,
        direction=direction,
        confidence=float(confidence),
        line_found=bool(line_found),
        zebra_detected=bool(zebra_detected),
        center_x=int(cx_full),
        debug={'zebra_score': float(zebra), 'bands': float(len(centers))},
    )


# -----------------------------
# Detección de semáforo
# -----------------------------

RED_LOWER1 = np.array([0, 100, 70])
RED_UPPER1 = np.array([12, 255, 255])
RED_LOWER2 = np.array([165, 100, 70])
RED_UPPER2 = np.array([180, 255, 255])

YELLOW_LOWER = np.array([18, 85, 80])
YELLOW_UPPER = np.array([38, 255, 255])

GREEN_LOWER = np.array([40, 70, 60])
GREEN_UPPER = np.array([88, 255, 255])


def _clean_color_mask(mask: np.ndarray) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return mask


def _best_color_candidate(mask: np.ndarray, min_area: float):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        aspect = w / float(h)
        if aspect < 0.25 or aspect > 3.2:
            continue
        extent = area / float(w * h)
        if extent < 0.18:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, w, h, area)
    return best


def detect_traffic_light(
    img: np.ndarray,
    roi_y_end: float = 0.70,
    min_area_ratio: float = 0.00065,
) -> TrafficLightResult:
    """
    Detecta semáforo rojo/amarillo/verde.

    Para evitar confundir colores del piso/pista con el semáforo, por defecto
    solo mira el 70% superior de la imagen.
    """
    h, w = img.shape[:2]
    y_end = int(h * roi_y_end)
    roi = img[:y_end, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2),
    )
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    red_mask = _clean_color_mask(red_mask)
    yellow_mask = _clean_color_mask(yellow_mask)
    green_mask = _clean_color_mask(green_mask)

    min_area = max(180.0, min_area_ratio * float(w * h))

    candidates = {
        'red': _best_color_candidate(red_mask, min_area),
        'yellow': _best_color_candidate(yellow_mask, min_area),
        'green': _best_color_candidate(green_mask, min_area),
    }

    # Prioridad segura: rojo > amarillo > verde. Así, si aparece rojo claro,
    # no se ignora por tener menor área que otra luz.
    for color, mult in [('red', 0.0), ('yellow', 0.5), ('green', 1.0)]:
        cand = candidates[color]
        if cand is not None:
            x, y, bw, bh, area = cand
            conf = min(1.0, area / (min_area * 6.0))
            return TrafficLightResult(
                color=color,
                speed_multiplier=mult,
                confidence=float(conf),
                area=float(area),
                bbox=(int(x), int(y), int(bw), int(bh)),
            )

    return TrafficLightResult(
        color='none',
        speed_multiplier=1.0,
        confidence=0.0,
        area=0.0,
        bbox=None,
    )


def draw_traffic_light(vis: np.ndarray, traffic: TrafficLightResult) -> np.ndarray:
    if traffic.bbox is None:
        return vis

    x, y, w, h = traffic.bbox
    colors = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'none': (255, 255, 255),
    }
    bgr = colors.get(traffic.color, (255, 255, 255))
    cv2.rectangle(vis, (x, y), (x + w, y + h), bgr, 3)
    cv2.putText(
        vis,
        f'{traffic.color} area={int(traffic.area)}',
        (x, max(22, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        bgr,
        2,
    )
    return vis

