import math
from typing import Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────
# Helpers de umbralización y bandas  (mecanismo de line_detector.py)
# ──────────────────────────────

def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _threshold_dark(roi: np.ndarray) -> np.ndarray:
    """Detecta zonas oscuras de manera adaptable para aguantar cambios de luz."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    block = _odd(min(151, max(41, roi.shape[1] // 5)))
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block, C=8,
    )

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark_by_value = cv2.inRange(v, 0, 115)

    mask = cv2.bitwise_or(adaptive, dark_by_value)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, erode_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    open_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_kernel,  iterations=1)
    return mask


def _find_runs(binary_1d: np.ndarray):
    """Regresa segmentos [inicio, fin] donde binary_1d es True."""
    runs   = []
    in_run = False
    start  = 0
    for i, val in enumerate(binary_1d):
        if val and not in_run:
            start  = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(binary_1d) - 1))
    return runs


def _zebra_score(mask: np.ndarray) -> Tuple[bool, float]:
    """Detecta patrón tipo paso de cebra para ignorarlo en find_line."""
    roi_h, roi_w = mask.shape[:2]
    contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_bars = 0
    horizontal_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        x, y, w, h  = cv2.boundingRect(cnt)
        wide         = w > 0.35 * roi_w
        flat         = h < 0.26 * roi_h
        not_too_thin = h > 3
        if wide and flat and not_too_thin:
            horizontal_bars += 1
            horizontal_area += area

    band_hits = 0
    n_bands   = 8
    for i in range(n_bands):
        y0       = int(i * roi_h / n_bands)
        y1       = int((i + 1) * roi_h / n_bands)
        band     = mask[y0:y1, :]
        coverage = cv2.countNonZero(band) / float(band.size)
        if coverage > 0.26:
            band_hits += 1

    score = min(1.0, (horizontal_bars / 3.0) * 0.65 + (band_hits / 4.0) * 0.35)
    return score > 0.32, score


def _scan_band_centers(mask: np.ndarray, previous_cx_roi: Optional[float] = None):
    """
    Busca centros de línea por bandas horizontales de abajo hacia arriba.
    - 3+ segmentos visibles: elige el central (mediana X) — línea del medio.
    - 1-2 segmentos (curva, lateral fuera de cuadro): usa proximidad al
      expected anterior para mantener continuidad.
    """
    roi_h, roi_w = mask.shape[:2]
    centers      = []
    expected     = previous_cx_roi if previous_cx_roi is not None else roi_w / 2.0

    band_ranges = [
        (0.82, 1.00, 1.00),
        (0.68, 0.84, 0.85),
        (0.54, 0.70, 0.65),
        (0.40, 0.56, 0.45),
        (0.26, 0.42, 0.30),
    ]

    for r0, r1, bottom_weight in band_ranges:
        y0   = int(r0 * roi_h)
        y1   = int(r1 * roi_h)
        band = mask[y0:y1, :]
        if band.size == 0:
            continue

        coverage = cv2.countNonZero(band) / float(band.size)
        if coverage > 0.55:
            continue

        col = np.sum(band > 0, axis=0).astype(np.float32)
        if float(np.max(col)) < 3.0:
            continue

        col       = cv2.GaussianBlur(col.reshape(1, -1), (1, 31), 0).flatten()
        threshold = max(3.0, 0.32 * float(np.max(col)))
        runs      = _find_runs(col > threshold)

        valid_runs = []
        for x0, x1 in runs:
            width = x1 - x0 + 1
            if width < 5 or width > 0.58 * roi_w:
                continue
            cx   = 0.5 * (x0 + x1)
            peak = float(np.max(col[x0:x1 + 1]))
            valid_runs.append((cx, peak, width))

        if not valid_runs:
            continue

        valid_runs.sort(key=lambda r: r[0])

        if len(valid_runs) >= 3:
            # Recta con 3 líneas visibles: tomar siempre la del medio
            cx, peak, width = valid_runs[len(valid_runs) // 2]
        else:
            # Curva: una lateral salió del cuadro, usar proximidad para continuidad
            cx, peak, width = min(valid_runs, key=lambda r: abs(r[0] - expected))

        expected = cx
        width_quality = 1.0 - min(1.0, width / (0.58 * roi_w))
        score = bottom_weight * peak * (1.0 + 0.1 * width_quality)
        centers.append((cx, score, y0, y1, width))

    return centers


def _fallback_contour_center(mask: np.ndarray, previous_cx_roi: Optional[float] = None):
    roi_h, roi_w = mask.shape[:2]
    contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    expected   = previous_cx_roi if previous_cx_roi is not None else roi_w / 2.0
    best       = None
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
        cx           = float(M['m10'] / M['m00'])
        bottom_bonus = (y + h) / roi_h
        proximity    = 1.0 - min(1.0, abs(cx - expected) / (0.5 * roi_w))
        score        = area * (0.55 * bottom_bonus + 0.45 * proximity)
        if score > best_score:
            best_score = score
            best = (cx, min(1.0, area / (0.08 * roi_w * roi_h)))

    return best


# ──────────────────────────────
# Detección de línea
# ──────────────────────────────

def find_line(
    img: np.ndarray,
    previous_direction: Optional[float] = None,
    roi_y_start: float = 0.58,
    roi_x_margin: float = 0.05,
):
    """
    Detecta línea oscura de forma robusta.

    Returns:
        vis        -- copy of the original image with annotations
        direction  -- float in [-1.0, +1.0]; negative=left, 0=center, positive=right
        line_found -- bool
    """
    h, w    = img.shape[:2]
    crop_y0 = int(h * roi_y_start)
    crop_x0 = int(w * roi_x_margin)
    crop_x1 = int(w * (1.0 - roi_x_margin))

    roi      = img[crop_y0:h, crop_x0:crop_x1].copy()
    mask     = _threshold_dark(roi)
    roi_h, roi_w = mask.shape[:2]

    # Guardia anti-zebra:
    # Si la ROI de seguimiento ya contiene varias franjas horizontales, NO
    # calculamos centroide de linea. Eso evita que el robot interprete el
    # paso zebra como una curva de 90 grados.
    zebra_like, zebra_score = _zebra_score(mask)
    if zebra_like:
        vis = img.copy()
        cx_ref = w // 2
        cv2.line(vis, (cx_ref, 0), (cx_ref, h - 1), (255, 0, 0), 1)
        cv2.rectangle(vis, (crop_x0, crop_y0), (crop_x1, h - 1), (0, 255, 255), 2)
        cv2.putText(
            vis, f'ZEBRA-GUARD score={zebra_score:.2f}',
            (10, max(24, crop_y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
        )
        return vis, 0.0, False

    previous_cx_roi = None
    if previous_direction is not None:
        prev_cx_full    = (previous_direction + 1.0) * (w / 2.0)
        previous_cx_roi = float(prev_cx_full - crop_x0)
        previous_cx_roi = max(0.0, min(float(roi_w - 1), previous_cx_roi))

    centers = _scan_band_centers(mask, previous_cx_roi)

    line_found = False
    confidence = 0.0
    cx_roi     = roi_w / 2.0

    if centers:
        weighted_sum = sum(c * s for c, s, *_ in centers)
        weight_total = sum(s for _, s, *_ in centers)
        cx_roi     = weighted_sum / max(1e-6, weight_total)
        confidence = min(1.0, len(centers) / 4.0 + min(0.35, weight_total / (roi_h * 40.0)))
        line_found = confidence > 0.15
    else:
        fallback = _fallback_contour_center(mask, previous_cx_roi)
        if fallback is not None:
            cx_roi, confidence = fallback
            line_found = confidence > 0.12

    cx_full   = int(round(cx_roi + crop_x0))
    cx_full   = max(0, min(w - 1, cx_full))
    direction = (cx_full / (w / 2.0)) - 1.0
    direction = max(-1.0, min(1.0, float(direction)))

    vis = img.copy()

    # Línea central de referencia (azul) — referencia visual de "ir recto"
    cx_ref = w // 2
    cv2.line(vis, (cx_ref, 0), (cx_ref, h - 1), (255, 0, 0), 1)

    cv2.rectangle(vis, (crop_x0, crop_y0), (crop_x1, h - 1), (0, 255, 0), 1)

    for c, score, y0, y1, width in centers:
        x   = int(c + crop_x0)
        yy0 = int(y0 + crop_y0)
        yy1 = int(y1 + crop_y0)
        cv2.circle(vis, (x, (yy0 + yy1) // 2), 4, (255, 0, 255), -1)
        cv2.line(vis, (x, yy0), (x, yy1), (255, 0, 255), 1)

    if line_found:
        cv2.line(vis, (cx_full, crop_y0), (cx_full, h - 1), (0, 0, 255), 2)
    else:
        cv2.line(vis, (w // 2, crop_y0), (w // 2, h - 1), (128, 128, 128), 1)

    return vis, direction, line_found


# ──────────────────────────────
# Detección de paso de cebra
# ──────────────────────────────

# Estos valores están pensados para el frame reducido (_PROC_SCALE=0.5).
# La versión anterior usaba min_blob_area=635, que en 320x180 puede ser
# demasiado grande para el segundo paso de zebra cuando aún aparece lejos.
_ZEBRA_DIST_TOLERANCE  = 22
_ZEBRA_RELATIVE_FACTOR = 2.8
_ZEBRA_MIN_ALIGNED     = 3
_ZEBRA_MAX_BLOB_AREA   = 9000


def _dist_point_to_line(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    denom  = math.sqrt(dx * dx + dy * dy)
    if denom == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom


def _zebra_mask(roi: np.ndarray) -> np.ndarray:
    """Máscara oscura específica para zebra; conserva franjas pequeñas/lejos."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    block = _odd(min(301, max(31, roi.shape[1] // 4)))
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block, C=7,
    )

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark_by_value = cv2.inRange(v, 0, 115)
    mask = cv2.bitwise_or(adaptive, dark_by_value)

    # Erosión más agresiva para eliminar falsos positivos durante vueltas
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    mask = cv2.erode(mask, erode_kernel, iterations=2)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return mask


def _find_horizontal_zebra_runs(mask: np.ndarray):
    """Busca varias franjas horizontales usando proyección por filas."""
    roi_h, roi_w = mask.shape[:2]
    row_cov = np.sum(mask > 0, axis=1).astype(np.float32) / float(max(1, roi_w))
    row_cov = cv2.GaussianBlur(row_cov.reshape(-1, 1), (1, 7), 0).flatten()

    # Una zebra ocupa bastante ancho; la línea guía normal no.
    thresh = max(0.10, min(0.32, float(np.percentile(row_cov, 84))))
    runs = _find_runs(row_cov > thresh)

    valid = []
    for y0, y1 in runs:
        height = y1 - y0 + 1
        if height < 2 or height > 0.28 * roi_h:
            continue
        strength = float(np.mean(row_cov[y0:y1 + 1]))
        if strength < 0.10:
            continue
        valid.append((y0, y1, strength))
    return valid


def _find_aligned_zebra_blobs(mask: np.ndarray, min_blob_area: float, max_blob_area: float):
    """RANSAC sencillo sobre centroides de blobs oscuros alineados."""
    roi_h, roi_w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_blob_area or area > max_blob_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Rechazar la línea central muy larga/vertical y manchas enormes de borde.
        if bh > 0.75 * roi_h and bw < 0.18 * roi_w:
            continue
        if bw > 0.85 * roi_w and bh > 0.45 * roi_h:
            continue
        if bw < 3 or bh < 2:
            continue

        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy, area, x, y, bw, bh))

    pts = [(cx, cy) for cx, cy, *_ in centroids]
    best_group = []
    tolerance = max(_ZEBRA_DIST_TOLERANCE, int(0.065 * roi_w))

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            # Evitar líneas casi verticales: la zebra cruza el camino, no sigue la línea guía.
            if abs(x2 - x1) < 0.18 * roi_w:
                continue
            with_dist = [
                (cx, cy, _dist_point_to_line(cx, cy, x1, y1, x2, y2))
                for cx, cy in pts
            ]
            candidates = [(cx, cy, d) for cx, cy, d in with_dist if d <= tolerance]
            if len(candidates) < _ZEBRA_MIN_ALIGNED:
                continue
            dvals = sorted(d for _, _, d in candidates)
            median_d = dvals[len(dvals) // 2]
            adaptive = max(median_d * _ZEBRA_RELATIVE_FACTOR, 2.0)
            group = [(cx, cy) for cx, cy, d in candidates if d <= adaptive]
            if len(group) >= _ZEBRA_MIN_ALIGNED and len(group) > len(best_group):
                best_group = group

    return centroids, best_group


def find_zebra(img, roi_y_start=0.34, min_blob_area=45, max_blob_area=_ZEBRA_MAX_BLOB_AREA):
    """
    Detecta paso de zebra combinando dos pistas:
      1) blobs oscuros alineados, útil cuando la zebra se ve inclinada/perspectiva;
      2) varias franjas horizontales por proyección de filas, útil para el segundo paso lejos.

    Returns:
        vis         -- copia del original con anotaciones
        detected    -- bool
        zebra_y_pos -- posición vertical normalizada: 0=centro, +1=arriba, -1=abajo
    """
    h, w = img.shape[:2]
    crop_start = int(h * roi_y_start)
    roi = img[crop_start:, :].copy()
    roi_h, roi_w = roi.shape[:2]

    mask = _zebra_mask(roi)

    # Umbrales dinámicos para que funcione con _PROC_SCALE=0.5 y también si lo pruebas full-res.
    roi_area = float(max(1, roi_h * roi_w))
    dyn_min_area = max(float(min_blob_area), 0.0008 * roi_area)
    dyn_max_area = min(float(max_blob_area), 0.18 * roi_area)

    centroids, zebra_group = _find_aligned_zebra_blobs(mask, dyn_min_area, dyn_max_area)
    row_runs = _find_horizontal_zebra_runs(mask)

    # Score por blobs alineados - ÚNICO criterio válido
    aligned_ok = len(zebra_group) >= _ZEBRA_MIN_ALIGNED
    detected = aligned_ok

    zebra_y_pos = 0.0
    if detected:
        if aligned_ok:
            mean_cy_roi = sum(cy for _, cy in zebra_group) / len(zebra_group)
        else:
            # Usar la franja más baja: sirve mejor para decidir cuándo ya está cerca.
            mean_cy_roi = max(0.5 * (y0 + y1) for y0, y1, _ in row_runs)
        mean_cy_img = mean_cy_roi + crop_start
        zebra_y_pos = -((mean_cy_img / (h / 2.0)) - 1.0)
        zebra_y_pos = max(-1.0, min(1.0, float(zebra_y_pos)))

    vis = img.copy()
    # Dibujar ROI y blobs candidatos.
    cv2.rectangle(vis, (0, crop_start), (w - 1, h - 1), (0, 255, 255), 1)
    for cx, cy, area, *_ in centroids:
        cv2.circle(vis, (cx, cy + crop_start), 4, (0, 0, 255), -1)

    for y0, y1, strength in row_runs:
        yy0, yy1 = y0 + crop_start, y1 + crop_start
        cv2.rectangle(vis, (0, yy0), (w - 1, yy1), (255, 255, 0), 1)

    if detected:
        if aligned_ok:
            for cx, cy in zebra_group:
                cv2.circle(vis, (cx, cy + crop_start), 8, (0, 255, 255), 2)
            gs = sorted(zebra_group, key=lambda p: p[0])
            p1 = (gs[0][0], gs[0][1] + crop_start)
            p2 = (gs[-1][0], gs[-1][1] + crop_start)
            cv2.line(vis, p1, p2, (0, 255, 255), 2)
            tx, ty = gs[len(gs) // 2][0], gs[len(gs) // 2][1] + crop_start
        else:
            tx, ty = 10, int(crop_start + max(0.5 * (y0 + y1) for y0, y1, _ in row_runs))
        cv2.putText(
            vis, f'ZEBRA y={zebra_y_pos:+.2f}',
            (int(tx) + 8, int(ty) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

    return vis, detected, zebra_y_pos