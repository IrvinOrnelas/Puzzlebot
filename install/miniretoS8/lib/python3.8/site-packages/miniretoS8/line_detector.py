import math

import cv2


def _threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=501, C=10
    )
    return binary


def _largest_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    biggest = max(contours, key=cv2.contourArea)
    mask = img * 0
    cv2.drawContours(mask, [biggest], -1, 255, cv2.FILLED)
    return mask


def find_line(img):
    """
    Process img to detect a dark line.

    Returns:
        vis       -- copy of the original image with annotations
        direction -- float in [-1.0, +1.0]; negative=left, 0=center, positive=right
    """
    h, w = img.shape[:2]
    v_crop = int(h * 0.75)
    h_crop = 0.75 * w
    cropped = img[v_crop:h, :].copy()
    # Ignorar el 10% de cada lado horizontalmente (usar solo el 80% central)
    x_start = int((w - h_crop) // 2)
    x_end   = int((w + h_crop) // 2)
    cropped = cropped[:, x_start:x_end]
    binary  = _threshold(cropped)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary  = cv2.erode(binary, kernel, iterations=2)
    binary  = cv2.dilate(binary, kernel, iterations=1)
    mask    = _largest_contour(binary)
    MIN_LINE_AREA = 3500

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Solo considerar contornos con area mayor al maximo de un blob de cebra
    contours = [c for c in contours if cv2.contourArea(c) > MIN_LINE_AREA]
    direction = 0.0
    cx = w // 2
    line_area = 0
    if contours:
        best = max(contours, key=cv2.contourArea)
        line_area = int(cv2.contourArea(best))
        M = cv2.moments(best)
        if M["m00"] != 0:
            cx_crop = int(M["m10"] / M["m00"])
            cx = cx_crop + x_start   # trasladar a coordenadas de imagen completa
            direction = (cx / (w / 2)) - 1.0

    # --- Debug: mostrar pasos intermedios ---
    # cv2.imshow("dbg: recorte", cropped)
    # cv2.imshow("dbg: binario", binary)
    # cv2.imshow("dbg: mascara", mask)
    # Contornos validos sobre el recorte
    dbg_cnt = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dbg_cnt, contours, -1, (0, 255, 0), 1)
    # cv2.imshow("dbg: contornos validos", dbg_cnt)
    print(f"  [find_line] area contorno = {line_area}  direction = {direction:+.3f}")
    # ----------------------------------------

    vis = img.copy()
    cv2.line(vis, (cx, v_crop), (cx, h), (0, 0, 255), 2)
    cv2.line(vis, (0, v_crop), (w, v_crop), (0, 255, 0), 1)
    cv2.line(vis, (w // 2, v_crop), (w // 2, h), (255, 0, 0), 1)
    cv2.line(vis, (x_start, v_crop), (x_start, h), (255, 255, 0), 1)
    cv2.line(vis, (x_end, v_crop), (x_end, h), (255, 255, 0), 1)

    line_found = line_area > 0
    return vis, direction, line_found


# -----------------------------
# Detección de paso de cebra
# -----------------------------

_ZEBRA_DIST_TOLERANCE  = 30    # umbral de distancia (px) para candidatos
_ZEBRA_RELATIVE_FACTOR = 2.5   # descarta si distancia > mediana * factor
_ZEBRA_MIN_ALIGNED     = 4     # mínimo de puntos alineados para confirmar
_ZEBRA_MAX_BLOB_AREA   = 2500  # área máxima de un centroide de cebra


def _dist_point_to_line(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    denom = math.sqrt(dx * dx + dy * dy)
    if denom == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom


def find_zebra(img, roi_y_start=0.60, min_blob_area=635, max_blob_area=_ZEBRA_MAX_BLOB_AREA):
    """
    Detecta paso de cebra buscando grupos de manchas oscuras alineadas.

    Returns:
        vis      -- copia del original con anotaciones
        detected -- bool
    """
    h, w = img.shape[:2]
    crop_start = int(h * roi_y_start)
    roi = img[crop_start:, :].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=501, C=10,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_blob_area < area < max_blob_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy, area))

    pts = [(cx, cy) for cx, cy, _ in centroids]
    zebra_group = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            with_dist = [
                (cx, cy, _dist_point_to_line(cx, cy, x1, y1, x2, y2))
                for cx, cy in pts
            ]
            candidates = [(cx, cy, d) for cx, cy, d in with_dist if d <= _ZEBRA_DIST_TOLERANCE]
            if len(candidates) < _ZEBRA_MIN_ALIGNED:
                continue
            dvals = sorted(d for _, _, d in candidates)
            median_d = dvals[len(dvals) // 2]
            adaptive = max(median_d * _ZEBRA_RELATIVE_FACTOR, 1.0)
            group = [(cx, cy) for cx, cy, d in candidates if d <= adaptive]
            if len(group) >= _ZEBRA_MIN_ALIGNED and len(group) > len(zebra_group):
                zebra_group = group

    detected = len(zebra_group) >= _ZEBRA_MIN_ALIGNED

    # Posicion vertical normalizada: 0=centro, +1=arriba, -1=abajo
    zebra_y_pos = 0.0
    if detected:
        mean_cy_roi = sum(cy for _, cy in zebra_group) / len(zebra_group)
        mean_cy_img = mean_cy_roi + crop_start          # coordenada en imagen completa
        # Invertir: arriba en imagen = y pequeño = positivo
        zebra_y_pos = -((mean_cy_img / (h / 2)) - 1.0)

    vis = img.copy()
    for cx, cy, _ in centroids:
        cv2.circle(vis, (cx, cy + crop_start), 5, (0, 0, 255), -1)

    if detected:
        for cx, cy in zebra_group:
            cv2.circle(vis, (cx, cy + crop_start), 8, (0, 255, 255), 2)
        gs = sorted(zebra_group, key=lambda p: p[0])
        p1 = (gs[0][0], gs[0][1] + crop_start)
        p2 = (gs[-1][0], gs[-1][1] + crop_start)
        cv2.line(vis, p1, p2, (0, 255, 255), 2)
        mid = gs[len(gs) // 2]
        cv2.putText(
            vis, f'ZEBRA  y={zebra_y_pos:+.2f}',
            (mid[0] + 8, mid[1] + crop_start - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        print(f"  [find_zebra] detectado  y_pos = {zebra_y_pos:+.3f}")

    return vis, detected, zebra_y_pos
