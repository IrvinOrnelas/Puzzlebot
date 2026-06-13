import cv2
import math
import numpy as np

# ============================================================
# Parametros de calibracion (parametros_camara.txt)
# ============================================================
K_CAM = np.array([[392.87750154,   0.,         315.86299253],
                  [  0.,         393.25106552, 179.03445968],
                  [  0.,           0.,           1.        ]])
DIST_COEFFS = np.array([[-0.35184753, 0.16420204, -0.00060158, -0.000373, -0.04176109]])

# Resolucion usada en la calibracion: K y dist solo son validas a este tamano
CALIB_SIZE = (640, 360)   # (ancho, alto)

# ============================================================
# Parametros de deteccion de zebra
# ============================================================
ROI_Y_START      = 0.60   # usar solo el 40% inferior
MIN_BLOB_AREA    = 635
MAX_BLOB_AREA    = 2500
DIST_TOLERANCE   = 10      # umbral de distancia (px) para candidatos
RELATIVE_FACTOR  = 2.5     # descarta si distancia > mediana * factor
MIN_ALIGNED      = 4       # minimo de puntos alineados para confirmar

# Interpolacion area esperada segun la coordenada y invertida (abajo->arriba).
# A mayor area, menor y. Dos puntos de referencia (y_inv, area_min, area_max):
AREA_Y_LOW   = (20,  1500, 2600)   # cerca del fondo de la imagen
AREA_Y_HIGH  = (128,  165,  240)   # mas arriba en la imagen
AREA_MARGIN  = 1.6                 # tolerancia +/- sobre el rango interpolado
MAX_ASPECT_RATIO = 6.0              # descartar si eje mayor / eje menor supera esto

# ============================================================
# Parametros de deteccion de linea
# ============================================================
LINE_V_CROP   = 0.75   # usar solo el 25% inferior de la imagen
LINE_H_CROP   = 0.75   # usar solo el 75% central horizontal
MIN_LINE_AREA = 3500   # area minima del contorno para considerarlo linea

SCALE = 0.5                # ventanas a la mitad del tamano

# Ventanas de debug: si True solo se muestran las ventanas finales (LAST_WINDOWS)
ONLY_LAST_WINDOW = True
LAST_WINDOWS = {
    "5) Deteccion zebra",
    "L1) Dedistorsion",
    "L2) Recorte linea",
    "L3) Umbral + morfologia",
    "L4) Deteccion linea",
}

_undistort_maps = {}
_windows = set()


def get_undistort_maps(w, h):
    key = (w, h)
    if key not in _undistort_maps:
        new_K, roi = cv2.getOptimalNewCameraMatrix(K_CAM, DIST_COEFFS, (w, h), alpha=0)
        map1, map2 = cv2.initUndistortRectifyMap(K_CAM, DIST_COEFFS, None, new_K, (w, h), cv2.CV_16SC2)
        _undistort_maps[key] = (map1, map2, roi)
    return _undistort_maps[key]


def dist_point_to_line(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    denom = math.sqrt(dx * dx + dy * dy)
    if denom == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom


def expected_area_range(y_inv):
    """Interpola el rango de area esperado (min, max) para una coordenada y
    invertida (abajo->arriba), segun los dos puntos de referencia."""
    y0, lo0, hi0 = AREA_Y_LOW
    y1, lo1, hi1 = AREA_Y_HIGH
    t = (y_inv - y0) / (y1 - y0) if y1 != y0 else 0.0
    t = max(0.0, min(1.0, t))           # limitar fuera de rango
    lo = lo0 + t * (lo1 - lo0)
    hi = hi0 + t * (hi1 - hi0)
    # aplicar margen de tolerancia
    return lo / AREA_MARGIN, hi * AREA_MARGIN


def show(name, img):
    """Muestra la imagen en una ventana redimensionable."""
    if ONLY_LAST_WINDOW and name not in LAST_WINDOWS:
        return
    if name not in _windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        h, w = img.shape[:2]
        cv2.resizeWindow(name, int(w * SCALE), int(h * SCALE))
        _windows.add(name)
    cv2.imshow(name, img)


def find_zebra(frame, debug=True):
    """Detecta un cruce de zebra en el frame. Retorna detected (bool).
    Si debug=True muestra las ventanas de cada paso del pipeline."""
    # Redimensionar a la resolucion de calibracion para que K sea valida
    if (frame.shape[1], frame.shape[0]) != CALIB_SIZE:
        frame = cv2.resize(frame, CALIB_SIZE, interpolation=cv2.INTER_AREA)

    undist,  vis1            = step1_undistort(frame)
    (roi_img, crop_y), vis2 = step2_crop(undist)
    (morph, contours), vis3 = step3_threshold(roi_img)
    centroids, vis4         = step4_filter(morph, contours)
    detected, vis5          = step5_detect(undist, centroids, crop_y)

    if debug:
        steps = [
            ("1) Dedistorsion", vis1),
            ("2) Recorte ROI", vis2),
            ("3) Umbral + morfologia (area)", vis3),
            ("4) Contornos filtrados", vis4),
            ("5) Deteccion zebra", vis5),
        ]
        for name, img in steps:
            if not ONLY_LAST_WINDOW or name in LAST_WINDOWS:
                show(name, img)

    return detected


def step1_undistort(frame):
    """1) Dedistorsion. Retorna (undist, frame_anotado)."""
    h0, w0 = frame.shape[:2]
    map1, map2, roi = get_undistort_maps(w0, h0)
    undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    rx, ry, rw, rh = roi
    undist = undist[ry:ry+rh, rx:rx+rw]
    return undist, undist.copy()


def step2_crop(undist):
    """2) Recorte (ROI inferior). Retorna ((roi_img, crop_start), frame_anotado)."""
    h = undist.shape[0]
    crop_start = int(h * ROI_Y_START)
    roi_img = undist[crop_start:, :].copy()
    return (roi_img, crop_start), roi_img.copy()


def step3_threshold(roi_img):
    """3) Umbral adaptativo + morfologia + area de cada contorno.
    Retorna ((morph, contornos), frame_anotado)."""
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=501, C=10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.erode(binary, kernel, iterations=2)

    roi_h = morph.shape[0]
    all_contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dbg_area = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cy_inv = roi_h - cy   # coordenada y invertida (de abajo hacia arriba)
            # Area en amarillo y coordenada y (invertida) en cyan, una debajo de la otra
            cv2.putText(dbg_area, f"a={int(area)}", (cx - 25, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(dbg_area, f"y={cy_inv}", (cx - 25, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return (morph, all_contours), dbg_area


def step4_filter(morph, all_contours):
    """4) Filtrar contornos por area interpolada segun su coordenada y.
    Retorna (centroids, frame_anotado)."""
    roi_h = morph.shape[0]
    centroids = []
    dbg_filt = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cy_inv = roi_h - cy
        area_min, area_max = expected_area_range(cy_inv)

        # Proporcion: descartar contornos muy alargados (eje mayor >> eje menor)
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        major = max(rw, rh)
        minor = min(rw, rh)
        aspect = major / minor if minor > 0 else float('inf')

        if area_min <= area <= area_max and aspect <= MAX_ASPECT_RATIO:
            centroids.append((cx, cy))
            cv2.drawContours(dbg_filt, [cnt], -1, (0, 255, 255), 1)
            cv2.circle(dbg_filt, (cx, cy), 4, (0, 0, 255), -1)
        else:
            cv2.drawContours(dbg_filt, [cnt], -1, (80, 80, 80), 1)
    return centroids, dbg_filt


def step5_detect(undist, centroids, crop_start):
    """5) Buscar alineacion para detectar zebra y anotar centro/angulo.
    Retorna (detected, frame_anotado)."""
    pts = centroids
    zebra_group = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            with_dist = [(cx, cy, dist_point_to_line(cx, cy, x1, y1, x2, y2))
                         for cx, cy in pts]
            candidates = [(cx, cy, d) for cx, cy, d in with_dist if d <= DIST_TOLERANCE]
            if len(candidates) < MIN_ALIGNED:
                continue
            dvals = sorted(d for _, _, d in candidates)
            median_d = dvals[len(dvals) // 2]
            adaptive = max(median_d * RELATIVE_FACTOR, 1.0)
            group = [(cx, cy) for cx, cy, d in candidates if d <= adaptive]
            if len(group) >= MIN_ALIGNED and len(group) > len(zebra_group):
                zebra_group = group

    detected = len(zebra_group) >= MIN_ALIGNED

    vis = undist.copy()
    for cx, cy in centroids:
        cv2.circle(vis, (cx, cy + crop_start), 5, (0, 0, 255), -1)

    if detected:
        gs = sorted(zebra_group, key=lambda p: p[0])
        p1 = (gs[0][0],  gs[0][1]  + crop_start)
        p2 = (gs[-1][0], gs[-1][1] + crop_start)

        # Centro (x,y) del cruce en coordenadas de imagen completa
        center_x = sum(p[0] for p in gs) / len(gs)
        center_y = sum(p[1] for p in gs) / len(gs) + crop_start
        # Angulo de la recta de alineacion (grados)
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

        for cx, cy in zebra_group:
            cv2.circle(vis, (cx, cy + crop_start), 8, (0, 255, 255), 2)
        cv2.line(vis, p1, p2, (0, 255, 255), 2)
        cv2.circle(vis, (int(center_x), int(center_y)), 6, (255, 0, 255), -1)

        text1 = f"ZEBRA x={center_x:.0f} y={center_y:.0f}"
        text2 = f"angulo={angle:+.1f} deg"
        cv2.putText(vis, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(f"[ZEBRA] centro=({center_x:.0f},{center_y:.0f})  angulo={angle:+.1f} deg")
    else:
        cv2.putText(vis, "sin zebra", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return detected, vis


# ============================================================
# Pipeline de deteccion de linea
# ============================================================
def find_line(frame, debug=True):
    """Detecta la linea sobre la imagen dedistorsionada.

    Retorna (detected, position, angle, vis):
        detected -- bool
        position -- float en [-1, +1] (izquierda<0, centro=0, derecha>0)
        angle    -- grados respecto a la vertical (positivo = inclinada a la derecha)
        vis      -- frame anotado (BGR)
    Si debug=True muestra las ventanas de cada paso del pipeline."""
    # Redimensionar a la resolucion de calibracion para que K sea valida
    if (frame.shape[1], frame.shape[0]) != CALIB_SIZE:
        frame = cv2.resize(frame, CALIB_SIZE, interpolation=cv2.INTER_AREA)

    undist, vis1                       = step1_undistort(frame)
    (roi_img, x_start, y_start), vis2  = line_step2_crop(undist)
    (mask, contours), vis3             = line_step3_threshold(roi_img)
    (detected, position, angle), vis4  = line_step4_detect(undist, mask, contours, x_start, y_start)

    if debug:
        steps = [
            ("L1) Dedistorsion", vis1),
            ("L2) Recorte linea", vis2),
            ("L3) Umbral + morfologia", vis3),
            ("L4) Deteccion linea", vis4),
        ]
        for name, img in steps:
            if not ONLY_LAST_WINDOW or name in LAST_WINDOWS:
                show(name, img)

    return detected, position, angle, vis4


def line_step2_crop(undist):
    """L2) Recorte: 25% inferior y 75% central horizontal.
    Retorna ((roi_img, x_start, y_start), frame_anotado)."""
    h, w = undist.shape[:2]
    y_start = int(h * LINE_V_CROP)
    crop_w = LINE_H_CROP * w
    x_start = int((w - crop_w) // 2)
    x_end = int((w + crop_w) // 2)
    roi_img = undist[y_start:h, x_start:x_end].copy()
    return (roi_img, x_start, y_start), roi_img.copy()


def line_step3_threshold(roi_img):
    """L3) Umbral adaptativo + morfologia + mayor contorno.
    Retorna ((mask, contornos_validos), frame_anotado)."""
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=501, C=10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.erode(binary, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Aislar el contorno mas grande
    all_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = binary * 0
    if all_contours:
        biggest = max(all_contours, key=cv2.contourArea)
        cv2.drawContours(mask, [biggest], -1, 255, cv2.FILLED)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_LINE_AREA]

    dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dbg, contours, -1, (0, 255, 0), 1)
    return (mask, contours), dbg


def line_step4_detect(undist, mask, contours, x_start, y_start):
    """L4) Estimar posicion y angulo de la linea con centros por fila.
    Retorna ((detected, position, angle), frame_anotado)."""
    h, w = undist.shape[:2]
    detected = False
    position = 0.0
    angle = 0.0
    cx = w // 2

    vis = undist.copy()

    if contours:
        best = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(best))

        # Mascara solo del mayor contorno (descarta ruido de otros blobs)
        line_mask = mask * 0
        cv2.drawContours(line_mask, [best], -1, 255, cv2.FILLED)

        # Centro horizontal de la linea en cada fila con pixeles blancos
        ys, xs = [], []
        roi_h = line_mask.shape[0]
        for row in range(roi_h):
            cols = np.where(line_mask[row] > 0)[0]
            if cols.size > 0:
                xs.append(cols.mean())
                ys.append(row)

        if len(xs) >= 2:
            detected = True
            xs = np.array(xs, dtype=np.float32)
            ys = np.array(ys, dtype=np.float32)

            # Posicion: centro en la fila inferior del recorte (mas cercana al robot)
            cx_crop = xs[-1]
            cx = int(cx_crop + x_start)
            position = (cx / (w / 2)) - 1.0

            # Angulo: ajustar recta a los centros (x = f(y)).
            # pendiente dx/dy -> angulo respecto a la vertical.
            pts = np.column_stack((xs, ys)).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            if vy < 0:                 # orientar el vector hacia abajo
                vx, vy = -vx, -vy
            angle = math.degrees(math.atan2(vx, vy))   # 0=vertical, + a la derecha

            # Dibujar la recta ajustada (en coords de imagen completa)
            x0i = x0 + x_start
            y0i = y0 + y_start
            length = roi_h
            p1 = (int(x0i - vx * length), int(y0i - vy * length))
            p2 = (int(x0i + vx * length), int(y0i + vy * length))
            cv2.line(vis, p1, p2, (0, 255, 255), 2)

            cy = int(ys[-1] + y_start)
            cv2.circle(vis, (cx, cy), 6, (255, 0, 255), -1)

            text1 = f"LINEA pos={position:+.2f}"
            text2 = f"angulo={angle:+.1f} deg"
            cv2.putText(vis, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"[LINEA] pos={position:+.2f}  angulo={angle:+.1f} deg  area={area}")

    if not detected:
        cv2.putText(vis, "sin linea", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Lineas guia del recorte
    cv2.line(vis, (0, y_start), (w, y_start), (0, 255, 0), 1)
    cv2.line(vis, (w // 2, y_start), (w // 2, h), (255, 0, 0), 1)
    cv2.line(vis, (x_start, y_start), (x_start, h), (255, 255, 0), 1)
    cv2.line(vis, (w - x_start, y_start), (w - x_start, h), (255, 255, 0), 1)

    return (detected, position, angle), vis
