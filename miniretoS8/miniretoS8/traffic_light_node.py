import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


WINDOW_TITLE = 'Traffic Light ULTRA FAR'

BGR = {
    'rojo': (0, 0, 255),
    'amarillo': (0, 255, 255),
    'verde': (0, 255, 0),
    'none': (180, 180, 180),
}

MULTIPLIER = {
    'rojo': 0.0,
    'amarillo': 0.5,
    'verde': 1.0,
    'none': 1.0,
}


@dataclass
class Candidate:
    color: str
    score: float
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    peak: float
    local_sum: float
    local_mean: float
    support_pixels: int
    dark_bonus: float
    housing_bonus: float
    mean_bgr: Tuple[float, float, float]
    mean_hsv: Tuple[float, float, float]


class TrafficLightNode(Node):
    """Detector de semáforo para distancias largas.

    Estrategia:
    1) No espera contornos grandes. Busca picos de color de 1 pixel o más.
    2) Usa puntajes suaves por color en vez de rangos HSV rígidos.
    3) Revisa evidencia local alrededor del punto y bonus de carcasa oscura.
    4) Usa confirmación temporal muy ligera para no perder detecciones lejanas.

    Para máxima distancia, publica la cámara en 1280x720 o 1920x1080.
    """

    def __init__(self):
        super().__init__('traffic_light_node')
        self._bridge = CvBridge()

        # ROI general. Conviene acotarlo en pista para evitar falsos positivos.
        self.declare_parameter('show_window', False)
        self.declare_parameter('roi_top_ratio', 0.00)
        self.declare_parameter('roi_bottom_ratio', 0.68)
        self.declare_parameter('roi_left_ratio', 0.00)
        self.declare_parameter('roi_right_ratio', 1.00)
        self.declare_parameter('process_scale', 1.0)
        self.declare_parameter('debug_every_n', 8)

        # ROI esperado dentro del ROI. Si no sabes dónde está, déjalo amplio.
        self.declare_parameter('expected_x_min', 0.00)
        self.declare_parameter('expected_x_max', 1.00)
        self.declare_parameter('expected_y_min', 0.00)
        self.declare_parameter('expected_y_max', 1.00)

        # Sensibilidad ULTRA FAR. Permite picos muy pequeños.
        self.declare_parameter('base_threshold_red', 0.020)
        self.declare_parameter('base_threshold_yellow', 0.025)
        self.declare_parameter('base_threshold_green', 0.020)
        self.declare_parameter('peak_fraction', 0.38)
        self.declare_parameter('min_peak', 0.055)
        self.declare_parameter('min_final_score', 0.42)
        self.declare_parameter('local_radius', 9)
        self.declare_parameter('housing_radius', 18)
        self.declare_parameter('max_candidates_per_color', 10)
        self.declare_parameter('nms_radius', 9)
        self.declare_parameter('score_blur_ksize', 3)

        # Si hay varios puntitos, rojo gana por seguridad.
        self.declare_parameter('red_priority_bonus', 1.50)
        self.declare_parameter('yellow_priority_bonus', 1.10)
        self.declare_parameter('green_priority_bonus', 1.00)

        # Control temporal.
        self.declare_parameter('confirm_frames', 1)
        self.declare_parameter('green_unlock_frames', 1)
        self.declare_parameter('lost_hold_sec', 1.20)
        self.declare_parameter('yellow_hold_sec', 0.80)
        self.declare_parameter('red_lock_enabled', True)

        # Cuando no hay detección, publicar esto. En carrera normalmente 1.0.
        # Para calibrar puedes usar 0.2 para notar visualmente cuando no detecta.
        self.declare_parameter('default_multiplier', 1.0)

        self._show_window = bool(self.get_parameter('show_window').value)
        self._roi_top_ratio = float(self.get_parameter('roi_top_ratio').value)
        self._roi_bottom_ratio = float(self.get_parameter('roi_bottom_ratio').value)
        self._roi_left_ratio = float(self.get_parameter('roi_left_ratio').value)
        self._roi_right_ratio = float(self.get_parameter('roi_right_ratio').value)
        self._process_scale = float(self.get_parameter('process_scale').value)
        self._debug_every_n = max(1, int(self.get_parameter('debug_every_n').value))

        self._expected_x_min = float(self.get_parameter('expected_x_min').value)
        self._expected_x_max = float(self.get_parameter('expected_x_max').value)
        self._expected_y_min = float(self.get_parameter('expected_y_min').value)
        self._expected_y_max = float(self.get_parameter('expected_y_max').value)

        self._base_threshold = {
            'rojo': float(self.get_parameter('base_threshold_red').value),
            'amarillo': float(self.get_parameter('base_threshold_yellow').value),
            'verde': float(self.get_parameter('base_threshold_green').value),
        }
        self._peak_fraction = float(self.get_parameter('peak_fraction').value)
        self._min_peak = float(self.get_parameter('min_peak').value)
        self._min_final_score = float(self.get_parameter('min_final_score').value)
        self._local_radius = int(self.get_parameter('local_radius').value)
        self._housing_radius = int(self.get_parameter('housing_radius').value)
        self._max_candidates_per_color = int(self.get_parameter('max_candidates_per_color').value)
        self._nms_radius = int(self.get_parameter('nms_radius').value)
        self._score_blur_ksize = int(self.get_parameter('score_blur_ksize').value)

        self._priority_bonus = {
            'rojo': float(self.get_parameter('red_priority_bonus').value),
            'amarillo': float(self.get_parameter('yellow_priority_bonus').value),
            'verde': float(self.get_parameter('green_priority_bonus').value),
        }

        self._confirm_frames = int(self.get_parameter('confirm_frames').value)
        self._green_unlock_frames = int(self.get_parameter('green_unlock_frames').value)
        self._lost_hold_sec = float(self.get_parameter('lost_hold_sec').value)
        self._yellow_hold_sec = float(self.get_parameter('yellow_hold_sec').value)
        self._red_lock_enabled = bool(self.get_parameter('red_lock_enabled').value)
        self._default_multiplier = float(self.get_parameter('default_multiplier').value)

        self._streak = {'rojo': 0, 'amarillo': 0, 'verde': 0}
        self._red_locked = False
        self._last_confirmed_color = 'none'
        self._last_confirmed_time = None
        self._last_multiplier = self._default_multiplier
        self._frame_count = 0
        self._last_best: Optional[Candidate] = None
        self._last_max_scores = {'rojo': 0.0, 'amarillo': 0.0, 'verde': 0.0}

        self._speed_pub = self.create_publisher(Float32, '/speed_multiplier', 10)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(Image, '/camera/image_raw', self._image_callback, qos)

        if self._show_window:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

        self.get_logger().info('TrafficLightNode ULTRA FAR listo')

    def _seconds_since(self, stamp) -> float:
        if stamp is None:
            return 999.0
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    @staticmethod
    def _hue_closeness(h: np.ndarray, center: float, width: float) -> np.ndarray:
        d = np.abs(h.astype(np.float32) - center)
        d = np.minimum(d, 180.0 - d)
        return np.clip(1.0 - d / max(width, 1e-6), 0.0, 1.0)

    def _score_maps(self, roi: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        bgr = roi.astype(np.float32) / 255.0
        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0

        # Componentes relativos. Esto ayuda cuando el foco lejano se ve como una mancha
        # desaturada/borrosa y no cae perfecto dentro de un rango HSV clásico.
        max_bg = np.maximum(b, g)
        max_rb = np.maximum(r, b)
        max_rg = np.maximum(r, g)

        red_hue = np.maximum(self._hue_closeness(h, 0.0, 34.0), self._hue_closeness(h, 179.0, 34.0))
        red_magenta = 0.75 * self._hue_closeness(h, 158.0, 34.0)
        red_dom = np.clip((r - 0.48 * g - 0.35 * b) / 0.42, 0.0, 1.0)
        red_soft = np.clip((r - 0.38) / 0.45, 0.0, 1.0) * np.clip((r - 0.70 * max_bg + 0.15) / 0.50, 0.0, 1.0)
        red_score = np.maximum.reduce([
            v * s * red_hue,
            v * s * red_magenta,
            v * red_dom,
            red_soft,
        ])

        yellow_hue = self._hue_closeness(h, 28.0, 28.0)
        yellow_dom = np.clip((np.minimum(r, g) - 0.58 * b) / 0.42, 0.0, 1.0)
        yellow_balance = np.clip(1.0 - np.abs(r - g) / 0.65, 0.0, 1.0)
        yellow_soft = np.clip((np.minimum(r, g) - 0.35) / 0.45, 0.0, 1.0) * yellow_balance
        yellow_score = np.maximum.reduce([
            v * s * yellow_hue,
            v * yellow_dom * yellow_balance,
            yellow_soft,
        ])

        green_hue = self._hue_closeness(h, 62.0, 44.0)
        green_dom = np.clip((g - 0.48 * r - 0.35 * b) / 0.42, 0.0, 1.0)
        green_soft = np.clip((g - 0.34) / 0.45, 0.0, 1.0) * np.clip((g - 0.68 * max_rb + 0.15) / 0.50, 0.0, 1.0)
        green_score = np.maximum.reduce([
            v * s * green_hue,
            v * green_dom,
            green_soft,
        ])

        scores = {
            'rojo': red_score.astype(np.float32),
            'amarillo': yellow_score.astype(np.float32),
            'verde': green_score.astype(np.float32),
        }

        k = max(1, int(self._score_blur_ksize))
        if k > 1:
            if k % 2 == 0:
                k += 1
            for color in scores:
                scores[color] = cv2.GaussianBlur(scores[color], (k, k), 0)

        return scores, hsv

    def _housing_bonus(self, roi: np.ndarray, cx: int, cy: int) -> Tuple[float, float]:
        h, w = roi.shape[:2]
        r = max(3, self._housing_radius)
        x0 = max(0, cx - 3 * r)
        x1 = min(w, cx + 3 * r + 1)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            return 1.0, 1.0

        patch = roi[y0:y1, x0:x1]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.mean(gray < 85))

        # Bonus por carcasa oscura alrededor.
        dark_bonus = 1.0 + 0.80 * min(1.0, dark_ratio / 0.34)

        # Bonus extra si hay negro a izquierda/derecha del foco. En semáforos miniatura
        # suele verse la caja negra aunque la luz sea pequeña.
        yy0 = max(0, cy - max(2, r // 2))
        yy1 = min(h, cy + max(2, r // 2) + 1)
        left = roi[yy0:yy1, max(0, cx - 4 * r):max(0, cx - r)]
        right = roi[yy0:yy1, min(w, cx + r):min(w, cx + 4 * r)]
        side_dark = 0.0
        vals = []
        for side in (left, right):
            if side.size > 0:
                vals.append(float(np.mean(cv2.cvtColor(side, cv2.COLOR_BGR2GRAY) < 90)))
        if vals:
            side_dark = sum(vals) / len(vals)
        housing_bonus = 1.0 + 0.60 * min(1.0, side_dark / 0.30)
        return dark_bonus, housing_bonus

    def _local_candidate(
        self,
        color: str,
        score: np.ndarray,
        roi: np.ndarray,
        hsv: np.ndarray,
        cx: int,
        cy: int,
    ) -> Optional[Candidate]:
        roi_h, roi_w = score.shape[:2]
        if roi_w <= 0 or roi_h <= 0:
            return None

        xn = cx / float(max(1, roi_w - 1))
        yn = cy / float(max(1, roi_h - 1))
        if not (self._expected_x_min <= xn <= self._expected_x_max):
            return None
        if not (self._expected_y_min <= yn <= self._expected_y_max):
            return None

        r = max(2, self._local_radius)
        x0 = max(0, cx - r)
        x1 = min(roi_w, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(roi_h, cy + r + 1)
        local = score[y0:y1, x0:x1]
        if local.size == 0:
            return None

        peak = float(np.max(local))
        if peak < self._min_peak:
            return None

        local_sum = float(np.sum(local))
        local_mean = float(np.mean(local))
        thr = max(self._base_threshold[color], peak * self._peak_fraction)
        support = (local >= thr).astype(np.uint8)
        support_pixels = int(np.count_nonzero(support))
        if support_pixels < 1:
            return None

        # bbox local del soporte; no se descarta si es 1 pixel.
        ys, xs = np.where(support > 0)
        bx0 = int(x0 + xs.min())
        by0 = int(y0 + ys.min())
        bx1 = int(x0 + xs.max())
        by1 = int(y0 + ys.max())
        bw = max(1, bx1 - bx0 + 1)
        bh = max(1, by1 - by0 + 1)

        # Rechazar líneas largas de ruido, pero permitir 1x1, 1x2, 2x1.
        aspect = bw / float(max(1, bh))
        if support_pixels >= 4 and (aspect > 7.0 or aspect < 0.14):
            return None

        mask = np.zeros(score.shape, dtype=np.uint8)
        mask[y0:y1, x0:x1] = support
        mean_bgr = cv2.mean(roi, mask=mask)[:3]
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]

        dark_bonus, housing_bonus = self._housing_bonus(roi, cx, cy)
        priority = self._priority_bonus[color]

        # Score diseñado para que un único pixel fuerte pueda pasar,
        # pero un área local real gane sobre ruido aislado.
        final_score = priority * dark_bonus * housing_bonus * (
            2.40 * peak
            + 0.45 * math.sqrt(max(0.0, local_sum))
            + 0.18 * support_pixels
            + 0.55 * local_mean
        )

        if final_score < self._min_final_score:
            return None

        return Candidate(
            color=color,
            score=final_score,
            center=(cx, cy),
            bbox=(bx0, by0, bw, bh),
            peak=peak,
            local_sum=local_sum,
            local_mean=local_mean,
            support_pixels=support_pixels,
            dark_bonus=dark_bonus,
            housing_bonus=housing_bonus,
            mean_bgr=mean_bgr,
            mean_hsv=mean_hsv,
        )

    def _peak_points(self, score: np.ndarray, color: str) -> List[Tuple[int, int]]:
        roi_h, roi_w = score.shape[:2]
        if roi_h == 0 or roi_w == 0:
            return []

        max_score = float(np.max(score))
        self._last_max_scores[color] = max_score
        threshold = max(self._base_threshold[color], max_score * self._peak_fraction, self._min_peak)
        if max_score < self._min_peak:
            return []

        # Máximos locales por dilatación.
        nms = max(3, self._nms_radius)
        if nms % 2 == 0:
            nms += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (nms, nms))
        dilated = cv2.dilate(score, kernel)
        maxima = (score >= threshold) & (score >= dilated - 1e-6)
        ys, xs = np.where(maxima)
        if len(xs) == 0:
            return []

        pts = [(int(x), int(y), float(score[y, x])) for x, y in zip(xs, ys)]
        pts.sort(key=lambda p: p[2], reverse=True)
        pts = pts[:max(1, self._max_candidates_per_color)]
        return [(x, y) for x, y, _ in pts]

    def _detect(self, roi: np.ndarray) -> Tuple[Optional[Candidate], List[Candidate], Dict[str, np.ndarray]]:
        scores, hsv = self._score_maps(roi)
        candidates: List[Candidate] = []

        for color in ('rojo', 'amarillo', 'verde'):
            for cx, cy in self._peak_points(scores[color], color):
                cand = self._local_candidate(color, scores[color], roi, hsv, cx, cy)
                if cand is not None:
                    candidates.append(cand)

        if not candidates:
            return None, [], scores

        def rank(c: Candidate) -> float:
            temporal = 1.0 + 0.16 * min(5, self._streak.get(c.color, 0))
            return c.score * temporal

        candidates.sort(key=rank, reverse=True)
        return candidates[0], candidates, scores

    def _update_streaks(self, detected_color: Optional[str]):
        for color in self._streak:
            self._streak[color] = self._streak[color] + 1 if detected_color == color else 0

    def _confirmed_color(self, detected_color: Optional[str]) -> Optional[str]:
        if detected_color is None:
            return None
        if self._streak.get(detected_color, 0) >= self._confirm_frames:
            return detected_color
        return None

    def _decide_multiplier(self, confirmed_color: Optional[str]) -> Tuple[float, str]:
        now = self.get_clock().now()

        if confirmed_color is not None:
            self._last_confirmed_color = confirmed_color
            self._last_confirmed_time = now

            if confirmed_color == 'rojo' and self._red_lock_enabled:
                self._red_locked = True
            elif confirmed_color == 'verde' and self._streak['verde'] >= self._green_unlock_frames:
                self._red_locked = False

            if self._red_locked:
                self._last_multiplier = 0.0
                return 0.0, f'{confirmed_color} LOCKED'

            self._last_multiplier = MULTIPLIER[confirmed_color]
            return self._last_multiplier, confirmed_color

        elapsed = self._seconds_since(self._last_confirmed_time)

        if self._red_locked:
            return 0.0, 'rojo LOCKED/sin vista'

        if self._last_confirmed_color == 'amarillo' and elapsed < self._yellow_hold_sec:
            return 0.5, 'amarillo HOLD'

        if elapsed < self._lost_hold_sec and self._last_confirmed_color in ('verde', 'amarillo'):
            return self._last_multiplier, f'{self._last_confirmed_color} HOLD'

        self._last_confirmed_color = 'none'
        self._last_multiplier = self._default_multiplier
        return self._default_multiplier, 'none'

    def _image_callback(self, msg: Image):
        self._frame_count += 1
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self._show_window and cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
            raise SystemExit

        if self._process_scale > 0 and abs(self._process_scale - 1.0) > 1e-3:
            proc = cv2.resize(frame, None, fx=self._process_scale, fy=self._process_scale,
                              interpolation=cv2.INTER_LINEAR)
        else:
            proc = frame

        h, w = proc.shape[:2]
        y0 = int(h * self._roi_top_ratio)
        y1 = int(h * self._roi_bottom_ratio)
        x0 = int(w * self._roi_left_ratio)
        x1 = int(w * self._roi_right_ratio)
        y0, y1 = max(0, y0), min(h, max(y0 + 1, y1))
        x0, x1 = max(0, x0), min(w, max(x0 + 1, x1))
        roi = proc[y0:y1, x0:x1].copy()

        best, candidates, _ = self._detect(roi)
        self._last_best = best
        detected_color = best.color if best is not None else None
        self._update_streaks(detected_color)
        confirmed = self._confirmed_color(detected_color)
        multiplier, state_label = self._decide_multiplier(confirmed)

        msg_out = Float32()
        msg_out.data = float(multiplier)
        self._speed_pub.publish(msg_out)

        should_log = (self._frame_count % self._debug_every_n == 0) or best is not None
        if should_log:
            if best is not None:
                self.get_logger().info(
                    f'TL raw={best.color} confirmed={confirmed or "-"} state={state_label} '
                    f'vel={multiplier:.1f} score={best.score:.2f} peak={best.peak:.3f} '
                    f'sum={best.local_sum:.2f} px={best.support_pixels} '
                    f'dark={best.dark_bonus:.2f} house={best.housing_bonus:.2f} '
                    f'center={best.center} bbox={best.bbox} streak={dict(self._streak)} '
                    f'max={self._last_max_scores}'
                )
            else:
                self.get_logger().info(
                    f'TL none vel={multiplier:.1f} state={state_label} '
                    f'max={self._last_max_scores} streak={dict(self._streak)}'
                )

        if self._show_window:
            display = proc.copy()
            cv2.rectangle(display, (x0, y0), (x1, y1), (255, 255, 0), 1)

            for cand in candidates[:15]:
                x, y, bw, bh = cand.bbox
                cx, cy = cand.center
                cv2.rectangle(display, (x0 + x, y0 + y), (x0 + x + bw, y0 + y + bh), BGR[cand.color], 1)
                cv2.circle(display, (x0 + cx, y0 + cy), 8, BGR[cand.color], 1)

            if best is not None:
                x, y, bw, bh = best.bbox
                cx, cy = best.center
                cv2.rectangle(display, (x0 + x, y0 + y), (x0 + x + bw, y0 + y + bh), BGR[best.color], 3)
                cv2.circle(display, (x0 + cx, y0 + cy), 13, BGR[best.color], 2)
                cv2.putText(display, f'{best.color} s={best.score:.1f} p={best.peak:.2f}',
                            (x0 + max(0, x - 12), max(22, y0 + y - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, BGR[best.color], 2)

            cv2.putText(display, f'TL: {state_label} vel={multiplier:.1f}',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        BGR.get(self._last_confirmed_color, BGR['none']), 2)
            cv2.putText(display,
                        f'max R/Y/G: {self._last_max_scores["rojo"]:.2f}/'
                        f'{self._last_max_scores["amarillo"]:.2f}/'
                        f'{self._last_max_scores["verde"]:.2f}',
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                raise SystemExit

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    