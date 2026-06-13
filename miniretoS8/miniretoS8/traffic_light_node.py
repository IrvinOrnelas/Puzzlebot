import cv2
import numpy as np
import rclpy
from typing import Optional
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

WINDOW_TITLE = 'Traffic Light'

MULTIPLIER = {'rojo': 0.0, 'amarillo': 0.5, 'verde': 1.0, 'none': 1.0}
BGR_COLOR = {
    'rojo': (0, 0, 255),
    'amarillo': (0, 255, 255),
    'verde': (0, 255, 0),
    'none': (180, 180, 180),
}


class TrafficLightNode(Node):
    """Detector de semáforo basado en máscara HSV + contornos.

    Solo procesa cuando recibe /at_zebra=True desde line_follower_node.
    El contorno con mayor área válida determina el color detectado.
    Se filtra por aspect ratio para distinguir luces circulares de señales.
    """

    def __init__(self):
        super().__init__('traffic_light_node')
        self._bridge = CvBridge()

        self.declare_parameter('show_window', True)
        self.declare_parameter('process_scale', 1.0)

        # ROI del frame donde aparece el semáforo
        self.declare_parameter('roi_top_ratio', 0.00)
        self.declare_parameter('roi_bottom_ratio', 0.55)
        self.declare_parameter('roi_left_ratio', 0.00)
        self.declare_parameter('roi_right_ratio', 0.50)

        # Umbrales HSV — rojo usa dos rangos de tono (envuelve en 180)
        self.declare_parameter('red_h_hi1', 10)
        self.declare_parameter('red_h_lo2', 165)
        self.declare_parameter('red_s_min', 100)
        self.declare_parameter('red_v_min', 80)

        self.declare_parameter('yellow_h_lo', 15)
        self.declare_parameter('yellow_h_hi', 38)
        self.declare_parameter('yellow_s_min', 100)
        self.declare_parameter('yellow_v_min', 80)

        self.declare_parameter('green_h_lo', 38)
        self.declare_parameter('green_h_hi', 90)
        self.declare_parameter('green_s_min', 80)
        self.declare_parameter('green_v_min', 80)

        # Morfología y contornos
        self.declare_parameter('dilate_kernel', 5)
        self.declare_parameter('dilate_iter', 2)
        self.declare_parameter('min_contour_area', 60.0)
        # Aspect ratio mínimo del bbox (w/h o h/w), filtra líneas y ruido alargado
        self.declare_parameter('min_aspect_ratio', 0.35)

        # Confirmación temporal
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('green_unlock_frames', 2)
        self.declare_parameter('lost_hold_sec', 1.5)
        self.declare_parameter('red_lock_enabled', True)
        self.declare_parameter('default_multiplier', 1.0)

        self._show_window = bool(self.get_parameter('show_window').value)
        self._proc_scale = float(self.get_parameter('process_scale').value)
        self._roi_top = float(self.get_parameter('roi_top_ratio').value)
        self._roi_bottom = float(self.get_parameter('roi_bottom_ratio').value)
        self._roi_left = float(self.get_parameter('roi_left_ratio').value)
        self._roi_right = float(self.get_parameter('roi_right_ratio').value)

        self._red_h_hi1 = int(self.get_parameter('red_h_hi1').value)
        self._red_h_lo2 = int(self.get_parameter('red_h_lo2').value)
        self._red_s_min = int(self.get_parameter('red_s_min').value)
        self._red_v_min = int(self.get_parameter('red_v_min').value)
        self._yellow_h_lo = int(self.get_parameter('yellow_h_lo').value)
        self._yellow_h_hi = int(self.get_parameter('yellow_h_hi').value)
        self._yellow_s_min = int(self.get_parameter('yellow_s_min').value)
        self._yellow_v_min = int(self.get_parameter('yellow_v_min').value)
        self._green_h_lo = int(self.get_parameter('green_h_lo').value)
        self._green_h_hi = int(self.get_parameter('green_h_hi').value)
        self._green_s_min = int(self.get_parameter('green_s_min').value)
        self._green_v_min = int(self.get_parameter('green_v_min').value)

        self._dilate_k = int(self.get_parameter('dilate_kernel').value)
        self._dilate_iter = int(self.get_parameter('dilate_iter').value)
        self._min_area = float(self.get_parameter('min_contour_area').value)
        self._min_aspect = float(self.get_parameter('min_aspect_ratio').value)

        self._confirm_frames = int(self.get_parameter('confirm_frames').value)
        self._green_unlock_frames = int(self.get_parameter('green_unlock_frames').value)
        self._lost_hold_sec = float(self.get_parameter('lost_hold_sec').value)
        self._red_lock_enabled = bool(self.get_parameter('red_lock_enabled').value)
        self._default_mult = float(self.get_parameter('default_multiplier').value)

        self._at_zebra = False
        self._streak = {'rojo': 0, 'amarillo': 0, 'verde': 0}
        self._red_locked = False
        self._last_confirmed = 'none'
        self._last_confirmed_time = None
        self._last_mult = self._default_mult

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._speed_pub = self.create_publisher(Float32, '/speed_multiplier', 10)
        self._sub = self.create_subscription(Image, '/camera/image_raw', self._image_callback, qos)
        self._zebra_sub = self.create_subscription(Bool, '/at_zebra', self._zebra_callback, 10)

        if self._show_window:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

        self.get_logger().info('TrafficLightNode (contour) listo — activo solo en zebra')

    # ------------------------------------------------------------------
    def _seconds_since(self, stamp) -> float:
        if stamp is None:
            return 999.0
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def _zebra_callback(self, msg: Bool):
        self._at_zebra = bool(msg.data)

    # ------------------------------------------------------------------
    def _build_masks(self, hsv: np.ndarray):
        k = max(1, self._dilate_k)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        r1 = cv2.inRange(hsv, (0, self._red_s_min, self._red_v_min),
                         (self._red_h_hi1, 255, 255))
        r2 = cv2.inRange(hsv, (self._red_h_lo2, self._red_s_min, self._red_v_min),
                         (180, 255, 255))
        red = cv2.dilate(cv2.bitwise_or(r1, r2), kernel, iterations=self._dilate_iter)

        yellow = cv2.dilate(
            cv2.inRange(hsv, (self._yellow_h_lo, self._yellow_s_min, self._yellow_v_min),
                        (self._yellow_h_hi, 255, 255)),
            kernel, iterations=self._dilate_iter)

        green = cv2.dilate(
            cv2.inRange(hsv, (self._green_h_lo, self._green_s_min, self._green_v_min),
                        (self._green_h_hi, 255, 255)),
            kernel, iterations=self._dilate_iter)

        return red, yellow, green

    def _best_contour(self, mask: np.ndarray):
        """Devuelve (area, contorno) del contorno más grande con forma no alargada."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area, best_cnt = 0.0, None
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue
            _, _, w, h = cv2.boundingRect(cnt)
            aspect = min(w, h) / float(max(w, h, 1))
            if aspect < self._min_aspect:
                continue
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_area, best_cnt

    # ------------------------------------------------------------------
    def _update_state(self, detected: Optional[str]):
        now = self.get_clock().now()

        for c in self._streak:
            self._streak[c] = self._streak[c] + 1 if c == detected else 0

        confirmed = None
        if detected and self._streak[detected] >= self._confirm_frames:
            confirmed = detected

        if confirmed:
            self._last_confirmed = confirmed
            self._last_confirmed_time = now
            if confirmed == 'rojo' and self._red_lock_enabled:
                self._red_locked = True
            elif confirmed == 'verde' and self._streak['verde'] >= self._green_unlock_frames:
                self._red_locked = False
            elif confirmed == 'amarillo' and self._streak['amarillo'] >= 3:
                self._red_locked = False

            if self._red_locked:
                return 0.0, f'{confirmed} LOCKED'
            self._last_mult = MULTIPLIER[confirmed]
            return self._last_mult, confirmed

        elapsed = self._seconds_since(self._last_confirmed_time)
        if self._red_locked:
            return 0.0, 'rojo LOCKED'
        if elapsed < self._lost_hold_sec and self._last_confirmed != 'none':
            return self._last_mult, f'{self._last_confirmed} HOLD'

        self._last_confirmed = 'none'
        self._last_mult = self._default_mult
        return self._default_mult, 'none'

    # ------------------------------------------------------------------
    def _image_callback(self, msg: Image):
        if not self._at_zebra:
            # Fuera de zebra: publicar default y resetear estado temporal
            self._streak = {'rojo': 0, 'amarillo': 0, 'verde': 0}
            out = Float32()
            out.data = float(self._default_mult)
            self._speed_pub.publish(out)
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if abs(self._proc_scale - 1.0) > 1e-3:
            frame = cv2.resize(frame, None, fx=self._proc_scale, fy=self._proc_scale,
                               interpolation=cv2.INTER_LINEAR)

        fh, fw = frame.shape[:2]
        y0 = int(fh * self._roi_top)
        y1 = int(fh * self._roi_bottom)
        x0 = int(fw * self._roi_left)
        x1 = int(fw * self._roi_right)
        roi = frame[y0:y1, x0:x1]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask, yellow_mask, green_mask = self._build_masks(hsv)

        red_area, red_cnt = self._best_contour(red_mask)
        yellow_area, yellow_cnt = self._best_contour(yellow_mask)
        green_area, green_cnt = self._best_contour(green_mask)

        areas = {'rojo': red_area, 'amarillo': yellow_area, 'verde': green_area}
        detected = max(areas, key=areas.get) if max(areas.values()) > 0 else None

        mult, label = self._update_state(detected)

        out = Float32()
        out.data = float(mult)
        self._speed_pub.publish(out)

        self.get_logger().info(
            f'TL det={detected or "-"} state={label} vel={mult:.1f} '
            f'R/Y/G={red_area:.0f}/{yellow_area:.0f}/{green_area:.0f} '
            f'streak={dict(self._streak)}'
        )

        if self._show_window:
            display = roi.copy()
            for color, cnt in [('rojo', red_cnt), ('amarillo', yellow_cnt), ('verde', green_cnt)]:
                if cnt is not None:
                    cv2.drawContours(display, [cnt], -1, BGR_COLOR[color], 2)
            color_label = detected if detected else 'none'
            cv2.putText(display, f'{label} vel={mult:.1f}',
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        BGR_COLOR[color_label], 2)
            cv2.imshow(WINDOW_TITLE, display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
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
