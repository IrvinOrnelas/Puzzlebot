import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32

try:
    from miniretoS8.yolo import configure as configure_yolo
except Exception:  # compatibilidad si yolo.py viejo no tiene configure()
    configure_yolo = None

from miniretoS8.yolo import get_signs


WINDOW_TITLE = 'YOLO Ultra Far'
DIRECTIONAL = {1, 2, 3}
SIGN_NAMES = {
    0: 'none',
    1: 'left',
    2: 'right',
    3: 'forward',
    4: 'stop',
    5: 'yield',
    6: 'roadwork',
    7: 'roundabout',
}


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class ProcessorNode(Node):
    """
    Processor YOLO para detectar señales desde más lejos.

    Publica:
      /stop_sign        Float32  área relativa del stop
      /slow_sign        Bool     construcción / roadwork
      /giveway_sign     Bool     ceda el paso
      /roundabout_sign  Bool     rotonda
      /sign_command     Int32    0 nada, 1 izq, 2 der, 3 adelante
    """

    def __init__(self):
        super().__init__('processor_node')

        # Ventana/modelo/inferencia
        self.declare_parameter('show_window', False)
        self.declare_parameter('model_path', '')
        self.declare_parameter('imgsz', 640)  # Reducido de 960 para velocidad
        self.declare_parameter('yolo_conf', 0.25)  # Aumentado de 0.10 para reducir falsos positivos
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('max_det', 80)
        self.declare_parameter('augment', False)

        # Modo ultra-far
        self.declare_parameter('far_mode', False)  # Desactivado para velocidad
        self.declare_parameter('yolo_upscale', 1.0)  # Reducido de 2.0 (sin upscaling)
        self.declare_parameter('tile_mode', 'off')  # Desactivado para velocidad (no procesar parches)
        self.declare_parameter('inference_every_n', 1)  # Corre YOLO en cada frame para detectar mejor

        # Confirmación/hold. Para señales lejanas usamos hold largo para guardarlas antes de zebra.
        self.declare_parameter('min_conf', 0.20)  # Valor intermedio: detecta más señales sin falsas positivas
        self.declare_parameter('fast_confirm_conf', 0.45)  # Requiere confianza decente
        self.declare_parameter('confirm_frames', 2)  # 2 frames para confirmar
        self.declare_parameter('hold_sec', 1.20)
        self.declare_parameter('direction_hold_sec', 7.00)
        self.declare_parameter('stop_hold_sec', 0.80)
        # Confianza mínima exclusiva para STOP (más alta para evitar falsos del semáforo rojo)
        self.declare_parameter('stop_min_conf', 0.92)

        # ROI de señales: conservar más imagen que antes para detectar desde lejos.
        self.declare_parameter('roi_top_ratio', 0.00)
        self.declare_parameter('roi_bottom_ratio', 0.90)  # Captura más arriba para señales lejanas
        self.declare_parameter('roi_x_margin', 0.00)

        # Score. Antes el área pesaba mucho; para lejos usamos raíz del área para no castigar tanto boxes pequeñas.
        self.declare_parameter('favor_near', 10.0)  # Reduce penalización de señales lejanas
        self.declare_parameter('log_only_changes', True)

        self._show_window = bool(self.get_parameter('show_window').value)
        self._show_window = False  # Visualización desactivada: solo salida por terminal
        self._model_path = str(self.get_parameter('model_path').value).strip()
        self._imgsz = int(self.get_parameter('imgsz').value)
        self._yolo_conf = float(self.get_parameter('yolo_conf').value)
        self._iou = float(self.get_parameter('iou').value)
        self._max_det = int(self.get_parameter('max_det').value)
        self._augment = bool(self.get_parameter('augment').value)
        self._far_mode = bool(self.get_parameter('far_mode').value)
        self._yolo_upscale = float(self.get_parameter('yolo_upscale').value)
        self._tile_mode = str(self.get_parameter('tile_mode').value)
        self._inference_every_n = max(1, int(self.get_parameter('inference_every_n').value))
        self._min_conf = float(self.get_parameter('min_conf').value)
        self._fast_confirm_conf = float(self.get_parameter('fast_confirm_conf').value)
        self._confirm_frames = max(1, int(self.get_parameter('confirm_frames').value))
        self._hold_sec = float(self.get_parameter('hold_sec').value)
        self._direction_hold_sec = float(self.get_parameter('direction_hold_sec').value)
        self._stop_hold_sec = float(self.get_parameter('stop_hold_sec').value)
        self._stop_min_conf = float(self.get_parameter('stop_min_conf').value)
        self._roi_top_ratio = float(self.get_parameter('roi_top_ratio').value)
        self._roi_bottom_ratio = float(self.get_parameter('roi_bottom_ratio').value)
        self._roi_x_margin = float(self.get_parameter('roi_x_margin').value)
        self._favor_near = float(self.get_parameter('favor_near').value)
        self._log_only_changes = bool(self.get_parameter('log_only_changes').value)

        if configure_yolo is not None:
            try:
                configure_yolo(
                    model_path=self._model_path if self._model_path else None,
                    conf=self._yolo_conf,
                    imgsz=self._imgsz,
                    iou=self._iou,
                    max_det=self._max_det,
                    far_mode=self._far_mode,
                    upscale=self._yolo_upscale,
                    tile_mode=self._tile_mode,
                    augment=self._augment,
                )
            except TypeError:
                # yolo.py integrado viejo: al menos configurar lo básico.
                configure_yolo(
                    model_path=self._model_path if self._model_path else None,
                    conf=self._yolo_conf,
                    imgsz=self._imgsz,
                )
        elif self._model_path:
            self.get_logger().warn('Tu yolo.py no tiene configure(); reemplaza yolo.py por yolo_ultra_far.py para usar modelo/rangos ultra-far.')

        self._bridge = CvBridge()
        self._frame_count = 0
        self._sign_streak = {sid: 0 for sid in range(1, 8)}
        self._last_confirmed_time = {}
        self._last_score = {sid: 0.0 for sid in range(1, 8)}
        self._last_area_ratio = {sid: 0.0 for sid in range(1, 8)}
        self._last_conf = {sid: 0.0 for sid in range(1, 8)}
        self._last_name = {sid: '' for sid in range(1, 8)}
        self._last_logged_state = None

        self._at_zebra = False
        self._stop_pub = self.create_publisher(Float32, '/stop_sign', 10)
        self._slow_pub = self.create_publisher(Bool, '/slow_sign', 10)
        self._giveway_pub = self.create_publisher(Bool, '/giveway_sign', 10)
        self._sign_cmd_pub = self.create_publisher(Int32, '/sign_command', 10)
        self._roundabout_pub = self.create_publisher(Bool, '/roundabout_sign', 10)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(Image, '/camera/image_raw', self._image_callback, qos)
        self._zebra_sub = self.create_subscription(Bool, '/at_zebra', self._zebra_callback, 10)

        if self._show_window:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

        self.get_logger().info(
            f'YOLO ultra-far listo | yolo_conf={self._yolo_conf:.2f} min_conf={self._min_conf:.2f} '
            f'imgsz={self._imgsz} upscale={self._yolo_upscale:.1f} tile={self._tile_mode} '
            f'every={self._inference_every_n} hold_dir={self._direction_hold_sec:.1f}s'
        )

    def _zebra_callback(self, msg: Bool):
        self._at_zebra = bool(msg.data)

    def _seconds_since(self, stamp):
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def _hold_for(self, sign_type: int) -> float:
        if sign_type in DIRECTIONAL:
            return self._direction_hold_sec
        if sign_type == 4:
            return self._stop_hold_sec
        return self._hold_sec

    def _is_active(self, sign_type: int) -> bool:
        stamp = self._last_confirmed_time.get(sign_type)
        if stamp is None:
            return False
        return self._seconds_since(stamp) <= self._hold_for(sign_type)

    def _confirm_detections(self, boxes, sign_types, confidences, class_names, img_area: float):
        now = self.get_clock().now()
        detected_this_frame = set()
        best_by_type = {}

        for i, sid_raw in enumerate(sign_types):
            sid = int(sid_raw)
            conf = float(confidences[i])
            if conf < self._min_conf:
                continue
            # STOP (sid=4): ignorar en cruce de zebra (semáforo rojo confunde a YOLO)
            # y exigir confianza alta fuera de cruce
            if sid == 4:
                if self._at_zebra:
                    continue
                if conf < self._stop_min_conf:
                    continue

            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, img_area)
            # Para detección lejana NO multiplicar conf*area directo, porque boxes pequeñas desaparecerían.
            score = conf * (1.0 + self._favor_near * math.sqrt(max(0.0, area_ratio)))
            name = class_names[i] if i < len(class_names) else ''
            detected_this_frame.add(sid)

            if sid not in best_by_type or score > best_by_type[sid][0]:
                best_by_type[sid] = (score, conf, area_ratio, name)

        for sid in range(1, 8):
            if sid in detected_this_frame:
                self._sign_streak[sid] = self._sign_streak.get(sid, 0) + 1
            else:
                self._sign_streak[sid] = 0

        for sid, (score, conf, area_ratio, name) in best_by_type.items():
            confirmed = self._sign_streak.get(sid, 0) >= self._confirm_frames or conf >= self._fast_confirm_conf
            if confirmed:
                self._last_confirmed_time[sid] = now
                self._last_score[sid] = score
                self._last_area_ratio[sid] = area_ratio
                self._last_conf[sid] = conf
                self._last_name[sid] = str(name)

    def _publish_state(self, display=None):
        stop_ratio = self._last_area_ratio[4] if self._is_active(4) else 0.0
        stop_msg = Float32()
        stop_msg.data = float(stop_ratio)
        self._stop_pub.publish(stop_msg)

        slow_msg = Bool()
        slow_msg.data = bool(self._is_active(6))
        self._slow_pub.publish(slow_msg)

        giveway_msg = Bool()
        giveway_msg.data = bool(self._is_active(5))
        self._giveway_pub.publish(giveway_msg)

        roundabout_msg = Bool()
        roundabout_msg.data = bool(self._is_active(7))
        self._roundabout_pub.publish(roundabout_msg)

        active_dirs = [sid for sid in DIRECTIONAL if self._is_active(sid)]
        sign_cmd = max(active_dirs, key=lambda sid: self._last_score.get(sid, 0.0)) if active_dirs else 0
        cmd_msg = Int32()
        cmd_msg.data = int(sign_cmd)
        self._sign_cmd_pub.publish(cmd_msg)

        state = (
            round(stop_ratio, 5),
            slow_msg.data,
            giveway_msg.data,
            roundabout_msg.data,
            sign_cmd,
        )
        if (not self._log_only_changes) or state != self._last_logged_state:
            self._last_logged_state = state

            # Mostrar solo la mejor señal detectada en este momento
            best_signal = 'NONE'
            best_conf = 0.0

            # Encontrar la mejor señal activa
            active_signals = []
            if self._is_active(4):  # STOP
                active_signals.append(('STOP', self._last_conf[4]))
            if self._is_active(6):  # ROADWORK
                active_signals.append(('ROADWORK', self._last_conf[6]))
            if self._is_active(5):  # YIELD
                active_signals.append(('YIELD', self._last_conf[5]))
            if self._is_active(7):  # ROUNDABOUT
                active_signals.append(('ROUNDABOUT', self._last_conf[7]))
            if self._is_active(1):  # LEFT
                active_signals.append(('LEFT', self._last_conf[1]))
            if self._is_active(2):  # RIGHT
                active_signals.append(('RIGHT', self._last_conf[2]))
            if self._is_active(3):  # FORWARD
                active_signals.append(('FORWARD', self._last_conf[3]))

            if active_signals:
                best_signal, best_conf = max(active_signals, key=lambda x: x[1])

            self.get_logger().info(f'Señal: {best_signal} conf={best_conf:.2f}')

        if self._show_window and display is not None:
            text = (
                f'cmd={SIGN_NAMES.get(sign_cmd, sign_cmd)} '
                f'stop={stop_ratio:.4f} slow={slow_msg.data} yield={giveway_msg.data} round={roundabout_msg.data}'
            )
            cv2.putText(display, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                raise SystemExit

    def _image_callback(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        display = frame.copy() if self._show_window else None

        if self._show_window and cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
            raise SystemExit

        self._frame_count += 1
        run_inference = (self._frame_count % self._inference_every_n) == 0

        y0 = int(_clamp(self._roi_top_ratio, 0.0, 0.95) * h)
        y1 = int(_clamp(self._roi_bottom_ratio, 0.05, 1.0) * h)
        x0 = int(_clamp(self._roi_x_margin, 0.0, 0.45) * w)
        x1 = int((1.0 - _clamp(self._roi_x_margin, 0.0, 0.45)) * w)
        if y1 <= y0 + 5 or x1 <= x0 + 5:
            y0, y1, x0, x1 = 0, int(0.85 * h), 0, w

        if display is not None:
            cv2.rectangle(display, (x0, y0), (x1, y1), (255, 255, 0), 1)

        if run_inference:
            roi = frame[y0:y1, x0:x1]
            draw_roi = display[y0:y1, x0:x1] if display is not None else None
            boxes, sign_types, confidences, class_names = get_signs(roi, drawing_frame=draw_roi)

            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0

            self._confirm_detections(boxes, sign_types, confidences, class_names, float(h * w))

            if len(sign_types) > 0 and not self._log_only_changes:
                pretty = [f'{SIGN_NAMES.get(int(s), s)}:{float(c):.2f}' for s, c in zip(sign_types, confidences)]
                self.get_logger().info(f'Raw YOLO ultra-far: {pretty} streak={dict(self._sign_streak)}')

        self._publish_state(display)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ProcessorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()