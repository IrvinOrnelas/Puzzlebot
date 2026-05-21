import math

import cv2
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from miniretoS6.camera_node import CSICamera
from miniretoS6.line_detector import (
    detect_traffic_light,
    draw_traffic_light,
    find_line,
)


WINDOW_TITLE = 'Robust Line Follower'


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # Cámara
        self.declare_parameter('sensor_id', 0)
        self.declare_parameter('framerate', 30)
        self.declare_parameter('flip_method', 0)
        self.declare_parameter('capture_width', 1280)
        self.declare_parameter('capture_height', 720)
        self.declare_parameter('display_width', 640)
        self.declare_parameter('display_height', 360)

        # Control
        self.declare_parameter('kp', 1.5)
        self.declare_parameter('linear', 0.10)
        self.declare_parameter('max_w', 2.0)
        self.declare_parameter('min_confidence', 0.18)
        self.declare_parameter('direction_alpha', 0.35)  # suavizado, 0=mantiene, 1=sin filtro

        # Robustez ante pérdidas/zebra
        self.declare_parameter('lost_timeout', 0.35)
        self.declare_parameter('zebra_speed_scale', 0.55)
        self.declare_parameter('lost_speed_scale', 0.25)
        self.declare_parameter('stop_when_lost', True)

        # Semáforo
        self.declare_parameter('traffic_hold_sec', 1.0)
        self.declare_parameter('traffic_roi_y_end', 0.70)
        self.declare_parameter('traffic_min_area_ratio', 0.00065)

        # Visualización
        self.declare_parameter('show_window', True)
        self.declare_parameter('publish_debug_text', True)

        self.sensor_id = int(self.get_parameter('sensor_id').value)
        self.framerate = int(self.get_parameter('framerate').value)
        self.flip_method = int(self.get_parameter('flip_method').value)

        self.kp = float(self.get_parameter('kp').value)
        self.linear = float(self.get_parameter('linear').value)
        self.max_w = float(self.get_parameter('max_w').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.direction_alpha = float(self.get_parameter('direction_alpha').value)

        self.lost_timeout = float(self.get_parameter('lost_timeout').value)
        self.zebra_speed_scale = float(self.get_parameter('zebra_speed_scale').value)
        self.lost_speed_scale = float(self.get_parameter('lost_speed_scale').value)
        self.stop_when_lost = bool(self.get_parameter('stop_when_lost').value)

        self.traffic_hold_sec = float(self.get_parameter('traffic_hold_sec').value)
        self.traffic_roi_y_end = float(self.get_parameter('traffic_roi_y_end').value)
        self.traffic_min_area_ratio = float(self.get_parameter('traffic_min_area_ratio').value)

        self.show_window = bool(self.get_parameter('show_window').value)
        self.publish_debug_text = bool(self.get_parameter('publish_debug_text').value)

        self.camera = CSICamera(
            sensor_id=self.sensor_id,
            capture_width=int(self.get_parameter('capture_width').value),
            capture_height=int(self.get_parameter('capture_height').value),
            display_width=int(self.get_parameter('display_width').value),
            display_height=int(self.get_parameter('display_height').value),
            framerate=self.framerate,
            flip_method=self.flip_method,
            logger=self.get_logger(),
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Estado temporal
        self.filtered_direction = 0.0
        self.last_valid_direction = 0.0
        self.last_line_time = self.get_clock().now()

        self.traffic_multiplier = 1.0
        self.last_traffic_multiplier = 1.0
        self.last_traffic_color = 'none'
        self.last_traffic_time = None
        self.red_locked = False  # bloqueado hasta detectar verde

        if self.show_window:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

        self.timer = self.create_timer(1.0 / self.framerate, self.timer_callback)
        self.get_logger().info('LineFollower robusto iniciado. Publicando en /cmd_vel')

    def _seconds_since(self, stamp):
        now = self.get_clock().now()
        return (now - stamp).nanoseconds * 1e-9

    def _update_traffic_state(self, traffic):
        now = self.get_clock().now()
        color = traffic.color

        # Verde libera el bloqueo de rojo
        if color == 'green' and self.red_locked:
            self.red_locked = False
            self.get_logger().info('Semáforo VERDE: reanudando marcha')

        # Rojo activa el bloqueo indefinido
        if color == 'red':
            self.red_locked = True
            self.last_traffic_color = 'red'
            self.last_traffic_time = now

        # Mientras esté bloqueado por rojo, parar siempre
        if self.red_locked:
            self.traffic_multiplier = 0.0
            return 'red', 0.0, False

        # Comportamiento normal para amarillo/verde/ninguno
        if color in ('yellow', 'green'):
            self.traffic_multiplier = traffic.speed_multiplier
            self.last_traffic_multiplier = traffic.speed_multiplier
            self.last_traffic_color = color
            self.last_traffic_time = now
            return color, self.traffic_multiplier, False

        # Mantener amarillo brevemente si se pierde de vista
        if self.last_traffic_time is not None and self.last_traffic_color == 'yellow':
            elapsed = (now - self.last_traffic_time).nanoseconds * 1e-9
            if elapsed < self.traffic_hold_sec:
                self.traffic_multiplier = self.last_traffic_multiplier
                return self.last_traffic_color, self.traffic_multiplier, True

        self.traffic_multiplier = 1.0
        return 'none', 1.0, False

    def _compute_line_control(self, line):
        now = self.get_clock().now()

        valid_line = line.line_found and line.confidence >= self.min_confidence

        if valid_line:
            alpha = clamp(self.direction_alpha, 0.0, 1.0)
            self.filtered_direction = (
                (1.0 - alpha) * self.filtered_direction + alpha * line.direction
            )
            self.last_valid_direction = self.filtered_direction
            self.last_line_time = now
            line_speed_scale = 1.0
            lost = False
        else:
            elapsed = self._seconds_since(self.last_line_time)
            lost = True

            if line.zebra_detected and elapsed < 0.90:
                # En pasos de cebra, no hacemos giros bruscos: mantenemos la
                # última dirección confiable y bajamos velocidad.
                self.filtered_direction = self.last_valid_direction
                line_speed_scale = self.zebra_speed_scale
            elif elapsed < self.lost_timeout:
                # Pérdida muy corta: seguir suave para no pararse por ruido.
                self.filtered_direction = self.last_valid_direction
                line_speed_scale = self.lost_speed_scale
            else:
                if self.stop_when_lost:
                    line_speed_scale = 0.0
                    self.filtered_direction = 0.0
                else:
                    line_speed_scale = self.lost_speed_scale
                    self.filtered_direction = self.last_valid_direction

        omega = -self.kp * self.filtered_direction
        omega = clamp(omega, -self.max_w, self.max_w)
        return omega, line_speed_scale, valid_line, lost

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def timer_callback(self):
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().warn('No se recibió frame')
            self._publish_stop()
            return

        # 1) Línea
        line = find_line(frame, previous_direction=self.last_valid_direction)

        # 2) Semáforo en el mismo frame, sin abrir otra cámara
        traffic = detect_traffic_light(
            frame,
            roi_y_end=self.traffic_roi_y_end,
            min_area_ratio=self.traffic_min_area_ratio,
        )
        traffic_color, traffic_mult, traffic_hold = self._update_traffic_state(traffic)

        # 3) Control
        omega, line_speed_scale, valid_line, lost = self._compute_line_control(line)

        v = self.linear * line_speed_scale * traffic_mult
        if traffic_mult <= 0.01:
            # Rojo: alto total, sin seguir girando.
            v = 0.0
            omega = 0.0

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

        # 4) Visualización/debug
        vis = line.vis
        vis = draw_traffic_light(vis, traffic)

        status = 'OK' if valid_line else ('ZEBRA' if line.zebra_detected else 'LOST')
        txt1 = (
            f'line:{status} dir:{self.filtered_direction:+.2f} '
            f'conf:{line.confidence:.2f} zebra:{line.zebra_detected}'
        )
        lock_tag = '(LOCKED)' if self.red_locked else ('(hold)' if traffic_hold else '')
        txt2 = (
            f'traffic:{traffic_color}{lock_tag}'
            f' mult:{traffic_mult:.2f} v:{v:.2f} w:{omega:+.2f}'
        )

        cv2.putText(vis, txt1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)
        cv2.putText(vis, txt2, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)

        if self.publish_debug_text:
            self.get_logger().info(
                f'{status} | dir={self.filtered_direction:+.2f} conf={line.confidence:.2f} '
                f'zebra={line.zebra_detected} | sem={traffic_color} mult={traffic_mult:.2f} '
                f'v={v:.3f} w={omega:.3f}'
            )

        if self.show_window:
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
                raise SystemExit
            cv2.imshow(WINDOW_TITLE, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                raise SystemExit

    def destroy_node(self):
        self._publish_stop()
        self.camera.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LineFollower()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(f'Error iniciando line_follower: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

