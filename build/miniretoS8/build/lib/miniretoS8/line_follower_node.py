import cv2
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from miniretoS8.line_detector import find_line


WINDOW_TITLE = 'Line Follower'


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class LineFollowerNode(Node):
    """Suscribe a /camera/image_raw, sigue la línea y publica /cmd_vel."""

    def __init__(self):
        super().__init__('line_follower_node')

        # Parámetros de control
        self.declare_parameter('kp',              1.5)
        self.declare_parameter('linear',          0.10)
        self.declare_parameter('max_w',           2.0)
        self.declare_parameter('direction_alpha', 0.35)   # suavizado EMA
        self.declare_parameter('lost_timeout',    0.35)   # seg antes de parar
        self.declare_parameter('lost_speed_scale', 0.25)  # velocidad al perder línea
        self.declare_parameter('stop_when_lost',  True)

        self._kp              = float(self.get_parameter('kp').value)
        self._linear          = float(self.get_parameter('linear').value)
        self._max_w           = float(self.get_parameter('max_w').value)
        self._alpha           = float(self.get_parameter('direction_alpha').value)
        self._lost_timeout    = float(self.get_parameter('lost_timeout').value)
        self._lost_scale      = float(self.get_parameter('lost_speed_scale').value)
        self._stop_when_lost  = bool(self.get_parameter('stop_when_lost').value)

        self._bridge = CvBridge()
        self._filtered_dir   = 0.0
        self._last_valid_dir = 0.0
        self._last_line_time = self.get_clock().now()

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        _qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            _qos,
        )

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        self.get_logger().info('LineFollowerNode listo, esperando imágenes en /camera/image_raw')

    # ------------------------------------------------------------------
    def _publish_stop(self):
        self._cmd_pub.publish(Twist())

    def _seconds_since(self, stamp):
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    # ------------------------------------------------------------------
    def _image_callback(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
            raise SystemExit

        vis, direction, line_found = find_line(frame)

        now = self.get_clock().now()

        if line_found:
            # Suavizado exponencial
            self._filtered_dir = (
                (1.0 - self._alpha) * self._filtered_dir + self._alpha * direction
            )
            self._last_valid_dir = self._filtered_dir
            self._last_line_time = now
            speed_scale = 1.0
            status = 'OK'
        else:
            elapsed = self._seconds_since(self._last_line_time)
            if elapsed < self._lost_timeout:
                # Pérdida corta: mantener dirección, bajar velocidad
                self._filtered_dir = self._last_valid_dir
                speed_scale = self._lost_scale
                status = 'LOST-SHORT'
            else:
                if self._stop_when_lost:
                    self._filtered_dir = 0.0
                    speed_scale = 0.0
                    status = 'LOST-STOP'
                else:
                    self._filtered_dir = self._last_valid_dir
                    speed_scale = self._lost_scale
                    status = 'LOST'

        omega = clamp(-self._kp * self._filtered_dir, -self._max_w, self._max_w)
        v = self._linear * speed_scale

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self._cmd_pub.publish(cmd)

        self.get_logger().info(
            f'{status} | dir={self._filtered_dir:+.2f} | v={v:.3f} w={omega:+.3f}'
        )

        # Overlay de estado en la ventana
        cv2.putText(
            vis,
            f'{status}  dir={self._filtered_dir:+.2f}  v={v:.2f}  w={omega:+.2f}',
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )
        cv2.imshow(WINDOW_TITLE, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            raise SystemExit

    # ------------------------------------------------------------------
    def destroy_node(self):
        self._publish_stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
