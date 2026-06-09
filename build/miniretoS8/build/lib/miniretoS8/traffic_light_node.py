import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge


WINDOW_TITLE = 'Traffic Light'
MIN_AREA = 10000

# Rangos HSV para rojo (envuelve los 0/180°, se necesitan dos rangos)
RED_LOWER1 = np.array([0,   120,  70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([160, 120,  70])
RED_UPPER2 = np.array([180, 255, 255])

# Rango HSV para amarillo
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

# Rango HSV para verde
GREEN_LOWER = np.array([40,  50,  50])
GREEN_UPPER = np.array([80, 255, 255])

COLOR_INFO = [
    (RED_LOWER1, RED_UPPER1, RED_LOWER2, RED_UPPER2, 'rojo',     (0,   0,   255)),
    (YELLOW_LOWER, YELLOW_UPPER, None, None,          'amarillo', (0,   255, 255)),
    (GREEN_LOWER,  GREEN_UPPER,  None, None,          'verde',    (0,   255,   0)),
]


class TrafficLightNode(Node):
    """Suscribe a /camera/image_raw, detecta colores del semáforo y publica /speed_multiplier."""

    def __init__(self):
        super().__init__('traffic_light_node')

        self._bridge = CvBridge()
        self._stopped_by_red = False  # latch: detenido hasta ver verde

        self._speed_pub = self.create_publisher(Float32, '/speed_multiplier', 10)
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
        self.get_logger().info('Nodo semáforo listo, esperando imágenes...')

    def _image_callback(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
            raise SystemExit

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best_contour = None
        best_area = 0
        best_color = None
        best_bgr = None

        for lower1, upper1, lower2, upper2, color_name, color_bgr in COLOR_INFO:
            mask = cv2.inRange(hsv, lower1, upper1)
            if lower2 is not None:
                mask |= cv2.inRange(hsv, lower2, upper2)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_AREA:
                    cv2.drawContours(frame, [cnt], -1, color_bgr, 2)
                    if area > best_area:
                        best_area = area
                        best_contour = cnt
                        best_color = color_name
                        best_bgr = color_bgr

        speed_msg = Float32()

        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), best_bgr, 3)
            cv2.putText(
                frame, f'{best_color} ({int(best_area)})',
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, best_bgr, 2,
            )

            if best_color == 'rojo':
                self._stopped_by_red = True
                speed_msg.data = 0.0
            elif best_color == 'verde':
                self._stopped_by_red = False
                speed_msg.data = 1.0
            else:  # amarillo
                speed_msg.data = 0.0 if self._stopped_by_red else 0.5

            self.get_logger().info(
                f'Color: {best_color} | area: {int(best_area)} | '
                f'stop_latch: {self._stopped_by_red} | vel: {speed_msg.data}'
            )
        else:
            speed_msg.data = 0.0 if self._stopped_by_red else 1.0
            self.get_logger().debug(
                f'Sin detección | stop_latch: {self._stopped_by_red} | vel: {speed_msg.data}'
            )

        self._speed_pub.publish(speed_msg)

        cv2.imshow(WINDOW_TITLE, frame)
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
