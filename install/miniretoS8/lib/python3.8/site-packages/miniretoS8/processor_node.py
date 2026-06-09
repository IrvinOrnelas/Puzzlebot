import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from miniretoS8.yolo import get_signs


WINDOW_TITLE = 'YOLO Detection'


class ProcessorNode(Node):
    """Suscribe a /camera/image_raw, corre YOLO y muestra el resultado en ventana OpenCV."""

    def __init__(self):
        super().__init__('processor_node')

        self._bridge = CvBridge()
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

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
        self.get_logger().info('Procesador YOLO listo, esperando imágenes...')

    def _image_callback(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
            raise SystemExit

        # Correr detección; dibujar sobre una copia para no modificar el original
        display = frame.copy()
        boxes, sign_types, confidences, class_names = get_signs(frame, drawing_frame=display)

        if len(sign_types) > 0:
            self.get_logger().info(
                f'Señales: {list(class_names)} conf={list(confidences.round(2))}'
            )

        cv2.imshow(WINDOW_TITLE, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            raise SystemExit

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
