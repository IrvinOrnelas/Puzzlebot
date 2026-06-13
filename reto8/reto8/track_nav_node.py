import cv2  # noqa: F401  (debe importarse antes que cv_bridge)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from reto8 import track_detector


class TrackNavNode(Node):
    """Sigue la linea publicando un Twist con omega proporcional a la
    posicion de la linea, de modo que esta quede siempre centrada."""

    def __init__(self):
        super().__init__('track_nav_node')

        self.declare_parameter('kp', 1.0)
        self.declare_parameter('debug', False)

        self.kp = float(self.get_parameter('kp').value)
        self.debug = bool(self.get_parameter('debug').value)

        self._bridge = CvBridge()
        self._sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 1)
        self._pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self._img_pub = self.create_publisher(Image, '/track_nav/image_annotated', 1)

        self.get_logger().info('track_nav_node iniciado')

    def image_callback(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        detected, position, angle, vis = track_detector.find_line(frame, debug=self.debug)

        twist = Twist()
        if detected:
            # position > 0 -> linea a la derecha -> girar a la derecha (omega < 0)
            twist.angular.z = -self.kp * position
        self._pub.publish(twist)

        # Publicar el frame anotado
        out = self._bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        out.header = msg.header
        self._img_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TrackNavNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
