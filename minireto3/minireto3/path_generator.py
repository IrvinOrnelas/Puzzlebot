import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


WAYPOINTS = [
    ( 0.4,    0.0   ),  # 0°
    ( 0.2,    0.3464),  # 60°
    (-0.2,    0.3464),  # 120°
    (-0.4,    0.0   ),  # 180°
    (-0.2,   -0.3464),  # 240°
    ( 0.2,   -0.3464),  # 300°
]

# Latched QoS so late-starting subscribers receive the last waypoint immediately
LATCH_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


class PathGenerator(Node):
    def __init__(self):
        super().__init__('path_generator')
        self.index = 0

        self.pub = self.create_publisher(Point, '/waypoint', LATCH_QOS)
        self.create_subscription(Bool, '/waypoint_reached', self.reached_cb, 10)

        self.get_logger().info(
            f'PathGenerator started. {len(WAYPOINTS)} waypoints (hexágono ⌀0.8 m).')
        self._send_current()

    def _send_current(self):
        wx, wy = WAYPOINTS[self.index]
        msg = Point()
        msg.x = float(wx)
        msg.y = float(wy)
        self.pub.publish(msg)
        self.get_logger().info(
            f'Waypoint {self.index + 1}/{len(WAYPOINTS)}: ({wx:.2f}, {wy:.2f})')

    def reached_cb(self, msg: Bool):
        if msg.data:
            self.index = (self.index + 1) % len(WAYPOINTS)
            self._send_current()


def main(args=None):
    rclpy.init(args=args)
    node = PathGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
