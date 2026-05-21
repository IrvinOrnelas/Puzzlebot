import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from std_msgs.msg import Bool, Float32
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
import math


LATCH_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


REACH_RADIUS = 0.05   # m — waypoint considered reached within 5 cm
MAX_V        = 0.2    # m/s
KP_W         = 2.0    # angular proportional gain
MAX_W        = 2.0    # rad/s
ALIGN_MIN    = 0.9    # alignment where authority = 0.0
ALIGN_MAX    = 1.0    # alignment where authority = 1.0


class GoToWaypoint(Node):
    def __init__(self):
        super().__init__('go_to_waypoint')

        # Pose state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Default waypoint
        self.wp_x = 0.0
        self.wp_y = 0.25
        self.done = False

        # Speed multiplier [0.0, 1.0]
        self.speed_multiplier = 1.0

        self.create_subscription(
            PoseStamped, '/estimated_pose', self.pose_cb, qos_profile_sensor_data)
        self.create_subscription(
            Point, '/waypoint', self.waypoint_cb, LATCH_QOS)
        self.create_subscription(
            Float32, '/speed_multiplier', self.speed_cb, 10)

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reached_pub = self.create_publisher(Bool, '/waypoint_reached', 10)
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            f'GoToWaypoint started. Default waypoint: ({self.wp_x}, {self.wp_y})')

    def pose_cb(self, msg: PoseStamped):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.theta = 2.0 * math.atan2(
            msg.pose.orientation.z, msg.pose.orientation.w)

    def speed_cb(self, msg: Float32):
        self.speed_multiplier = max(0.0, min(1.0, msg.data))
        self.get_logger().info(f'Speed multiplier: {self.speed_multiplier:.2f}')

    def waypoint_cb(self, msg: Point):
        if abs(msg.x - self.wp_x) < 0.001 and abs(msg.y - self.wp_y) < 0.001:
            return  # same waypoint, ignore
        self.wp_x = msg.x
        self.wp_y = msg.y
        self.done = False
        self.get_logger().info(
            f'New waypoint: ({self.wp_x:.3f}, {self.wp_y:.3f})')

    def control_loop(self):
        if self.done:
            return

        dx = self.wp_x - self.x
        dy = self.wp_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < REACH_RADIUS:
            self.pub.publish(Twist())
            self.done = True
            reached_msg = Bool()
            reached_msg.data = True
            self.reached_pub.publish(reached_msg)
            self.get_logger().info(
                f'Waypoint reached! dist={dist:.3f} m  pos=({self.x:.3f}, {self.y:.3f})')
            return

        # Desired heading toward waypoint
        target_angle = math.atan2(dy, dx)

        # Angular error in [-pi, pi]
        error = (target_angle - self.theta + math.pi) % (2 * math.pi) - math.pi

        error_deg = abs(math.degrees(error))
        alignment = 1.0 - error_deg / 180.0

        # Authority: 0.0 at alignment=0.9, 1.0 at alignment=1.0
        authority = (alignment - ALIGN_MIN) / (ALIGN_MAX - ALIGN_MIN)
        authority = max(0.0, min(1.0, authority))

        v = authority * MAX_V * self.speed_multiplier
        w = max(-MAX_W, min(MAX_W, KP_W * error))

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.pub.publish(cmd)

        print(
            f"dist: {dist:.3f} m | err: {math.degrees(error):7.2f}° | "
            f"align: {alignment:.3f} | auth: {authority:.3f} | "
            f"spd: {self.speed_multiplier:.2f} | "
            f"v: {v:.3f} m/s | w: {math.degrees(w):.1f}°/s"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoToWaypoint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
