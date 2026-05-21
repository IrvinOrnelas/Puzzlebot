import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import qos_profile_sensor_data
import math


TARGET_DEG = 90.0   # degrees — change as needed
KP = 1.5            # proportional gain
MAX_W = 1.5         # rad/s
TOLERANCE_DEG = 2.0 # degrees


class RotateToAngle(Node):
    def __init__(self):
        super().__init__('rotate_to_angle')
        self.theta = 0.0
        self.done = False

        self.target_rad = math.radians(TARGET_DEG)

        self.sub = self.create_subscription(
            PoseStamped, '/estimated_pose', self.pose_cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f'RotateToAngle started. Target = {TARGET_DEG}°')

    def pose_cb(self, msg: PoseStamped):
        self.theta = 2.0 * math.atan2(
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )

    def control_loop(self):
        if self.done:
            return

        # Angular error normalized to [-pi, pi]
        error = self.target_rad - self.theta
        error = (error + math.pi) % (2 * math.pi) - math.pi

        error_deg = math.degrees(abs(error))
        alignment = 1.0 - error_deg / 180.0

        print(
            f"Theta: {math.degrees(self.theta):7.2f}° | "
            f"Error: {math.degrees(error):7.2f}° | "
            f"Alignment: {alignment:.3f}"
        )

        if error_deg < TOLERANCE_DEG:
            self.pub.publish(Twist())
            self.done = True
            self.get_logger().info(
                f'Target reached! Theta = {math.degrees(self.theta):.2f}° | Alignment = {alignment:.3f}'
            )
            return

        w = KP * error
        w = max(-MAX_W, min(MAX_W, w))

        cmd = Twist()
        cmd.angular.z = w
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RotateToAngle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
