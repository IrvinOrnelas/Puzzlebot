import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import qos_profile_sensor_data


class GoForward(Node):
    def __init__(self):
        super().__init__('go_forward')
        self.x = 0.0
        self.target_x = 0.2
        self.done = False

        self.sub = self.create_subscription(
            PoseStamped, '/estimated_pose', self.pose_cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info('GoForward started. Target x = 0.2 m')

    def pose_cb(self, msg: PoseStamped):
        self.x = msg.pose.position.x

    def control_loop(self):
        if self.done:
            return

        if self.x >= self.target_x:
            self.pub.publish(Twist())  # stop
            self.done = True
            self.get_logger().info(f'Target reached! x = {self.x:.3f} m')
            return

        cmd = Twist()
        cmd.linear.x = 0.1  # m/s
        self.pub.publish(cmd)
        self.get_logger().info(f'Moving... x = {self.x:.3f} m', throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = GoForward()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
