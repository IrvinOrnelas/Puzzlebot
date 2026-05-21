import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import TwistStamped, PoseStamped, TransformStamped
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster
import math

class PoseEstimator(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        # Dynamic parameter for orientation calibration
        self.declare_parameter('angular_factor', 0.9636593357174979)
        self.angular_factor = self.get_parameter('angular_factor').value
        self.add_on_set_parameters_callback(self._on_param_update)

        # State of velocities
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # State of the pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_update_time = self.get_clock().now()

        # Subscription to robot velocities
        self.subscription = self.create_subscription(
            TwistStamped,
            '/robot_vel',
            self.robot_vel_callback,
            qos_profile_sensor_data
        )

        # Publisher for estimated pose
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/estimated_pose',
            10
        )

        # Broadcast transformations with TF2
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timers: sampling (high freq) and printing (low freq)
        self.create_timer(1 / 400, self.sampling_loop)
        self.create_timer(1 / 4, self.print_loop)

        self.get_logger().info('PoseEstimator is active and listening to /robot_vel.')

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'angular_factor':
                self.angular_factor = p.value
        return SetParametersResult(successful=True)

    def robot_vel_callback(self, msg: TwistStamped):
        self.linear_velocity = msg.twist.linear.x
        self.angular_velocity = msg.twist.angular.z

    def sampling_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds * 1e-9

        # Correct angular velocity before integration
        corrected_w = self.angular_velocity * self.angular_factor
        self.theta += corrected_w * dt

        # Normalize theta to [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

        # Integrate position using the current velocity and orientation
        self.x += self.linear_velocity * math.cos(self.theta) * dt
        self.y += self.linear_velocity * math.sin(self.theta) * dt
        self.last_update_time = now

        # Publish PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now.to_msg()
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = self.x
        pose_msg.pose.position.y = self.y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = math.sin(self.theta / 2.0)
        pose_msg.pose.orientation.w = math.cos(self.theta / 2.0)
        self.pose_publisher.publish(pose_msg)

        # Broadcast the transform from odom to base_link
        tf = TransformStamped()
        tf.header.stamp = now.to_msg()
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = 0.0
        tf.transform.rotation = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

    def print_loop(self):
        theta_deg = self.theta * 180.0 / math.pi
        print(
            f"V: {self.linear_velocity:.2f} m/s | W: {self.angular_velocity * 180/math.pi:.2f}°/s | "
            f"Theta: {theta_deg:.2f}° | X: {self.x:.2f} | Y: {self.y:.2f} | "
            f"Factor: {self.angular_factor:.3f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
