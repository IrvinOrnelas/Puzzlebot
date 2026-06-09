from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='miniretoS8',
            executable='camera_node',
            name='camera_node',
            output='screen',
        ),
        Node(
            package='miniretoS8',
            executable='processor_node',
            name='processor_node',
            output='screen',
        ),
        Node(
            package='miniretoS8',
            executable='traffic_light_node',
            name='traffic_light_node',
            output='screen',
        ),
        Node(
            package='miniretoS8',
            executable='line_follower_node',
            name='line_follower_node',
            output='screen',
        ),
    ])
