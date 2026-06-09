from launch import LaunchDescription
from launch.actions import ExecuteProcess
import time


def generate_launch_description():
    return LaunchDescription([
        # Terminal 1: micro_ros_agent (tmux window 0)
        ExecuteProcess(
            cmd=['tmux', 'new-session', '-d', '-s', 'miniretoS8_setup', '-x', '200', '-y', '50'],
            output='screen',
        ),

        ExecuteProcess(
            cmd=['tmux', 'send-keys', '-t', 'miniretoS8_setup', 'source /home/puzzlebot/ros2_packages_ws/install/setup.bash && ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0', 'Enter'],
            output='screen',
        ),

        # Terminal 2: teleop_twist_keyboard (tmux window 1)
        ExecuteProcess(
            cmd=['tmux', 'new-window', '-t', 'miniretoS8_setup'],
            output='screen',
        ),

        ExecuteProcess(
            cmd=['tmux', 'send-keys', '-t', 'miniretoS8_setup', 'ros2 topic list && ros2 run teleop_twist_keyboard teleop_twist_keyboard', 'Enter'],
            output='screen',
        ),

        # Terminal 3: camera_node (tmux window 2)
        ExecuteProcess(
            cmd=['tmux', 'new-window', '-t', 'miniretoS8_setup'],
            output='screen',
        ),

        ExecuteProcess(
            cmd=['tmux', 'send-keys', '-t', 'miniretoS8_setup', 'source ~/ros2_ws/install/setup.bash && ros2 run miniretoS8 camera_node', 'Enter'],
            output='screen',
        ),
    ])


