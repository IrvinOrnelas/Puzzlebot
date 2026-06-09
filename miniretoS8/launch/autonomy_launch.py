from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        # Create tmux session
        ExecuteProcess(
            cmd=['tmux', 'new-session', '-d', '-s', 'miniretoS8_autonomy', '-x', '200', '-y', '50'],
            output='screen',
        ),

        # Terminal 1: traffic_light_node (tmux window 0)
        ExecuteProcess(
            cmd=['tmux', 'send-keys', '-t', 'miniretoS8_autonomy', 'source ~/ros2_ws/install/setup.bash && ros2 run miniretoS8 traffic_light_node', 'Enter'],
            output='screen',
        ),

        # Terminal 2: processor_node (tmux window 1)
        ExecuteProcess(
            cmd=['tmux', 'new-window', '-t', 'miniretoS8_autonomy'],
            output='screen',
        ),

        ExecuteProcess(
            cmd=['tmux', 'send-keys', '-t', 'miniretoS8_autonomy', 'source ~/ros2_ws/install/setup.bash && ros2 run miniretoS8 processor_node', 'Enter'],
            output='screen',
        ),

        # Terminal 3: line_follower_node (tmux window 2)
        ExecuteProcess(
            cmd=['tmux', 'new-window', '-t', 'miniretoS8_autonomy'],
            output='screen',
        ),

        ExecuteProcess(
            cmd=['tmux', 'send-keys', '-t', 'miniretoS8_autonomy', 'source ~/ros2_ws/install/setup.bash && ros2 run miniretoS8 line_follower_node', 'Enter'],
            output='screen',
        ),
    ])


