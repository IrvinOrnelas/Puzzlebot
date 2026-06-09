from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    # Inicia el agente micro-ROS usando el workspace de ros2_packages_ws
    micro_ros_agent = ExecuteProcess(
        cmd=[
            'bash', '-c',
            'source /home/puzzlebot/ros2_packages_ws/install/setup.bash && '
            'ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0'
        ],
        output='screen',
    )

    # Muestra los topics disponibles 3 segundos después del arranque
    topic_list = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'topic', 'list'],
                output='screen',
            )
        ],
    )

    # Teleop en una ventana xterm propia (requiere entrada de teclado)
    teleop = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix='xterm -e',
        emulate_tty=True,
    )

    # Nodo grabador de pista
    grabador = Node(
        package='grabacionpista',
        executable='grabador',
        name='grabador_pista',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        micro_ros_agent,
        topic_list,
        teleop,
        grabador,
    ])
