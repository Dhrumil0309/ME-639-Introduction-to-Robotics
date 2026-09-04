import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('hw01_tf_demo')
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'demo.rviz')

    tf_broadcaster_node = Node(
        package='hw01_tf_demo',
        executable='tf_broadcaster_node',
        name='hw01_tf_broadcaster',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'compose_frame': 'current',
            'anim_duration': 1.5,
        }]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        tf_broadcaster_node,
        rviz_node,
    ])
