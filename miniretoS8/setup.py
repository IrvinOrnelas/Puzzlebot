from setuptools import find_packages, setup

package_name = 'miniretoS8'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/miniretoS8_launch.py',
            'launch/setup_launch.py',
            'launch/autonomy_launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='puzzlebot',
    maintainer_email='Irvin Ornelas',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = miniretoS8.camera_node:main',
            'processor_node = miniretoS8.processor_node:main',
            'traffic_light_node = miniretoS8.traffic_light_node:main',
            'line_follower_node = miniretoS8.line_follower_node:main',
        ],
    },
)
