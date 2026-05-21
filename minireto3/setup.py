from setuptools import find_packages, setup

package_name = 'minireto3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='puzzlebot',
    maintainer_email='puzzlebot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimator = minireto3.pose_estimator:main',
            'go_forward = minireto3.go_forward:main',
            'rotate_to_angle = minireto3.rotate_to_angle:main',
            'go_to_waypoint = minireto3.go_to_waypoint:main',
            'path_generator = minireto3.path_generator:main',
        ],
    },
)
