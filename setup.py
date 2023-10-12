from setuptools import find_packages, setup
import os 
from glob import glob

package_name = 'aqua_station_keeping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kvirji',
    maintainer_email='khalil.virji@mail.mcgill.ca',
    description='Aqua Station Keeping Project',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'underwater_adversary_command_publisher = aqua_station_keeping.underwater_adversary_command_publisher:main',
            'downward_camera_subscriber = aqua_station_keeping.downward_camera_subscriber:main',
            'pid_controllers = aqua_station_keeping.pid_controllers:main',
            'manual_controller = aqua_station_keeping.manual_controller:main',
            'yolo_detector = aqua_station_keeping.yolo_detector:main'

        ],
    },
)
