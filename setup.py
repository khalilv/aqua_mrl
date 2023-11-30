from setuptools import find_packages, setup
import os 
from glob import glob

package_name = 'aqua_pipeline_inspection'

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
    description='Aqua Pipeline Inspection Project',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'underwater_adversary_command_publisher = aqua_pipeline_inspection.underwater_adversary_command_publisher:main',
            'downward_camera_subscriber = aqua_pipeline_inspection.downward_camera_subscriber:main',
            'pid_controllers = aqua_pipeline_inspection.pid_controllers:main',
            'manual_controller = aqua_pipeline_inspection.manual_controller:main',
            'yolo_detector = aqua_pipeline_inspection.yolo_detector:main',
            'pipeline_segmentation = aqua_pipeline_inspection.pipeline_segmentation:main',
            'pipeline_parameters = aqua_pipeline_inspection.pipeline_parameters:main',
            'pid_pipeline_inspection = aqua_pipeline_inspection.pid_pipeline_inspection:main',
            'dqn_controller = aqua_pipeline_inspection.dqn_controller:main',
            'dqn_controller_eval = aqua_pipeline_inspection.dqn_controller_eval:main',

        ],
    },
)
