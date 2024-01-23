from setuptools import find_packages, setup
import os 
from glob import glob

package_name = 'aqua_rl'

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
    description='Aqua Reinforcement Learning Project',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'underwater_adversary_command_publisher = aqua_rl.underwater_adversary_command_publisher:main',
            'pid_controller = aqua_rl.pid_controller:main',
            'manual_controller = aqua_rl.manual_controller:main',
            'segmentation = aqua_rl.segmentation:main',
            'pid_parameters = aqua_rl.pid_parameters:main',
            'pid_tuning = aqua_rl.pid_tuning:main',
            'dqn_controller = aqua_rl.dqn_controller:main',
            'dqn_controller_eval = aqua_rl.dqn_controller_eval:main',
            'dqn_controller_adv = aqua_rl.dqn_controller_adv:main',
        ],
    },
)
