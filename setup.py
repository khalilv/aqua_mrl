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
            'pid_controller = aqua_rl.pid_controller:main',
            'dqn_controller = aqua_rl.dqn_controller:main',
            'detect = aqua_rl.detect:main',
            'diver_controller = aqua_rl.diver_controller:main',
            'autopilot = aqua_rl.autopilot:main',
            'current_controller = aqua_rl.current_controller:main',
            'td3_adversary = aqua_rl.td3_adversary:main',

        ],
    },
)
