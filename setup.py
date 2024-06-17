from setuptools import find_packages, setup
import os 
from glob import glob

package_name = 'aqua_mrl'

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
            'pid_controller = aqua_mrl.pid_controller:main',
            'dqn_controller = aqua_mrl.dqn_controller:main',
            'detect = aqua_mrl.detect:main',
            'diver_controller = aqua_mrl.diver_controller:main',
            'autopilot = aqua_mrl.autopilot:main',
            'evaluation = aqua_mrl.evaluation:main',
        ],
    },
)
