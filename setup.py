from setuptools import find_packages, setup

package_name = 'aqua_station_keeping'

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
    maintainer='kvirji',
    maintainer_email='khalil.virji@mail.mcgill.ca',
    description='Aqua Station Keeping Project',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'underwater_adversary_command_publisher = aqua_station_keeping.underwater_adversary_command_publisher:main',
            'aqua_pose_subscriber = aqua_station_keeping.aqua_pose_subscriber:main'
        ],
    },
)
