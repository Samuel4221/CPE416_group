from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'project2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),  # <-- install here
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='calpoly',
    maintainer_email='calpoly@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 
            "ekf_node = project2.EKF:main"
        ],
    },
)
