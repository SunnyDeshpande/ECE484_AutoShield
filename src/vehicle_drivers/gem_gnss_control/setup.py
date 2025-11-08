from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'gem_gnss_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gem',
    maintainer_email='gem@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit = gem_gnss_control.pure_pursuit:main',
            'pure_pursuit_test = gem_gnss_control.pure_pursuit_test:main',
        ],
    },
)
