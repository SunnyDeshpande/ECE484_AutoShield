from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'basic_launch'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *[(os.path.join('share', package_name, 'launch', os.path.relpath(os.path.dirname(path), 'launch')), [path]) for path in glob('launch/**/*launch.py', recursive=True)],
        *[(os.path.join('share', package_name, 'config', os.path.relpath(os.path.dirname(path), 'config')), [path]) for path in glob('config/**/*.yaml', recursive=True)],

        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hang Cui',
    maintainer_email='hangcui1201@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
