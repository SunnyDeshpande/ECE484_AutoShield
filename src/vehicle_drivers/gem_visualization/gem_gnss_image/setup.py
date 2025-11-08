from setuptools import setup
import os
from glob import glob

package_name = 'gem_gnss_image'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'images'), glob(os.path.join('images', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Hang Cui',
    author_email='hangcui1201@gmail.com',
    maintainer='Hang Cui',
    maintainer_email='hangcui1201@gmail.com',
    keywords=['ROS2'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gem_gnss_image = gem_gnss_image.gem_gnss_image:main',   
        ],
    },
)
