# setup.py

from setuptools import setup, find_packages

setup(
    name='your_package_name',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'tensorflow',
        'keras-segmentation',  # Add any other dependencies
    ],
    entry_points={
        'console_scripts': [
            'your_node_name = your_package_name.detection_node:main',  # Adjust the entry point
        ],
    },
)
