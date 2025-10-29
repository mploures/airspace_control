from setuptools import setup, find_packages

setup(
    name='airspace_control',
    version='0.1.0',
    packages=find_packages(include=['ultrades_lib', 'ultrades_lib.*']),
    install_requires=[],
)
