from setuptools import setup, find_packages
import os

def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as file:
        return [line.strip() for line in file.readlines() if not line.startswith('#')]

setup(
    name='DeepED',  # Replace 'my_project' with the name of your project
    version='0.1.0',  # The current version of your project
    packages=find_packages(where='src'),  # Tells setuptools to find packages under src
    package_dir={'': 'src'},  # Specifies that packages are under src directory
    install_requires=read_requirements(),  # Use the read_requirements function
)
