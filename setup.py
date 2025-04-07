from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> list[str]:
    """This function will return the list of requirements."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Strip whitespace and newline characters
        requirements = [req.strip() for req in requirements if req.strip()]
        # Filter out '-e .' and any invalid entries
        requirements = [req for req in requirements if req != '-e .']
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Adwait',
    author_email='adwaittiwari99@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)