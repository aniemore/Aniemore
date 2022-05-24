from setuptools import setup
from os import path

requirements_path = path.join(path.dirname(__file__), "requirements.txt")

with open(requirements_path, "r") as f:
    requirements = [x for x in f.read().split("\n") if x]

setup(
    name='Aniemore',
    install_requires=requirements,
    version='0.1.0',
    packages=['Aniemore', 'Aniemore.Text', 'Aniemore.Tests', 'Aniemore.Utils', 'Aniemore.Voice', 'Aniemore.Datasets'],
    package_data={
        "Aniemore": ["*.yml", "*.md"],
        "Aniemore.Text": ["*.yml", "*.md"],
        "Aniemore.Voice": ["*.yml", "*.md"],
        "Aniemore.Datasets": ["*.yml", "*.md"],
        "Aniemore.Tests": ["*.yml", "*.md"]
    },
    url='https://github.com/aniemore/Aniemore',
    license='GPL-3.0 license ',
    author='toiletsandpaper',
    author_email='lii291001@gmail.com',
    description=''
)
