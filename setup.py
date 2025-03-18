from setuptools import find_packages
from setuptools import setup
import platform
import os

requirements_files = [
    'requirements.txt'
]

if os.environ.get('WITH_GPU'):
    arch_requirements_files = ['requirements_gpu.txt']
else:
    arch = platform.machine().lower()[0:3]  # should return 'arm' or 'x86'
    arch_requirements_files = {
        'arm': ['requirements_arm.txt'],
        'x86': ['requirements_intel.txt']
    }.get(arch, [])

requirements = []
for file in requirements_files + arch_requirements_files:
    with open(file) as f:
        lines = f.readlines()
    requirements.extend( [x.strip() for x in lines if "git+" not in x] )

setup(
    name='smearly',
    version="0.0.1",
    description="Smearly project",
    install_requires=requirements,
    packages=find_packages()
)
