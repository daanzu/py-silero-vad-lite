from setuptools import setup, find_packages
from setuptools.dist import Distribution
import os
import sys
import platform

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

def get_lib_extension():
    if platform.system() == "Windows":
        return ".dll"
    elif platform.system() == "Darwin":
        return ".dylib"
    else:
        return ".so"

setup(
    name='silero-vad-lite',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A lightweight wrapper for Silero VAD using ONNX Runtime',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/silero-vad-lite',
    packages=find_packages(),
    package_data={
        'silero_vad_lite': ['*' + get_lib_extension()],
    },
    distclass=BinaryDistribution,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
    ],
    python_requires='>=3.6',
)
