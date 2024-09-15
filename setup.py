import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
import os
import sys
import platform
import subprocess
import urllib.request
import tarfile

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(os.path.join(sourcedir, 'silero_vad_lite'))

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))

        self.download_onnxruntime()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir, '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(f"CMake: Configuring: {['cmake', ext.sourcedir] + cmake_args} in {self.build_temp}")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        print(f"CMake: Building: {['cmake', '--build', '.'] + build_args} in {self.build_temp}")
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

    def download_onnxruntime(self):
        onnxruntime_version = '1.19.2'
        onnxruntime_url = f'https://github.com/microsoft/onnxruntime/releases/download/v{onnxruntime_version}/onnxruntime-linux-x64-{onnxruntime_version}.tgz'
        onnxruntime_dir = os.path.join(self.build_temp, 'onnxruntime')
        if not os.path.exists(onnxruntime_dir):
            os.makedirs(onnxruntime_dir)
            print(f"Downloading ONNXRuntime from {onnxruntime_url}")
            file_path, _ = urllib.request.urlretrieve(onnxruntime_url)
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=onnxruntime_dir)
            # Move the contents of the extracted directory (version-specific-named) to the parent directory to ease building
            extracted_dir = os.path.join(onnxruntime_dir, f'onnxruntime-linux-x64-{onnxruntime_version}')
            for item in os.listdir(extracted_dir):
                shutil.move(os.path.join(extracted_dir, item), onnxruntime_dir)
            os.rmdir(extracted_dir)
        return onnxruntime_dir

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
    ext_modules=[CMakeExtension('silero_vad_lite', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
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
