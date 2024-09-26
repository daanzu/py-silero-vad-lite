import glob
import platform
import shutil
import zipfile
from setuptools import setup, find_namespace_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
import os
import sys
import subprocess
import urllib.request
import tarfile

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(os.path.join(sourcedir, 'src', 'silero_vad_lite'))

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extension_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        extension_dir = os.path.join(extension_dir, 'silero_vad_lite', 'data')
        cmake_args = ['-DPACKAGE_DATA_OUTPUT_DIRECTORY=' + extension_dir, '-DPYTHON_EXECUTABLE=' + sys.executable]

        # self.debug = True
        # cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        if not platform.system() == 'Windows':
            build_args += ['--', '-j2']

        onnxruntime_static_default = 'OFF' if platform.system() == 'Darwin' else 'ON'
        onnxruntime_static = os.environ.get('SILERO_VAD_LITE_ONNXRUNTIME_STATIC', onnxruntime_static_default) == 'ON'
        cmake_args += [f'-DONNXRUNTIME_STATIC={"ON" if onnxruntime_static else "OFF"}']
        onnxruntime_dir = self.download_onnxruntime(onnxruntime_static)
        # If using shared onnxruntime, copy the onnxruntime library to the extension directory so that it can be found at runtime
        if not onnxruntime_static:
            onnxruntime_lib_name = 'onnxruntime.dll' if platform.system() == 'Windows' else 'libonnxruntime.dylib' if platform.system() == 'Darwin' else 'libonnxruntime.so'
            shutil.copyfile(os.path.join(onnxruntime_dir, 'lib', onnxruntime_lib_name), os.path.join(extension_dir, onnxruntime_lib_name))

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(f"CMake: Configuring: {['cmake', ext.sourcedir] + cmake_args} in {self.build_temp}")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        print(f"CMake: Building: {['cmake', '--build', '.'] + build_args} in {self.build_temp}")
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

    def download_onnxruntime(self, onnxruntime_static):
        onnxruntime_dir = os.path.join(self.build_temp, 'onnxruntime')
        if os.path.exists(onnxruntime_dir):
            return onnxruntime_dir
        os.makedirs(onnxruntime_dir)

        if not onnxruntime_static:
            # Releases · microsoft/onnxruntime (https://github.com/microsoft/onnxruntime/releases)
            onnxruntime_version = '1.19.2'
            if platform.system() == 'Windows':
                onnxruntime_platform = 'win-x64'
                onnxruntime_extension = 'zip'
            elif platform.system() == 'Darwin':
                onnxruntime_platform = 'osx-universal2'  # TODO: choose between 'osx-x64' and 'osx-arm64'
                onnxruntime_extension = 'tgz'
            elif platform.system() == 'Linux':
                onnxruntime_platform = 'linux-x64'
                onnxruntime_extension = 'tgz'
            else:
                raise ValueError(f"Unsupported platform: {platform.system()}")
            onnxruntime_url = f'https://github.com/microsoft/onnxruntime/releases/download/v{onnxruntime_version}/onnxruntime-{onnxruntime_platform}-{onnxruntime_version}.{onnxruntime_extension}'
        else:
            # Releases · csukuangfj/onnxruntime-libs (https://github.com/csukuangfj/onnxruntime-libs/releases)
            onnxruntime_version = '1.19.0'
            if platform.system() == 'Windows':
                onnxruntime_platform = 'win-x64-static_lib'
                onnxruntime_extension = 'tar.bz2'
            elif platform.system() == 'Darwin':
                onnxruntime_platform = 'osx-universal2-static_lib'  # TODO: choose between 'osx-x64-static_lib' and 'osx-arm64-static_lib'
                onnxruntime_extension = 'zip'
            elif platform.system() == 'Linux':
                onnxruntime_platform = 'linux-x64-static_lib'
                onnxruntime_extension = 'zip'
            else:
                raise ValueError(f"Unsupported platform: {platform.system()}")
            onnxruntime_suffix = '-glibc2_17' if platform.system() == 'Linux' else ''
            onnxruntime_url = f'https://github.com/csukuangfj/onnxruntime-libs/releases/download/v{onnxruntime_version}/onnxruntime-{onnxruntime_platform}-{onnxruntime_version}{onnxruntime_suffix}.{onnxruntime_extension}'

        print(f"Downloading ONNXRuntime from {onnxruntime_url}")
        file_path, _ = urllib.request.urlretrieve(onnxruntime_url)
        if onnxruntime_extension == 'tgz':
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=onnxruntime_dir)
        elif onnxruntime_extension == 'tar.bz2':
            with tarfile.open(file_path, 'r:bz2') as tar:
                tar.extractall(path=onnxruntime_dir)
        elif onnxruntime_extension == 'zip':
            with zipfile.ZipFile(file_path, 'r') as zip:
                zip.extractall(path=onnxruntime_dir)
        else:
            raise ValueError(f"Unsupported archive extension: {onnxruntime_extension}")

        # Move the contents of the extracted directory (version-specific-named) to the parent directory, to ease building
        contents = glob.glob(os.path.join(onnxruntime_dir, '*'))
        assert len(contents) == 1 and os.path.isdir(contents[0])
        extracted_dir = contents[0]
        for item in os.listdir(extracted_dir):
            shutil.move(os.path.join(extracted_dir, item), onnxruntime_dir)
        os.rmdir(extracted_dir)

        return onnxruntime_dir

class BinaryDistribution(Distribution):
    # Ensure generation of platform-specific wheels
    def has_ext_modules(self):
        return True

setup(
    name='silero-vad-lite',
    version='0.1.0',
    author='David Zurow',
    author_email='daanzu@gmail.com',
    description='Lightweight wrapper for Silero VAD using internal ONNX Runtime and with no python package dependencies',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/daanzu/py-silero-vad-lite',
    packages=find_namespace_packages(where='src'),
    ext_modules=[CMakeExtension('silero_vad_lite', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    package_dir={'': 'src'},
    include_package_data=True,
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
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
    ],
    python_requires='>=3.6',
    extras_require={
        'dev': [
            'pytest >= 7.0.0',
        ],
    },
)
