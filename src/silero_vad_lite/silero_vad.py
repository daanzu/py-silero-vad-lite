import ctypes
import os
import platform

class SileroVAD:
    def __init__(self, sample_rate, model_path=None):
        if model_path is None:
            model_path = self._get_model_path()

        # Load the shared library
        self.lib = ctypes.CDLL(self._get_lib_path())

        # Define function prototypes
        self.lib.SileroVAD_new.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.SileroVAD_new.restype = ctypes.c_void_p

        self.lib.SileroVAD_delete.argtypes = [ctypes.c_void_p]

        self.lib.SileroVAD_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self.lib.SileroVAD_process.restype = ctypes.c_float

        # Create the C++ object
        self.obj = self.lib.SileroVAD_new(model_path.encode('utf-8'), sample_rate)

    def __del__(self):
        if self.obj:
            self.lib.SileroVAD_delete(self.obj)

    def process(self, data):
        float_array = (ctypes.c_float * len(data))(*data)
        return self.lib.SileroVAD_process(self.obj, float_array, len(data))

    @staticmethod
    def _get_lib_name():
        base_name = 'silero_vad_lite'
        if platform.system() == 'Windows':
            return base_name + '.dll'
        elif platform.system() == 'Darwin':
            return base_name + '.dylib'
        else:
            return base_name + '.so'

    @classmethod
    def _get_lib_path(cls):
        # TODO: Implement a proper way to get the library path
        # Data Files Support - setuptools 75.1.0.post20240916 documentation (https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime)
        return os.path.join(os.path.dirname(__file__), 'data', cls._get_lib_name())

    @staticmethod
    def _get_model_path():
        # TODO: Implement a proper way to get the model path
        # Data Files Support - setuptools 75.1.0.post20240916 documentation (https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime)
        return os.path.join(os.path.dirname(__file__), 'data', 'silero_vad.onnx')
