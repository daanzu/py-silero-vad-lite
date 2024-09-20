import array
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

        self.lib.SileroVAD_get_window_size_samples.argtypes = [ctypes.c_void_p]
        self.lib.SileroVAD_get_window_size_samples.restype = ctypes.c_size_t

        self.lib.SileroVAD_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self.lib.SileroVAD_process.restype = ctypes.c_float

        # Create the C++ object
        self.obj = self.lib.SileroVAD_new(model_path.encode('utf-8'), sample_rate)
        self._sample_rate = sample_rate  # Constant
        self._window_size_samples = self.lib.SileroVAD_get_window_size_samples(self.obj)  # Constant

    def __del__(self):
        if self.obj:
            self.lib.SileroVAD_delete(self.obj)

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def window_size_samples(self):
        return self._window_size_samples

    def process(self, data):
        length = len(data)
        if length <= 0:
            raise ValueError("Data must not be empty")
        if isinstance(data, (bytes, bytearray)):
            if isinstance(data, bytes):
                data = bytearray(data)
            if length % ctypes.sizeof(ctypes.c_float) != 0:
                raise ValueError(f"Data length must be a multiple of the size of a float ({ctypes.sizeof(ctypes.c_float)} bytes)")
            length = length // ctypes.sizeof(ctypes.c_float)
            float_array = (ctypes.c_float * length).from_buffer(data)
        elif isinstance(data, memoryview):
            if data.ndim != 1:
                raise ValueError("Memoryview must be one-dimensional")
            if data.itemsize != ctypes.sizeof(ctypes.c_float):
                raise ValueError(f"Memoryview item size must be the size of a float ({ctypes.sizeof(ctypes.c_float)} bytes)")
            if not data.contiguous:
                raise ValueError("Memoryview must be contiguous")
            if data.readonly:
                raise ValueError("Memoryview must be writable")
            float_array = (ctypes.c_float * length).from_buffer(data)
        elif isinstance(data, array.array):
            if data.typecode != 'f':
                raise ValueError("Array must be of type 'f' (float)")
            float_array = (ctypes.c_float * length).from_buffer(data)
        elif isinstance(data, ctypes.Array):
            float_array = data
        else:
            float_array = (ctypes.c_float * length)(*data)
        return self.lib.SileroVAD_process(self.obj, float_array, length)

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
