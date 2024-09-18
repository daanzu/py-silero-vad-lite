import ctypes
import os
import platform

class SileroVAD:
    def __init__(self, model_path):
        if model_path is None:
            model_path = self._get_model_path()

        # Load the shared library
        lib_name = self._get_lib_name()
        lib_path = os.path.join(os.path.dirname(__file__), lib_name)
        self.lib = ctypes.CDLL(lib_path)

        # Define function prototypes
        self.lib.SileroVAD_new.argtypes = [ctypes.c_char_p]
        self.lib.SileroVAD_new.restype = ctypes.c_void_p

        self.lib.SileroVAD_delete.argtypes = [ctypes.c_void_p]

        self.lib.SileroVAD_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        self.lib.SileroVAD_process.restype = ctypes.c_float

        # Create the C++ object
        self.obj = self.lib.SileroVAD_new(model_path.encode('utf-8'))

    def __del__(self):
        if self.obj:
            self.lib.SileroVAD_delete(self.obj)

    def process(self, data, sample_rate):
        float_array = (ctypes.c_float * len(data))(*data)
        return self.lib.SileroVAD_process(self.obj, float_array, len(data), sample_rate)

    @staticmethod
    def _get_lib_name():
        if platform.system() == "Windows":
            return "silero_vad.dll"
        elif platform.system() == "Darwin":
            return "silero_vad.dylib"
        else:
            return "silero_vad.so"

    @staticmethod
    def _get_model_path():
        # TODO: Implement a proper way to get the model path
        # Data Files Support - setuptools 75.1.0.post20240916 documentation (https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime)
        return os.path.join(os.path.dirname(__file__), 'data', 'silero_vad.onnx')
