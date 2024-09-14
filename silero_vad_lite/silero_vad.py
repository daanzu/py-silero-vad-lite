import ctypes
import os

class SileroVAD:
    def __init__(self, model_path):
        # Load the shared library
        lib_path = os.path.join(os.path.dirname(__file__), 'silero_vad.so')
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
