import array
import copy
import ctypes
import math

import pytest

from silero_vad_lite import SileroVAD

@pytest.fixture
def silero_vad():
    return SileroVAD(16000)

def _generate_audio_data_array(silero_vad):
    num_samples = silero_vad.window_size_samples
    sample_rate = silero_vad.sample_rate
    def audio_data_generator():
        for i in range(num_samples):
            t = i / sample_rate
            yield math.sin(2 * math.pi * 440 * t)
    return array.array('f', audio_data_generator())

@pytest.mark.parametrize('sample_rate', [8000, 16000])
@pytest.mark.parametrize('data_type', [array.array, bytes, bytearray, memoryview, ctypes.Array, list, tuple])
def test_silero_vad_process(sample_rate, data_type):
    silero_vad = SileroVAD(sample_rate)
    audio_data = _generate_audio_data_array(silero_vad)
    if data_type == array.array:
        pass
    elif data_type in [bytes, bytearray, memoryview, list, tuple]:
        audio_data = data_type(audio_data)
    elif data_type == ctypes.Array:
        audio_data = (ctypes.c_float * len(audio_data))(*audio_data)
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    if data_type not in [memoryview, ctypes.Array]:
        # test_silero_vad_process[memoryview-8000] - TypeError: cannot pickle memoryview objects
        audio_data_orig = copy.deepcopy(audio_data)
    result = silero_vad.process(audio_data)
    assert isinstance(result, float)
    assert 0 <= result <= 1
    if data_type not in [memoryview, ctypes.Array]:
        # test_silero_vad_process[Array-8000] - assert <silero_vad_l...x7fee623695b0> == <silero_vad_l...x7fee62369520>
        assert audio_data == audio_data_orig

def test_silero_vad_process_invalid_input(silero_vad):
    with pytest.raises(TypeError):
        silero_vad.process('invalid input')
    with pytest.raises(TypeError):
        silero_vad.process(1.0)
    with pytest.raises(ValueError):
        silero_vad.process([])
