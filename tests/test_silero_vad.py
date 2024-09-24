import array
import copy
import ctypes
import math
import os
import struct
import wave

import pytest

from silero_vad_lite import SileroVAD


def _generate_audio_data_array(silero_vad):
    num_samples = silero_vad.window_size_samples
    sample_rate = silero_vad.sample_rate
    def audio_data_generator():
        for i in range(num_samples):
            t = i / sample_rate
            yield math.sin(2 * math.pi * 440 * t)
    return array.array('f', audio_data_generator())

def _load_wav_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        num_channels = wav_file.getnchannels()
        assert num_channels == 1
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(num_frames)
    return audio_data, num_frames, sample_rate, sample_width


@pytest.fixture
def silero_vad():
    return SileroVAD(16000)

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

def test_silero_vad_process_wav_file():
    file_path = os.path.join(os.path.dirname(__file__), 'sample.wav')
    audio_data, num_frames, sample_rate, sample_width = _load_wav_file(file_path)
    assert sample_width == 2
    audio_data = struct.pack(f'<{num_frames}f', *(sample / 32768.0 for sample in struct.unpack(f'<{num_frames}h', audio_data)))
    sample_width = 4
    silero_vad = SileroVAD(sample_rate)
    window_size_bytes = silero_vad.window_size_samples * sample_width
    chunks = [audio_data[i:i + window_size_bytes] for i in range(0, len(audio_data), window_size_bytes)]
    if chunks[-1] != window_size_bytes:
        chunks = chunks[:-1]
    results = []
    for chunk in chunks:
        result = silero_vad.process(chunk)
        assert isinstance(result, float)
        assert 0 <= result <= 1
        results.append(result)
    # print(results)
    expected_results = [0.31846824288368225, 0.12080410122871399, 0.9278429746627808, 0.9227734804153442, 0.9691531658172607, 0.9847737550735474, 0.9906067848205566, 0.9805426597595215, 0.97320556640625, 0.9933459758758545, 0.9977824687957764, 0.9969353675842285, 0.9895951747894287, 0.9930758476257324, 0.9968366622924805, 0.9980421662330627, 0.9967591762542725, 0.9882574081420898, 0.9961190819740295, 0.9822508096694946, 0.9960722923278809, 0.9989539384841919, 0.9985291957855225, 0.9767082929611206, 0.9802166223526001, 0.9991974830627441, 0.998380184173584, 0.9981842041015625, 0.9984550476074219, 0.9984889030456543, 0.9990912079811096, 0.9931062459945679, 0.9294931888580322, 0.5672889947891235, 0.342951238155365, 0.1822890043258667, 0.09109050035476685]
    assert len(results) == len(expected_results)
    # Check if the results are close enough within a margin of error
    for result, expected_result in zip(results, expected_results):
        assert math.isclose(result, expected_result, abs_tol=1e-6)

def test_silero_vad_process_invalid_input(silero_vad):
    with pytest.raises(TypeError):
        silero_vad.process('invalid input')
    with pytest.raises(TypeError):
        silero_vad.process(1.0)
    with pytest.raises(ValueError):
        silero_vad.process([])
