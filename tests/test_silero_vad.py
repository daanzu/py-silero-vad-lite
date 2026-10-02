import array
import copy
import ctypes
import math
import json
from pathlib import Path
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
    if len(chunks[-1]) != window_size_bytes:
        chunks = chunks[:-1]
    results = []
    for chunk in chunks:
        result = silero_vad.process(chunk)
        assert isinstance(result, float)
        assert 0 <= result <= 1
        results.append(result)
    # print(results)
    reference = json.loads((Path(__file__).parent / 'fixtures/context_v5_1.json').read_text())
    expected_results = reference['probabilities'][str(sample_rate)]
    assert len(results) == len(expected_results)
    # Check if the results are close enough within a margin of error
    for result, expected_result in zip(results, expected_results):
        assert math.isclose(result, expected_result, abs_tol=1e-6, rel_tol=1e-5)

def test_silero_vad_process_invalid_input(silero_vad):
    with pytest.raises(TypeError):
        silero_vad.process('invalid input')
    with pytest.raises(TypeError):
        silero_vad.process(1.0)
    with pytest.raises(ValueError):
        silero_vad.process([])
