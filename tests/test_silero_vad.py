import array
import copy
import math

import pytest

from silero_vad_lite import SileroVAD

@pytest.fixture
def silero_vad():
    return SileroVAD(16000)

def test_silero_vad_process(silero_vad):
    # Generate some dummy audio data
    num_samples = silero_vad.window_size_samples
    sample_rate = 16000
    def audio_data_generator():
        for i in range(num_samples):
            t = i / sample_rate
            yield math.sin(2 * math.pi * 440 * t)
    audio_data = array.array('f', audio_data_generator())

    # Process the audio data
    audio_data_orig = copy.deepcopy(audio_data)
    result = silero_vad.process(audio_data)

    # Check if the result is a float between 0 and 1
    assert isinstance(result, float)
    assert 0 <= result <= 1

    # Check that the data was not modified
    assert audio_data == audio_data_orig

def test_silero_vad_invalid_input(silero_vad):
    with pytest.raises(TypeError):
        silero_vad.process("invalid input")
    with pytest.raises(TypeError):
        silero_vad.process(1.0)
    with pytest.raises(ValueError):
        silero_vad.process([])
