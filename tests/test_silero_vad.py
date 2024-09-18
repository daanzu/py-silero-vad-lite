import pytest
import numpy as np
from silero_vad_lite import SileroVAD

@pytest.fixture
def silero_vad():
    return SileroVAD(16000)

def test_silero_vad_process(silero_vad):
    # Generate some dummy audio data
    sample_rate = 16000
    duration = 1  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Process the audio data
    result = silero_vad.process(audio_data)

    # Check if the result is a float between 0 and 1
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_silero_vad_invalid_input(silero_vad):
    with pytest.raises(TypeError):
        silero_vad.process("invalid input")

    with pytest.raises(ValueError):
        silero_vad.process([])

    with pytest.raises(ValueError):
        silero_vad.process([1.0, 2.0, 3.0])
