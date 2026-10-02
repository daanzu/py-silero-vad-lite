"""Numerical v6.2.3 parity without numpy, torch or Python ONNX Runtime."""
import array
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import wave

import pytest

from silero_vad_lite import SileroVAD

ROOT = Path(__file__).resolve().parent
REFERENCE = json.loads((ROOT / 'fixtures/context_v6_2_3.json').read_text())


def windows(sample_rate):
    with wave.open(str(ROOT / 'sample.wav'), 'rb') as wav:
        pcm = wav.readframes(wav.getnframes())
    data = array.array('f', (value / 32768.0 for (value,) in struct.iter_unpack('<h', pcm)))
    data = data[::16000 // sample_rate]
    size = sample_rate * 32 // 1000
    return [data[i:i + size] for i in range(0, len(data) - size + 1, size)]


def test_reference_inputs_unchanged():
    assert hashlib.sha256(Path(SileroVAD._get_model_path()).read_bytes()).hexdigest() == REFERENCE['model_sha256']
    assert hashlib.sha256((ROOT / 'sample.wav').read_bytes()).hexdigest() == REFERENCE['audio_sha256']


def test_bundled_model_license():
    license_path = Path(SileroVAD._get_model_path()).with_name('LICENSE.silero')
    assert hashlib.sha256(license_path.read_bytes()).hexdigest() == REFERENCE['license_sha256']


@pytest.mark.parametrize('sample_rate', [8000, 16000])
def test_upstream_context_parity(sample_rate):
    vad = SileroVAD(sample_rate)
    actual = [vad.process(chunk) for chunk in windows(sample_rate)]
    # Covers zero context in the first frame and carried context/state thereafter.
    assert actual == pytest.approx(REFERENCE['probabilities'][str(sample_rate)], abs=1e-6, rel=1e-5)


@pytest.mark.parametrize('sample_rate', [8000, 16000])
def test_reset_matches_fresh_instance(sample_rate):
    chunks = windows(sample_rate)
    vad = SileroVAD(sample_rate)
    fresh = SileroVAD(sample_rate)
    vad.reset()  # Reset before first use is supported.
    for chunk in chunks:
        vad.process(chunk)
    vad.reset()
    vad.reset()  # Idempotent; both recurrent state and context must be cleared.
    assert vad.sample_rate == sample_rate
    assert vad.window_size_samples == sample_rate * 32 // 1000
    for chunk in chunks:
        assert vad.process(chunk) == fresh.process(chunk)


def test_interleaved_sample_rates_keep_independent_state():
    streams = {rate: SileroVAD(rate) for rate in (8000, 16000)}
    chunks = {rate: windows(rate) for rate in streams}
    for index in range(min(map(len, chunks.values()))):
        for rate, vad in streams.items():
            assert vad.process(chunks[rate][index]) == pytest.approx(
                REFERENCE['probabilities'][str(rate)][index], abs=1e-6, rel=1e-5)


@pytest.mark.parametrize('sample_rate', [8000, 16000])
def test_invalid_window_does_not_advance_state(sample_rate):
    vad = SileroVAD(sample_rate)
    fresh = SileroVAD(sample_rate)
    chunks = windows(sample_rate)
    assert vad.process(chunks[0]) == fresh.process(chunks[0])
    with pytest.raises(ValueError):
        vad.process(chunks[1][:-1])
    assert vad.process(chunks[1]) == fresh.process(chunks[1])


@pytest.mark.parametrize('sample_rate', [0, -16000, 44100])
def test_invalid_sample_rate(sample_rate):
    with pytest.raises(ValueError, match='Sample rate'):
        SileroVAD(sample_rate)


def test_missing_model_raises_instead_of_crashing(tmp_path):
    with pytest.raises(RuntimeError, match='Failed to initialize'):
        SileroVAD(16000, str(tmp_path / 'missing.onnx'))


def test_runtime_does_not_import_numerical_packages():
    # A separate interpreter also catches accidental eager imports, even when
    # the test environment happens to have numerical packages installed.
    code = '''
import sys
class BlockNumericalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'onnxruntime'}:
            raise AssertionError('Unexpected runtime dependency: ' + fullname)
sys.meta_path.insert(0, BlockNumericalImports())
import array
from silero_vad_lite import SileroVAD
for rate in (8000, 16000):
    vad = SileroVAD(rate)
    frame = array.array('f', [0.0]) * vad.window_size_samples
    score = vad.process(frame)
    vad.reset()
    assert vad.process(frame) == score
'''
    subprocess.run([sys.executable, '-c', code], check=True)
