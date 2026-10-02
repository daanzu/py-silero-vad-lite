"""Regenerate the reference with test-only numpy and onnxruntime==1.19.0.

Run from the repository root: python tests/generate_context_fixture.py
The NumPy-only reference below implements OnnxWrapper.__call__ from:
https://github.com/snakers4/silero-vad/blob/v5.1/src/silero_vad/utils_vad.py
No package implementation is imported or used to generate expected results.
"""
import hashlib
import json
from pathlib import Path
import struct
import wave

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = '2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f'


def main():
    assert ort.__version__ == '1.19.0', ort.__version__
    model = ROOT / 'src/silero_vad_lite/data/silero_vad.onnx'
    assert hashlib.sha256(model.read_bytes()).hexdigest() == MODEL_SHA256
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    wav_path = ROOT / 'tests/sample.wav'
    with wave.open(str(wav_path), 'rb') as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        pcm = wav.readframes(wav.getnframes())
    samples = [value / 32768.0 for (value,) in struct.iter_unpack('<h', pcm)]
    fixture = {
        'reference': 'https://github.com/snakers4/silero-vad/blob/v5.1/src/silero_vad/utils_vad.py',
        'model_source': 'https://github.com/snakers4/silero-vad/blob/v5.1/src/silero_vad/data/silero_vad.onnx',
        'model_git_blob': 'b3e3a900c0d70e67b5e2b90a33ad856ee7947930',
        'model_sha256': MODEL_SHA256,
        'audio_sha256': hashlib.sha256(wav_path.read_bytes()).hexdigest(),
        'onnxruntime_version': ort.__version__,
        'description': 'sample.wav, normalized int16 PCM; every second sample for 8 kHz (deterministic test input, not production resampling); complete 32 ms windows only',
        'probabilities': {},
    }
    for rate in (8000, 16000):
        signal = np.asarray(samples[::16000 // rate], dtype=np.float32)
        window = rate * 32 // 1000
        context_size = 64 if rate == 16000 else 32
        context = np.zeros((1, context_size), dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        probabilities = []
        for start in range(0, len(signal) - window + 1, window):
            x = np.concatenate((context, signal[start:start + window].reshape(1, -1)), axis=1)
            out, state = session.run(None, {'input': x, 'state': state,
                                           'sr': np.array(rate, dtype=np.int64)})
            probabilities.append(float(out[0, 0]))
            context = x[:, -context_size:]
        fixture['probabilities'][str(rate)] = probabilities
    (ROOT / 'tests/fixtures/context_v5_1.json').write_text(json.dumps(fixture, indent=2) + '\n')


if __name__ == '__main__':
    main()
