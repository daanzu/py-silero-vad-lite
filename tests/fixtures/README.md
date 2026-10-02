# Silero v5.1 context reference

`context_v5_1.json` contains deterministic CPU ONNX Runtime 1.19.0 scores for the
bundled model and `tests/sample.wav`, with SHA-256 hashes for both inputs. The
independent reference follows the upstream v5.1 `OnnxWrapper` algorithm: zero
initial state `(2, 1, 128)`, zero initial context, prepend the last 64 samples at
16 kHz or 32 samples at 8 kHz, run the model, then carry forward state and context.

The reference processes complete 32 ms frames only. The 8 kHz input is obtained
by selecting every second sample of the supplied 16 kHz WAV; this deterministic
test signal is not a recommendation for production audio resampling. PCM int16
values are divided by 32768 and converted to float32.

Regenerate from the repository root in a development environment with NumPy and
`onnxruntime==1.19.0` installed:

```sh
python tests/generate_context_fixture.py
```

The generator does not import this package and uses the model directly. The
normal parity tests read the checked-in scores and need neither NumPy, PyTorch,
nor the Python ONNX Runtime package. They allow `abs=1e-6, rel=1e-5` for numerical
variation across supported CPU architectures. The reset tests additionally
require exact equality to a fresh instance on the same machine.
