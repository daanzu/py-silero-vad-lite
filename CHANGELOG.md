# Changelog

## 0.3.0

### Compatibility

- Requires Python 3.10 or newer. Tested wheels target standard, GIL-enabled CPython 3.10–3.14 on Linux x86-64 (glibc 2.17+), Windows x86-64, and macOS Intel/Apple Silicon. PyPy and free-threaded CPython are outside the wheel test matrix.
- Streaming probabilities change: the wrapper now supplies the preceding 4 ms of audio required by Silero v5.1. Recheck thresholds tuned against earlier releases.
- The bundled Silero v5.1 model and ONNX Runtime 1.19.0 are unchanged. No runtime Python dependencies were added.

### Changes

- Add `SileroVAD.reset()` to clear recurrent state and audio context between independent streams without reloading the model.
- Include missing C++/CMake sources and test fixtures in source distributions; build and test release wheels from the actual source archive.
- Correct Linux executable-stack metadata inherited from ONNX Runtime assembly so the library can load under non-executable-stack restrictions.
- Raise Python exceptions for invalid sample rates and native model initialization failures instead of dereferencing a null native object.
- Refresh wheel builders and runners, with numerical parity and reset tests at 8 and 16 kHz.
