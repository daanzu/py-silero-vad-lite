
# Silero VAD Lite

[![PyPI Version](https://img.shields.io/pypi/v/silero-vad-lite.svg)](https://pypi.python.org/pypi/silero-vad-lite/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/silero-vad-lite.svg)](https://pypi.python.org/pypi/silero-vad-lite/)
[![Wheel Support](https://img.shields.io/pypi/wheel/silero-vad-lite.svg)](https://pypi.python.org/pypi/silero-vad-lite/)
<!-- [![Downloads per Month](https://img.shields.io/pypi/dm/silero-vad-lite.svg?logo=python)](https://pypi.python.org/pypi/silero-vad-lite/) -->
[![Build Status](https://github.com/daanzu/py-silero-vad-lite/actions/workflows/build_and_publish.yml/badge.svg)](https://github.com/daanzu/py-silero-vad-lite/actions/workflows/build_and_publish.yml)
[![Donate via PayPal](https://img.shields.io/badge/donate-PayPal-green.svg)](https://paypal.me/daanzu)
[![Sponsor on GitHub](https://img.shields.io/badge/sponsor-GitHub-pink.svg)](https://github.com/sponsors/daanzu)

Silero VAD Lite is a **lightweight Python wrapper** for the high-quality [Silero Voice Activity Detection (VAD)](https://github.com/snakers4/silero-vad) model using ONNX Runtime.

- **Simple interface** to use Silero VAD in Python, supporting **streaming audio** processing
- **Binary wheels** for **Windows, Linux, and MacOS** for easy installation
- **Zero dependencies** for the installable package, because it includes internally:
    - The Silero VAD model in ONNX format, so you don't need to supply it separately
    - The C++ ONNX Runtime (CPU), so the Python package for ONNX Runtime is not required

## Installation

You can install Silero VAD Lite using pip:

```
python -m pip install silero-vad-lite
```

This should install the package from the provided binary wheels, which are highly recommended. Installing from source is somewhat brittle and requires a C++ compiler.

## Usage

Here's a simple example of how to use Silero VAD Lite:

```python
from silero_vad_lite import SileroVAD
vad = SileroVAD(16000)  # sample_rate = 16000 Hz
speech_probability = vad.process(audio_data)
print(f"Voice activity detection probability of speech: {speech_probability}")  # 0 <= speech_probability <= 1
```

Requirements:
- Sample rate must be either 8000 Hz or 16000 Hz.
- Audio data must be 32-bit float PCM samples, normalized to the range [-1, 1], mono channel.
- Audio data must be supplied with length of 32ms (512 samples for 16kHz, 256 samples for 8kHz).
- Audio data can be supplied as: `bytes`, `bytearray`, `memoryview`, `array.array`, or `ctypes.Array`.

See docstrings in the code for more details.

## License

This project is licensed under the MIT License: see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Silero Team](https://github.com/snakers4/silero-vad) for the original Silero VAD model
- Microsoft for the [ONNX Runtime](https://github.com/microsoft/onnxruntime)

## Contributing

To build Silero VAD Lite from source:

1. Clone the repository:
    ```
    git clone https://github.com/daanzu/py-silero-vad-lite.git
    cd silero-vad-lite
    ```

2. Install the package (editable mode likely won't work):
    ```
    pip install .[dev]
    ```

This will compile the C++ extension and install the package, including the development dependencies for testing, which can be run with:

```
python -m pytest
```

Contributions are welcome! Please feel free to submit a Pull Request.
