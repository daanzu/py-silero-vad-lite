# Silero VAD Lite

Silero VAD Lite is a lightweight Python wrapper for the Silero Voice Activity Detection (VAD) model using ONNX Runtime. This package provides a simple interface to use the Silero VAD model without the need for the full ONNX Runtime Python package.

## Installation

You can install Silero VAD Lite using pip:

```
pip install silero-vad-lite
```

Note: This package requires CMake and a C++ compiler to be installed on your system.

## Usage

Here's a simple example of how to use Silero VAD Lite:

```python
from silero_vad_lite import SileroVAD
import numpy as np

# Initialize the SileroVAD object with the path to your ONNX model
vad = SileroVAD("path/to/silero_vad.onnx")

# Generate some dummy audio data (replace this with your actual audio data)
sample_rate = 16000
duration = 1  # 1 second
t = np.linspace(0, duration, int(sample_rate * duration), False)
audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Process the audio data
result = vad.process(audio_data, sample_rate)

print(f"Voice activity detection result: {result}")
```

The `process` method returns a float value between 0 and 1, indicating the probability of voice activity in the given audio segment.

## Building from Source

To build Silero VAD Lite from source:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/silero-vad-lite.git
   cd silero-vad-lite
   ```

2. Install the package in editable mode:
   ```
   pip install -e .
   ```

This will compile the C++ extension and install the package.

## Running Tests

To run the tests, first install pytest:

```
pip install pytest
```

Then, from the root directory of the project, run:

```
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Silero Team](https://github.com/snakers4/silero-vad) for the original Silero VAD model
- ONNX Runtime team for the C++ API

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
