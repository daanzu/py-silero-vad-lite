# Bundled Silero model

The package bundles the **streaming** `silero_vad.onnx` from
[Silero VAD v6.2.3](https://github.com/snakers4/silero-vad/releases/tag/v6.2.3),
the latest non-prerelease upstream release checked on 2026-10-02. That release
was published on 2026-09-23. Its optional audio-backend changes apply to the
upstream Python package; this package does not depend on that package.
The streaming weights are the v6.2 model, last changed by upstream commit
[`bfdc0193023f121ea5b3cc7b176dbed570a68a59`](https://github.com/snakers4/silero-vad/commit/bfdc0193023f121ea5b3cc7b176dbed570a68a59)
on 2025-11-06 and retained in v6.2.3.

## Provenance

- Upstream repository: <https://github.com/snakers4/silero-vad>
- Release commit: `5cd7945676eb32225748052e2e6a0580e4686a08`
- [Exact model source](https://github.com/snakers4/silero-vad/blob/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.onnx)
- Git blob: `80c5592ef1f4c9ede3e357bbd02eb863358a6a9d`
- SHA-256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- Size: 2,327,524 bytes
- ONNX IR version 8; default-domain opset 16
- [Upstream MIT license](https://github.com/snakers4/silero-vad/blob/5cd7945676eb32225748052e2e6a0580e4686a08/LICENSE), copied byte-for-byte to `src/silero_vad_lite/data/LICENSE.silero`
- License SHA-256: `2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b`

The checked-in binary is copied without conversion, quantization, or other
modification. Builds use that local file and do not fetch a moving upstream
model. The test fixture locks both the model and license checksums.

## Streaming compatibility

The model retains the interface used by the previous v5.1 model:

- Mono float32 PCM at 8,000 or 16,000 Hz
- Exactly 256 or 512 new samples per call (32 ms)
- 32 or 64 preceding samples (4 ms) prepended by the wrapper
- Float32 recurrent state of shape `(2, 1, 128)` for a single stream
- Inputs `input`, `state`, and int64 `sr`; outputs `output` and `stateN`
- Zero context and recurrent state at initialization and after `reset()`

ONNX Runtime 1.19.0 can load and run this graph, so no runtime upgrade or
Python numerical dependency is needed. The existing C++ streaming integration
is unchanged. The new upstream `silero_vad_16k_sequence.onnx` is a separate
offline, 16 kHz-only model and is not used by this streaming API.

Updating the weights changes probabilities relative to v5.1, even for the same
audio and correct context. Users should recheck previously tuned thresholds.
This update makes no measured accuracy claim.

## Validation

`tests/generate_context_fixture.py` independently implements the pinned upstream
[`OnnxWrapper`](https://github.com/snakers4/silero-vad/blob/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/utils_vad.py)
streaming algorithm using NumPy and CPU ONNX Runtime 1.19.0. It does not import
the package under test. The fixture covers zero initial context and consecutive
stateful windows at both sample rates. Installed-wheel tests compare the native
wrapper with those scores, verify the bundled model/license bytes, test reset
and independent interleaved streams, and reject unexpected numerical imports.
See [fixture regeneration](../tests/fixtures/README.md).
