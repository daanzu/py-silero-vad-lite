# Linux stack permissions

The Linux native target links with `-Wl,-z,noexecstack`. This declares an ordinary
read/write, non-executable stack; it does not relax a loader or OS security policy.
Windows and macOS linking are unchanged.

## Why this is needed with ONNX Runtime 1.19

The existing static dependency is
[`onnxruntime-linux-x64-static_lib-1.19.0-glibc2_17.zip`](https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.19.0/onnxruntime-linux-x64-static_lib-1.19.0-glibc2_17.zip).
The inspected archive's SHA-256 is
`9b3d4ecbe8e8b2e401af18891e0260d2ce37a8a80749adeb809e39eb051d7823`.

`readelf -SW libonnxruntime.a` shows 793 objects. Of these, 38 MLAS assembly
objects (`*.S.o`) lack `.note.GNU-stack`; none has a note requesting an executable
stack. For example, `SgemmKernelSse2.S.o`, `SgemmKernelAvx.S.o`, and
`QgemmU8S8KernelAmx.S.o` omit the note. Older GNU linkers conservatively treat
missing notes as an executable-stack requirement. The published 0.2.1 Linux
x86-64 wheel consequently has `GNU_STACK RWE`, and cannot be loaded where the
loader is forbidden to enable an executable stack.

The upstream [SSE2 kernel](https://github.com/microsoft/onnxruntime/blob/v1.19.0/onnxruntime/core/mlas/lib/x86_64/SgemmKernelSse2.S)
and [assembly macros](https://github.com/microsoft/onnxruntime/blob/v1.19.0/onnxruntime/core/mlas/lib/x86_64/asmmacro.h)
place instructions in `.text` and omit the GNU-stack annotation. These are native
matrix/vector kernels, not stack-generated trampolines. The linker flag corrects
this missing metadata at the final shared-library link while retaining the
existing ONNX Runtime version and static linking choice.

## Regression checks

`python -m pytest tests/test_linux_stack.py` inspects ELF program headers directly,
without requiring `readelf` or an extra Python dependency. It requires an explicit
non-executable `PT_GNU_STACK` on the packaged Linux native library. Run it against the
installed wheel and against a wheel built from the source distribution. The
regular inference tests also verify that the library loads and executes with
these permissions.

For manual inspection:

```sh
readelf -W -l path/to/silero_vad_lite/data/silero_vad_lite.so | grep GNU_STACK
```

The expected flags are `RW`, without `E`. Recheck the archive metadata and run
inference tests whenever updating ONNX Runtime or changing native toolchains.
