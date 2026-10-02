"""Check the release archive, including files needed to build and run the tests."""
import sys
import tarfile

required = {
    "pyproject.toml", "setup.py", "README.md", "LICENSE",
    "src/silero_vad_lite/CMakeLists.txt",
    "src/silero_vad_lite/silero_vad.cpp",
    "src/silero_vad_lite/silero_vad.py",
    "src/silero_vad_lite/data/silero_vad.onnx",
    "tests/test_silero_vad.py", "tests/sample.wav",
}
with tarfile.open(sys.argv[1], "r:gz") as archive:
    files = {name.partition("/")[2] for name in archive.getnames()}
missing = required - files
if missing:
    raise SystemExit(f"Source distribution is missing: {sorted(missing)}")
print(f"Source distribution contains all {len(required)} required build/test files")
