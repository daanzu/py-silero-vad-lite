"""Checks that also run against the installed, repaired release wheels."""
from importlib.metadata import distribution
from pathlib import Path

from silero_vad_lite import SileroVAD


def test_installed_distribution():
    metadata = distribution("silero-vad-lite")
    assert metadata.metadata["Requires-Python"] == ">=3.10"
    assert not [item for item in metadata.requires or [] if "extra ==" not in item]
    assert Path(SileroVAD._get_model_path()).is_file()
    assert Path(SileroVAD._get_lib_path()).is_file()
