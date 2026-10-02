"""Check the installed native library stack permissions."""
import pathlib
import struct
import sys

import pytest
import silero_vad_lite


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU_STACK is ELF-specific')
def test_native_library_does_not_require_executable_stack():
    package_dir = pathlib.Path(silero_vad_lite.__file__).parent
    libraries = [package_dir / 'data' / 'silero_vad_lite.so']
    for library in libraries:
        data = library.read_bytes()
        assert data[:4] == b'\x7fELF', str(library)
        assert data[4] in (1, 2), 'Unsupported ELF class'
        assert data[5] in (1, 2), 'Unsupported ELF byte order'
        endian = '<' if data[5] == 1 else '>'
        if data[4] == 2:  # ELF64
            phoff = struct.unpack_from(endian + 'Q', data, 32)[0]
            phentsize, phnum = struct.unpack_from(endian + 'HH', data, 54)
            flags_offset = 4
        else:  # ELF32
            phoff = struct.unpack_from(endian + 'I', data, 28)[0]
            phentsize, phnum = struct.unpack_from(endian + 'HH', data, 42)
            flags_offset = 24
        stack_flags = []
        for index in range(phnum):
            offset = phoff + index * phentsize
            segment_type = struct.unpack_from(endian + 'I', data, offset)[0]
            if segment_type == 0x6474E551:  # PT_GNU_STACK
                stack_flags.append(struct.unpack_from(
                    endian + 'I', data, offset + flags_offset)[0])
        assert len(stack_flags) == 1, f'{library}: missing GNU_STACK declaration'
        assert not stack_flags[0] & 1, f'{library}: executable stack requested'
