cmake_minimum_required(VERSION 3.12)
project(silero_vad_lite)

set(CMAKE_CXX_STANDARD 14)

# Find ONNX Runtime package
find_package(ONNXRuntime REQUIRED)

# Add the shared library target
add_library(silero_vad SHARED silero_vad.cpp)

# Link against ONNX Runtime
target_link_libraries(silero_vad PRIVATE ONNXRuntime::ONNXRuntime)

# Set the output name of the library
set_target_properties(silero_vad PROPERTIES
    OUTPUT_NAME "silero_vad"
    PREFIX ""
)

# Install the library
install(TARGETS silero_vad DESTINATION .)
