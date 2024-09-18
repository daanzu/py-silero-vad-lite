cmake_minimum_required(VERSION 3.12)
project(silero_vad_lite)

set(CMAKE_CXX_STANDARD 14)

# Set ONNX Runtime path
set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Add the shared library target
add_library(silero_vad SHARED silero_vad.cpp)

# Link against ONNX Runtime
target_link_libraries(silero_vad PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)

# Set the output name of the library
if(WIN32)
    set_target_properties(silero_vad PROPERTIES
        OUTPUT_NAME "silero_vad"
        PREFIX ""
        SUFFIX ".dll"
    )
elseif(APPLE)
    set_target_properties(silero_vad PROPERTIES
        OUTPUT_NAME "silero_vad"
        PREFIX ""
        SUFFIX ".dylib"
    )
else()
    set_target_properties(silero_vad PROPERTIES
        OUTPUT_NAME "silero_vad"
        PREFIX ""
        SUFFIX ".so"
    )
endif()

# Install the library
install(TARGETS silero_vad DESTINATION .)

# Print debug info
# message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
