cmake_minimum_required(VERSION 3.12)
project(silero_vad_lite)

set(CMAKE_CXX_STANDARD 14)

# Set the output directory, so that the library is built directly into the package directory
# NOTE: We need to set "library" on linux for the .so, and "runtime" on windows for the .dll (and "_config" to avoid placing it in a "Release" or "Debug" subdirectory)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PACKAGE_DATA_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PACKAGE_DATA_OUTPUT_DIRECTORY})
# For multi-config builds (e.g., Visual Studio)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PACKAGE_DATA_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PACKAGE_DATA_OUTPUT_DIRECTORY})
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

# Set ONNX Runtime path
set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Add the shared library target
add_library(silero_vad_lite SHARED silero_vad.cpp)

# Link against ONNX Runtime statically
if(WIN32)
    target_link_libraries(silero_vad_lite PRIVATE ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
elseif(APPLE)
    target_link_libraries(silero_vad_lite PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.a)
else()
    target_link_libraries(silero_vad_lite PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.a)
endif()

# Add additional system libraries required for static linking
if(UNIX AND NOT APPLE)
    target_link_libraries(silero_vad_lite PRIVATE pthread dl)
endif()

# Set the output name of the library
if(WIN32)
    set_target_properties(silero_vad_lite PROPERTIES
        OUTPUT_NAME "silero_vad_lite"
        PREFIX ""
        SUFFIX ".dll"
    )
elseif(APPLE)
    set_target_properties(silero_vad_lite PROPERTIES
        OUTPUT_NAME "silero_vad_lite"
        PREFIX ""
        SUFFIX ".dylib"
    )
else()
    set_target_properties(silero_vad_lite PROPERTIES
        OUTPUT_NAME "silero_vad_lite"
        PREFIX ""
        SUFFIX ".so"
    )
endif()

# Install the library
install(TARGETS silero_vad_lite DESTINATION .)

# Print debug info
# message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
# message(STATUS "PACKAGE_OUTPUT_DIRECTORY: ${PACKAGE_OUTPUT_DIRECTORY}")
