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

# Link against ONNX Runtime
option(ONNXRUNTIME_STATIC "Link against ONNX Runtime statically" OFF)
if(WIN32)
    target_link_libraries(silero_vad_lite PRIVATE ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
    if(ONNXRUNTIME_STATIC)
        target_compile_options(silero_vad_lite PRIVATE $<$<CONFIG:Debug>:/MTd> $<$<CONFIG:Release>:/MT>)
    endif()
elseif(APPLE)
    if(ONNXRUNTIME_STATIC)
        target_link_libraries(silero_vad_lite PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.a)
    else()
        target_link_libraries(silero_vad_lite PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib)
    endif()
else()
    if(ONNXRUNTIME_STATIC)
        target_link_libraries(silero_vad_lite PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.a)
    else()
        target_link_libraries(silero_vad_lite PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
        set_target_properties(silero_vad_lite PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()

# Prevent prefixing the library with "lib" on linux & macos
set_target_properties(silero_vad_lite PROPERTIES PREFIX "")

# Export all symbols on Windows, to avoid needing to export each function with __declspec(dllexport)
# set_target_properties(silero_vad_lite PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

# Install the library
install(TARGETS silero_vad_lite DESTINATION .)

# Print debug info
# message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
# message(STATUS "PACKAGE_OUTPUT_DIRECTORY: ${PACKAGE_OUTPUT_DIRECTORY}")
