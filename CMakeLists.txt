cmake_minimum_required(VERSION 3.15)
project(vision_detector VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTS "Build unit tests" OFF)
option(USE_GPU_DELEGATE "Use GPU delegate for TFLite" ON)
option(USE_TFLITE "Enable TensorFlow Lite inference" OFF)  # OFF for testing without TFLite

# Platform detection
if(APPLE)
    set(PLATFORM_MACOS TRUE)
    message(STATUS "Platform: macOS")
elseif(UNIX)
    set(PLATFORM_LINUX TRUE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(PLATFORM_JETSON TRUE)
        message(STATUS "Platform: Linux (Jetson)")
    else()
        message(STATUS "Platform: Linux (x86_64)")
    endif()
endif()

# Dependencies
# TensorFlow Lite (optional for testing)
find_package(PkgConfig QUIET)

if(USE_TFLITE)
    # Try to find TFLite via pkg-config or manual path
    if(DEFINED TFLITE_ROOT)
        set(TFLITE_INCLUDE_DIRS "${TFLITE_ROOT}/include")
        set(TFLITE_LIBRARIES "${TFLITE_ROOT}/lib/libtensorflowlite.so")
    else()
        # Fallback: try common locations
        find_path(TFLITE_INCLUDE_DIRS tensorflow/lite/interpreter.h
            HINTS /usr/local/include /opt/tensorflow/include
        )
        find_library(TFLITE_LIBRARIES tensorflowlite tensorflow-lite
            HINTS /usr/local/lib /opt/tensorflow/lib
        )
    endif()

    if(TFLITE_INCLUDE_DIRS AND TFLITE_LIBRARIES)
        message(STATUS "TFLite found: ${TFLITE_LIBRARIES}")
        add_compile_definitions(USE_TFLITE=1)
    else()
        message(FATAL_ERROR "TFLite not found but USE_TFLITE=ON")
    endif()
else()
    message(STATUS "TFLite disabled (USE_TFLITE=OFF) - using placeholder inference")
    set(TFLITE_INCLUDE_DIRS "")
    set(TFLITE_LIBRARIES "")
endif()

# Shared protocol submodule
add_subdirectory(third_party/detector-protocol)

# Source files
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/preprocessor.cpp
    src/server.cpp
)

set(HEADERS
    include/detector.h
    include/preprocessor.h
    include/server.h
)

# Main executable
add_executable(vision_detector ${SOURCES} ${HEADERS})

target_include_directories(vision_detector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TFLITE_INCLUDE_DIRS}
)

target_link_libraries(vision_detector PRIVATE
    detector_protocol
    ${TFLITE_LIBRARIES}
    pthread
)

# Platform-specific libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(vision_detector PRIVATE rt)
endif()

# Installation
install(TARGETS vision_detector
    RUNTIME DESTINATION bin
)

install(DIRECTORY private/models/
    DESTINATION share/vision_detector/models
    FILES_MATCHING PATTERN "*.tflite"
)

install(DIRECTORY private/config/
    DESTINATION share/vision_detector/config
    FILES_MATCHING PATTERN "*.json"
)

# Summary
message(STATUS "")
message(STATUS "=== Vision Detector Configuration ===")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  TFLite: ${USE_TFLITE}")
message(STATUS "  GPU Delegate: ${USE_GPU_DELEGATE}")
message(STATUS "======================================")
message(STATUS "")