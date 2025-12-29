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
    # Try to find TFLite - check for local build first
    set(TFLITE_LOCAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow")
    set(TFLITE_LOCAL_BUILD "${TFLITE_LOCAL_ROOT}/tflite_build")

    if(EXISTS "${TFLITE_LOCAL_BUILD}/libtensorflow-lite.a")
        # Use locally built TFLite
        message(STATUS "Using local TFLite build: ${TFLITE_LOCAL_BUILD}")
        set(TFLITE_INCLUDE_DIRS
            "${TFLITE_LOCAL_ROOT}"
            "${TFLITE_LOCAL_BUILD}/flatbuffers/include"
        )
        set(TFLITE_LIBRARIES "${TFLITE_LOCAL_BUILD}/libtensorflow-lite.a")

        # TFLite dependencies from the CMake build
        set(TFLITE_DEP_LIBS
            "${TFLITE_LOCAL_BUILD}/_deps/flatbuffers-build/libflatbuffers.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/strings/libabsl_strings.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/strings/libabsl_strings_internal.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/base/libabsl_base.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/base/libabsl_spinlock_wait.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/base/libabsl_throw_delegate.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/base/libabsl_raw_logging_internal.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/base/libabsl_log_severity.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/numeric/libabsl_int128.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/synchronization/libabsl_synchronization.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/synchronization/libabsl_graphcycles_internal.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/time/libabsl_time.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/time/libabsl_civil_time.a"
            "${TFLITE_LOCAL_BUILD}/_deps/abseil-cpp-build/absl/time/libabsl_time_zone.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_ctx.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_context.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_context_get_ctx.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_frontend.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_trmul.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_block_map.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_cpuinfo.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_allocator.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_tune.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_wait.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_thread_pool.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_blocking_counter.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_prepare_packed_matrices.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_prepacked_cache.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_apply_multiplier.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_system_aligned_alloc.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_kernel_arm.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_pack_arm.a"
            "${TFLITE_LOCAL_BUILD}/_deps/ruy-build/ruy/libruy_denormal.a"
            "${TFLITE_LOCAL_BUILD}/_deps/cpuinfo-build/libcpuinfo.a"
            "${TFLITE_LOCAL_BUILD}/_deps/farmhash-build/libfarmhash.a"
            "${TFLITE_LOCAL_BUILD}/_deps/fft2d-build/libfft2d_fftsg.a"
            "${TFLITE_LOCAL_BUILD}/_deps/fft2d-build/libfft2d_fftsg2d.a"
            "${TFLITE_LOCAL_BUILD}/pthreadpool/libpthreadpool.a"
        )
    elseif(DEFINED TFLITE_ROOT)
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
    set(TFLITE_DEP_LIBS "")
endif()

# Shared protocol submodule
add_subdirectory(third_party/detector-protocol)

# Source files
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/preprocessor.cpp
    src/server.cpp
    src/model_manager.cpp
    src/evaluator.cpp
)

set(HEADERS
    include/detector.h
    include/preprocessor.h
    include/server.h
    include/model_manager.h
    include/evaluator.h
    include/json_utils.h
)

# Main executable
add_executable(vision_detector ${SOURCES} ${HEADERS})

target_include_directories(vision_detector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
    ${TFLITE_INCLUDE_DIRS}
)

target_link_libraries(vision_detector PRIVATE
    detector_protocol
    ${TFLITE_LIBRARIES}
    ${TFLITE_DEP_LIBS}
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