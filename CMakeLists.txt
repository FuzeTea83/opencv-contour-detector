cmake_minimum_required(VERSION 3.25)
project(OpenCVHomework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress developer and deprecation warnings from external dependencies like OpenCV
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
# Fix potential SSL certificate verification issues during third-party downloads
set(CMAKE_TLS_VERIFY OFF CACHE BOOL "" FORCE)


# Default policies for subprojects to suppress compatibility warnings
foreach(_policy CMP0177 CMP0218 CMP0219 CMP0146 CMP0148 CMP0135)
    if(POLICY ${_policy})
        set(CMAKE_POLICY_DEFAULT_${_policy} NEW)
    endif()
endforeach()

include(FetchContent)

find_package(OpenCV 4.10 CONFIG QUIET COMPONENTS core imgproc imgcodecs highgui)

if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found -> FetchContent")

    # Only build the strictly necessary OpenCV modules
    set(BUILD_LIST "core,imgproc,imgcodecs,highgui" CACHE STRING "List of OpenCV modules" FORCE)
    set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)

    # Disable unnecessary modules, features and downloads
    set(BUILD_EXAMPLES      OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS         OFF CACHE BOOL "" FORCE)
    set(BUILD_PERF_TESTS    OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS          OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_apps   OFF CACHE BOOL "" FORCE)
    set(BUILD_JAVA          OFF CACHE BOOL "" FORCE)
    set(BUILD_JAVA_WRAPPERS OFF CACHE BOOL "" FORCE)
    set(BUILD_FAT_JAVA_LIB  OFF CACHE BOOL "" FORCE)
    set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python2  OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python3  OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_dnn      OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_gapi     OFF CACHE BOOL "" FORCE)
    set(BUILD_PACKAGE         OFF CACHE BOOL "" FORCE)
    set(BUILD_IPP_IW          OFF CACHE BOOL "" FORCE)
    set(BUILD_ITT             OFF CACHE BOOL "" FORCE)
    set(CV_TRACE              OFF CACHE BOOL "" FORCE)

    # Disable plugin loading to prevent missing DLL search logs at runtime
    set(PARALLEL_ENABLE_PLUGINS OFF CACHE BOOL "" FORCE)
    set(HIGHGUI_ENABLE_PLUGINS  OFF CACHE BOOL "" FORCE)
    set(VIDEOIO_ENABLE_PLUGINS  OFF CACHE BOOL "" FORCE)

    # Disable all AVX/AVX2/AVX512 generation on OpenCV side
    set(CPU_BASELINE                  "" CACHE STRING "" FORCE)
    set(CPU_DISPATCH                  "" CACHE STRING "" FORCE)
    set(CPU_BASELINE_WITH             "SSE;SSE2;SSE3;SSSE3;SSE4_1;SSE4_2" CACHE STRING "" FORCE)
    set(ENABLE_AVX2                   OFF CACHE BOOL "" FORCE)
    set(ENABLE_AVX512                 OFF CACHE BOOL "" FORCE)
    set(ENABLE_SSE                    ON  CACHE BOOL "" FORCE)
    set(ENABLE_SSE2                   ON  CACHE BOOL "" FORCE)
    set(ENABLE_SSE3                   ON  CACHE BOOL "" FORCE)
    set(ENABLE_SSSE3                  ON  CACHE BOOL "" FORCE)
    set(ENABLE_SSE41                  ON  CACHE BOOL "" FORCE)
    set(ENABLE_SSE42                  ON  CACHE BOOL "" FORCE)
    set(ENABLE_FMA3                   OFF CACHE BOOL "" FORCE)

    # Disable 3rdparty dependencies that cause download failures or warnings
    set(WITH_ADE            OFF CACHE BOOL "" FORCE)
    set(WITH_FFMPEG         OFF CACHE BOOL "" FORCE)
    set(WITH_GSTREAMER      OFF CACHE BOOL "" FORCE)
    set(WITH_1394           OFF CACHE BOOL "" FORCE)
    set(WITH_CUDA           OFF CACHE BOOL "" FORCE)
    set(WITH_DIRECTX        OFF CACHE BOOL "" FORCE)
    set(WITH_OPENCL         OFF CACHE BOOL "" FORCE)
    set(WITH_LAPACK         OFF CACHE BOOL "" FORCE)
    set(ENABLE_LAPACK       OFF CACHE BOOL "" FORCE)
    set(HAVE_LAPACK         OFF CACHE BOOL "" FORCE)
    set(LAPACK_LIBRARIES    "" CACHE STRING "" FORCE)
    set(WITH_EIGEN          OFF CACHE BOOL "" FORCE)
    set(WITH_VTK            OFF CACHE BOOL "" FORCE)
    set(WITH_PROTOBUF       OFF CACHE BOOL "" FORCE)
    set(WITH_FLATBUFFERS    OFF CACHE BOOL "" FORCE)
    set(WITH_QUIRC          OFF CACHE BOOL "" FORCE)
    set(WITH_TIFF           OFF CACHE BOOL "" FORCE)
    set(WITH_OPENJPEG       OFF CACHE BOOL "" FORCE)
    set(WITH_JASPER         OFF CACHE BOOL "" FORCE)
    set(WITH_WEBP           OFF CACHE BOOL "" FORCE)
    set(WITH_OPENEXR        OFF CACHE BOOL "" FORCE)
    set(WITH_IPP            OFF CACHE BOOL "" FORCE)
    set(WITH_ITT            OFF CACHE BOOL "" FORCE)
    set(WITH_CAROTENE       OFF CACHE BOOL "" FORCE)
    set(WITH_CPUFEATURES    OFF CACHE BOOL "" FORCE)
    set(WITH_TBB            OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
            opencv
            GIT_REPOSITORY https://github.com/opencv/opencv.git
            GIT_TAG        4.10.0
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
    )
    FetchContent_MakeAvailable(opencv)
endif()

foreach(hw homework1 homework2 homework3)
    add_executable(${hw} ${hw}/main.cpp)
    target_include_directories(${hw} PRIVATE
            ${hw}
            ${OpenCV_INCLUDE_DIRS}
            ${opencv_SOURCE_DIR}/include
            ${opencv_SOURCE_DIR}/modules/core/include
            ${opencv_SOURCE_DIR}/modules/imgproc/include
            ${opencv_SOURCE_DIR}/modules/imgcodecs/include
            ${opencv_SOURCE_DIR}/modules/highgui/include
            ${opencv_BINARY_DIR}
            ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(${hw} PRIVATE
            opencv_core
            opencv_imgproc
            opencv_imgcodecs
            opencv_highgui
    )
endforeach()