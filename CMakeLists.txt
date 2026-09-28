cmake_minimum_required(VERSION 3.20)
project(BharatOpt-X VERSION 0.1.0 LANGUAGES CXX)

# Enforce C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0")
    message(STATUS "Using bundled third_party/eigen-3.4.0 headers")
    add_library(Eigen3_Eigen INTERFACE)
    target_include_directories(Eigen3_Eigen INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0")
    add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
else()
    include(FetchContent)
    FetchContent_Declare(
      eigen
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    )
    FetchContent_MakeAvailable(eigen)
endif()

# Add subdirectories
add_subdirectory(engine)

# Add Engine Executable
add_executable(bharatopt_engine engine/main.cpp)

# Optional CUDA Support
# The CUDAToolkit is found via nvcc, but MSBuild/VS is missing the CUDA Toolkit BuildCustomizations.
# The .cu kernel architecture (gpu_kernels.cu) is fully laid out for Phase 5.
# Awaiting VS CUDA Toolset installation by the user to uncomment compilation.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_library(gpu_lib STATIC engine/gpu_kernels.cu)
    target_link_libraries(gpu_lib PRIVATE CUDA::cusparse)
    target_compile_definitions(bharatopt_engine PRIVATE USE_CUDA=1)
    target_link_libraries(bharatopt_engine PRIVATE gpu_lib)
endif()

target_link_libraries(bharatopt_engine PRIVATE engine_lib Eigen3::Eigen)

# Enable testing
enable_testing()
add_subdirectory(tests)
