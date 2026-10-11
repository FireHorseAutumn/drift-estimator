cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
project(drift-estimator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_library(de INTERFACE)
target_include_directories(de INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(de INTERFACE
    CUDA::cudart CUDA::cublas Eigen3::Eigen
)

# TODO: add targets as sources land
# add_executable(test_ekf_cpu_ref tests/test_ekf_cpu_ref.cpp)
# add_executable(bench_drift benchmarks/bench_drift.cpp)

enable_testing()
