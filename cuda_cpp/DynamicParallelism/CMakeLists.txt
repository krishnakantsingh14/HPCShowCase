cmake_minimum_required(VERSION 3.10)

project(dynamicParallel VERSION 1.0 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(dynamicParallel helloworld.cu)

set_target_properties(dynamicParallel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)

target_link_libraries(dynamicParallel ${CUDA_LIBRARIES})

# Add OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dynamicParallel OpenMP::OpenMP_CXX)
endif()