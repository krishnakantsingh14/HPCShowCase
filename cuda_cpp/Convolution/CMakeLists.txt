cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(Convolution1DSharedMemory VERSION 1.0 LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA
find_package(CUDA REQUIRED)

# Add the executable
add_executable(convolution_1d_shared_memory conv1d.cu)

# Set the CUDA architecture
set_target_properties(convolution_1d_shared_memory PROPERTIES CUDA_ARCHITECTURES "all")

# Set the compile options
target_compile_options(convolution_1d_shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --extended-lambda
>)

# Link the CUDA runtime library
target_link_libraries(convolution_1d_shared_memory ${CUDA_LIBRARIES})
#target_link_libraries(convolution_1d_shared_memory CUDA::cudart)