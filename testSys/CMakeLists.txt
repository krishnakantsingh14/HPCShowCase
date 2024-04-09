cmake_minimum_required(VERSION 3.10)
project(TimeIt)

enable_testing()

# Find the MPI package
find_package(MPI REQUIRED)

# Set the C++ compiler to mpicxx
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_STANDARD 17)

add_executable(TimeIt TimeIt.cpp)
add_executable(MPITimeIt MPITimeIt.cpp)

# Need an include path to the mpi.h file and to the MPI library
target_include_directories(MPITimeIt PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(MPITimeIt ${MPI_LIBRARIES})

target_link_libraries(TimeIt PRIVATE pthread)