cmake_minimum_required(VERSION 3.22)
project(ucc_test)

set(CMAKE_CXX_STANDARD 17)

find_package (Threads REQUIRED)

# mpi
find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "MPI include dir: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI libs: ${MPI_CXX_LIBRARIES}")

# ucx 
if (NOT UCX_INSTALL_PREFIX)
    message(FATAL_ERROR "UCX_INSTALL_PREFIX is missing")
endif ()
find_package(ucx REQUIRED CONFIG HINTS ${UCX_INSTALL_PREFIX})
message(STATUS "UCX include dirs: ${UCX_INCLUDE_DIRS}")
message(STATUS "UCX libs: ${UCX_LIBRARIES}")

# ucc
if (NOT UCC_INSTALL_PREFIX)
    message(FATAL_ERROR "UCC_INSTALL_PREFIX is missing")
endif ()

find_package(ucc REQUIRED CONFIG HINTS ${UCC_INSTALL_PREFIX})
message(STATUS "UCC include dirs: ${UCC_INCLUDE_DIRS}")
message(STATUS "UCC libs: ${UCC_LIBRARIES}")

include_directories(${MPI_CXX_INCLUDE_PATH} ${UCX_INCLUDE_DIRS} ${UCC_INCLUDE_DIRS})
link_directories("${UCX_LIBRARIES}/ucx" ${UCC_LIBRARIES} ${UCC_LIBRARIES}/ucc)

add_executable(ucc_test main.cpp)

target_link_libraries(ucc_test ${MPI_CXX_LIBRARIES} ucx::ucp ucc ${CMAKE_THREAD_LIBS_INIT})
