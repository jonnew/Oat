cmake_minimum_required (VERSION 2.8)
project (OAT)

# C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# Boost
set (BOOST_ROOT /opt/boost_1_57_0 )
find_package (Boost REQUIRED system thread program_options filesystem)
link_directories (${Boost_LIBRARY_DIR})

# OpenCV
find_package (OpenCV REQUIRED)

# oat shared memory lib
add_subdirectory("../lib/shmem" "${CMAKE_CURRENT_BINARY_DIR}/shmem_build")

# oat components
add_subdirectory(cleaner)
add_subdirectory(decorator)


