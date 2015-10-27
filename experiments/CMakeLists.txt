cmake_minimum_required (VERSION 2.8)

project (Oat-Experiments)
 
# C++11
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# Boost
find_package (Boost 1.53.0  REQUIRED system thread program_options filesystem)
include_directories (${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIR})

# OpenCV
find_package (OpenCV REQUIRED)
if (OpenCV_FOUND)
    message (STATUS "Found OpenCV")
else (OpenCV_FOUND)
    message (CRITICAL_ERROR "OpenCV not found")
endif (OpenCV_FOUND)

# Shmem dir
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src)

# All executables should be installed in Oat/oat/libexec
set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Default install path" FORCE)
