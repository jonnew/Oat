cmake_minimum_required (VERSION 2.8)
project (oat)

# C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# Boost
set (BOOST_ROOT /opt/boost_1_57_0 )
find_package (Boost REQUIRED system thread program_options filesystem)
link_directories (${Boost_LIBRARY_DIR})

# OpenCV
find_package (OpenCV REQUIRED)

# oat shared memory lib
add_subdirectory(lib/shmem "${CMAKE_CURRENT_BINARY_DIR}/shmem")

# oat components
add_subdirectory(src/cleaner)
add_subdirectory(src/decorator)
add_subdirectory(src/framefilter)
add_subdirectory(src/frameserver)
add_subdirectory(src/frameviewer)
add_subdirectory(src/positioncombiner)
add_subdirectory(src/positiondetector)
add_subdirectory(src/positionfilter)
add_subdirectory(src/positiontester)
add_subdirectory(src/recorder)

# All executables should be installed in oat/libexec
set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../oat/libexec" CACHE PATH "default install path" FORCE ) 

