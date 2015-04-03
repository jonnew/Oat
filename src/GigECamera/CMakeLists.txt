cmake_minimum_required (VERSION 2.8)
project (GigECamer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

set (BOOST_ROOT /opt/boost_1_57_0 )
set (CPP_PTHREAD_LINK_FLAG "-lpthread")
set (CPP_RT_LINK_FLAG "-lrt")

find_package (Boost 1.57.0 EXACT REQUIRED)

include_directories ("/usr/include/flycapture")
find_library(FLYCAPTURE2 flycapture)

add_subdirectory("../../lib/shmem" "${CMAKE_CURRENT_BINARY_DIR}/shmem_build")

find_package (OpenCV REQUIRED)

add_executable (camserv_ge CameraControl.cpp main.cpp )
target_link_libraries (camserv_ge shmem ${OpenCV_LIBS} ${FLYCAPTURE2} ${CPP_PTHREAD_LINK_FLAG} ${CPP_RT_LINK_FLAG})
