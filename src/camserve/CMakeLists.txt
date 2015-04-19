cmake_minimum_required (VERSION 2.8)
project (CamServe)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

set (BOOST_ROOT /opt/boost_1_57_0 )
find_package (Boost REQUIRED system thread program_options)
link_directories (${Boost_LIBRARY_DIR})

include_directories ("/usr/include/flycapture")
find_library(FLYCAPTURE2 flycapture)

add_subdirectory("../../lib/shmem" "${CMAKE_CURRENT_BINARY_DIR}/shmem_build")

find_package (OpenCV REQUIRED)

add_executable (camserv CameraControl.cpp WebCam.cpp FileReader.cpp main.cpp )
target_link_libraries (camserv shmem ${OpenCV_LIBS} ${FLYCAPTURE2} ${Boost_LIBRARIES})
