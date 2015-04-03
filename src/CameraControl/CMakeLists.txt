cmake_minimum_required (VERSION 2.8)
project (GigECameraServe)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

include_directories ("/usr/include/flycapture")
find_library(FLYCAPTURE2 flycapture)

find_package (OpenCV REQUIRED)
add_executable (ge_camserv main.cpp )
target_link_libraries (ge_camserv CameraControl ${OpenCV_LIBS} ${FLYCAPTURE2})
