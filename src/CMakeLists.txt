cmake_minimum_required (VERSION 2.8)
project (SimpleTracker)

# Put outputs into separate folders so they can be gitignored
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Version number
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/SimpleTrackerConfig.h.in"
  "${PROJECT_BINARY_DIR}/SimpleTrackerConfig.h"
  )

# Add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

include_directories ("/usr/include/flycapture")
find_library(FLYCAPTURE2 flycapture)

include_directories ("${PROJECT_SOURCE_DIR}/CameraControl")
add_subdirectory (CameraControl) 

find_package (OpenCV REQUIRED)
add_executable (SimpleTracker SimpleTracker.cpp )
target_link_libraries (SimpleTracker CameraControl ${OpenCV_LIBS} ${FLYCAPTURE2})
