cmake_minimum_required (VERSION 2.8)
project (SimpleTracker)

# Put outputs into separate folders so they can be gitignored
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Version number
set (SimpleTracker_VERSION_MAJOR 1)
set (SimpleTracker_VERSION_MINOR 0)

# Configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/SimpleTrackerConfig.h.in"
  "${PROJECT_BINARY_DIR}/SimpleTrackerConfig.h"
  )

# Add the binary tree to the search path for include files
# so that we will find SimpleTrackerConfig.h
include_directories("${PROJECT_BINARY_DIR}")

include_directories ("/usr/include/flycapture")
find_library(FLYCAPTURE2 flycapture)

include_directories ("${PROJECT_SOURCE_DIR}/CameraControl")
add_subdirectory (CameraControl) 

include_directories ("${PROJECT_SOURCE_DIR}/Tracker")
add_subdirectory (Tracker) 

find_package (OpenCV REQUIRED)
add_executable (SimpleTracker SimpleTracker.cpp )
target_link_libraries (SimpleTracker CameraControl Tracker ${OpenCV_LIBS} ${FLYCAPTURE2})
