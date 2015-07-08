cmake_minimum_required (VERSION 2.8)

project (Oat)
set (Oat_VERSION_MAJOR 0)
set (Oat_VERSION_MINOR 1)

# Configure a header file to pass CMake settings to source code
configure_file (
  lib/utility/OatConfig.h.in
  "${PROJECT_BINARY_DIR}/OatConfig.h"
  )

# Add the binary tree to the search path for include files
# so that we will find OatConfig.h
include_directories("${PROJECT_BINARY_DIR}")
 
# C++11
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# Add preprossor defintion to compile with OpenGL support
option (OAT_USE_FLYCAP "Compile with support for Point-Grey cameras" ON)
option (OAT_USE_OPENGL "Use OpenGL functionality in OpenCV" OFF)
option (OAT_USE_CUDA "Use CUDA GPU functionality in OpenCV" OFF)

# Boost
set (BOOST_ROOT /opt/boost_1_57_0 )
find_package (Boost REQUIRED system thread program_options filesystem)
link_directories (${Boost_LIBRARY_DIR})

# OpenCV
find_package (OpenCV REQUIRED)
if (OpenCV_FOUND)
    message (STATUS "Found OpenCV")
else (OpenCV_FOUND)
    message (CRITICAL_ERROR "OpenCV not found")
endif (OpenCV_FOUND)

# Flycapture
if (${OAT_USE_FLYCAP})
    # Required for point-grey cameras
    # TODO: super fragile. Need to provide a user flag to specify flycapture dir
    include_directories ("/usr/include/flycapture")
    find_library(FLYCAPTURE2 flycapture)
    
    if (FLYCAPTURE2)
         message (STATUS "Found Flycapture.")
     else (FLYCAPTURE2)
         message (CRITICAL_ERROR "Flycapture not found. 
            Did you install in /usr/include/flycapture?")
     endif (FLYCAPTURE2)
endif (${OAT_USE_FLYCAP})

# oat shared memory lib
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/lib/shmem 
				 "${CMAKE_CURRENT_BINARY_DIR}/shmem")

# oat components
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/cleaner)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/decorator)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/framefilter)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/frameserver)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/frameviewer)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/positioncombiner)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/positiondetector)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/positionfilter)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/positiontester)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/recorder)

# oat API documentation
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/doc")

# All executables should be installed in oat/libexec
set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../oat/libexec" CACHE PATH "Default install path" FORCE)
