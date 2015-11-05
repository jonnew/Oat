cmake_minimum_required (VERSION 2.8)

project (Oat)
set (Oat_VERSION_MAJOR 1)
set (Oat_VERSION_MINOR 0)
# TODO: Add commit SHA and reference it in header of recordings
set (Oat_VERSION_LIST "${Oat_VERSION_MAJOR}" ".${Oat_VERSION_MINOR}")

# Configure a header file to pass CMake settings to source code
configure_file (lib/utility/OatConfig.h.in "${PROJECT_BINARY_DIR}/OatConfig.h")
include_directories("${PROJECT_BINARY_DIR}") # Config.h ends up in BINARY_DIR

# C++11
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

# Add preprocessor definition to compile with OpenGL support
option (OAT_USE_FLYCAP "Compile with support for Point-Grey cameras" OFF)
option (OAT_USE_OPENGL "Use OpenGL functionality in OpenCV" OFF)
option (OAT_USE_CUDA "Use CUDA GPU functionality in OpenCV" OFF)

# Show options summary
message (STATUS "Oat version: " ${Oat_VERSION_LIST})
message (STATUS "Compilation options:" )
message (STATUS "  Compile with Point Grey Support: ${OAT_USE_FLYCAP}")
message (STATUS "  Compile with CUDA Support: ${OAT_USE_CUDA}")
message (STATUS "  Compile with OpenGL Support: ${OAT_USE_OPENGL}")

# Boost
find_package (Boost 1.53.0 REQUIRED system thread program_options filesystem)
include_directories (${Boost_INCLUDE_DIRS})
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


# Include dirs 
#set (COMMON_INCLUDES ${PROJECT_SOURCE_DIR}) 
set (EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)

# Include cpptoml
add_subdirectory (${EXT_PROJECTS_DIR}/cpptoml)
include_directories (${CPPTOML_INCLUDE_DIR})

#Include paths common to all components
#set (COMMON_INCLUDES
#     "${CPPTOML_INCLUDE_DIR}/cpptoml.h")

#message (STATUS "common include path: ${COMMON_INCLUDES}")

# Oat libshmem 
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/lib/shmem "${CMAKE_CURRENT_BINARY_DIR}/shmem")

# Oat components
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
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/positionsocket)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/calibrator)

# All executables should be installed in Oat/oat/libexec
set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../oat/libexec" CACHE PATH "Default install path" FORCE)

# Catch (and enables unit testing)
add_subdirectory(${EXT_PROJECTS_DIR}/catch)
include_directories(${CATCH_INCLUDE_DIR} ${COMMON_INCLUDES})
enable_testing(true)

set(TESTING_INCLUDES ${CATCH_INCLUDE_DIR} )
function(add_oat_test name)
  include_directories(${TESTING_INCLUDES} ${COMMON_INCLUDES})
  add_executable(${name}_test ${name}_test.cpp)
  add_test(${name}_test ${name}_test)
endfunction(add_oat_test)

# TODO: add this 
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test) 

# TODO: remove on merge into master
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/experiments/) 

# API documentation
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/doc")

# CPack Deb package
set (CPACK_GENERATOR "DEB")
set (CPACK_DEBIAN_PACKAGE_MAINTAINER "jonnew") #required
