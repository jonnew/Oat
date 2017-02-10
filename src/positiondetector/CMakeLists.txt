# Include the directory itself as a path to include directories
#set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-posidet_SOURCE
     PositionDetector.cpp
     Tuner.cpp
     #DetectorFunc.cpp
     #DifferenceDetector.cpp
     #HSVDetector.cpp
     #SimpleThreshold.cpp
     main.cpp)

set (oat-posidet_LIBS
     oat-posidet
     oat-utility
     oat-base
     oat-types
     ${OatCommon_LIBS})

# Aruco requires opencv contrib modules
include (CheckIncludeFileCXX)
check_include_file_cxx (opencv2/aruco.hpp ARUCO_FOUND)
if (${ARUCO_FOUND})
    set (oat-posidet_SOURCE ${oat-posidet_SOURCE} ArucoBoard.cpp)
endif ()

# RPG pose estimation requires Eigen3
if (${EIGEN3_FOUND})
    #add_subdirectory (./rpgposeest)
    #set (oat-posidet_SOURCE ${oat-posidet_SOURCE} RPGPoseEst.cpp)
    #set (oat-posidet_LIBS ${oat-posidet_LIBS} rpg-pose-est)
endif ()

# Target
add_executable (oat-posidet ${oat-posidet_SOURCE})
target_link_libraries (${oat-posidet_LIBS})
add_dependencies (oat-posidet cpptoml rapidjson)

# Installation
install (TARGETS oat-posidet DESTINATION ../../oat/libexec COMPONENT oat-processors)
