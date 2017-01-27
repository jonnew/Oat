# Include the directory itself as a path to include directories
#set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-posidet_SOURCE
     PositionDetector.cpp
     Tuner.cpp
     DetectorFunc.cpp
     DifferenceDetector.cpp
     HSVDetector.cpp
     SimpleThreshold.cpp
     ArucoBoard.cpp
     main.cpp)

if (${EIGEN3_FOUND})
    # RPG pose estimation library
    add_subdirectory (./rpgposeest)
    set (oat-posidet_SOURCE ${oat-posidet_SOURCE} RPGPoseEst.cpp)
endif ()

# Target
add_executable (oat-posidet ${oat-posidet_SOURCE})
target_link_libraries (oat-posidet
                       oat-utility
                       oat-base
                       rpg-pose-est
                       ${OatCommon_LIBS})
add_dependencies (oat-posidet cpptoml rapidjson)

# Installation
install (TARGETS oat-posidet DESTINATION ../../oat/libexec COMPONENT oat-processors)
