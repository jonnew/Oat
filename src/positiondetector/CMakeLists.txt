# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-posidet_SOURCE
     Aruco.cpp
     PositionDetector.cpp
     DetectorFunc.cpp
     DifferenceDetector.cpp
     HSVDetector.cpp
     SimpleThreshold.cpp
     main.cpp)

# Target
add_executable (oat-posidet ${oat-posidet_SOURCE})
target_link_libraries (oat-posidet ${OatCommon_LIBS})

# Installation
install (TARGETS oat-posidet DESTINATION ../../oat/libexec COMPONENT oat-processors)
