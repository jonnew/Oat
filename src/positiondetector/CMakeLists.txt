# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-posidet_SOURCE
     PositionDetector.cpp
     DetectorFunc.cpp
     DifferenceDetector.cpp
     HSVDetector.cpp
     main.cpp)

# Target
add_executable (oat-posidet ${oat-posidet_SOURCE})
target_link_libraries (oat-posidet shmem ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Installation
install (TARGETS oat-posidet DESTINATION ../../oat/libexec COMPONENT oat-processors)
