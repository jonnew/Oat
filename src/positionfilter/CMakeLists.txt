# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCES variable containing all required .cpp files:
set (oat-posifilt_SOURCE
     PositionFilter.cpp
     KalmanFilter2D.cpp
     HomographyTransform2D.cpp
     RegionFilter2D.cpp main.cpp)

# Target
add_executable (oat-posifilt ${oat-posifilt_SOURCE})
target_link_libraries (oat-posifilt
                       ${OpenCV_LIBS}
                       ${Boost_LIBRARIES}
                       ${Thread_LIBS})

# Installation
install (TARGETS oat-posifilt DESTINATION ../../oat/libexec COMPONENT oat-processors)
