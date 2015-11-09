# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-view_SOURCE Viewer.cpp main.cpp)

# Target
add_executable (oat-view ${oat-view_SOURCE})
target_link_libraries (oat-view ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Installation
install (TARGETS oat-view DESTINATION ../../oat/libexec COMPONENT oat-processors)
