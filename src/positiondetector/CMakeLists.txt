# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCES variable containing all required .cpp files:
set (oat-posidet_SOURCE DifferenceDetector2D.cpp HSVDetector2D.cpp main.cpp)

# Target
add_executable (oat-posidet ${oat-posidet_SOURCE})
target_link_libraries (oat-posidet shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	
# Installation
install (TARGETS oat-posidet DESTINATION ../../oat/libexec COMPONENT oat-processors)
