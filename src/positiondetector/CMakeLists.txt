# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCES variable containing all required .cpp files:
set (oat-detect_SOURCE DifferenceDetector2D.cpp HSVDetector2D.cpp main.cpp)

# Target
add_executable (oat-detect ${oat-detect_SOURCE})
target_link_libraries (oat-detect shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	
# Installation
install (TARGETS oat-detect DESTINATION ../../oat/libexec COMPONENT oat-processors)
