# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCES variable containing all required .cpp files:
set (oat-positest_SOURCE RandomAccel2D.cpp main.cpp)

# Target
add_executable (oat-positest ${oat-positest_SOURCE})
target_link_libraries (oat-positest shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	
# Installation
install (TARGETS oat-positest DESTINATION ../../oat/libexec COMPONENT oat-processors)
