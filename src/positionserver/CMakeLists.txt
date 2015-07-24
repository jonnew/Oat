# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCES variable containing all required .cpp files:
set (oat-posiserve_SOURCE PositionUDPServer.cpp main.cpp)

# Target
add_executable (oat-posiserve ${oat-posiserve_SOURCE})
target_link_libraries (oat-posiserve shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	
# Installation
install (TARGETS oat-posiserve DESTINATION ../../oat/libexec COMPONENT oat-processors)
