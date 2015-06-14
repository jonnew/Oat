# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCE variable containing all required .cpp files:
set (oat-decorate_SOURCE Decorator.cpp main.cpp)

# Target
add_executable (oat-decorate ${oat-decorate_SOURCE})
target_link_libraries (oat-decorate shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	
# Installation
install (TARGETS oat-decorate DESTINATION ../../oat/libexec COMPONENT oat-processors)
