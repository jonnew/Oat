# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCE variable containing all required .cpp files:
set (oat-framefilt_SOURCE 
	 BackgroundSubtractor.cpp 
	 #BackgroundSubtractorCUDA.cpp 
	 FrameMasker.cpp 
	 main.cpp)

# Target
add_executable (oat-framefilt ${oat-framefilt_SOURCE})
target_link_libraries (oat-framefilt shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	
# Installation
install (TARGETS oat-framefilt DESTINATION ../../oat/libexec COMPONENT oat-processors)


