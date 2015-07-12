# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCE variable containing all required .cpp filesj
set (oat-framefilt_SOURCE 
     BackgroundSubtractor.cpp 
     FrameMasker.cpp 
     main.cpp)

# Target
add_executable (oat-framefilt ${oat-framefilt_SOURCE})
target_link_libraries (oat-framefilt shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	
# Installation
install (TARGETS oat-framefilt DESTINATION ../../oat/libexec COMPONENT oat-processors)


