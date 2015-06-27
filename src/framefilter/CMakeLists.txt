# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCE variable containing all required .cpp files:
if (${OAT_USE_CUDA})
    set (oat-framefilt_SOURCE 
             BackgroundSubtractorCUDA.cpp 
             FrameMasker.cpp 
             main.cpp)
else (${OAT_USE_CUDA})
    set (oat-framefilt_SOURCE 
             BackgroundSubtractor.cpp 
             FrameMasker.cpp 
             main.cpp)
endif (${OAT_USE_CUDA})

# Target
add_executable (oat-framefilt ${oat-framefilt_SOURCE})
target_link_libraries (oat-framefilt shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	
# Installation
install (TARGETS oat-framefilt DESTINATION ../../oat/libexec COMPONENT oat-processors)


