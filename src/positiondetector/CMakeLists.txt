# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCE variable containing all required .cpp files:
if (${OAT_USE_CUDA})
    set (oat-posidet_SOURCE 
             DifferenceDetector.cpp 
             HSVDetectorCUDA.cpp
             main.cpp)
else (${OAT_USE_CUDA})
    set (oat-posidet_SOURCE 
             DifferenceDetector.cpp 
             HSVDetector.cpp
             main.cpp)
endif (${OAT_USE_CUDA})

# Target
add_executable (oat-posidet ${oat-posidet_SOURCE})
target_link_libraries (oat-posidet shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	
# Installation
install (TARGETS oat-posidet DESTINATION ../../oat/libexec COMPONENT oat-processors)
